"""Bounded mappings from rendered history to previously served token IDs."""

from __future__ import annotations

from array import array
import asyncio
import base64
import binascii
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import logging
import re
import string
import sys
import time
from typing import Sequence
import zlib

logger = logging.getLogger(__name__)

COMPACT_PREFIX_VERSION = "sha256-u32be-v1"
COMPACT_PREFIX_MAX_CANDIDATES = 1024
COMPACT_PREFIX_MAX_REPLACEMENT_IDS = 262_144
COMPACT_PREFIX_RECORD_LIMIT = 4 * 1024 * 1024
COMPACT_PREFIX_RESPONSE_LIMIT = 8 * 1024 * 1024
COMPACT_PREFIX_HEADER_LIMIT = 6 * 1024
COMPACT_PREFIX_HEADER_DECODE_LIMIT = 64 * 1024
# Keep the original wire domain so deployed prefix stores remain compatible.
_DIGEST_DOMAIN = b"caladan-token-prefix\0" + COMPACT_PREFIX_VERSION.encode() + b"\0"
_MAX_PREFIX_TOKENS = 262_144
_TOKEN_DIGEST = re.compile(r"[0-9a-fA-F]{64}")
_SharedPrefixEntry = tuple[str, list[int], list[int], str, tuple["PrefixEdit", ...]]


@dataclass(frozen=True, slots=True)
class PrefixEdit:
    start: int
    stop: int
    replacement: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CompactPrefixCandidate:
    rendered_length: int
    rendered_digest: str
    raw_digest: str
    edits: tuple[PrefixEdit, ...]


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    rendered_length: int
    raw_prefix: tuple[int, ...]
    edits: tuple[PrefixEdit, ...] = field(default=(), compare=False)


def _token_bytes(token_ids: Sequence[int]) -> bytes:
    # Engine and wire boundaries validate bool/type explicitly. Let the C array
    # conversion enforce the numeric/range constraint without another O(n)
    # Python scan on every 128K lookup.
    try:
        values = array("I", token_ids)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("token IDs must be unsigned 32-bit integers") from error
    if values.itemsize != 4:
        raise RuntimeError("this platform does not provide 32-bit unsigned integers")
    if sys.byteorder == "little":
        values.byteswap()
    return values.tobytes()


def token_prefix_digests(
    token_ids: Sequence[int], lengths: Sequence[int]
) -> dict[int, str]:
    """Hash several token prefixes in one pass over a fixed-width encoding."""
    requested = sorted(set(lengths))
    if any(
        not isinstance(length, int)
        or isinstance(length, bool)
        or not 0 <= length <= len(token_ids)
        for length in requested
    ):
        raise ValueError("invalid token prefix length")
    if not requested:
        return {}
    encoded = _token_bytes(token_ids[: requested[-1]])
    digest = hashlib.sha256(_DIGEST_DOMAIN)
    previous = 0
    result: dict[int, str] = {}
    for length in requested:
        digest.update(encoded[previous * 4 : length * 4])
        result[length] = digest.hexdigest()
        previous = length
    return result


def token_digest(token_ids: Sequence[int]) -> str:
    return token_prefix_digests(token_ids, [len(token_ids)])[len(token_ids)]


def apply_prefix_edits(
    rendered_prefix: Sequence[int], edits: Sequence[PrefixEdit]
) -> list[int]:
    result: list[int] = []
    cursor = 0
    for edit in edits:
        if (
            not 0 <= edit.start <= edit.stop <= len(rendered_prefix)
            or edit.start < cursor
        ):
            raise ValueError("token-prefix edits must be ordered and non-overlapping")
        result.extend(rendered_prefix[cursor : edit.start])
        result.extend(edit.replacement)
        cursor = edit.stop
    result.extend(rendered_prefix[cursor:])
    return result


def prefix_edits(
    rendered_prefix: Sequence[int], raw_prefix: Sequence[int]
) -> tuple[PrefixEdit, ...]:
    """Return an exact, deterministic edit script without nonlinear alignment."""
    if rendered_prefix == raw_prefix:
        return ()
    if len(rendered_prefix) == len(raw_prefix):
        edits: list[PrefixEdit] = []
        start: int | None = None
        for index, (rendered, raw) in enumerate(
            zip(rendered_prefix, raw_prefix, strict=True)
        ):
            if rendered != raw and start is None:
                start = index
            elif rendered == raw and start is not None:
                edits.append(PrefixEdit(start, index, tuple(raw_prefix[start:index])))
                start = None
        if start is not None:
            edits.append(PrefixEdit(start, len(raw_prefix), tuple(raw_prefix[start:])))
        return tuple(edits)

    start = 0
    limit = min(len(rendered_prefix), len(raw_prefix))
    while start < limit and rendered_prefix[start] == raw_prefix[start]:
        start += 1
    suffix = 0
    while (
        suffix < len(rendered_prefix) - start
        and suffix < len(raw_prefix) - start
        and rendered_prefix[-1 - suffix] == raw_prefix[-1 - suffix]
    ):
        suffix += 1
    raw_stop = len(raw_prefix) - suffix
    return (
        PrefixEdit(
            start,
            len(rendered_prefix) - suffix,
            tuple(raw_prefix[start:raw_stop]),
        ),
    )


def compact_prefix_candidate(
    rendered_prefix: Sequence[int],
    raw_prefix: Sequence[int],
    edits: Sequence[PrefixEdit] | None = None,
) -> CompactPrefixCandidate:
    exact_edits = tuple(
        prefix_edits(rendered_prefix, raw_prefix) if edits is None else edits
    )
    if apply_prefix_edits(rendered_prefix, exact_edits) != list(raw_prefix):
        raise ValueError("token-prefix edits do not reconstruct the raw prefix")
    return CompactPrefixCandidate(
        rendered_length=len(rendered_prefix),
        rendered_digest=token_digest(rendered_prefix),
        raw_digest=token_digest(raw_prefix),
        edits=exact_edits,
    )


def resolve_compact_prefix(
    rendered: Sequence[int], candidates: Sequence[CompactPrefixCandidate]
) -> PrefixMatch | None:
    eligible = sorted(
        (
            candidate
            for candidate in candidates
            if 0 < candidate.rendered_length <= len(rendered)
        ),
        key=lambda candidate: candidate.rendered_length,
        reverse=True,
    )
    if not eligible:
        return None
    digests = token_prefix_digests(
        rendered, [candidate.rendered_length for candidate in eligible]
    )
    for candidate in eligible:
        if digests[candidate.rendered_length] != candidate.rendered_digest:
            continue
        prefix = rendered[: candidate.rendered_length]
        raw = apply_prefix_edits(prefix, candidate.edits)
        if token_digest(raw) != candidate.raw_digest:
            continue
        return PrefixMatch(
            rendered_length=candidate.rendered_length,
            raw_prefix=tuple(raw),
            edits=candidate.edits,
        )
    return None


def compact_candidate_payload(
    candidate: CompactPrefixCandidate,
) -> dict[str, object]:
    return {
        "rendered_length": candidate.rendered_length,
        "rendered_digest": candidate.rendered_digest,
        "raw_digest": candidate.raw_digest,
        "edits": [
            [edit.start, edit.stop, list(edit.replacement)] for edit in candidate.edits
        ],
    }


def compact_candidate_from_payload(value: object) -> CompactPrefixCandidate:
    if not isinstance(value, dict):
        raise ValueError("token-prefix candidate must be an object")
    payload: dict[str, object] = {
        key: item for key, item in value.items() if isinstance(key, str)
    }
    if len(payload) != len(value):
        raise ValueError("token-prefix candidate keys must be strings")
    rendered_length = payload.get("rendered_length")
    rendered_digest = payload.get("rendered_digest")
    raw_digest = payload.get("raw_digest")
    raw_edits = payload.get("edits")
    if (
        not isinstance(rendered_length, int)
        or isinstance(rendered_length, bool)
        or not 0 < rendered_length <= _MAX_PREFIX_TOKENS
        or not isinstance(rendered_digest, str)
        or not isinstance(raw_digest, str)
        or (
            (
                _TOKEN_DIGEST.fullmatch(rendered_digest) is None
                or _TOKEN_DIGEST.fullmatch(raw_digest) is None
            )
            if type(rendered_digest) is str and type(raw_digest) is str
            else any(
                len(digest) != 64
                or any(character not in string.hexdigits for character in digest)
                for digest in (rendered_digest, raw_digest)
            )
        )
        or not isinstance(raw_edits, list)
        or len(raw_edits) > 256
    ):
        raise ValueError("invalid token-prefix candidate")
    edits: list[PrefixEdit] = []
    cursor = 0
    replacement_count = 0
    for raw_edit in raw_edits:
        if not isinstance(raw_edit, list) or len(raw_edit) != 3:
            raise ValueError("invalid token-prefix edit")
        start = raw_edit[0]
        stop = raw_edit[1]
        replacement = raw_edit[2]
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
            or not isinstance(replacement, list)
        ):
            raise ValueError("invalid token-prefix edit")
        replacement_ids: list[int] = []
        for token_id in replacement:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError("invalid token-prefix edit")
            replacement_ids.append(token_id)
        if not cursor <= start <= stop <= rendered_length:
            raise ValueError("token-prefix edits must be ordered and non-overlapping")
        _token_bytes(replacement_ids)
        replacement_count += len(replacement_ids)
        if replacement_count > COMPACT_PREFIX_MAX_REPLACEMENT_IDS:
            raise ValueError("token-prefix candidate has too many replacement tokens")
        edits.append(PrefixEdit(start, stop, tuple(replacement_ids)))
        cursor = stop
    return CompactPrefixCandidate(
        rendered_length=rendered_length,
        rendered_digest=rendered_digest.lower(),
        raw_digest=raw_digest.lower(),
        edits=tuple(edits),
    )


def compact_candidates_header(
    candidates: Sequence[CompactPrefixCandidate],
) -> str | None:
    """Encode a complete candidate snapshot, or decline unsafe header growth."""
    prefix = f'{{"version":"{COMPACT_PREFIX_VERSION}","candidates":['
    suffix = "]}"
    parts = [prefix]
    size = len(prefix) + len(suffix)
    for candidate in candidates:
        encoded = json.dumps(
            compact_candidate_payload(candidate), separators=(",", ":")
        )
        separator = int(len(parts) > 1)
        size += separator + len(encoded)
        if size > COMPACT_PREFIX_HEADER_DECODE_LIMIT:
            return None
        if separator:
            parts.append(",")
        parts.append(encoded)
    parts.append(suffix)
    encoded = base64.urlsafe_b64encode(
        gzip.compress("".join(parts).encode(), compresslevel=1)
    ).decode()
    return encoded if len(encoded) <= COMPACT_PREFIX_HEADER_LIMIT else None


def compact_candidates_from_header(
    value: str,
) -> tuple[CompactPrefixCandidate, ...]:
    if len(value) > COMPACT_PREFIX_HEADER_LIMIT:
        raise ValueError("token-prefix candidate header is too large")
    try:
        compressed = base64.b64decode(value, altchars=b"-_", validate=True)
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        decoded = decompressor.decompress(
            compressed, COMPACT_PREFIX_HEADER_DECODE_LIMIT + 1
        )
        if (
            len(decoded) > COMPACT_PREFIX_HEADER_DECODE_LIMIT
            or decompressor.unconsumed_tail
        ):
            raise ValueError("token-prefix candidate header expands beyond its limit")
        decoded += decompressor.flush(
            COMPACT_PREFIX_HEADER_DECODE_LIMIT + 1 - len(decoded)
        )
        if not decompressor.eof or decompressor.unused_data:
            raise ValueError("token-prefix candidate header is malformed")
        payload = json.loads(decoded)
    except (
        binascii.Error,
        json.JSONDecodeError,
        UnicodeDecodeError,
        zlib.error,
    ) as error:
        raise ValueError("token-prefix candidate header is malformed") from error
    if (
        not isinstance(payload, dict)
        or payload.get("version") != COMPACT_PREFIX_VERSION
        or not isinstance(payload.get("candidates"), list)
        or len(payload["candidates"]) > COMPACT_PREFIX_MAX_CANDIDATES
    ):
        raise ValueError("token-prefix candidate header is malformed")
    return tuple(
        compact_candidate_from_payload(candidate) for candidate in payload["candidates"]
    )


class _Edge:
    __slots__ = ("label", "child")

    def __init__(self, label: tuple[int, ...], child: _Node) -> None:
        self.label = label
        self.child = child


@dataclass(slots=True)
class _Variant:
    lineages: OrderedDict[str, int]
    edits: tuple[PrefixEdit, ...]


class _Node:
    __slots__ = ("children", "parent", "parent_token", "rendered_length", "variants")

    def __init__(
        self,
        parent: _Node | None = None,
        parent_token: int | None = None,
        rendered_length: int = 0,
    ) -> None:
        self.children: dict[int, _Edge] = {}
        self.parent = parent
        self.parent_token = parent_token
        self.rendered_length = rendered_length
        self.variants: dict[tuple[int, ...], _Variant] = {}


def _common_prefix_length(
    values: Sequence[int], start: int, label: tuple[int, ...]
) -> int:
    limit = min(len(values) - start, len(label))
    index = 0
    while index < limit and values[start + index] == label[index]:
        index += 1
    return index


class TokenPrefixCache:
    """A bounded radix trie retaining distinct observed raw tokenizations."""

    def __init__(
        self,
        *,
        max_variants: int = 4096,
        max_token_ids: int = 2_000_000,
        max_lineages_per_variant: int = 64,
    ) -> None:
        if min(max_variants, max_token_ids, max_lineages_per_variant) <= 0:
            raise ValueError("token-prefix cache bounds must be positive")
        self._root = _Node()
        self._lru: OrderedDict[_Node, int] = OrderedDict()
        self._variants = 0
        self._max_variants = max_variants
        self._max_token_ids = max_token_ids
        self._max_lineages_per_variant = max_lineages_per_variant
        self._token_ids = 0
        self._observation = 0

    def _matching_nodes(self, rendered_tokens: Sequence[int]) -> list[_Node]:
        node = self._root
        index = 0
        candidates: list[_Node] = []
        while index < len(rendered_tokens):
            edge = node.children.get(rendered_tokens[index])
            if edge is None:
                break
            matched = _common_prefix_length(rendered_tokens, index, edge.label)
            if matched != len(edge.label):
                break
            index += matched
            node = edge.child
            if node.variants:
                candidates.append(node)
        return candidates

    def lookup(
        self, rendered_tokens: Sequence[int], lineage: str | None
    ) -> PrefixMatch | None:
        for candidate in reversed(self._matching_nodes(rendered_tokens)):
            variants = candidate.variants
            # Without a rollout identifier, never choose between distinct raw
            # tokenizations of the same rendered history.
            if lineage is None:
                if len(variants) == 1:
                    raw, variant = next(iter(variants.items()))
                    self._lru.move_to_end(candidate)
                    return PrefixMatch(candidate.rendered_length, raw, variant.edits)
                continue
            matching = [
                (variant.lineages[lineage], raw, variant.edits)
                for raw, variant in variants.items()
                if lineage in variant.lineages
            ]
            if not matching:
                continue
            _, raw_prefix, edits = max(matching, key=lambda value: value[0])
            self._lru.move_to_end(candidate)
            lineages = variants[raw_prefix].lineages
            lineages.move_to_end(lineage)
            return PrefixMatch(candidate.rendered_length, raw_prefix, edits)
        return None

    def insert(
        self,
        rendered_prefix: Sequence[int],
        raw_prefix: Sequence[int],
        lineage: str,
        edits: Sequence[PrefixEdit] | None = None,
    ) -> bool:
        if not rendered_prefix or not raw_prefix or not lineage:
            return False
        raw = tuple(raw_prefix)
        cost = len(rendered_prefix) + len(raw)
        if cost > self._max_token_ids:
            return False
        node = self._root
        index = 0
        while index < len(rendered_prefix):
            token = rendered_prefix[index]
            edge = node.children.get(token)
            if edge is None:
                child = _Node(
                    parent=node,
                    parent_token=token,
                    rendered_length=len(rendered_prefix),
                )
                node.children[token] = _Edge(tuple(rendered_prefix[index:]), child)
                node = child
                break

            matched = _common_prefix_length(rendered_prefix, index, edge.label)
            if matched == len(edge.label):
                index += matched
                node = edge.child
                continue

            middle = _Node(
                parent=node,
                parent_token=token,
                rendered_length=index + matched,
            )
            old_suffix = edge.label[matched:]
            old_child = edge.child
            old_child.parent = middle
            old_child.parent_token = old_suffix[0]
            middle.children[old_suffix[0]] = _Edge(old_suffix, old_child)
            edge.label = edge.label[:matched]
            edge.child = middle
            node = middle
            index += matched
            if index < len(rendered_prefix):
                new_token = rendered_prefix[index]
                child = _Node(
                    parent=node,
                    parent_token=new_token,
                    rendered_length=len(rendered_prefix),
                )
                node.children[new_token] = _Edge(tuple(rendered_prefix[index:]), child)
                node = child
            break

        exact_edits = tuple(
            prefix_edits(rendered_prefix, raw_prefix) if edits is None else edits
        )
        if apply_prefix_edits(rendered_prefix, exact_edits) != list(raw_prefix):
            raise ValueError("token-prefix edits do not reconstruct the raw prefix")
        variant = node.variants.get(raw)
        if variant is None:
            variant = node.variants[raw] = _Variant(OrderedDict(), exact_edits)
            self._lru[node] = self._lru.get(node, 0) + cost
            self._variants += 1
            self._token_ids += cost
        else:
            variant.edits = exact_edits
            self._lru.move_to_end(node)
        lineages = variant.lineages
        self._observation += 1
        lineages[lineage] = self._observation
        lineages.move_to_end(lineage)
        while len(lineages) > self._max_lineages_per_variant:
            lineages.popitem(last=False)
        if node in self._lru:
            self._lru.move_to_end(node)
        self._evict()
        return True

    def _evict(self) -> None:
        while self._lru and (
            self._variants > self._max_variants or self._token_ids > self._max_token_ids
        ):
            self.evict_oldest()

    @property
    def token_ids(self) -> int:
        return self._token_ids

    def evict_oldest(self) -> bool:
        if not self._lru:
            return False
        node, cost = self._lru.popitem(last=False)
        self._token_ids -= cost
        self._variants -= len(node.variants)
        node.variants.clear()
        self._prune(node)
        return True

    def _prune(self, node: _Node) -> None:
        while node.parent is not None:
            parent = node.parent
            token = node.parent_token
            assert token is not None
            if not node.variants and not node.children:
                del parent.children[token]
                node = parent
                continue
            if not node.variants and len(node.children) == 1:
                child_edge = next(iter(node.children.values()))
                parent_edge = parent.children[token]
                parent_edge.label += child_edge.label
                parent_edge.child = child_edge.child
                child_edge.child.parent = parent
                child_edge.child.parent_token = token
                node = parent
                continue
            break


class TokenPrefixStore:
    """LRU-bounded collection of model/tokenizer-scoped prefix tries."""

    def __init__(
        self,
        *,
        max_scopes: int = 64,
        max_variants_per_scope: int = 4096,
        max_token_ids_per_scope: int = 2_000_000,
        max_token_ids: int = 4_000_000,
    ) -> None:
        if min(max_scopes, max_token_ids) <= 0:
            raise ValueError("token-prefix store bounds must be positive")
        self._max_scopes = max_scopes
        self._max_variants = max_variants_per_scope
        self._max_token_ids = max_token_ids_per_scope
        self._max_total_token_ids = max_token_ids
        self._token_ids = 0
        self._scopes: OrderedDict[str, TokenPrefixCache] = OrderedDict()

    def _cache(self, scope: str) -> TokenPrefixCache:
        cache = self._scopes.get(scope)
        if cache is None:
            cache = TokenPrefixCache(
                max_variants=self._max_variants,
                max_token_ids=self._max_token_ids,
            )
            self._scopes[scope] = cache
            while len(self._scopes) > self._max_scopes:
                _, removed = self._scopes.popitem(last=False)
                self._token_ids -= removed.token_ids
        else:
            self._scopes.move_to_end(scope)
        return cache

    def lookup(
        self, scope: str, rendered_tokens: Sequence[int], lineage: str | None
    ) -> PrefixMatch | None:
        cache = self._scopes.get(scope)
        if cache is None:
            return None
        self._scopes.move_to_end(scope)
        return cache.lookup(rendered_tokens, lineage)

    def insert(
        self,
        scope: str,
        rendered_prefix: Sequence[int],
        raw_prefix: Sequence[int],
        lineage: str,
        edits: Sequence[PrefixEdit] | None = None,
    ) -> bool:
        cache = self._cache(scope)
        before = cache.token_ids
        inserted = cache.insert(rendered_prefix, raw_prefix, lineage, edits)
        self._token_ids += cache.token_ids - before
        while self._token_ids > self._max_total_token_ids and self._scopes:
            oldest_scope, oldest = next(iter(self._scopes.items()))
            before = oldest.token_ids
            if not oldest.evict_oldest():
                self._scopes.pop(oldest_scope)
                continue
            self._token_ids -= before - oldest.token_ids
            if oldest.token_ids == 0:
                self._scopes.pop(oldest_scope)
        return inserted


class CompactPrefixStore:
    """Bounded shared snapshots keyed by scope and logical rollout lineage."""

    def __init__(
        self,
        *,
        max_candidates: int = 262_144,
        max_candidates_per_lineage: int = COMPACT_PREFIX_MAX_CANDIDATES,
        max_edit_token_ids: int = 1_000_000,
        max_serialized_bytes_per_lineage: int = COMPACT_PREFIX_RESPONSE_LIMIT - 1024,
        max_staged_attempts: int = 262_144,
        staged_ttl: float = 300,
    ) -> None:
        if (
            min(
                max_candidates,
                max_candidates_per_lineage,
                max_edit_token_ids,
                max_serialized_bytes_per_lineage,
                max_staged_attempts,
                staged_ttl,
            )
            <= 0
        ):
            raise ValueError("compact token-prefix store bounds must be positive")
        self._max_candidates = max_candidates
        self._max_per_lineage = max_candidates_per_lineage
        self._max_edit_token_ids = max_edit_token_ids
        self._max_serialized_bytes_per_lineage = max_serialized_bytes_per_lineage
        self._edit_token_ids = 0
        self._entries: OrderedDict[
            tuple[str, str, int, str], tuple[CompactPrefixCandidate, int, int]
        ] = OrderedDict()
        self._lineages: dict[
            tuple[str, str], OrderedDict[tuple[str, str, int, str], None]
        ] = {}
        self._lineage_bytes: dict[tuple[str, str], int] = {}
        self._max_staged_attempts = max_staged_attempts
        self._staged_ttl = staged_ttl
        self._staged: OrderedDict[
            str,
            tuple[
                float,
                list[tuple[str, str, CompactPrefixCandidate]],
                int,
            ],
        ] = OrderedDict()
        self._staged_candidates = 0
        self._staged_edit_token_ids = 0
        self._staged_evictions = 0
        self._missing_commits = 0

    @property
    def staged_evictions(self) -> int:
        return self._staged_evictions

    @property
    def missing_commits(self) -> int:
        return self._missing_commits

    def insert(
        self,
        scope: str,
        lineage: str,
        candidate: CompactPrefixCandidate,
    ) -> bool:
        cost = sum(len(edit.replacement) for edit in candidate.edits)
        serialized = (
            len(
                json.dumps(
                    compact_candidate_payload(candidate), separators=(",", ":")
                ).encode()
            )
            + 1
        )
        if (
            cost > self._max_edit_token_ids
            or serialized > self._max_serialized_bytes_per_lineage
        ):
            return False
        key = (
            scope,
            lineage,
            candidate.rendered_length,
            candidate.rendered_digest,
        )
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._edit_token_ids -= previous[1]
            self._lineage_bytes[(scope, lineage)] -= previous[2]
        self._entries[key] = (candidate, cost, serialized)
        self._edit_token_ids += cost
        lineage_key = (scope, lineage)
        self._lineage_bytes[lineage_key] = (
            self._lineage_bytes.get(lineage_key, 0) + serialized
        )
        entries = self._lineages.setdefault(lineage_key, OrderedDict())
        entries.pop(key, None)
        entries[key] = None
        while (
            len(entries) > self._max_per_lineage
            or self._lineage_bytes[lineage_key] > self._max_serialized_bytes_per_lineage
        ):
            self._remove(next(iter(entries)))
        while (
            len(self._entries) > self._max_candidates
            or self._edit_token_ids > self._max_edit_token_ids
        ):
            self._remove(next(iter(self._entries)))
        return key in self._entries

    def candidates(
        self, scope: str, lineage: str, max_rendered_length: int
    ) -> list[CompactPrefixCandidate]:
        entries = self._lineages.get((scope, lineage))
        if entries is None:
            return []
        ordered = [
            self._entries[key][0]
            for key in reversed(entries)
            if key in self._entries
            and self._entries[key][0].rendered_length <= max_rendered_length
        ]
        # Python's sort is stable, so equal-length variants remain newest first.
        ordered.sort(key=lambda candidate: candidate.rendered_length, reverse=True)
        return ordered

    def stage(
        self,
        attempt: str,
        observations: Sequence[tuple[str, str, CompactPrefixCandidate]],
    ) -> bool:
        """Retain observations invisibly until their upstream attempt commits."""
        now = time.monotonic()
        self._expire_staged(now)
        existing = self._staged.pop(attempt, None)
        entries = [] if existing is None else existing[1]
        cost = 0 if existing is None else existing[2]
        if existing is not None:
            self._staged_candidates -= len(entries)
            self._staged_edit_token_ids -= cost
        combined = OrderedDict(
            (
                (
                    scope,
                    lineage,
                    candidate.rendered_length,
                    candidate.rendered_digest,
                ),
                (scope, lineage, candidate),
            )
            for scope, lineage, candidate in entries
        )
        for scope, lineage, candidate in observations:
            candidate_cost = sum(len(edit.replacement) for edit in candidate.edits)
            if candidate_cost <= self._max_edit_token_ids:
                key = (
                    scope,
                    lineage,
                    candidate.rendered_length,
                    candidate.rendered_digest,
                )
                combined.pop(key, None)
                combined[key] = (scope, lineage, candidate)
        entries = list(combined.values())
        cost = sum(
            len(edit.replacement)
            for _, _, candidate in entries
            for edit in candidate.edits
        )
        if entries:
            self._staged[attempt] = (now, entries, cost)
            self._staged_candidates += len(entries)
            self._staged_edit_token_ids += cost
        while self._staged and (
            len(self._staged) > self._max_staged_attempts
            or self._staged_candidates > self._max_candidates
            or self._staged_edit_token_ids > self._max_edit_token_ids
        ):
            self._remove_staged(next(iter(self._staged)))
            self._staged_evictions += 1
        return attempt in self._staged

    def commit(self, attempt: str) -> bool:
        """Publish only observations produced by the selected upstream attempt."""
        self._expire_staged(time.monotonic())
        staged = self._staged.get(attempt)
        if staged is None:
            self._missing_commits += 1
            return False
        observations = list(staged[1])
        self._remove_staged(attempt)
        inserted = [
            self.insert(scope, lineage, candidate)
            for scope, lineage, candidate in observations
        ]
        return all(inserted)

    def _expire_staged(self, now: float) -> None:
        while self._staged:
            attempt, (created, _, _) = next(iter(self._staged.items()))
            if now - created < self._staged_ttl:
                break
            self._remove_staged(attempt)

    def _remove_staged(self, attempt: str) -> None:
        removed = self._staged.pop(attempt, None)
        if removed is None:
            return
        self._staged_candidates -= len(removed[1])
        self._staged_edit_token_ids -= removed[2]

    def _remove(self, key: tuple[str, str, int, str]) -> None:
        removed = self._entries.pop(key, None)
        if removed is None:
            return
        self._edit_token_ids -= removed[1]
        lineage_key = key[:2]
        self._lineage_bytes[lineage_key] -= removed[2]
        entries = self._lineages[lineage_key]
        entries.pop(key, None)
        if not entries:
            del self._lineages[lineage_key]
            del self._lineage_bytes[lineage_key]


class TokenPrefixRuntime:
    """Local hot cache with shared lookup and batched replication."""

    def __init__(
        self,
        *,
        lookup_shared: Callable[[str, list[int], str], Awaitable[PrefixMatch | None]],
        insert_shared_many: Callable[[list[_SharedPrefixEntry]], Awaitable[None]],
        max_pending_batches: int = 32,
    ) -> None:
        if max_pending_batches <= 0:
            raise ValueError("replication bound must be positive")
        self._local = TokenPrefixStore()
        self._lookup_shared = lookup_shared
        self._insert_shared_many = insert_shared_many
        self._pending: asyncio.Queue[list[_SharedPrefixEntry]] = asyncio.Queue(
            max_pending_batches
        )
        self._replicator: asyncio.Task[None] | None = None
        self._closed = False

    async def rewrite_with_edits(
        self,
        scope: str,
        rendered_tokens: Sequence[int],
        lineage: str,
        *,
        fallback_lineage: str | None = None,
        shared_candidate: bool,
        shared_match: PrefixMatch | None = None,
        shared_resolved: bool = False,
    ) -> tuple[list[int], list[int], tuple[PrefixEdit, ...]]:
        """Return canonical/input tokens plus their exact sparse substitution."""
        canonical = list(rendered_tokens)
        local_match = self._local.lookup(scope, canonical, lineage)
        if local_match is None and fallback_lineage is not None:
            local_match = self._local.lookup(scope, canonical, fallback_lineage)
            if local_match is not None:
                self._local.insert(
                    scope,
                    canonical[: local_match.rendered_length],
                    local_match.raw_prefix,
                    lineage,
                    local_match.edits,
                )
        if shared_candidate and not shared_resolved:
            shared_match = await self._lookup_shared(scope, canonical, lineage)
            if shared_match is None and fallback_lineage is not None:
                shared_match = await self._lookup_shared(
                    scope, canonical, fallback_lineage
                )
        if shared_match is not None and (
            local_match is None
            or shared_match.rendered_length > local_match.rendered_length
        ):
            exact_edits = shared_match.edits
            rendered_prefix = canonical[: shared_match.rendered_length]
            if not exact_edits and list(shared_match.raw_prefix) != rendered_prefix:
                exact_edits = prefix_edits(rendered_prefix, shared_match.raw_prefix)
            self._local.insert(
                scope,
                rendered_prefix,
                shared_match.raw_prefix,
                lineage,
                exact_edits,
            )
            match = PrefixMatch(
                shared_match.rendered_length,
                shared_match.raw_prefix,
                exact_edits,
            )
        else:
            match = local_match
        if match is None:
            return canonical, canonical.copy(), ()
        return (
            canonical,
            [*match.raw_prefix, *canonical[match.rendered_length :]],
            match.edits,
        )

    async def insert_many(
        self,
        entries: Sequence[_SharedPrefixEntry],
    ) -> None:
        normalized: list[_SharedPrefixEntry] = []
        for scope, rendered, raw, lineage, edits in entries:
            exact_edits = tuple(edits)
            self._local.insert(scope, rendered, raw, lineage, exact_edits)
            normalized.append((scope, rendered, raw, lineage, exact_edits))
        if not normalized or self._closed:
            return
        try:
            self._pending.put_nowait(normalized)
        except asyncio.QueueFull:
            # Let the already-scheduled replicator drain a simultaneous burst
            # once before dropping cross-replica work.
            self._ensure_replicator()
            await asyncio.sleep(0)
            if self._closed:
                return
            try:
                self._pending.put_nowait(normalized)
            except asyncio.QueueFull:
                logger.warning(
                    "shared token-prefix replication queue is full; "
                    "retaining local mappings only"
                )
                return
        self._ensure_replicator()

    def _ensure_replicator(self) -> None:
        if self._replicator is None or self._replicator.done():
            self._replicator = asyncio.create_task(
                self._replicate(), name="caladan-token-prefix-replication"
            )

    async def _replicate(self) -> None:
        try:
            while not self._pending.empty():
                batches = [self._pending.get_nowait()]
                while True:
                    try:
                        batches.append(self._pending.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                try:
                    entries = [entry for batch in batches for entry in batch]
                    await self._insert_shared_many(entries)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.warning(
                        "shared token-prefix insertion failed; "
                        "retaining local mappings",
                        exc_info=True,
                    )
                finally:
                    for _ in batches:
                        self._pending.task_done()
        finally:
            if self._replicator is asyncio.current_task():
                self._replicator = None
                if not self._closed and not self._pending.empty():
                    self._ensure_replicator()

    async def close(self, *, timeout: float = 2.0) -> bool:
        """Drain pending replication for at most timeout seconds, then stop."""
        self._closed = True
        drained = await self.flush(timeout=timeout)
        replicator = self._replicator
        if replicator is not None:
            replicator.cancel()
            with suppress(asyncio.CancelledError):
                await replicator
            if self._replicator is replicator:
                self._replicator = None
        while not self._pending.empty():
            self._pending.get_nowait()
            self._pending.task_done()
        return drained

    async def flush(self, *, timeout: float | None = None) -> bool:
        """Wait until queued shared replication has completed."""
        if self._replicator is None:
            return True
        try:
            await asyncio.wait_for(self._pending.join(), timeout)
        except TimeoutError:
            logger.warning("token-prefix replication did not drain before timeout")
            return False
        return True
