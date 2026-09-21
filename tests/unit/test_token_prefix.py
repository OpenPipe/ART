import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import replace
import json
import random
from typing import LiteralString

import pytest

from art_inference.token_prefix import (
    COMPACT_PREFIX_VERSION,
    CompactPrefixStore,
    PrefixEdit,
    PrefixMatch,
    TokenPrefixCache,
    TokenPrefixRuntime,
    TokenPrefixStore,
    apply_prefix_edits,
    compact_candidate_from_payload,
    compact_candidate_payload,
    compact_candidates_from_header,
    compact_candidates_header,
    compact_prefix_candidate,
    prefix_edits,
    resolve_compact_prefix,
    token_digest,
    token_prefix_digests,
)

_SharedEntry = tuple[str, list[int], list[int], str, tuple[PrefixEdit, ...]]


def _compact_entry(
    rendered: list[int],
    raw: list[int],
    *,
    scope: str = "a" * 64,
    lineage: str = "rollout",
) -> dict[str, object]:
    return {
        "scope": scope,
        "lineage": lineage,
        **compact_candidate_payload(compact_prefix_candidate(rendered, raw)),
    }


def _batch(
    insert: Callable[[str, list[int], list[int], str], Awaitable[object]],
) -> Callable[[list[_SharedEntry]], Awaitable[None]]:
    async def insert_many(entries: list[_SharedEntry]) -> None:
        for scope, rendered, raw, lineage, _ in entries:
            await insert(scope, rendered, raw, lineage)

    return insert_many


async def _discard(_entries: list[_SharedEntry]) -> None:
    pass


async def _runtime_insert(
    runtime: TokenPrefixRuntime,
    scope: str,
    rendered: list[int],
    raw: list[int],
    lineage: str = "rollout",
) -> None:
    await runtime.insert_many(
        [(scope, rendered, raw, lineage, prefix_edits(rendered, raw))]
    )


def test_store_retains_only_a_complete_bounded_lineage_snapshot() -> None:
    store = CompactPrefixStore(max_serialized_bytes_per_lineage=512)
    candidates = [compact_prefix_candidate([index], [index + 1]) for index in range(4)]
    for candidate in candidates:
        store.insert("a" * 64, "rollout", candidate)

    retained = store.candidates("a" * 64, "rollout", 4)
    payload = {
        "version": COMPACT_PREFIX_VERSION,
        "candidates": [compact_candidate_payload(value) for value in retained],
    }
    assert len(json.dumps(payload, separators=(",", ":")).encode()) <= 512
    assert 0 < len(retained) < len(candidates)
    assert retained[0] == candidates[-1]


def test_default_store_retains_busy_lineage_turns() -> None:
    store = CompactPrefixStore()
    candidates = [
        compact_prefix_candidate([index], [index + 1]) for index in range(240)
    ]
    for candidate in candidates:
        assert store.insert("a" * 64, "shared-lineage", candidate)

    retained = store.candidates("a" * 64, "shared-lineage", 240)
    assert len(retained) == len(candidates)
    assert candidates[0] in retained


def test_complete_candidate_headers_round_trip_or_fall_back() -> None:
    candidate = compact_prefix_candidate([1, 500], [1, 101, 102])

    encoded = compact_candidates_header([candidate])
    empty = compact_candidates_header([])
    dense = compact_candidates_header(
        [compact_prefix_candidate([index], [index]) for index in range(65)]
    )
    oversized = compact_candidates_header(
        [
            compact_prefix_candidate(
                [index],
                [index, *range(1_000 + index * 1_000, 2_000 + index * 1_000)],
            )
            for index in range(32)
        ]
    )

    assert encoded is not None
    assert compact_candidates_from_header(encoded) == (candidate,)
    assert empty is not None
    assert compact_candidates_from_header(empty) == ()
    assert dense is not None
    assert len(compact_candidates_from_header(dense)) == 65

    assert oversized is None


def test_candidate_header_rejects_malformed_or_partial_snapshots() -> None:
    for value in (
        "not-json",
        json.dumps({"version": "old", "candidates": []}),
        json.dumps({"version": COMPACT_PREFIX_VERSION}),
    ):
        try:
            compact_candidates_from_header(value)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid candidate header was accepted")


def test_compact_edits_cover_deletions_expansions_and_sparse_128k_changes() -> None:
    cases = (
        ([9, 1, 2], [1, 2]),
        ([1, 2], [9, 1, 2]),
        ([1, 2, 3], [1, 8, 9, 3]),
    )
    for rendered, raw in cases:
        edits = prefix_edits(rendered, raw)
        assert apply_prefix_edits(rendered, edits) == raw
        candidate = compact_prefix_candidate(rendered, raw, edits)
        assert resolve_compact_prefix([*rendered, 77], [candidate]) == PrefixMatch(
            len(rendered), tuple(raw)
        )

    rendered = list(range(128_000))
    raw = rendered.copy()
    raw[10] = 200_010
    raw[64_000] = 264_000
    raw[-2] = 327_998
    edits = prefix_edits(rendered, raw)
    payload = compact_candidate_payload(compact_prefix_candidate(rendered, raw, edits))

    assert len(edits) == 3
    assert len(json.dumps(payload, separators=(",", ":")).encode()) < 512
    assert resolve_compact_prefix(
        [*rendered, 9], [compact_candidate_from_payload(payload)]
    ) == PrefixMatch(len(rendered), tuple(raw))


def test_compact_resolution_is_longest_newest_and_corruption_tolerant() -> None:
    shorter = compact_prefix_candidate([1, 2], [7, 8])
    longer = compact_prefix_candidate([1, 2, 3], [7, 8, 9])
    corrupt = replace(longer, raw_digest="0" * 64)

    assert resolve_compact_prefix(
        [1, 2, 3, 4], [shorter, corrupt, longer]
    ) == PrefixMatch(3, (7, 8, 9))

    first = compact_prefix_candidate([1, 2], [10, 11])
    newest = compact_prefix_candidate([1, 2], [20, 21])
    assert resolve_compact_prefix([1, 2, 3], [newest, first]) == PrefixMatch(
        2, (20, 21)
    )


def test_compact_identity_candidate_shadows_a_shorter_divergence() -> None:
    store = CompactPrefixStore()
    store.insert(
        "scope",
        "lineage",
        compact_prefix_candidate([1, 2], [7, 8]),
    )
    store.insert(
        "scope",
        "lineage",
        compact_prefix_candidate([1, 2, 3], [1, 2, 3]),
    )

    candidates = store.candidates("scope", "lineage", 4)
    assert candidates[0].edits == ()
    assert resolve_compact_prefix([1, 2, 3, 4], candidates) == PrefixMatch(3, (1, 2, 3))


def test_compact_payload_rejects_invalid_token_and_edit_bounds() -> None:
    payload = compact_candidate_payload(compact_prefix_candidate([1], [1]))
    for edit in (
        [[0, 1, [True]]],
        [[0, 1, [-1]]],
        [[0, 1, [0x1_0000_0000]]],
        [[1, 1, []], [0, 1, []]],
    ):
        invalid = payload | {"edits": edit}
        try:
            compact_candidate_from_payload(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid edit: {edit}")

    try:
        compact_candidate_from_payload(payload | {"rendered_length": 262_145})
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unreachable rendered length")


@pytest.mark.parametrize("field", ["rendered_digest", "raw_digest"])
@pytest.mark.parametrize(
    "digest", ["0123456789abcdef" * 4, "0123456789ABCDEF" * 4, "0123456789aBcDeF" * 4]
)
def test_compact_payload_normalizes_ascii_hex_digests(field: str, digest: str) -> None:
    candidate = compact_prefix_candidate([1, 2, 3], [8, 9, 3])
    payload = compact_candidate_payload(candidate) | {field: digest}

    assert compact_candidate_from_payload(payload) == replace(
        candidate, **{field: digest.lower()}
    )


@pytest.mark.parametrize("field", ["rendered_digest", "raw_digest"])
@pytest.mark.parametrize(
    "digest",
    [
        "",
        "a" * 63,
        "a" * 65,
        "a" * 64 + "\n",
        *[
            character + "a" * 63
            for character in ("g", "Ｇ", "ａ", "١", "é", "\n", "\0", "\ud800")
        ],
        None,
        True,
        0,
        b"a" * 64,
        [],
        {},
    ],
)
def test_compact_payload_rejects_invalid_digests(field: str, digest: object) -> None:
    payload = compact_candidate_payload(compact_prefix_candidate([1, 2], [3, 2]))

    with pytest.raises(ValueError, match="^invalid token-prefix candidate$"):
        compact_candidate_from_payload(payload | {field: digest})


@pytest.mark.parametrize("field", ["rendered_digest", "raw_digest"])
def test_compact_payload_preserves_digest_subclass_validation(field: str) -> None:
    class ValidLength(str):
        def __len__(self) -> int:
            return 64

    class InvalidLength(str):
        def __len__(self) -> int:
            return 63

    class ValidCharacters(str):
        def __iter__(self) -> Iterator[LiteralString]:
            return iter(("A",) * 64)

    class InvalidCharacters(str):
        def __iter__(self) -> Iterator[LiteralString]:
            return iter(("g",))

    candidate = compact_prefix_candidate([1, 2], [3, 2])
    payload = compact_candidate_payload(candidate)
    for digest in (ValidLength("A" * 63), ValidCharacters("g" * 64)):
        assert compact_candidate_from_payload(payload | {field: digest}) == replace(
            candidate, **{field: digest.lower()}
        )
    for digest in (InvalidLength("A" * 64), InvalidCharacters("A" * 64)):
        with pytest.raises(ValueError, match="^invalid token-prefix candidate$"):
            compact_candidate_from_payload(payload | {field: digest})


def test_compact_prefix_hashes_requested_lengths_consistently() -> None:
    values = list(range(128_000))
    lengths = [128_000, 32, 64_000, 32]
    digests = token_prefix_digests(values, lengths)

    assert list(digests) == [32, 64_000, 128_000]
    for length, digest in digests.items():
        assert (
            digest
            == compact_prefix_candidate(
                values[:length], values[:length]
            ).rendered_digest
        )


def test_compact_digest_encoding_has_a_stable_golden_vector() -> None:
    assert token_digest([]) == (
        "ff127bd383d2204e4e5509b3c4001b75988e0256b4116ffb904e0864848b04c1"
    )
    assert token_digest([0, 1, 0xFFFFFFFF]) == (
        "4cc20a06621a8b2cec34aef4b495b6439931b0faa260e9308a7298cfe9e862b5"
    )


def test_compact_candidates_match_a_randomized_reference() -> None:
    randomizer = random.Random(1)
    for _ in range(1_000):
        rendered = [
            randomizer.randrange(32) for _ in range(randomizer.randrange(1, 65))
        ]
        raw = rendered.copy()
        start = randomizer.randrange(len(raw) + 1)
        stop = randomizer.randrange(start, len(raw) + 1)
        raw[start:stop] = [
            randomizer.randrange(32) for _ in range(randomizer.randrange(0, 12))
        ]
        suffix = [randomizer.randrange(32) for _ in range(randomizer.randrange(0, 12))]
        candidate = compact_prefix_candidate(rendered, raw)

        assert resolve_compact_prefix([*rendered, *suffix], [candidate]) == PrefixMatch(
            len(rendered), tuple(raw)
        )


def test_longest_observed_prefix_rewrites_only_rendered_prefix() -> None:
    cache = TokenPrefixCache()
    cache.insert([1, 500], [1, 101, 102], "rollout")
    cache.insert([1, 500, 3, 4], [1, 101, 102, 3, 4], "rollout")

    assert cache.lookup([1, 500, 3, 4, 5], "rollout") == PrefixMatch(
        rendered_length=4,
        raw_prefix=(1, 101, 102, 3, 4),
    )


def test_cache_matches_a_reference_model_across_random_prefixes() -> None:
    randomizer = random.Random(0)
    cache = TokenPrefixCache(
        max_variants=10_000,
        max_token_ids=1_000_000,
        max_lineages_per_variant=100,
    )
    reference: dict[tuple[int, ...], dict[tuple[int, ...], dict[str, int]]] = (
        defaultdict(lambda: defaultdict(dict))
    )

    for observation in range(2_000):
        rendered = tuple(
            randomizer.randrange(8) for _ in range(randomizer.randrange(1, 12))
        )
        raw = tuple(randomizer.randrange(8) for _ in range(randomizer.randrange(1, 12)))
        lineage = f"lineage-{randomizer.randrange(12)}"
        cache.insert(rendered, raw, lineage)
        reference[rendered][raw][lineage] = observation

        query = tuple(
            randomizer.randrange(8) for _ in range(randomizer.randrange(1, 16))
        )
        query_lineage = f"lineage-{randomizer.randrange(12)}"
        expected = None
        candidates: list[tuple[int, ...]] = sorted(
            (
                prefix
                for prefix in reference
                if len(prefix) <= len(query) and query[: len(prefix)] == prefix
            ),
            key=lambda prefix: len(prefix),
            reverse=True,
        )
        for prefix in candidates:
            variants = reference[prefix]
            matching = [
                (lineages[query_lineage], variant)
                for variant, lineages in variants.items()
                if query_lineage in lineages
            ]
            if not matching:
                continue
            _, selected = max(matching)
            expected = PrefixMatch(len(prefix), selected)
            break

        assert cache.lookup(query, query_lineage) == expected


def test_runtime_preserves_the_entire_prompt_on_a_miss() -> None:
    async def exercise() -> None:
        runtime = TokenPrefixRuntime(
            lookup_shared=lambda _scope, _rendered, _lineage: asyncio.sleep(
                0, result=None
            ),
            insert_shared_many=_discard,
        )
        prompt = [1, 2, 3, 4]

        canonical, engine_input, _ = await runtime.rewrite_with_edits(
            "model", prompt, "rollout", shared_candidate=True
        )

        assert canonical == prompt
        assert engine_input == prompt
        assert canonical is not prompt
        assert engine_input is not canonical

    asyncio.run(exercise())


def test_runtime_changes_exactly_the_matched_prefix() -> None:
    async def exercise() -> None:
        shared = TokenPrefixStore()

        async def lookup(scope, rendered, lineage):
            return shared.lookup(scope, rendered, lineage)

        async def insert(scope, rendered, raw, lineage):
            shared.insert(scope, rendered, raw, lineage)

        runtime = TokenPrefixRuntime(
            lookup_shared=lookup, insert_shared_many=_batch(insert)
        )
        cases = (
            ([500], [101, 102], []),
            ([1, 500], [1, 101, 102], [3]),
            ([1, 2, 3, 4], [7], [5, 6, 7, 8]),
            (list(range(128)), [9, 10, 11], [200, 201]),
        )
        for index, (rendered_prefix, raw_prefix, suffix) in enumerate(cases):
            scope = f"model-{index}"
            await _runtime_insert(runtime, scope, rendered_prefix, raw_prefix)
            await runtime.flush()
            prompt = [*rendered_prefix, *suffix]

            canonical, engine_input, _ = await runtime.rewrite_with_edits(
                scope, prompt, "rollout", shared_candidate=True
            )

            assert canonical == prompt
            assert engine_input == [*raw_prefix, *suffix]
            assert engine_input[len(raw_prefix) :] == suffix

    asyncio.run(exercise())


def test_replication_is_async_while_local_continuations_are_immediate() -> None:
    async def exercise() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        shared = TokenPrefixStore()

        async def insert(scope, rendered, raw, lineage):
            started.set()
            await release.wait()
            shared.insert(scope, rendered, raw, lineage)

        runtime = TokenPrefixRuntime(
            lookup_shared=lambda scope, rendered, lineage: asyncio.sleep(
                0, result=shared.lookup(scope, rendered, lineage)
            ),
            insert_shared_many=_batch(insert),
        )
        await _runtime_insert(runtime, "model", [1, 500], [1, 101, 102])

        assert not started.is_set()
        continuation = asyncio.create_task(
            runtime.rewrite_with_edits(
                "model",
                [1, 500, 3],
                "rollout",
                shared_candidate=True,
            )
        )
        _, rewritten, _ = await continuation
        assert rewritten == [1, 101, 102, 3]
        await started.wait()
        release.set()
        assert await runtime.close()

    asyncio.run(exercise())


def test_completed_replication_does_not_delay_a_later_continuation() -> None:
    async def exercise() -> None:
        shared = TokenPrefixStore()

        async def insert(scope, rendered, raw, lineage):
            shared.insert(scope, rendered, raw, lineage)

        runtime = TokenPrefixRuntime(
            lookup_shared=lambda scope, rendered, lineage: asyncio.sleep(
                0, result=shared.lookup(scope, rendered, lineage)
            ),
            insert_shared_many=_batch(insert),
        )
        await _runtime_insert(runtime, "model", [1, 500], [1, 101, 102])
        assert await runtime.flush()
        assert runtime._replicator is None

        _, rewritten, _ = await asyncio.wait_for(
            runtime.rewrite_with_edits(
                "model",
                [1, 500, 3],
                "rollout",
                shared_candidate=True,
            ),
            0.1,
        )
        assert rewritten == [1, 101, 102, 3]
        await runtime.close()

    asyncio.run(exercise())


def test_failed_replication_can_reuse_an_older_matching_shared_prefix() -> None:
    async def exercise() -> None:
        shared = TokenPrefixStore()
        shared.insert("model", [1, 500], [1, 41, 42], "rollout")

        async def fail(_scope, _rendered, _raw, _lineage):
            raise RuntimeError("unavailable")

        first = TokenPrefixRuntime(
            lookup_shared=lambda scope, rendered, lineage: asyncio.sleep(
                0, result=shared.lookup(scope, rendered, lineage)
            ),
            insert_shared_many=_batch(fail),
        )
        await _runtime_insert(first, "model", [1, 500], [1, 101, 102])
        assert await first.flush()
        second = TokenPrefixRuntime(
            lookup_shared=lambda scope, rendered, lineage: asyncio.sleep(
                0, result=shared.lookup(scope, rendered, lineage)
            ),
            insert_shared_many=_discard,
        )

        canonical, rewritten, _ = await second.rewrite_with_edits(
            "model", [1, 500, 3], "rollout", shared_candidate=True
        )
        assert canonical == [1, 500, 3]
        assert rewritten == [1, 41, 42, 3]
        await first.close()
        await second.close()

    asyncio.run(exercise())


def test_replication_queue_is_bounded_and_drops_only_shared_work() -> None:
    async def exercise() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        replicated: list[str] = []

        async def insert(scope, _rendered, _raw, _lineage):
            replicated.append(scope)
            if scope == "first":
                started.set()
                await release.wait()

        runtime = TokenPrefixRuntime(
            lookup_shared=lambda _scope, _rendered, _lineage: asyncio.sleep(
                0, result=None
            ),
            insert_shared_many=_batch(insert),
            max_pending_batches=1,
        )
        await _runtime_insert(runtime, "first", [1], [11])
        await started.wait()
        await _runtime_insert(runtime, "second", [2], [22])
        await _runtime_insert(runtime, "dropped", [3], [33])

        _, rewritten, _ = await runtime.rewrite_with_edits(
            "dropped", [3, 4], "rollout", shared_candidate=False
        )
        assert rewritten == [33, 4]

        release.set()
        assert await runtime.close()
        assert replicated == ["first", "second"]

    asyncio.run(exercise())


def test_shutdown_waits_for_replication_then_stops_the_worker() -> None:
    async def exercise() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def insert(_scope, _rendered, _raw, _lineage):
            started.set()
            await release.wait()

        runtime = TokenPrefixRuntime(
            lookup_shared=lambda _scope, _rendered, _lineage: asyncio.sleep(
                0, result=None
            ),
            insert_shared_many=_batch(insert),
        )
        await _runtime_insert(runtime, "model", [1], [2])
        await started.wait()
        closing = asyncio.create_task(runtime.close(timeout=1))
        await asyncio.sleep(0)
        assert not closing.done()

        release.set()
        assert await closing

    asyncio.run(exercise())


def test_replication_coalesces_and_shutdown_timeout_is_bounded() -> None:
    async def exercise() -> None:
        calls = []

        async def insert_many(entries):
            calls.append(entries)

        runtime = TokenPrefixRuntime(
            lookup_shared=lambda _scope, _rendered, _lineage: asyncio.sleep(
                0, result=None
            ),
            insert_shared_many=insert_many,
        )
        await _runtime_insert(runtime, "first", [1], [11])
        await _runtime_insert(runtime, "second", [2], [22])
        assert await runtime.flush()
        assert len(calls) == 1
        assert [entry[0] for entry in calls[0]] == ["first", "second"]

        blocked = TokenPrefixRuntime(
            lookup_shared=lambda _scope, _rendered, _lineage: asyncio.sleep(
                0, result=None
            ),
            insert_shared_many=_batch(lambda *_args: asyncio.Event().wait()),
        )
        await _runtime_insert(blocked, "model", [3], [33])
        assert not await blocked.close(timeout=0.001)

    asyncio.run(exercise())


def test_target_concurrency_burst_is_retained_while_replication_is_blocked() -> None:
    async def exercise() -> None:
        replicated = []
        started = asyncio.Event()
        release = asyncio.Event()

        async def insert_many(entries):
            replicated.extend(entries)
            if len(replicated) == 1:
                started.set()
                await release.wait()

        runtime = TokenPrefixRuntime(
            lookup_shared=lambda _scope, _rendered, _lineage: asyncio.sleep(
                0, result=None
            ),
            insert_shared_many=insert_many,
        )
        await runtime.insert_many([("model-0", [1], [1], "rollout", ())])
        await started.wait()
        await asyncio.gather(
            *(
                runtime.insert_many(
                    [
                        (
                            f"model-{index}",
                            [index + 1],
                            [index + 1],
                            "rollout",
                            (),
                        )
                    ]
                )
                for index in range(1, 32)
            )
        )
        release.set()
        assert await runtime.flush()
        assert len(replicated) == 32
        await runtime.close()

    asyncio.run(exercise())


def test_identical_observations_are_idempotent() -> None:
    cache = TokenPrefixCache()
    cache.insert([1, 500], [1, 101, 102], "first")
    cache.insert([1, 500], [1, 101, 102], "second")

    expected = PrefixMatch(2, (1, 101, 102))
    assert cache.lookup([1, 500, 3], "first") == expected
    assert cache.lookup([1, 500, 3], "second") == expected
    assert cache.lookup([1, 500, 3], "unknown") is None


def test_eviction_removes_all_variants_at_the_rendered_prefix() -> None:
    cache = TokenPrefixCache(max_variants=2, max_token_ids=12)
    cache.insert([1, 500], [1, 101, 102], "shared")
    cache.insert([1, 500], [1, 500], "shared")
    cache.insert([9], [9], "third")

    assert cache.lookup([1, 500, 3], "shared") is None
    assert cache.lookup([9], "third") == PrefixMatch(1, (9,))


def test_lineage_presence_tracks_only_retained_observations() -> None:
    cache = TokenPrefixCache()
    cache.insert([1], [1], "shared")
    cache.insert([2], [2], "shared")

    assert cache.lookup([1], "shared") == PrefixMatch(1, (1,))
    assert cache.evict_oldest() is True
    assert cache.lookup([1], "shared") == PrefixMatch(1, (1,))
    assert cache.evict_oldest() is True
    assert cache.lookup([1], "shared") is None


def test_lineage_trimming_is_variant_local() -> None:
    cache = TokenPrefixCache(max_lineages_per_variant=1)
    cache.insert([1, 500], [1, 101, 102], "shared")
    cache.insert([1, 500], [1, 500], "shared")
    cache.insert([1, 500], [1, 101, 102], "new")

    assert cache.lookup([1, 500, 3], "shared") == PrefixMatch(2, (1, 500))
    assert cache.lookup([1, 500, 3], "new") == PrefixMatch(2, (1, 101, 102))


def test_distinct_lineages_resolve_observed_tokenization_collision() -> None:
    cache = TokenPrefixCache()
    cache.insert([1], [1], "first")
    cache.insert([1, 500], [1, 101, 102], "first")
    cache.insert([1, 500], [1, 500], "second")

    assert cache.lookup([1, 500, 3], "first") == PrefixMatch(2, (1, 101, 102))
    assert cache.lookup([1, 500, 3], "second") == PrefixMatch(2, (1, 500))


def test_collision_uses_most_recent_variant_at_longest_known_prefix() -> None:
    cache = TokenPrefixCache()
    cache.insert([1], [9], "shared")
    cache.insert([1, 500], [9, 101, 102], "shared")
    cache.insert([1, 500], [9, 500], "shared")

    assert cache.lookup([1, 500, 3], "shared") == PrefixMatch(2, (9, 500))
    cache.insert([1, 500], [9, 101, 102], "shared")
    assert cache.lookup([1, 500, 3], "shared") == PrefixMatch(2, (9, 101, 102))


def test_collision_without_shorter_prefix_uses_most_recent_variant() -> None:
    cache = TokenPrefixCache()
    cache.insert([1, 500], [1, 101, 102], "shared")
    cache.insert([1, 500], [1, 500], "shared")

    assert cache.lookup([1, 500, 3], "shared") == PrefixMatch(2, (1, 500))


def test_store_partitions_model_keys() -> None:
    store = TokenPrefixStore()
    store.insert("base", [1], [2], "lineage")
    store.insert("checkpoint", [1], [3], "lineage")

    assert store.lookup("base", [1, 4], "lineage") == PrefixMatch(1, (2,))
    assert store.lookup("checkpoint", [1, 4], "lineage") == PrefixMatch(1, (3,))


def test_store_enforces_per_scope_and_global_token_budgets() -> None:
    store = TokenPrefixStore(
        max_scopes=4,
        max_variants_per_scope=4,
        max_token_ids_per_scope=6,
        max_token_ids=8,
    )

    assert store.insert("oversized", [1, 2, 3, 4], [5, 6, 7], "lineage") is False
    assert store.lookup("oversized", [1, 2, 3, 4], "lineage") is None

    assert store.insert("first", [1, 2], [3, 4], "lineage") is True
    assert store.insert("second", [5, 6], [7, 8], "lineage") is True
    assert store.insert("third", [9], [10], "lineage") is True

    assert store.lookup("first", [1, 2], "lineage") is None
    assert store.lookup("second", [5, 6], "lineage") == PrefixMatch(2, (7, 8))
    assert store.lookup("third", [9], "lineage") == PrefixMatch(1, (10,))


def test_shared_store_reuses_identity_across_engine_replicas() -> None:
    async def exercise() -> None:
        shared = TokenPrefixStore()

        async def lookup(scope, rendered, lineage):
            return shared.lookup(scope, rendered, lineage)

        async def insert(scope, rendered, raw, lineage):
            shared.insert(scope, rendered, raw, lineage)

        first = TokenPrefixRuntime(
            lookup_shared=lookup, insert_shared_many=_batch(insert)
        )
        second = TokenPrefixRuntime(
            lookup_shared=lookup, insert_shared_many=_batch(insert)
        )
        await _runtime_insert(first, "model", [1, 500], [1, 101, 102])
        await first.flush()

        canonical, rewritten, _ = await second.rewrite_with_edits(
            "model",
            [1, 500, 3],
            "rollout",
            shared_candidate=True,
        )

        assert canonical == [1, 500, 3]
        assert rewritten == [1, 101, 102, 3]

    asyncio.run(exercise())


def test_runtime_migrates_an_implicit_lineage_without_a_client_session() -> None:
    async def exercise() -> None:
        shared = TokenPrefixStore()
        shared.insert("model", [1, 500], [1, 101, 102], "scenario")
        lookups = []

        async def lookup(scope, rendered, lineage):
            lookups.append(lineage)
            return shared.lookup(scope, rendered, lineage)

        async def insert_many(_entries):
            return None

        runtime = TokenPrefixRuntime(
            lookup_shared=lookup,
            insert_shared_many=insert_many,
        )
        canonical, rewritten, _ = await runtime.rewrite_with_edits(
            "model",
            [1, 500, 3],
            "tool-call",
            fallback_lineage="scenario",
            shared_candidate=True,
        )

        assert canonical == [1, 500, 3]
        assert rewritten == [1, 101, 102, 3]
        assert lookups == ["tool-call", "scenario"]
        _, local, _ = await runtime.rewrite_with_edits(
            "model",
            [1, 500, 4],
            "tool-call",
            shared_candidate=False,
        )
        assert local == [1, 101, 102, 4]
        await runtime.close()

    asyncio.run(exercise())


def test_shared_store_restores_text_equivalent_noncanonical_sampled_ids() -> None:
    async def exercise() -> None:
        shared = TokenPrefixStore()

        async def lookup(scope, rendered, lineage):
            return shared.lookup(scope, rendered, lineage)

        async def insert(scope, rendered, raw, lineage):
            shared.insert(scope, rendered, raw, lineage)

        first = TokenPrefixRuntime(
            lookup_shared=lookup, insert_shared_many=_batch(insert)
        )
        second = TokenPrefixRuntime(
            lookup_shared=lookup, insert_shared_many=_batch(insert)
        )
        # Qwen3.5 decodes both forms as " roaming", but only the two-token form
        # carries the sampled logprobs that ART must train against.
        await _runtime_insert(first, "model", [10, 65945], [10, 897, 6267])
        await first.flush()

        canonical, rewritten, edits = await second.rewrite_with_edits(
            "model", [10, 65945, 20], "rollout", shared_candidate=True
        )

        assert canonical == [10, 65945, 20]
        assert rewritten == [10, 897, 6267, 20]
        assert edits == (PrefixEdit(1, 2, (897, 6267)),)

    asyncio.run(exercise())


def test_replication_restarts_when_an_entry_arrives_during_shutdown() -> None:
    async def exercise() -> None:
        second = ("model", [3, 4], [3, 5], "rollout", (PrefixEdit(1, 2, (5,)),))

        class RacingQueue(asyncio.Queue):
            empty_calls = 0

            def empty(self) -> bool:
                self.empty_calls += 1
                if self.empty_calls == 2:
                    self.put_nowait([second])
                    return True
                return super().empty()

        replicated: list[
            tuple[str, list[int], list[int], str, tuple[PrefixEdit, ...]]
        ] = []

        async def insert_many(entries):
            replicated.extend(entries)

        runtime = TokenPrefixRuntime(
            lookup_shared=lambda *_args: asyncio.sleep(0, result=None),
            insert_shared_many=insert_many,
        )
        setattr(runtime, "_pending", RacingQueue(maxsize=16))

        await _runtime_insert(runtime, "model", [1, 2], [1, 9])
        assert await runtime.flush(timeout=1)

        assert [entry[1] for entry in replicated] == [[1, 2], [3, 4]]
        assert await runtime.close()

    asyncio.run(exercise())


def test_pending_replication_can_reuse_an_older_matching_shared_prefix() -> None:
    async def exercise() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        shared = TokenPrefixStore()

        async def lookup(scope, rendered, lineage):
            return shared.lookup(scope, rendered, lineage)

        async def insert(scope, rendered, raw, lineage):
            started.set()
            await release.wait()
            shared.insert(scope, rendered, raw, lineage)

        shared.insert("model", [1, 500], [1, 41, 42], "rollout")
        first = TokenPrefixRuntime(
            lookup_shared=lookup, insert_shared_many=_batch(insert)
        )
        await _runtime_insert(first, "model", [1, 500], [1, 101, 102])
        await started.wait()
        second = TokenPrefixRuntime(
            lookup_shared=lookup, insert_shared_many=_batch(insert)
        )
        canonical, rewritten, _ = await second.rewrite_with_edits(
            "model", [1, 500, 3], "rollout", shared_candidate=True
        )
        assert canonical == [1, 500, 3]
        assert rewritten == [1, 41, 42, 3]

        release.set()
        assert await first.close()
        await second.close()

    asyncio.run(exercise())


def test_shared_candidate_compares_shared_with_local_identity() -> None:
    async def exercise() -> None:
        shared_lookups = 0

        async def lookup(_scope, _rendered, _lineage):
            nonlocal shared_lookups
            shared_lookups += 1
            return None

        runtime = TokenPrefixRuntime(
            lookup_shared=lookup,
            insert_shared_many=_discard,
        )
        await _runtime_insert(runtime, "model", [1, 2], [1, 2])

        canonical, rewritten, _ = await runtime.rewrite_with_edits(
            "model", [1, 2, 3], "rollout", shared_candidate=True
        )

        assert canonical == rewritten == [1, 2, 3]
        assert shared_lookups == 1

    asyncio.run(exercise())


def test_local_nonidentity_and_latest_ambiguity_choice_are_immediate() -> None:
    async def exercise() -> None:
        shared_lookups = 0

        async def lookup(_scope, _rendered, _lineage):
            nonlocal shared_lookups
            shared_lookups += 1
            return None

        runtime = TokenPrefixRuntime(
            lookup_shared=lookup,
            insert_shared_many=_discard,
        )
        await _runtime_insert(runtime, "model", [1, 500], [1, 101, 102])
        _, first, _ = await runtime.rewrite_with_edits(
            "model", [1, 500, 3], "rollout", shared_candidate=True
        )
        await _runtime_insert(runtime, "model", [1, 500], [1, 500])
        _, second, _ = await runtime.rewrite_with_edits(
            "model", [1, 500, 3], "rollout", shared_candidate=True
        )

        assert first == [1, 101, 102, 3]
        assert second == [1, 500, 3]
        assert shared_lookups == 2
        await runtime.close()

    asyncio.run(exercise())


def test_known_lineage_without_a_matching_prefix_consults_shared_lookup() -> None:
    async def exercise() -> None:
        shared_lookups = 0

        async def lookup(_scope, _rendered, _lineage):
            nonlocal shared_lookups
            shared_lookups += 1
            return None

        runtime = TokenPrefixRuntime(
            lookup_shared=lookup,
            insert_shared_many=_discard,
        )
        await _runtime_insert(runtime, "model", [1, 2], [1, 2])

        canonical, rewritten, _ = await runtime.rewrite_with_edits(
            "model", [8, 9], "rollout", shared_candidate=True
        )

        assert canonical == rewritten == [8, 9]
        assert shared_lookups == 1

    asyncio.run(exercise())


def test_broad_lineage_without_a_local_match_consults_shared_lookup() -> None:
    async def exercise() -> None:
        shared_lookups = 0

        async def lookup(_scope, _rendered, _lineage):
            nonlocal shared_lookups
            shared_lookups += 1
            return PrefixMatch(2, (8, 10))

        runtime = TokenPrefixRuntime(
            lookup_shared=lookup,
            insert_shared_many=_discard,
        )
        await _runtime_insert(runtime, "model", [1, 2], [1, 2])

        canonical, rewritten, _ = await runtime.rewrite_with_edits(
            "model",
            [8, 9, 11],
            "rollout",
            shared_candidate=True,
        )

        assert canonical == [8, 9, 11]
        assert rewritten == [8, 10, 11]
        assert shared_lookups == 1

    asyncio.run(exercise())


def test_longer_shared_prefix_wins_after_routing_returns_to_stale_worker() -> None:
    async def exercise() -> None:
        shared = PrefixMatch(
            4,
            (1, 101, 102, 3, 201, 202),
            prefix_edits(
                [1, 500, 3, 600],
                [1, 101, 102, 3, 201, 202],
            ),
        )
        runtime = TokenPrefixRuntime(
            lookup_shared=lambda *_args: asyncio.sleep(0, result=shared),
            insert_shared_many=_discard,
        )
        await _runtime_insert(runtime, "model", [1, 500], [1, 101, 102])

        canonical, rewritten, _ = await runtime.rewrite_with_edits(
            "model",
            [1, 500, 3, 600, 4],
            "rollout",
            shared_candidate=True,
        )

        assert canonical == [1, 500, 3, 600, 4]
        assert rewritten == [1, 101, 102, 3, 201, 202, 4]
        await runtime.close()

    asyncio.run(exercise())


def test_unknown_lineage_still_consults_shared_lookup() -> None:
    async def exercise() -> None:
        shared_lookups = 0

        async def lookup(_scope, _rendered, _lineage):
            nonlocal shared_lookups
            shared_lookups += 1
            return None

        runtime = TokenPrefixRuntime(
            lookup_shared=lookup,
            insert_shared_many=_discard,
        )
        await _runtime_insert(runtime, "model", [1, 2], [1, 2], "known")
        await runtime.rewrite_with_edits(
            "model",
            [8, 9],
            "unknown",
            shared_candidate=True,
        )

        assert shared_lookups == 1

    asyncio.run(exercise())
