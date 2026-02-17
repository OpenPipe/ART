from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import struct
from typing import Sequence

import datrie

_TRIE_ALPHABET = "0123456789abcdef"


@dataclass(frozen=True)
class PrefixEntry:
    rendered_len: int
    raw_prefix: tuple[int, ...]


class LRUTrieCache:
    """LRU-bounded prefix trie for token sequence rewrites."""

    def __init__(self, max_entries: int = 1024) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._trie: datrie.Trie = datrie.Trie(_TRIE_ALPHABET)
        self._lru: OrderedDict[str, None] = OrderedDict()
        self._max_entries = max_entries

    @staticmethod
    def _encode_tokens(tokens: Sequence[int]) -> str:
        packed = bytearray()
        for token in tokens:
            packed.extend(struct.pack(">I", token))
        return packed.hex()

    def lookup(self, rendered_tokens: Sequence[int]) -> PrefixEntry | None:
        key = self._encode_tokens(rendered_tokens)
        match: tuple[str, PrefixEntry] | None = None
        for item in self._trie.prefix_items(key):
            match = item
        if match is None:
            return None
        match_key, entry = match
        self._lru.move_to_end(match_key)
        return entry

    def insert(self, rendered_prefix: Sequence[int], raw_prefix: Sequence[int]) -> None:
        key = self._encode_tokens(rendered_prefix)
        entry = PrefixEntry(
            rendered_len=len(rendered_prefix), raw_prefix=tuple(raw_prefix)
        )
        self._trie[key] = entry
        self._lru[key] = None
        self._lru.move_to_end(key)
        self._evict()

    def _evict(self) -> None:
        while len(self._lru) > self._max_entries:
            old_key, _ = self._lru.popitem(last=False)
            if old_key in self._trie:
                del self._trie[old_key]
