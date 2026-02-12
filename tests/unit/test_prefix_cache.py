"""Tests for the LRUTrieCache prefix rewrite helper."""

import pytest

pytest.importorskip("datrie")

from art.tinker.prefix_cache import LRUTrieCache


class TestLRUTrieCache:
    def test_longest_prefix_match(self) -> None:
        cache = LRUTrieCache(max_entries=10)
        cache.insert([1, 2], [10, 11])
        cache.insert([1, 2, 3], [20, 21, 22])

        entry = cache.lookup([1, 2, 3, 4])

        assert entry is not None
        assert entry.rendered_len == 3
        assert entry.raw_prefix == (20, 21, 22)

    def test_lru_eviction(self) -> None:
        cache = LRUTrieCache(max_entries=2)
        cache.insert([1], [10])
        cache.insert([2], [20])

        assert cache.lookup([1, 99]) is not None

        cache.insert([3], [30])

        assert cache.lookup([2, 0]) is None
        assert cache.lookup([1, 0]) is not None

    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError):
            LRUTrieCache(max_entries=0)
