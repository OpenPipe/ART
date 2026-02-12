"""Tests for strip_logprobs utility function."""

import copy
import logging
from unittest.mock import MagicMock

import pytest

from art.utils.strip_logprobs import strip_logprobs


class TestStripLogprobs:
    """Test suite for strip_logprobs function."""

    def test_strip_dict_with_logprobs(self):
        """Test stripping logprobs from dictionary."""
        input_dict = {
            "data": "value",
            "logprobs": [0.1, 0.2, 0.3],
            "nested": {"key": "val", "logprobs": {"nested_log": 0.5}},
        }
        expected = {"data": "value", "nested": {"key": "val"}}

        result = strip_logprobs(input_dict)

        assert result == expected
        assert input_dict["logprobs"] == [0.1, 0.2, 0.3]  # Original unchanged

    def test_strip_nested_dict(self):
        """Test stripping logprobs from deeply nested dictionaries."""
        input_dict = {
            "level1": {
                "level2": {
                    "level3": {"data": 1, "logprobs": "remove_me"},
                    "logprobs": [1, 2, 3],
                }
            },
            "logprobs": None,
        }
        expected = {"level1": {"level2": {"level3": {"data": 1}}}}

        result = strip_logprobs(input_dict)

        assert result == expected

    def test_strip_list_with_logprobs(self):
        """Test stripping logprobs from lists."""
        input_list = [
            {"item": 1, "logprobs": 0.1},
            {"item": 2, "logprobs": 0.2},
            {"item": 3},
        ]
        expected = [{"item": 1}, {"item": 2}, {"item": 3}]

        result = strip_logprobs(input_list)

        assert result == expected

    def test_strip_tuple_with_logprobs(self):
        """Test stripping logprobs from tuples."""
        input_tuple = (
            {"item": 1, "logprobs": 0.1},
            {"item": 2},
            {"nested": {"logprobs": "remove"}},
        )
        expected = ({"item": 1}, {"item": 2}, {"nested": {}})

        result = strip_logprobs(input_tuple)

        assert result == expected
        assert isinstance(result, tuple)

    def test_strip_object_with_logprobs(self):
        """Test stripping logprobs from objects with __dict__."""

        class TestObject:
            def __init__(self):
                self.data = "value"
                self.logprobs = [0.1, 0.2]
                self.nested = {"key": "val", "logprobs": "remove"}

        obj = TestObject()
        result = strip_logprobs(obj)

        assert result.data == "value"
        assert result.logprobs is None  # Set to None for objects
        assert result.nested == {"key": "val"}

    def test_strip_mixed_nested_structure(self):
        """Test stripping logprobs from mixed nested structures."""
        input_data = {
            "list": [
                {"logprobs": 1},
                [{"nested_list": True, "logprobs": 2}],
            ],
            "tuple": ({"logprobs": 3}, {"keep": "me"}),
            "dict": {"nested": {"logprobs": 4, "data": "keep"}},
        }
        expected = {
            "list": [{}, [{"nested_list": True}]],
            "tuple": ({}, {"keep": "me"}),
            "dict": {"nested": {"data": "keep"}},
        }

        result = strip_logprobs(input_data)

        assert result == expected

    def test_strip_empty_structures(self):
        """Test stripping logprobs from empty structures."""
        assert strip_logprobs({}) == {}
        assert strip_logprobs([]) == []
        assert strip_logprobs(()) == ()

    def test_strip_none_and_primitives(self):
        """Test stripping logprobs from None and primitive values."""
        assert strip_logprobs(None) is None
        assert strip_logprobs(42) == 42
        assert strip_logprobs("string") == "string"
        assert strip_logprobs(3.14) == 3.14
        assert strip_logprobs(True) is True

    def test_no_logprobs_unchanged(self):
        """Test that structures without logprobs remain unchanged."""
        input_dict = {
            "data": "value",
            "nested": {"key": "val"},
            "list": [1, 2, 3],
        }

        result = strip_logprobs(input_dict)

        assert result == input_dict

    def test_deepcopy_behavior(self):
        """Test that the function creates a deep copy."""
        nested_list = [1, 2, 3]
        input_dict = {
            "data": nested_list,
            "logprobs": "remove",
        }

        result = strip_logprobs(input_dict)

        result["data"].append(4)
        assert nested_list == [1, 2, 3]  # Original unchanged
        assert result["data"] == [1, 2, 3, 4]

    def test_deepcopy_failure_returns_original(self, caplog):
        """Test that deepcopy failure returns original object and logs warning."""

        class UnCopyableObject:
            def __init__(self):
                self.data = "value"
                self.logprobs = "should_remain"

            def __deepcopy__(self, memo):
                raise RuntimeError("Cannot deepcopy this object")

        obj = UnCopyableObject()

        with caplog.at_level(logging.WARNING):
            result = strip_logprobs(obj)

        # Should return the original object unchanged
        assert result is obj
        assert result.logprobs == "should_remain"

        # Check that warning was logged
        assert len(caplog.records) == 1
        assert "Failed to deepcopy object in strip_logprobs" in caplog.text
        assert "Cannot deepcopy this object" in caplog.text
        assert "Returning original object unchanged" in caplog.text

    def test_deepcopy_failure_with_recursion_error(self, caplog):
        """Test handling of RecursionError during deepcopy."""

        class RecursiveObject:
            def __init__(self):
                self.data = "value"
                self.logprobs = [1, 2, 3]

            def __deepcopy__(self, memo):
                raise RecursionError("maximum recursion depth exceeded")

        obj = RecursiveObject()

        with caplog.at_level(logging.WARNING):
            result = strip_logprobs(obj)

        # Should return the original object unchanged
        assert result is obj
        assert result.logprobs == [1, 2, 3]

        # Check that warning was logged
        assert "Failed to deepcopy object in strip_logprobs" in caplog.text
        assert "maximum recursion depth exceeded" in caplog.text
