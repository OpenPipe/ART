"""Tests for TrajectoryGroup copy and deepcopy functionality."""

import copy

from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import pytest

from art.trajectories import PydanticException, Trajectory, TrajectoryGroup


@pytest.fixture
def sample_trajectory():
    """Create a sample trajectory for testing."""
    return Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "Hello"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="Hi there!",
                    refusal=None,
                ),
            ),
        ],
        tools=None,
        reward=1.0,
        metrics={"accuracy": 0.95},
        metadata={"test": "value"},
    )


@pytest.fixture
def sample_trajectory_group(sample_trajectory):
    """Create a sample trajectory group for testing."""
    trajectory2 = Trajectory(
        messages_and_choices=[
            {"role": "user", "content": "How are you?"},
            Choice(
                finish_reason="stop",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    role="assistant",
                    content="I'm doing well!",
                    refusal=None,
                ),
            ),
        ],
        tools=None,
        reward=0.8,
    )
    return TrajectoryGroup(
        trajectories=[sample_trajectory, trajectory2],
        exceptions=[],
    )


def test_shallow_copy(sample_trajectory_group):
    """Test that shallow copy works correctly."""
    copied = copy.copy(sample_trajectory_group)

    # Should be a different object
    assert copied is not sample_trajectory_group

    # Trajectories should be a new list (shallow copy of list)
    assert copied.trajectories is not sample_trajectory_group.trajectories

    # But the trajectory objects themselves should be the same (shallow copy)
    assert copied.trajectories[0] is sample_trajectory_group.trajectories[0]
    assert copied.trajectories[1] is sample_trajectory_group.trajectories[1]

    # Exceptions should be a new list with same contents
    assert copied.exceptions is not sample_trajectory_group.exceptions
    assert copied.exceptions == sample_trajectory_group.exceptions


def test_deep_copy(sample_trajectory_group):
    """Test that deep copy works correctly."""
    copied = copy.deepcopy(sample_trajectory_group)

    # Should be a different object
    assert copied is not sample_trajectory_group

    # Should have different trajectories list (deep copy)
    assert copied.trajectories is not sample_trajectory_group.trajectories

    # Trajectories themselves should be different objects
    assert copied.trajectories[0] is not sample_trajectory_group.trajectories[0]
    assert copied.trajectories[1] is not sample_trajectory_group.trajectories[1]

    # But should have same content
    assert len(copied.trajectories) == len(sample_trajectory_group.trajectories)
    assert (
        copied.trajectories[0].reward == sample_trajectory_group.trajectories[0].reward
    )
    assert (
        copied.trajectories[1].reward == sample_trajectory_group.trajectories[1].reward
    )

    # Exceptions should also be deep copied
    assert copied.exceptions is not sample_trajectory_group.exceptions


def test_deep_copy_with_exceptions():
    """Test that deep copy works with exceptions."""
    group = TrajectoryGroup(
        trajectories=[
            Trajectory(
                messages_and_choices=[{"role": "user", "content": "test"}],
                tools=None,
                reward=1.0,
            )
        ],
        exceptions=[ValueError("test error")],
    )

    copied = copy.deepcopy(group)

    # Should be different objects
    assert copied is not group
    assert copied.exceptions is not group.exceptions

    # Should have same exception content
    assert len(copied.exceptions) == len(group.exceptions)
    assert copied.exceptions[0].message == group.exceptions[0].message


def test_deep_copy_circular_reference():
    """Test that deep copy handles circular references correctly."""
    group = TrajectoryGroup(
        trajectories=[
            Trajectory(
                messages_and_choices=[{"role": "user", "content": "test"}],
                tools=None,
                reward=1.0,
            )
        ],
        exceptions=[],
    )

    # Create a memo dict with a circular reference
    memo = {}
    copied = copy.deepcopy(group, memo)

    # Should be in memo
    assert id(group) in memo
    assert memo[id(group)] is copied

    # Copying again with same memo should return the same object
    copied2 = copy.deepcopy(group, memo)
    assert copied2 is copied


def test_deep_copy_preserves_metadata(sample_trajectory_group):
    """Test that deep copy preserves trajectory metadata."""
    copied = copy.deepcopy(sample_trajectory_group)

    # Check that metadata is preserved
    assert (
        copied.trajectories[0].metrics
        == sample_trajectory_group.trajectories[0].metrics
    )
    assert (
        copied.trajectories[0].metadata
        == sample_trajectory_group.trajectories[0].metadata
    )

    # But should be different dict objects
    assert (
        copied.trajectories[0].metrics
        is not sample_trajectory_group.trajectories[0].metrics
    )
    assert (
        copied.trajectories[0].metadata
        is not sample_trajectory_group.trajectories[0].metadata
    )


def test_copy_empty_group():
    """Test copying an empty trajectory group."""
    empty_group = TrajectoryGroup(trajectories=[], exceptions=[])

    shallow = copy.copy(empty_group)
    assert shallow is not empty_group
    assert len(shallow.trajectories) == 0

    deep = copy.deepcopy(empty_group)
    assert deep is not empty_group
    assert len(deep.trajectories) == 0
