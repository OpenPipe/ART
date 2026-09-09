"""Tests for trajectory and group copy semantics."""

import copy
from datetime import UTC, datetime

from anthropic.types import Message
from openai.types import Completion
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.responses import Response
import pytest

from art.trajectories import (
    ChatCompletionsExchange,
    CompletionsExchange,
    MessagesExchange,
    PydanticException,
    ResponsesExchange,
    Trajectory,
    TrajectoryExchanges,
    TrajectoryGroup,
)


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


@pytest.fixture
def captured_trajectory():
    class TaggedExchanges(TrajectoryExchanges):
        tag: str = "exchanges"

    class TaggedTrajectory(Trajectory):
        tag: str = "trajectory"

    protocols = {}
    for name, exchange_type, response_type in (
        ("chat_completions", ChatCompletionsExchange, ChatCompletion),
        ("completions", CompletionsExchange, Completion),
        ("responses", ResponsesExchange, Response),
        ("messages", MessagesExchange, Message),
    ):
        protocols[name] = [
            exchange_type.model_construct(
                request={"model": model} if model not in ("world", None) else {},
                response=response_type.model_construct(
                    model=model if model in ("world", None) else "served"
                ),
                start_time=datetime.now(UTC),
                end_time=datetime.now(UTC),
            )
            for model in ("policy@1", "policy@2", "policy@*", "Policy@3", "world", None)
        ]
    trajectory = TaggedTrajectory(exchanges=TaggedExchanges.model_validate(protocols))
    trajectory.metadata["exchange"] = trajectory.exchanges.chat_completions[0]
    trajectory._policy_token_counts = {1: 3}
    return trajectory


@pytest.mark.parametrize("owner", ["trajectory", "exchanges"])
@pytest.mark.parametrize(
    "include,exclude,indices",
    [
        (None, None, range(6)),
        ("policy@*", None, [2]),
        ("policy@?", "policy@2", [0, 2]),
        (["policy@1", "world"], None, [0, 4]),
        ("*", "policy@*", [0, 1, 3, 4]),
        (None, ["world", "Policy@*"], [0, 1, 2, 5]),
        ([], None, []),
        (None, [], range(6)),
        (None, "*", [5]),
        ("missing", None, []),
        (None, "served", range(6)),
    ],
)
def test_model_copy_filters(captured_trajectory, owner, include, exclude, indices):
    original = captured_trajectory
    source = original if owner == "trajectory" else original.exchanges
    copied = source.model_copy(
        include_models=iter(include) if isinstance(include, list) else include,
        exclude_models=iter(exclude) if isinstance(exclude, list) else exclude,
    )
    assert type(copied) is type(source)
    exchanges = copied.exchanges if owner == "trajectory" else copied
    assert type(exchanges) is type(original.exchanges)
    for name in TrajectoryExchanges.model_fields:
        before, after = getattr(original.exchanges, name), getattr(exchanges, name)
        assert after is not before
        assert [id(item) for item in after] == [id(before[index]) for index in indices]
        after.clear()
        assert len(before) == 6


@pytest.mark.parametrize("deep", [False, True])
def test_model_copy_depth_and_aliases(captured_trajectory, deep):
    copied = captured_trajectory.model_copy(include_models="policy@1", deep=deep)
    original_exchange = captured_trajectory.exchanges.chat_completions[0]
    copied_exchange = copied.exchanges.chat_completions[0]
    assert copied.metadata["exchange"] is copied_exchange
    assert (copied_exchange is original_exchange) is not deep
    assert (copied_exchange.request is original_exchange.request) is not deep
    assert (copied.metadata is captured_trajectory.metadata) is not deep
    assert (copied.metrics is captured_trajectory.metrics) is not deep
    assert (
        copied._policy_token_counts is captured_trajectory._policy_token_counts
    ) is not deep


@pytest.mark.parametrize("deep", [False, True])
@pytest.mark.parametrize("exchange_deep", [False, True])
def test_model_copy_explicit_exchanges(captured_trajectory, deep, exchange_deep):
    exchanges = captured_trajectory.exchanges.model_copy(deep=exchange_deep)
    copied = captured_trajectory.model_copy(
        exclude_models="*", update={"exchanges": exchanges}, deep=deep
    )
    assert copied.exchanges is exchanges
    assert (copied.metrics is captured_trajectory.metrics) is not deep
    for name in TrajectoryExchanges.model_fields:
        before, after = (
            getattr(captured_trajectory.exchanges, name),
            getattr(exchanges, name),
        )
        assert after is not before
        assert (after[0] is before[0]) is not exchange_deep


@pytest.mark.parametrize("deep", [False, True])
def test_model_copy_updates_are_unvalidated(captured_trajectory, deep):
    replacement = object()
    update = {"exchanges": replacement, "reward": replacement}
    copied = captured_trajectory.model_copy(update=update, deep=deep)
    assert copied.exchanges is copied.reward is replacement
    assert update == {"exchanges": replacement, "reward": replacement}
    exchanges = captured_trajectory.exchanges.model_copy(
        include_models=[], update={"messages": replacement}, deep=deep
    )
    assert exchanges.messages is replacement
    assert not exchanges.chat_completions


def test_model_copy_filters_before_deepcopy(captured_trajectory):
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise AssertionError("Excluded exchange was copied")

    for name in TrajectoryExchanges.model_fields:
        getattr(captured_trajectory.exchanges, name)[-1].request["extra"] = Uncopyable()
    for source in (captured_trajectory, captured_trajectory.exchanges):
        source.model_copy(include_models="policy@1", deep=True)


@pytest.mark.parametrize("deep", [False, True])
def test_model_copy_filters_leave_legacy_history(sample_trajectory, deep):
    copied = sample_trajectory.model_copy(include_models=[], deep=deep)
    assert copied.messages_and_choices == sample_trajectory.messages_and_choices
    assert (
        copied.messages_and_choices is sample_trajectory.messages_and_choices
    ) is not deep
