from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

import art
from art.verifiers import (
    trajectory_to_verifiers_state,
    verifiers_state_to_trajectory,
    verifiers_states_to_trajectory_group,
)


def assistant_choice(content: str, *, finish_reason: str = "stop") -> Choice:
    return Choice(
        finish_reason=finish_reason,
        index=0,
        logprobs=None,
        message=ChatCompletionMessage(
            role="assistant",
            content=content,
            refusal=None,
        ),
    )


def test_trajectory_to_verifiers_state_splits_choice_turns():
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "First?"},
            assistant_choice("one"),
            {"role": "user", "content": "Second?"},
            assistant_choice("two", finish_reason="length"),
        ],
        reward=0.75,
        metrics={"exact": 1.0},
        metadata={"scenario": "smoke"},
        logs=["converted"],
    )

    state = trajectory_to_verifiers_state(
        trajectory,
        task={"prompt": [{"role": "user", "content": "First?"}], "info": {"id": "t1"}},
        example_id="ex-1",
        answer="two",
    )

    assert state["reward"] == 0.75
    assert state["metrics"] == {"exact": 1.0}
    assert state["metadata"] == {"scenario": "smoke"}
    assert state["logs"] == ["converted"]
    assert state["example_id"] == "ex-1"
    assert state["answer"] == "two"
    assert state["info"] == {"id": "t1"}

    steps = state["trajectory"]
    assert len(steps) == 2
    assert steps[0]["prompt"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "First?"},
    ]
    assert steps[0]["completion"] == [{"role": "assistant", "content": "one"}]
    assert steps[0]["reward"] is None
    assert steps[1]["prompt"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "Second?"},
    ]
    assert steps[1]["completion"] == [{"role": "assistant", "content": "two"}]
    assert steps[1]["is_truncated"] is True
    assert steps[1]["reward"] == 0.75
    assert steps[1]["extras"]["art_metrics"] == {"exact": 1.0}
    assert state["is_truncated"] is True


def test_verifiers_state_to_trajectory_reconstructs_transcript():
    state = {
        "prompt": [{"role": "user", "content": "First?"}],
        "completion": [{"role": "assistant", "content": "one"}],
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "First?"}],
                "completion": [{"role": "assistant", "content": "one"}],
            },
            {
                "prompt": [
                    {"role": "user", "content": "First?"},
                    {"role": "assistant", "content": "one"},
                    {"role": "user", "content": "Second?"},
                ],
                "completion": [{"role": "assistant", "content": "two"}],
            },
        ],
        "reward": 1.0,
        "metrics": {"answer": 1.0},
        "metadata": {"task_id": "t1"},
        "stop_condition": "done",
    }

    trajectory = verifiers_state_to_trajectory(state)

    assert trajectory.reward == 1.0
    assert trajectory.metrics == {"answer": 1.0}
    assert trajectory.metadata == {"task_id": "t1", "stop_condition": "done"}
    assert trajectory.messages_and_choices == [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "two"},
    ]


def test_verifiers_state_to_trajectory_preserves_completion_tail_after_steps():
    state = {
        "prompt": [{"role": "user", "content": "First?"}],
        "completion": [
            {"role": "assistant", "content": "one"},
            {"role": "tool", "content": "tail result"},
        ],
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "First?"}],
                "completion": [{"role": "assistant", "content": "one"}],
            },
        ],
    }

    trajectory = verifiers_state_to_trajectory(state)

    assert trajectory.messages_and_choices == [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "one"},
        {"role": "tool", "content": "tail result"},
    ]


def test_verifiers_state_to_trajectory_can_restore_trainable_choices():
    def choice_factory(message, step):
        assert step["reward"] == 1.0
        return assistant_choice(message["content"])

    trajectory = verifiers_state_to_trajectory(
        {
            "trajectory": [
                {
                    "prompt": [{"role": "user", "content": "Say hi"}],
                    "completion": [{"role": "assistant", "content": "hi"}],
                    "reward": 1.0,
                }
            ],
            "reward": 1.0,
        },
        choice_factory=choice_factory,
    )

    assert trajectory.messages_and_choices[0] == {"role": "user", "content": "Say hi"}
    assert isinstance(trajectory.messages_and_choices[1], Choice)
    assert trajectory.messages() == [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "hi"},
    ]


def test_verifiers_states_to_trajectory_group():
    group = verifiers_states_to_trajectory_group(
        [
            {
                "prompt": [{"role": "user", "content": "A"}],
                "completion": [{"role": "assistant", "content": "B"}],
                "reward": 0.25,
            },
            {
                "prompt": [{"role": "user", "content": "C"}],
                "completion": [{"role": "assistant", "content": "D"}],
                "reward": 0.5,
            },
        ]
    )

    assert len(group.trajectories) == 2
    assert [trajectory.reward for trajectory in group.trajectories] == [0.25, 0.5]
