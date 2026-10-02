import inspect
import sys
import types

import pytest

import art
from art.verifiers import (
    normalize_verifiers_rollout_output,
    normalize_verifiers_rollout_outputs,
    rollout_output_from_trajectory,
    rollout_outputs_from_trajectory_group,
    rollout_with_verifiers_environment,
    trajectory_from_verifiers_rollout,
    trajectory_group_from_verifiers_outputs,
    trajectory_group_with_verifiers_environment,
)


def test_trajectory_from_verifiers_rollout_reconstructs_multiturn_steps():
    output = {
        "example_id": 7,
        "prompt": [{"role": "user", "content": "Find the invoice"}],
        "reward": 1.0,
        "metrics": {"accuracy": 1.0, "notes": "ignored"},
        "is_completed": True,
        "is_truncated": False,
        "stop_condition": "answer_ready",
        "tool_defs": [
            {
                "name": "search",
                "description": "Search mail",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "Find the invoice"}],
                "completion": [{"role": "assistant", "content": "Searching"}],
            },
            {
                "prompt": [
                    {"role": "user", "content": "Find the invoice"},
                    {"role": "assistant", "content": "Searching"},
                    {"role": "tool", "tool_call_id": "t1", "content": "Invoice #42"},
                ],
                "completion": [{"role": "assistant", "content": "Invoice #42"}],
            },
        ],
    }

    trajectory = trajectory_from_verifiers_rollout(output)

    assert trajectory.reward == 1.0
    assert trajectory.metrics["accuracy"] == 1.0
    assert trajectory.metadata["verifiers_example_id"] == 7
    assert trajectory.metadata["verifiers_stop_condition"] == "answer_ready"
    assert trajectory.messages_and_choices == [
        {"role": "user", "content": "Find the invoice"},
        {"role": "assistant", "content": "Searching"},
        {"role": "tool", "tool_call_id": "t1", "content": "Invoice #42"},
        {"role": "assistant", "content": "Invoice #42"},
    ]
    assert trajectory.tools == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search mail",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_trajectory_from_verifiers_rollout_falls_back_to_prompt_completion():
    output = {
        "prompt": [{"role": "user", "content": "2 + 2?"}],
        "completion": [{"role": "assistant", "content": "4"}],
        "reward": 1,
    }

    trajectory = trajectory_from_verifiers_rollout(output)

    assert trajectory.messages_and_choices == [
        {"role": "user", "content": "2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_rollout_output_from_trajectory_splits_prompt_and_completion():
    trajectory = art.Trajectory(
        messages_and_choices=[
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "2 + 2?"},
            {"role": "assistant", "content": "4"},
        ],
        reward=0.75,
        metrics={"accuracy": 1.0},
        metadata={"trajectory_id": "abc"},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    output = rollout_output_from_trajectory(trajectory, example_id=3)

    assert output["example_id"] == 3
    assert output["prompt"] == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "2 + 2?"},
    ]
    assert output["completion"] == [{"role": "assistant", "content": "4"}]
    assert output["reward"] == 0.75
    assert output["metrics"] == {"accuracy": 1.0}
    assert output["tool_defs"] == [
        {
            "name": "calculator",
            "description": "Calculate",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    assert output["trajectory"][0]["trajectory_id"] == "abc"


def test_trajectory_group_from_verifiers_outputs():
    group = trajectory_group_from_verifiers_outputs(
        [
            {"prompt": "first", "completion": [{"role": "assistant", "content": "a"}]},
            {"prompt": "second", "completion": [{"role": "assistant", "content": "b"}]},
        ]
    )

    assert len(group) == 2
    assert group.trajectories[0].messages_and_choices[0] == {
        "role": "user",
        "content": "first",
    }


def test_rollout_outputs_from_trajectory_group_assigns_example_ids():
    group = art.TrajectoryGroup(
        [
            art.Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "a"},
                ],
                reward=1.0,
            ),
            art.Trajectory(
                messages_and_choices=[
                    {"role": "user", "content": "second"},
                    {"role": "assistant", "content": "b"},
                ],
                reward=0.5,
            ),
        ]
    )

    outputs = rollout_outputs_from_trajectory_group(group, first_example_id=10)

    assert [output["example_id"] for output in outputs] == [10, 11]
    assert [output["reward"] for output in outputs] == [1.0, 0.5]


def test_normalize_verifiers_rollout_output_round_trips_through_art():
    output = {
        "example_id": 42,
        "prompt": [{"role": "user", "content": "Use a tool"}],
        "completion": [{"role": "assistant", "content": "done"}],
        "reward": 0.25,
        "metrics": {"score": 0.25, "label": "ignored"},
        "logs": ["started", "scored"],
        "answer": "done",
        "stop_condition": "final_answer",
        "is_completed": True,
        "is_truncated": True,
        "timing": {"total": 1.5},
        "tool_defs": [
            {
                "name": "lookup",
                "description": "Lookup records",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }

    normalized = normalize_verifiers_rollout_output(output)

    assert normalized["example_id"] == 42
    assert normalized["prompt"] == [{"role": "user", "content": "Use a tool"}]
    assert normalized["completion"] == [{"role": "assistant", "content": "done"}]
    assert normalized["reward"] == 0.25
    assert normalized["metrics"]["score"] == 0.25
    assert "label" not in normalized["metrics"]
    assert normalized["logs"] == ["started", "scored"]
    assert normalized["answer"] == "done"
    assert normalized["stop_condition"] == "final_answer"
    assert normalized["is_completed"] is True
    assert normalized["is_truncated"] is True
    assert normalized["timing"] == {"total": 1.5}
    assert normalized["tool_defs"] == output["tool_defs"]
    assert normalized["trajectory"][0]["extras"]["art_metadata"]["verifiers_example_id"] == 42


def test_normalize_verifiers_rollout_outputs_handles_iterables():
    outputs = normalize_verifiers_rollout_outputs(
        [
            {"example_id": 1, "prompt": "first", "completion": [{"role": "assistant", "content": "a"}], "reward": 1},
            {"example_id": 2, "prompt": "second", "completion": [{"role": "assistant", "content": "b"}], "reward": 0},
        ],
        include_trajectory=False,
    )

    assert [output["example_id"] for output in outputs] == [1, 2]
    assert "trajectory" not in outputs[0]
    assert outputs[0]["completion"] == [{"role": "assistant", "content": "a"}]


def test_real_verifiers_package_contract_if_installed():
    verifiers = pytest.importorskip("verifiers")
    client_module = pytest.importorskip(
        "verifiers.clients.openai_chat_completions_client"
    )
    env_module = pytest.importorskip("verifiers.envs.environment")

    assert hasattr(client_module, "OpenAIChatCompletionsClient")

    rollout_params = inspect.signature(env_module.Environment.run_rollout).parameters
    assert {"input", "client", "model", "sampling_args"} <= set(rollout_params)
    assert {"max_retries", "state_columns"} <= set(rollout_params)

    group_params = inspect.signature(env_module.Environment.run_group).parameters
    assert {"group_inputs", "client", "model", "sampling_args"} <= set(group_params)
    assert {"max_retries", "state_columns"} <= set(group_params)

    output = verifiers.RolloutOutput(
        example_id=7,
        prompt=[{"role": "user", "content": "hi"}],
        completion=[{"role": "assistant", "content": "hello"}],
        reward=1.0,
        timing={"total": 0.1},
        is_completed=True,
        is_truncated=False,
        metrics={"score": 1.0},
        answer="hello",
        info={},
        error=None,
        stop_condition="final_answer",
        trajectory=[],
        tool_defs=[],
        token_usage={},
    )

    trajectory = trajectory_from_verifiers_rollout(output)

    assert trajectory.reward == 1.0
    assert trajectory.metrics["score"] == 1.0
    assert trajectory.metadata["verifiers_example_id"] == 7
    assert trajectory.messages_and_choices == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


async def test_rollout_with_verifiers_environment_uses_art_model_client(monkeypatch):
    _install_fake_verifiers_client(monkeypatch)
    env = _FakeVerifiersEnv()
    model = _FakeArtModel()

    trajectory = await rollout_with_verifiers_environment(
        env,
        model,
        {"prompt": [{"role": "user", "content": "hi"}], "example_id": 1},
        sampling_args={"temperature": 0.2},
        state_columns=("trajectory", "custom"),
    )

    assert trajectory.reward == 1.0
    assert trajectory.messages_and_choices[-1] == {
        "role": "assistant",
        "content": "done",
    }
    assert env.last_rollout_call["client"].raw_client == "art-openai-client"
    assert env.last_rollout_call["model"] == "art-model"
    assert env.last_rollout_call["sampling_args"] == {"temperature": 0.2}
    assert env.last_rollout_call["state_columns"] == ["trajectory", "custom"]


async def test_trajectory_group_with_verifiers_environment(monkeypatch):
    _install_fake_verifiers_client(monkeypatch)
    env = _FakeVerifiersEnv()
    model = _FakeArtModel()

    group = await trajectory_group_with_verifiers_environment(
        env,
        model,
        [{"prompt": "a"}, {"prompt": "b"}],
    )

    assert len(group) == 2
    assert env.last_group_call["client"].raw_client == "art-openai-client"
    assert env.last_group_call["model"] == "art-model"
    assert group.trajectories[0].messages_and_choices[-1] == {
        "role": "assistant",
        "content": "group done",
    }


class _FakeArtModel:
    def openai_client(self):
        return "art-openai-client"

    def get_inference_name(self):
        return "art-model"


class _FakeOpenAIChatCompletionsClient:
    def __init__(self, raw_client):
        self.raw_client = raw_client


class _FakeVerifiersEnv:
    def __init__(self):
        self.last_rollout_call = None
        self.last_group_call = None

    async def run_rollout(self, **kwargs):
        self.last_rollout_call = kwargs
        return {
            "prompt": kwargs["input"]["prompt"],
            "completion": [{"role": "assistant", "content": "done"}],
            "reward": 1.0,
        }

    async def run_group(self, **kwargs):
        self.last_group_call = kwargs
        return [
            {
                "prompt": input_value["prompt"],
                "completion": [{"role": "assistant", "content": "group done"}],
                "reward": 1.0,
            }
            for input_value in kwargs["group_inputs"]
        ]


def _install_fake_verifiers_client(monkeypatch):
    verifiers_module = types.ModuleType("verifiers")
    clients_module = types.ModuleType("verifiers.clients")
    client_module = types.ModuleType("verifiers.clients.openai_chat_completions_client")
    client_module.OpenAIChatCompletionsClient = _FakeOpenAIChatCompletionsClient

    monkeypatch.setitem(sys.modules, "verifiers", verifiers_module)
    monkeypatch.setitem(sys.modules, "verifiers.clients", clients_module)
    monkeypatch.setitem(
        sys.modules,
        "verifiers.clients.openai_chat_completions_client",
        client_module,
    )
