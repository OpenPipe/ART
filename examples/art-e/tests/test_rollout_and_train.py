import asyncio

from art_e.data import validation_scenarios
from art_e.rollout import EmailScenario, rollout
from art_e.train import build_model


class FakeModel:
    def get_inference_name(self) -> str:
        return "fake-email-agent"


class FakeAgent:
    def __init__(self, tools):
        self.tools = {tool.name: tool for tool in tools}

    async def ainvoke(self, *_args, **_kwargs):
        return self.tools["return_final_answer_tool"].invoke(
            {
                "answer": "The team offsite is April 9 at the North Pier studio.",
                "source_ids": ["msg_offsite"],
            }
        )


def test_rollout_uses_langgraph_tools_without_provider(monkeypatch) -> None:
    captured = {}

    def fake_init_chat_model(name: str, temperature: float):
        captured["model_name"] = name
        captured["temperature"] = temperature
        return object()

    def fake_create_react_agent(_chat_model, tools):
        captured["tool_names"] = [tool.name for tool in tools]
        return FakeAgent(tools)

    monkeypatch.setattr("art_e.rollout.init_chat_model", fake_init_chat_model)
    monkeypatch.setattr("art_e.rollout.create_react_agent", fake_create_react_agent)

    trajectory = asyncio.run(
        rollout(
            FakeModel(),
            EmailScenario(step=3, scenario=validation_scenarios[0]),
        )
    )

    assert captured == {
        "model_name": "fake-email-agent",
        "temperature": 1.0,
        "tool_names": [
            "search_inbox_tool",
            "read_email_tool",
            "return_final_answer_tool",
        ],
    }
    assert trajectory.reward == 1.0
    assert trajectory.metrics["answer_match"] is True
    assert trajectory.metrics["cited_expected_message"] is True
    assert trajectory.final_answer is not None


def test_build_model_uses_art_e_environment(monkeypatch) -> None:
    monkeypatch.setenv("ART_E_MODEL_NAME", "custom-art-e")
    monkeypatch.setenv("ART_E_BASE_MODEL", "base-model")
    monkeypatch.setenv("ART_E_INFERENCE_MODEL", "inference-model")
    monkeypatch.setenv("ART_E_INFERENCE_BASE_URL", "https://inference.example")
    monkeypatch.setenv("ART_E_INFERENCE_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    model = build_model()

    assert model.name == "custom-art-e"
    assert model.project == "art-e"
    assert model.base_model == "base-model"
    assert model.inference_model_name == "inference-model"
    assert model.inference_base_url == "https://inference.example"
    assert model.inference_api_key == "test-key"
