from types import SimpleNamespace
from typing import Any, cast

from art.megatron.runtime.executor import (
    MCoreRunSlotExecutor,
    _ResidentCommandRun,
)
from art.megatron.runtime.specs import TrainingRunSpec


class _Record:
    def model_dump(self, *, mode: str) -> dict[str, int]:
        assert mode == "json"
        return {"rank": 0}


class _Publisher:
    def __init__(self) -> None:
        self.adapter_config: dict[str, Any] | None = None

    def submit_command(
        self,
        spec: Any,
        *,
        adapter_config: dict[str, Any],
        sink: Any,
    ) -> dict[str, float]:
        del spec
        self.adapter_config = adapter_config
        sink.future.set_result(_Record())
        return {"snapshot": 0.5}


def _publication_spec() -> Any:
    return SimpleNamespace(
        run_id="run",
        generation=SimpleNamespace(
            policy_step=3,
            training_session_id="session",
            generation_id="generation",
        ),
        optimizer_state_path="optimizer",
        staging_adapter_path="adapter",
        publication_targets=(object(),),
    )


def test_command_publication_uses_resident_adapter_config() -> None:
    adapter_config = {"r": 4, "target_modules": ["q_proj", "v_proj"]}
    publisher = _Publisher()
    executor = object.__new__(MCoreRunSlotExecutor)
    executor.runtime = SimpleNamespace(rank=0)
    executor._runs = {
        "run": _ResidentCommandRun(
            spec=cast(
                TrainingRunSpec,
                SimpleNamespace(training_session_id="session"),
            ),
            learner_version=3,
            gradients=SimpleNamespace(contribution_ids=()),
            adapter_config=adapter_config,
        )
    }
    executor._publisher = publisher

    result = executor.publish_generation(_publication_spec())

    assert publisher.adapter_config is adapter_config
    assert result["record"] == {"rank": 0}
    assert result["metrics"] == {"snapshot": 0.5}
