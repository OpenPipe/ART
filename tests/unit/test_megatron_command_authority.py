import ast
from pathlib import Path

_ROOT = Path(__file__).parents[2]
_SERVICE = _ROOT / "src/art/megatron/distributed_service.py"
_EXECUTOR = _ROOT / "src/art/megatron/runtime/executor.py"
_MONARCH = _ROOT / "src/art/megatron/runtime/monarch.py"
_COORDINATOR = _ROOT / "src/art/megatron/slot_coordinator.py"


def _class_methods(path: Path, class_name: str) -> dict[str, str]:
    source = path.read_text()
    tree = ast.parse(source)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name: ast.get_source_segment(source, node) or ""
        for node in owner.body
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef))
    }


def test_normal_training_uses_one_mcore_command_authority() -> None:
    service = _class_methods(_SERVICE, "DistributedMegatronService")
    actor = _class_methods(_MONARCH, "MonarchTrainerActor")
    legacy_executor = _class_methods(_EXECUTOR, "MegatronTrainJobExecutor")
    trainer = _class_methods(_MONARCH, "MonarchTrainerRun")
    coordinator = _class_methods(_COORDINATOR, "MegatronSlotCoordinator")

    assert "start_pipeline_forward_backward(" in service["train_packed"]
    assert "start_pipeline_optimizer(" in service["train_packed"]
    assert "trainer.train(" not in service["train_packed"]
    assert "_start_pipeline_forward_backward_command(" in service["train_sft"]
    assert "start_pipeline_optimizer(" in service["train_sft"]
    assert "trainer.train_sft(" not in service["train_sft"]

    optimizer = service["start_pipeline_optimizer"]
    assert "forward.completion" not in optimizer
    assert "prefetch_command_run_residency(" in optimizer
    assert "admit_command_run_residency(" not in optimizer
    assert "start_command_generation_publication(" in optimizer
    assert "publish_command_generation(" not in optimizer
    assert "trainer=trainer" in optimizer
    actor_optimizer = actor["execute_optimizer"]
    assert actor_optimizer.index("admit_residency(") < actor_optimizer.index(
        "execute_optimizer(job)"
    )

    for name in ("execute_forward_backward", "execute_sft_forward_backward"):
        source = actor[name]
        assert source.index("ready_port.send(") < source.index(
            "self._run_slot_executor.execute_"
        )
        assert "self._executor.execute_" not in source

    assert "start_resident_forward_backward" not in trainer
    assert "resident_optim_step_and_publish" not in trainer
    assert "execute_resident_optimizer_and_publish" not in actor
    assert "publish_split_generation" not in legacy_executor

    assert (
        "state.open_forward_backward_ids" not in trainer["publish_command_generation"]
    )
    assert (
        "state.open_forward_backward_ids"
        not in trainer["export_command_run_checkpoint"]
    )
    assert "state.open_forward_backward_ids" in trainer["record_control_command"]
    assert "state.open_forward_backward_ids" in trainer["drain_command_run"]
    assert "retained_contribution_inputs()" not in coordinator["export_run_checkpoint"]
