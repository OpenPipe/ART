from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path

import pytest

from ..model_support.oracle_harness import (
    LoraConfig,
    PackedTensorConfig,
    Topology,
    VariantRunner,
    VariantSpec,
    _prune_case_artifacts,
    _prune_topology_artifacts,
    available_gpu_count,
    case_config,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
LOG_PATH = REPO_ROOT / ".local" / "qwen35_gdn_cp_topology_oracle.log"


_CP_SIZES = (2, 4, 8)


@pytest.mark.parametrize("cp_size", _CP_SIZES)
def test_qwen35_gdn_prefix_tree_cp_topology_oracle(
    cp_size: int,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runs a real Qwen3.5 GDN-only RL stack under CP without self-attn CP."""
    gpu_count = available_gpu_count()
    if gpu_count < cp_size:
        pytest.skip(f"Need {cp_size} GPUs for CP{cp_size}; found {gpu_count}.")

    topology = Topology(tp=1, ep=1, etp=1, dp=1, sp=False, cp=cp_size)
    config = case_config(base_model="Qwen/Qwen3.5-35B-A3B").model_copy(
        update={
            "num_layers": 1,
            "precision": "bf16",
            "grad_accumulation_sequences": 1,
            "lora": LoraConfig(
                rank=1,
                alpha=32,
                target_modules=[
                    "in_proj_qkv",
                    "in_proj_z",
                    "out_proj",
                ],
            ),
            "packed_tensors": PackedTensorConfig(
                num_sequences=2,
                sequence_length=24,
                prefill_tokens=4,
                completion_branches_per_prefix=2,
                decode_tokens=3,
                decode_tokens_jitter=1,
                vocab_high=128,
            ),
        }
    )
    variant = VariantSpec(
        name=f"qwen35_gdn_prefix_tree_cp{cp_size}",
        objective="rl",
        topology=topology,
    )

    monkeypatch.setenv("ART_MEGATRON_RECOMPUTE_GRANULARITY", "disabled")
    monkeypatch.setenv("ART_MEGATRON_RECOMPUTE_METHOD", "disabled")
    monkeypatch.setenv("ART_MEGATRON_RECOMPUTE_NUM_LAYERS", "disabled")
    monkeypatch.setenv("ART_MEGATRON_RECOMPUTE_MODULES", "disabled")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with capsys.disabled():
        print(f"\nQwen3.5 GDN CP topology oracle log: {LOG_PATH}", flush=True)
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            runner = VariantRunner(objective="rl", case_config=config)
            topology_dir = runner._run_topology(
                topology=topology,
                output_slug=variant.resolved_output_slug(),
                mutation=None,
                replay_bundle_dir=None,
                capture_bundle_dir=None,
                regenerate=True,
            )
    manifest = json.loads((topology_dir / "manifest.json").read_text())
    assert manifest["topology"] == topology.slug()
    assert manifest["num_layers"] == 1
    assert len(manifest["steps"]) == config.num_steps
    assert "finished step_index=0" in (topology_dir / "worker.log").read_text()
    _prune_topology_artifacts(topology_dir)
    _prune_case_artifacts(Path(runner.case_artifacts.case_dir))
