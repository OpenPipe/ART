from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from .oracle_harness import (  # noqa: E402
    LIVE_TRAINING_LOG_PATH,
    MetricThresholdRule,
    OracleCaseConfig,
    OracleObjective,
    PhasePassFn,
    VariantReport,
    VariantRunner,
    VariantSpec,
    available_gpu_count,
    selected_oracle_objectives,
    selected_suite_topologies,
)

BASE_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
NUM_LAYERS = 4
REPO_ROOT = Path(__file__).resolve().parents[4]
CORRECTNESS_LOG_PATH = REPO_ROOT / ".local" / "correctness.log"
ORACLE_LIVE_TRAINING_LOG_ENV = "ART_ORACLE_LIVE_TRAINING_LOG"
_EXPECTED_COMPRESS_RATIOS = [0, 0, 4, 128]
_EXPECTED_LAYER_TYPES = [
    "sliding_attention",
    "sliding_attention",
    "compressed_sparse_attention",
    "heavily_compressed_attention",
]
_EXPECTED_MLP_LAYER_TYPES = ["hash_moe", "hash_moe", "hash_moe", "moe"]


def test_dsv4_real_path_bf16_correctness(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runs the real DSV4 ART Megatron path through the standard oracle harness."""
    _assert_representative_dsv4_layers()
    gpu_count = available_gpu_count()
    reports: list[VariantReport] = []
    with capsys.disabled():
        print(
            f"\nDSV4 real-path bf16 correctness log: {CORRECTNESS_LOG_PATH}",
            flush=True,
        )
        print(
            f"DSV4 real-path bf16 live training log: {LIVE_TRAINING_LOG_PATH}",
            flush=True,
        )
    CORRECTNESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_TRAINING_LOG_PATH.write_text(
        (
            "DSV4 real-path bf16 live training log.\n"
            "Topology worker output is appended below. If no topology sections "
            "appear, complete cached artifacts were reused.\n"
        ),
        encoding="utf-8",
    )
    with _temporary_env(**{ORACLE_LIVE_TRAINING_LOG_ENV: str(LIVE_TRAINING_LOG_PATH)}):
        with CORRECTNESS_LOG_PATH.open("w", encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                print("DSV4 real-path bf16 correctness")
                print(f"base_model={BASE_MODEL}")
                print(f"num_layers={NUM_LAYERS}")
                print(f"precision=bf16")
                print(f"visible_gpus={gpu_count}")
                print(f"live_training_log={LIVE_TRAINING_LOG_PATH}")
                for objective in selected_oracle_objectives():
                    runner = VariantRunner(
                        objective=objective,
                        case_config=OracleCaseConfig(
                            base_model=BASE_MODEL,
                            precision="bf16",
                            num_layers=NUM_LAYERS,
                        ),
                        use_fp32_lora_reference=False,
                    )
                    variants = _dsv4_bf16_variants(
                        objective=objective,
                        max_world_size=gpu_count,
                    )
                    if variants:
                        reports.extend(runner.run_suite(variants))
    if not reports:
        CORRECTNESS_LOG_PATH.write_text(
            f"DSV4 real-path bf16 correctness skipped. Need at least 2 GPUs; found {gpu_count}.\n",
            encoding="utf-8",
        )
        pytest.skip(f"Need at least 2 GPUs for DSV4 correctness; found {gpu_count}.")
    assert all(report.signal == "pass" for report in reports)


@contextmanager
def _temporary_env(**updates: str) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _dsv4_bf16_variants(
    *,
    objective: OracleObjective,
    max_world_size: int,
) -> list[VariantSpec]:
    phase_pass = _dsv4_bf16_phase_pass_fns()
    variants: list[VariantSpec] = []
    for topology in selected_suite_topologies(is_moe=True, cp_supported=False)[1:]:
        if topology.world_size() > max_world_size:
            continue
        variants.append(
            VariantSpec(
                name=f"{objective}_dsv4_bf16_topology_{topology.slug()}",
                objective=objective,
                topology=topology,
                pass_fn_by_phase=phase_pass,
            )
        )
    return variants


def _dsv4_bf16_phase_pass_fns() -> dict[str, PhasePassFn]:
    non_zero_scales = {"typical_abs_scale": 0.0, "candidate_abs_scale": 0.0}
    fwd = MetricThresholdRule(
        limits={"mean_abs_pct": 3.0},
        minimums=non_zero_scales,
    )
    loss = MetricThresholdRule(limits={"mean_abs_pct": 3.0})
    grad = MetricThresholdRule(
        limits={"mean_abs_pct": 5.0},
        minimums=non_zero_scales,
    )
    router_topk = MetricThresholdRule(
        limits={"topk_mismatch_fraction": 0.0, "top1_mismatch_fraction": 0.0}
    )
    return {
        "forward": fwd,
        "outputs": fwd,
        "losses": loss,
        "grads": grad,
        "deltas": grad,
        "router_scores": fwd,
        "router_topk_ids": router_topk,
    }


def _assert_representative_dsv4_layers() -> None:
    from transformers import AutoConfig

    from art.megatron.dsv4.hf_config import ensure_dsv4_hf_model_registered

    ensure_dsv4_hf_model_registered()
    config = AutoConfig.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert list(config.compress_ratios[:NUM_LAYERS]) == _EXPECTED_COMPRESS_RATIOS
    assert list(config.layer_types[:NUM_LAYERS]) == _EXPECTED_LAYER_TYPES
    assert list(config.mlp_layer_types[:NUM_LAYERS]) == _EXPECTED_MLP_LAYER_TYPES
