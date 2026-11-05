"""Parity failures remain blocking while complete evidence reaches F8."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from tests.integration.megatron.glm5next_parity_diagnostics import (
    capture_sublayers,
    margin_top1_gate,
    sublayer_metrics,
    top1_diagnostics,
)
from tests.integration.megatron.glm5next_qualification_launch import has_grant, pipeline


@pytest.mark.parametrize(
    "native,hf,eligible",
    [
        ([16.0, 15.75], [16.0, 15.75], False),  # exactly 2 ULPs: strict >
        ([16.0, 15.5], [16.0, 15.5], True),
        ([16.0, 15.5], [16.0, 16.0], False),  # BOTH margins required
        ([-16.0, -16.25], [-16.0, -16.25], False),
        ([0.0, 0.0], [0.0, 0.0], False),
        ([2.0**-133, -(2.0**-133)], [2.0**-133, -(2.0**-133)], False),
        ([2.0**-131, 0.0], [2.0**-131, 0.0], True),
        ([128.0, 127.0], [128.0, 127.0], False),  # logprob ULP would be wrong
    ],
)
def test_margin_gate_requires_both_raw_margins_strictly_above_two_ulps(
    native, hf, eligible
):
    actual = {"top1": torch.tensor([0]), "top2.logits": torch.tensor([native])}
    reference = {"top1": torch.tensor([1]), "top2.logits": torch.tensor([hf])}
    report = margin_top1_gate(actual, reference)
    assert report["eligible_positions"] == int(eligible)
    assert report["eligible_disagreements"] == int(eligible)
    assert report["status"] == ("fail" if eligible else "pass")
    actual["top1"] = reference["top1"]
    assert margin_top1_gate(actual, reference)["status"] == "pass"


@pytest.mark.parametrize(
    "values", [None, [[float("nan"), 0.0]], [[0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0]]]
)
def test_margin_gate_fails_closed_on_bad_or_logprob_only_data(values):
    actual = {"top1": torch.tensor([0]), "top2.values": torch.tensor([[-1.0, -2.0]])}
    if values is not None:
        actual["top2.logits"] = torch.tensor(values)
    with pytest.raises(ValueError):
        margin_top1_gate(actual, actual)


def test_raw_top1_miss_is_report_only_but_margin_and_logprob_gates_block(tmp_path):
    from safetensors.torch import save_file

    from tests.integration.megatron.glm5next_qualification import compare

    candidate = tmp_path / "candidate"
    oracle = tmp_path / "dumps/F8"
    candidate.mkdir()
    oracle.mkdir(parents=True)
    actual = {
        "target_logprobs": torch.tensor([-2.0]),
        "top1": torch.tensor([0]),
        "top2.logits": torch.tensor([[128.0, 127.0]]),
    }
    reference = {
        "target_logprobs": torch.tensor([-2.0]),
        "top64.indices": torch.ones(1, 2, 64, dtype=torch.int64),
        "top64.values": torch.zeros(1, 2, 64),
    }
    reference["top64.values"][0, 0, :2] = torch.tensor([128.0, 127.0])
    save_file(actual, str(candidate / "doc-rank0.safetensors"))
    save_file(reference, str(oracle / "doc.safetensors"))
    report = compare(candidate, tmp_path, "F8", tmp_path / "report.json")
    assert report["status"] == "pass" and report["rows"][0]["top1"] == 0.0
    actual["target_logprobs"] = torch.tensor([-2.03])
    save_file(actual, str(candidate / "doc-rank0.safetensors"))
    assert (
        compare(candidate, tmp_path, "F8", tmp_path / "report.json")["status"] == "fail"
    )
    actual["target_logprobs"] = torch.tensor([-2.0])
    actual["top2.logits"][0] = torch.tensor([128.0, 124.0])
    reference["top64.values"][0, 0, :2] = torch.tensor([128.0, 124.0])
    save_file(actual, str(candidate / "doc-rank0.safetensors"))
    save_file(reference, str(oracle / "doc.safetensors"))
    assert (
        compare(candidate, tmp_path, "F8", tmp_path / "report.json")["status"] == "fail"
    )


def test_top1_reports_long_document_counts_and_both_margins():
    count = 513
    actual = {
        "top1": torch.zeros(count, dtype=torch.int64),
        "top2.values": torch.tensor([[-0.1, -0.1001]]).repeat(count, 1),
    }
    reference = {
        "top64.indices": torch.zeros(1, count + 1, 64, dtype=torch.int64),
        "top64.values": torch.zeros(1, count + 1, 64),
    }
    reference["top64.indices"][0, 512, 0] = 1
    reference["top64.values"][0, 512, :2] = torch.tensor([3.0, 2.999])
    report = top1_diagnostics(actual, reference)
    assert report["positions"] == 513 and report["agreements"] == 512
    assert report["disagreement_details"][0]["position"] == 512
    assert report["disagreement_details"][0][
        "native_top1_top2_logit_margin"
    ] == pytest.approx(0.0001, abs=1e-7)
    assert report["disagreement_details"][0][
        "hf_top1_top2_logit_margin"
    ] == pytest.approx(0.001, abs=1e-6)


def test_routing_compares_sets_and_weights_aligned_by_id():
    actual = {
        "layers.3.router.indices": torch.tensor([[3, 1], [2, 1]]),
        "layers.3.router.weights": torch.tensor([[0.2, 0.8], [0.4, 0.6]]),
    }
    reference = {
        "layers.3.router.indices": torch.tensor([[1, 3], [1, 4]]),
        "layers.3.router.weights": torch.tensor([[0.8, 0.2], [0.6, 0.4]]),
    }
    report = sublayer_metrics(actual, reference, 2)
    assert report["layers.3.router.indices"]["disagreement_positions"] == [1]
    assert report["layers.3.router.weights"]["max_abs"] == pytest.approx(0.4)
    # Ordering and invalid-sentinel padding do not manufacture disagreements.
    actual = {"layers.3.indexer.indices": torch.tensor([[[2, 1, -1, -1]]])}
    reference = {"layers.3.indexer.indices": torch.tensor([[[1, 2, -1]]])}
    assert (
        sublayer_metrics(actual, reference, 1)["layers.3.indexer.indices"][
            "exact_set_positions"
        ]
        == 1
    )


@pytest.mark.parametrize("hard_bug", [False, True])
@pytest.mark.parametrize("order", [("F4", "F8"), ("F8", "F4")])
def test_pipeline_keeps_f4_failure_and_runs_f8_unless_hard_bug(
    tmp_path, monkeypatch, hard_bug, order
):
    import tests.integration.megatron.glm5next_qualification_launch as launch

    commands = [
        {"name": name, "argv": [name], "seconds": 1, "env": {}} for name in order
    ]
    monkeypatch.setattr(launch, "steps", lambda *_args, **_kwargs: commands)
    calls = []

    def run(argv, *_args, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        if "--task" in argv:
            task = argv[argv.index("--task") + 1]
            output = argv[argv.index("--output") + 1]
            launch.write(
                launch.Path(output) / "comparison.json",
                {"status": "fail", "hard_bug": hard_bug},
            )
            calls.append(task)
        else:
            calls.append(argv[0])
        return {}

    monkeypatch.setattr(launch, "run", run)
    seal = {
        "kind": "workstation",
        "window_seconds": 7200,
        "pss_limit_bytes": 120 << 30,
        "native_python": "python",
        "fixtures": str(tmp_path),
        "fixture_order": order,
    }
    if hard_bug:
        with pytest.raises(ValueError, match="hard bug"):
            pipeline(seal, tmp_path, tmp_path, tiny=True)
        assert calls == [order[0], "compare-" + order[0]]
        assert any(
            row["name"] == order[1] and row["status"] == "not_run"
            for row in json.loads((tmp_path / "results/pipeline.json").read_text())
        )
    else:
        result = pipeline(seal, tmp_path, tmp_path, tiny=True)
        assert calls == [
            order[0],
            "compare-" + order[0],
            order[1],
            "compare-" + order[1],
        ]
        assert result["status"] == "fail"
        assert [row["comparison"] for row in result["steps"]] == ["fail", "fail"]


def test_new_lane_requires_its_own_exact_grant():
    seal = {"kind": "workstation", "lane": "glm-m4-f4loc"}
    assert has_grant(seal, "abc", "GRANT glm-m4-f4loc workstation seal_sha256=abc")
    assert not has_grant(seal, "abc", "GRANT glm-m4-model workstation seal_sha256=abc")


def test_nonfinite_outputs_are_saved_before_the_hard_stop(tmp_path):
    from safetensors.torch import load_file

    from tests.integration.megatron.glm5next_qualification import experiment

    (tmp_path / "prompts.json").write_text(
        json.dumps(
            {
                "packs": [
                    {"id": "packed-3", "input_ids": [1, 2, 3], "cu_seqlens": [0, 3]}
                ]
            }
        )
    )
    engine = SimpleNamespace(
        rank=0,
        layers={},
        forward=lambda *_args, **_kwargs: {
            "logprobs": torch.tensor([float("nan"), 0.0]),
            "logits": None,
        },
    )
    with pytest.raises(ValueError, match="Non-finite output"):
        experiment(engine, "parity", tmp_path, tmp_path / "F4")
    saved = load_file(str(tmp_path / "F4/packed-3-doc0-rank0.safetensors"))
    assert torch.isnan(saved["target_logprobs"][0])
    assert (
        json.loads((tmp_path / "F4/failure-rank0.json").read_text())["status"]
        == "hard_bug"
    )


def test_hooks_observe_all_positions_without_replaying_router():
    class Router(torch.nn.Module):
        topk = 2
        calls = 0

        def forward(self, x):  # ty: ignore[invalid-mutable-override]
            self.calls += 1
            probabilities = torch.tensor([[0.1, 0.0, 0.9]]).repeat(x.shape[0], 1)
            return probabilities, probabilities != 0

    class Mlp(torch.nn.Identity):
        def __init__(self):
            super().__init__()
            self.router = Router()

    class Util:
        def layer_pre(self, x, *_args):
            return x, x.transpose(0, 1), x.transpose(0, 1)

        def layer_post(self, x):
            return x

    layer: Any = SimpleNamespace(
        layer_number=4,
        self_attention=torch.nn.Identity(),
        mlp=Mlp(),
        hc_util=Util(),
        hc_attn_fn=object(),
    )
    original = layer.hc_util.layer_pre
    values = {}
    handles = capture_sublayers(layer, values)
    x = torch.arange(24).reshape(8, 1, 3).float()
    layer.self_attention(x)
    layer.mlp(x)
    layer.mlp.router(x[:, 0])
    layer.hc_util.layer_pre(x, layer.hc_attn_fn, None, None)
    layer.hc_util.layer_post(x)
    assert values["layers.3.attention"].shape == (1, 8, 3)
    assert values["layers.3.mlp"].shape == (1, 8, 3)
    assert values["layers.3.attn_hc.streams"].shape == (1, 8, 3)
    assert values["layers.3.router.indices"].shape == (8, 2)
    assert layer.mlp.router.calls == 1
    for handle in handles:
        handle.remove()
    assert layer.hc_util.layer_pre == original


def test_blocking_pass_never_installs_observers_and_survives_diagnostic_failure(
    tmp_path, monkeypatch
):
    from safetensors.torch import load_file

    import tests.integration.megatron.glm5next_qualification as worker

    (tmp_path / "prompts.json").write_text(
        json.dumps(
            {
                "packs": [
                    {
                        "id": "packed-3",
                        "input_ids": [1, 2, 3],
                        "cu_seqlens": [0, 3],
                    }
                ]
            }
        )
    )
    engine = worker.TinyEngine(tmp_path, tmp_path / "F4")
    production_hooks = dict(engine.head._forward_hooks)
    assert not engine.handles
    worker.experiment(engine, "parity", tmp_path, tmp_path / "F4")
    path = tmp_path / "F4/packed-3-doc0-rank0.safetensors"
    before = path.read_bytes()
    assert not any(k.startswith("layers.") for k in load_file(str(path)))
    assert dict(engine.head._forward_hooks) == production_hooks

    def fail(*_args):
        raise AssertionError("diagnostic hook failed")

    monkeypatch.setattr(worker, "capture_layer_outputs", fail)
    with pytest.raises(AssertionError, match="diagnostic hook failed"):
        worker.experiment(
            engine, "parity-diagnostics", tmp_path, tmp_path / "F4-diagnostics"
        )
    assert path.read_bytes() == before
    assert (
        json.loads((tmp_path / "F4/result-rank0.json").read_text())["status"] == "pass"
    )


@pytest.mark.parametrize("diagnostic_failure", ["worker", "comparison", "repeat"])
def test_report_only_worker_and_comparator_cannot_invalidate_blocking_results(
    tmp_path, monkeypatch, diagnostic_failure
):
    import tests.integration.megatron.glm5next_qualification_launch as launch

    seal = {
        "kind": "workstation",
        "window_seconds": 7200,
        "pss_limit_bytes": 120 << 30,
        "native_python": "python",
        "oracle_python": "python",
        "fixtures": str(tmp_path),
        "supplemental_f8": True,
    }
    commands = launch.steps(seal, tmp_path, tmp_path, tiny=True)
    names = [s["name"] for s in commands]
    assert (
        names.index("F4")
        < names.index("F8")
        < names.index("hf-stage0")
        < names.index("F4-diagnostics")
    )
    assert all(s["report_only"] for s in commands[names.index("hf-stage0") :])
    calls = []

    def run(argv, *_args, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        task = argv[argv.index("--task") + 1]
        output = launch.Path(argv[argv.index("--output") + 1])
        calls.append((task, output.name))
        if task == "parity-diagnostics" and diagnostic_failure == "worker":
            raise AssertionError("legacy Dynamo assertion")
        if task.startswith("compare-diagnostics-"):
            if diagnostic_failure == "comparison":
                raise ValueError("missing diagnostic reference")
            launch.write(
                output / "diagnostics.json",
                {"status": "fail", "hard_bug": diagnostic_failure == "repeat"},
            )
        elif task.startswith("compare-"):
            launch.write(
                output / "comparison.json", {"status": "pass", "hard_bug": False}
            )
        return {}

    monkeypatch.setattr(launch, "run", run)
    result = launch.pipeline(seal, tmp_path, tmp_path, tiny=True)
    assert result["status"] == "pass" and result["diagnostics_status"] == "fail"
    assert len([row for row in result["steps"] if row.get("comparison") == "pass"]) == 2
    assert ("parity", "F8") in calls and (
        "parity-diagnostics",
        "F8-diagnostics",
    ) in calls
    assert (
        json.loads((tmp_path / "results/F4-comparison.json").read_text())["status"]
        == "pass"
    )


@pytest.mark.parametrize("delta, hard_bug", [(0.0, False), (0.2, True)])
def test_repeat_delta_is_report_only_and_missing_diagnostic_file_is_recorded(
    tmp_path, delta, hard_bug
):
    from safetensors.torch import save_file

    from tests.integration.megatron.glm5next_qualification import compare_diagnostics

    blocking, diagnostic = tmp_path / "F4", tmp_path / "F4-diagnostics"
    blocking.mkdir()
    diagnostic.mkdir()
    clean = {"target_logprobs": torch.full((2,), -1.0)}
    for name in ("present", "missing"):
        save_file(clean, str(blocking / f"{name}-rank0.safetensors"))
    save_file(
        {"target_logprobs": torch.full((2,), -1.0 + delta)},
        str(diagnostic / "present-rank0.safetensors"),
    )
    report = compare_diagnostics(
        diagnostic, tmp_path, "F4", tmp_path / "diagnostics.json", tiny=True
    )
    assert report["status"] == "fail" and report["hard_bug"] == hard_bug
    present = next(r for r in report["rows"] if r["document"].startswith("present"))
    assert present["repeat_logprobs"]["logprob_abs_mean"] == pytest.approx(delta)
    assert (
        next(r for r in report["rows"] if r["document"].startswith("missing"))["status"]
        == "fail"
    )


def test_blocking_comparator_fails_closed_without_raw_margins(tmp_path):
    from safetensors.torch import save_file

    from tests.integration.megatron.glm5next_qualification import compare

    candidate, reference = tmp_path / "F4", tmp_path / "dumps/F4"
    candidate.mkdir()
    reference.mkdir(parents=True)
    save_file(
        {"target_logprobs": torch.full((2,), -1.0), "top1": torch.tensor([0, 1])},
        str(candidate / "doc-rank0.safetensors"),
    )
    save_file(
        {"target_logprobs": torch.full((2,), -1.0), "top1": torch.tensor([0, 0])},
        str(reference / "doc.safetensors"),
    )
    report = compare(
        candidate,
        tmp_path,
        "F4",
        tmp_path / "comparison.json",
        supplemental=tmp_path / "absent",
    )
    assert report["status"] == "fail" and not report["hard_bug"]
    assert report["rows"][0]["top1"] == 0.5
    assert report["rows"][0]["margin_top1_gate"]["status"] == "fail"
    assert "raw top-2 logits" in report["rows"][0]["margin_top1_gate"]["error"]
    assert report["rows"][0]["top1_diagnostics"]["policy"] == "report-only"


@pytest.mark.parametrize("index", [0, 3])
def test_real_cpu_layer_captures_match_hf_sites_at_every_position(index):
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.modeling_glm5_next import (
        Glm5NextTextModel,
    )

    from tests.unit.test_glm5next_model import cpu_layer, tiny_config

    torch.manual_seed(51)
    config = tiny_config()
    reference = Glm5NextTextModel(config).layers[index]
    layer = cpu_layer(config, reference, index)
    values = {}
    from tests.integration.megatron.glm5next_qualification import capture_layer_outputs

    wrapped = SimpleNamespace(
        module=SimpleNamespace(decoder=SimpleNamespace(layers=[layer]))
    )
    handles = capture_layer_outputs([wrapped], values)
    expected = {}
    prefix = f"layers.{index}"

    def hook(key):
        def capture(_module, _args, result):
            if key.endswith("_hc"):
                for field, tensor in zip(
                    ("post", "comb", "collapsed"), result, strict=True
                ):
                    expected[prefix + "." + key + "." + field] = tensor.detach()
            else:
                expected[prefix + "." + key] = (
                    result[0] if isinstance(result, tuple) else result
                ).detach()

        return capture

    for key, module in (
        ("attention", reference.self_attn),
        ("mlp", reference.mlp),
        ("attn_hc", reference.attn_hc),
        ("ffn_hc", reference.ffn_hc),
    ):
        handles.append(module.register_forward_hook(hook(key)))
    x = torch.randn(1, 9, 4, 16) * 0.02
    expected[prefix + ".input_streams"] = x
    from art.megatron.glm5next.hf_oracle import cpu_reference_kernels

    with torch.no_grad(), cpu_reference_kernels():
        expected[prefix + ".streams"] = reference(
            x, attention_mask=torch.ones(1, 9, dtype=torch.bool)
        )[0]
        for handle in handles[-4:]:
            handle.remove()
        layer(x.transpose(0, 1))
    report = sublayer_metrics(values, expected, 9)
    assert len(report) == 11
    assert all(
        metric["positions"] == 9 and metric["max_abs"] < 2e-6
        for metric in report.values()
    )
    if index == 0:
        from tests.integration.megatron.glm5next_parity_diagnostics import (
            validate_captures,
        )

        validate_captures(values, 1, 9)
        values["layers.0.attention"] = values["layers.0.attention"][:, :4]
        with pytest.raises(ValueError, match="Incomplete position capture"):
            validate_captures(values, 1, 9)
    for handle in handles:
        handle.remove()
