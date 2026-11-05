from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.integration.megatron.glm5next_e2e import install_capture, metrics
from tests.integration.megatron.glm5next_e2e_launch import (
    has_grant,
    literals,
    pending_state,
    pending_stop,
)


def test_positive_control_detects_m0_signature_and_nonfinite() -> None:
    x = torch.zeros(2048)
    y = x.clone()
    y[145:401] = 5
    y[1024 + 147 : 1024 + 403] = 5
    result = metrics(x, y, [1024, 1024])
    assert result["hard_bug"] and result["m0_bimodal_signature"]
    assert result["first_bad_position_per_document"] == [145, 147]
    assert result["frac_gt1"] == 0.25 and result["p50_abs"] == 0
    assert not metrics(x, x, [1024, 1024])["hard_bug"]
    y[0] = torch.nan
    assert metrics(x, y, [1024, 1024])["hard_bug"]


def test_capture_keeps_mhc_streams_in_channel_axis(tmp_path: Path) -> None:
    layer = torch.nn.Identity()
    values = torch.arange(32).reshape(4, 1, 4, 2)
    trainer = SimpleNamespace(
        runtime=SimpleNamespace(
            model=[SimpleNamespace(decoder=SimpleNamespace(layers=[layer]))]
        ),
        _decoder_hidden=lambda prepared: layer(values),
    )
    prepared = SimpleNamespace(
        token_uids=torch.arange(4),
        position_ids=torch.arange(4),
        attention_state=SimpleNamespace(),
    )
    install_capture(trainer, tmp_path / "capture.pt")
    trainer._decoder_hidden(prepared)
    saved = torch.load(tmp_path / "capture.pt", weights_only=True)
    torch.testing.assert_close(saved["layer-0"], values.reshape(4, 8))
    assert saved["layer-0.uids"].tolist() == [0, 1, 2, 3]


def test_exact_grant_requires_every_literal() -> None:
    seal = dict(
        context="use15a",
        namespace="qual-flash",
        pool="caladan-b200-spot",
        target_node="node313",
        window_seconds=9000,
    )
    terms = literals(seal, "a" * 64)
    grant = "GRANT glm-m4-model " + " ".join(terms)
    assert has_grant(seal, "a" * 64, grant)
    assert not has_grant(seal, "b" * 64, grant)
    assert not has_grant(seal, "a" * 64, "NEEDS-LEAD " + grant)
    for term in terms:
        assert not has_grant(seal, "a" * 64, grant.replace(term, ""))


@pytest.mark.parametrize(
    "gpu,scaleup,limit",
    [(False, False, 300), (False, True, 2700), (True, False, 300), (True, True, 300)],
)
def test_split_pending_boundaries(gpu: bool, scaleup: bool, limit: int) -> None:
    assert not pending_stop(limit - 1, gpu=gpu, own_scaleup=scaleup)
    assert pending_stop(limit, gpu=gpu, own_scaleup=scaleup)


def test_pending_uses_own_uid_and_assigned_startup_clock() -> None:
    pod = {
        "metadata": {
            "uid": "own",
            "name": "run-stage0",
            "creationTimestamp": "2026-11-05T14:00:00Z",
        },
        "spec": {},
        "status": {
            "conditions": [
                {
                    "type": "PodScheduled",
                    "status": "True",
                    "lastTransitionTime": "2026-11-05T14:28:00Z",
                }
            ]
        },
    }
    now = datetime(2026, 11, 5, 14, 30, tzinfo=timezone.utc)
    events = {
        "items": [{"involvedObject": {"uid": "foreign"}, "reason": "TriggeredScaleUp"}]
    }
    assert pending_state(pod, events, now=now)["stopped"]
    events["items"][0]["involvedObject"]["uid"] = "own"
    assert not pending_state(pod, events, now=now)["stopped"]
    pod["spec"]["nodeName"] = "node"
    assigned = pending_state(pod, events, now=now)
    assert (
        assigned["age_s"] == 120
        and not assigned["own_TriggeredScaleUp"]
        and not assigned["stopped"]
    )
