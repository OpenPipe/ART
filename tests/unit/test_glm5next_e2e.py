from __future__ import annotations

import pytest
import torch

from tests.integration.megatron.glm5next_e2e import metrics
from tests.integration.megatron.glm5next_e2e_launch import (
    has_grant,
    literals,
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
