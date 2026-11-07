"""Fail-closed detection of native and custom router auxiliary-loss settings."""

from types import SimpleNamespace

import pytest
import torch

from art.trainer_rank import TrainerRank
from art.trainer_rank._planner_cost import GDN_MOE_H2048_GEOMETRY, QWEN35_4B_GEOMETRY


@pytest.mark.parametrize(
    "fields, refused",
    [
        ({"moe_aux_loss_coeff": 0.0}, False),
        (
            {"moe_router_load_balancing_type": "aux_loss", "moe_aux_loss_coeff": 0.3},
            False,
        ),
        (
            {
                "moe_router_load_balancing_type": "global_aux_loss",
                "moe_aux_loss_coeff": 0.0,
            },
            False,
        ),
        (
            {
                "moe_router_load_balancing_type": "global_aux_loss",
                "moe_aux_loss_coeff": 0.3,
            },
            True,
        ),
        (
            {
                "moe_router_load_balancing_type": "global_aux_loss",
                "moe_aux_loss_coeff": -0.3,
            },
            True,
        ),
        (
            {
                "moe_router_load_balancing_type": ["aux_loss", "global_aux_loss"],
                "moe_aux_loss_coeff": [0.3, 0.0],
            },
            False,
        ),
        (
            {
                "moe_router_load_balancing_type": ["aux_loss", "global_aux_loss"],
                "moe_aux_loss_coeff": [0.0, 0.3],
            },
            True,
        ),
        (
            {
                "moe_router_load_balancing_type": ["aux_loss", "global_aux_loss"],
                "moe_aux_loss_coeff": [0.0],
            },
            True,
        ),
        ({"moe_router_load_balancing_type": "global_aux_loss"}, True),
        ({"moe_aux_loss_coeff": 0.3}, True),
        (
            {
                "moe_router_load_balancing_type": ["global_aux_loss"],
                "moe_aux_loss_coeff": 0.0,
            },
            True,
        ),
        ({}, True),
    ],
)
def test_global_aux_settings_are_explicit_or_refused(monkeypatch, fields, refused):
    trainer = TrainerRank.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(
        provider=SimpleNamespace(**fields),
        model=[],
        model_support_handler=SimpleNamespace(),
    )
    trainer._geometry = GDN_MOE_H2048_GEOMETRY
    trainer._moe_layers = 1
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    error = trainer._global_aux_admission_scope_error(None)
    assert (error is not None) is refused


def test_native_transformer_config_defaults_disable_global_aux(monkeypatch):
    config_module = pytest.importorskip("megatron.core.transformer.transformer_config")
    config = config_module.TransformerConfig(
        num_layers=1, hidden_size=16, num_attention_heads=1
    )
    assert config.moe_router_load_balancing_type == "aux_loss"
    assert config.moe_aux_loss_coeff == 0.0
    trainer = TrainerRank.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(
        provider=config, model=[], model_support_handler=SimpleNamespace()
    )
    trainer._geometry = GDN_MOE_H2048_GEOMETRY
    trainer._moe_layers = 1
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert trainer._global_aux_admission_scope_error(None) is None


@pytest.mark.parametrize("indicator", ["provider", "handler", "spec"])
def test_incomplete_moe_metadata_is_not_treated_as_default(monkeypatch, indicator):
    trainer = TrainerRank.__new__(TrainerRank)
    trainer.runtime = SimpleNamespace(
        provider=SimpleNamespace(),
        model=[],
        model_support_handler=SimpleNamespace(),
        model_support_spec=SimpleNamespace(),
    )
    if indicator == "provider":
        trainer.runtime.provider.num_moe_experts = 4
    elif indicator == "handler":
        trainer.runtime.model_support_handler.is_moe = True
    else:
        trainer.runtime.model_support_spec.is_moe = True
    trainer._geometry = QWEN35_4B_GEOMETRY
    trainer._moe_layers = 0
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert "configuration is unavailable" in trainer._global_aux_admission_scope_error(
        None
    )
