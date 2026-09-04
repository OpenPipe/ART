"""The fitted layout score applies only to the execution classes it was measured on.

A calibrated table admits a runtime when its device class, parameter dtype,
model geometry (read from the Megatron config, never a model name) and
parallel shape (TP x CP x EP x ETP) all appear in the table's measured sets;
everything else keeps the version-1 score.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from art.trainer_rank._planner_cost import (
    COEFFICIENT_VERSION,
    COEFFICIENT_VERSION_FALLBACK,
    DEFAULT_TABLE,
    DENSE_H2560_TABLE,
    QWEN3_4B_GEOMETRY,
    QWEN35_4B_GEOMETRY,
    DeviceClass,
    ModelGeometry,
    ParallelShape,
    prefix_tree_layout_score,
    select_scoring,
)
from art.trainer_rank._prefix_tree_planner import (
    build_canonical_prefix_tree,
    prefix_tree_layout_candidates,
    select_prefix_tree_layout,
)

H200_MEMORY_BYTES = 143 * 1024**3
H100_MEMORY_BYTES = 80 * 1024**3


def _selection(
    *,
    device_capability: tuple[int, int] | None = (9, 0),
    device_memory_bytes: int | None = H200_MEMORY_BYTES,
    param_dtype: str = "torch.bfloat16",
    geometry: ModelGeometry = QWEN35_4B_GEOMETRY,
    shape: ParallelShape = ParallelShape(tp=1, cp=4),
):
    """The measured configuration, with one fact overridden at a time."""

    return select_scoring(
        device_capability=device_capability,
        device_memory_bytes=device_memory_bytes,
        param_dtype=param_dtype,
        geometry=geometry,
        shape=shape,
    )


def test_measured_execution_classes_are_admitted() -> None:
    for geometry in (QWEN35_4B_GEOMETRY, QWEN3_4B_GEOMETRY):
        for shape in DENSE_H2560_TABLE.shapes:
            selection = _selection(geometry=geometry, shape=shape)
            assert selection.version == COEFFICIENT_VERSION
            assert selection.table_id == DENSE_H2560_TABLE.table_id
            assert selection.coefficients is DENSE_H2560_TABLE.coefficients_milli_us
    # Memory unknown (older driver): the capability check alone applies.
    assert _selection(device_memory_bytes=None).version == COEFFICIENT_VERSION


def test_unmeasured_classes_fall_back_to_version_one() -> None:
    fallback = COEFFICIENT_VERSION_FALLBACK
    # Ampere; H100 (shares capability 9.0, not the H200 memory system); fp16.
    assert _selection(device_capability=(8, 0)).version == fallback
    assert _selection(device_memory_bytes=H100_MEMORY_BYTES).version == fallback
    assert _selection(param_dtype="torch.float16").version == fallback
    # Any geometry change: a wider model (Qwen3-8B-like), a neighbouring
    # width, a different FFN ratio or head geometry, GDN state shape, or MoE.
    dense = QWEN3_4B_GEOMETRY
    for geometry in (
        ModelGeometry(4_096, 12_288, 32, 8, 128),
        ModelGeometry(3_072, 8_192, 24, 8, 128),
        ModelGeometry(**{**dense.as_dict(), "ffn_hidden_size": 8_192}),
        ModelGeometry(**{**dense.as_dict(), "num_query_groups": 4}),
        ModelGeometry(**{**QWEN35_4B_GEOMETRY.as_dict(), "gdn_value_heads": 16}),
        ModelGeometry(
            **{
                **dense.as_dict(),
                "moe_experts": 128,
                "moe_topk": 8,
                "moe_ffn_hidden_size": 768,
            }
        ),
    ):
        assert _selection(geometry=geometry).version == fallback, geometry
    # Unmeasured parallel shapes: TP2 x CP2, CP8, TP4, expert parallelism.
    for shape in (
        ParallelShape(tp=2, cp=2),
        ParallelShape(tp=1, cp=8),
        ParallelShape(tp=4, cp=1),
        ParallelShape(tp=1, cp=2, ep=2),
        ParallelShape(tp=2, cp=1, etp=2),
    ):
        selection = _selection(shape=shape)
        assert selection.version == fallback, shape
        assert selection.table_id is None and selection.coefficients is None


def test_cpu_only_planning_uses_the_default_table() -> None:
    # No CUDA device (unit tests): the calibrated domain describes GPU
    # execution, so the device is not a reason to fall back.
    selection = select_scoring(
        device_capability=None,
        device_memory_bytes=None,
        param_dtype="torch.float16",
        geometry=ModelGeometry(8, 32, 2, 2, 4),
        shape=ParallelShape(),
    )
    assert selection.version == COEFFICIENT_VERSION
    assert selection.table_id == DEFAULT_TABLE.table_id


def test_geometry_is_read_from_the_megatron_config() -> None:
    # Megatron's TransformerConfig carries GDN defaults (linear_num_value_heads
    # = 32) even for attention-only models; the geometry zeroes them unless the
    # model has GDN layers, and likewise for expert fields.
    qwen3_4b = SimpleNamespace(
        hidden_size=2_560,
        ffn_hidden_size=9_728,
        num_attention_heads=32,
        num_query_groups=8,
        kv_channels=128,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_moe_experts=None,
        moe_router_topk=2,
    )
    assert ModelGeometry.from_config(qwen3_4b, has_gdn=False, has_moe=False) == (
        QWEN3_4B_GEOMETRY
    )
    qwen35_4b = SimpleNamespace(
        hidden_size=2_560,
        ffn_hidden_size=9_216,
        num_attention_heads=16,
        num_query_groups=4,
        kv_channels=256,
        linear_num_value_heads=32,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    assert ModelGeometry.from_config(qwen35_4b, has_gdn=True, has_moe=False) == (
        QWEN35_4B_GEOMETRY
    )
    moe = SimpleNamespace(
        hidden_size=2_048,
        ffn_hidden_size=6_144,
        num_attention_heads=32,
        num_query_groups=4,
        kv_channels=128,
        num_moe_experts=128,
        moe_router_topk=8,
        moe_ffn_hidden_size=768,
        moe_shared_expert_intermediate_size=None,
    )
    geometry = ModelGeometry.from_config(moe, has_gdn=False, has_moe=True)
    assert (geometry.moe_experts, geometry.moe_topk, geometry.moe_ffn_hidden_size) == (
        128,
        8,
        768,
    )
    assert geometry.moe_shared_expert_ffn == 0
    # Fake providers in unit tests carry only hidden_size / num_layers.
    tiny = ModelGeometry.from_config(
        SimpleNamespace(hidden_size=8, num_layers=4), has_gdn=False, has_moe=False
    )
    assert tiny == ModelGeometry(8, 0, 0, 0, 0)


def test_device_class_memory_buckets() -> None:
    assert DeviceClass.for_device((9, 0), H200_MEMORY_BYTES).memory_class == "hbm-141g"
    assert DeviceClass.for_device((9, 0), H100_MEMORY_BYTES).memory_class == "hbm-80g"
    assert DeviceClass.for_device((9, 0), None).memory_class is None


def _tree():
    prefix = torch.arange(1, 2_001)
    return build_canonical_prefix_tree(
        tuple(
            torch.cat((prefix, torch.arange(10_000 + i * 100, 10_100 + i * 100)))
            for i in range(4)
        )
    )


def test_scores_follow_the_selected_table() -> None:
    tree = _tree()
    layout = prefix_tree_layout_candidates(tree)[0].layout
    shipped = prefix_tree_layout_score(layout, cp_size=1, layers=32, uses_gdn=True)
    other = {
        name: 2 * value
        for name, value in DENSE_H2560_TABLE.coefficients_milli_us.items()
    }
    doubled = prefix_tree_layout_score(
        layout, cp_size=1, layers=32, uses_gdn=True, coefficients=other
    )
    assert doubled[0] == 2 * shipped[0]
    assert doubled[1:] == shipped[1:]
    fallback = prefix_tree_layout_score(
        layout,
        cp_size=1,
        layers=32,
        uses_gdn=True,
        coefficient_version=COEFFICIENT_VERSION_FALLBACK,
        coefficients=other,
    )
    assert fallback != doubled
    selected = select_prefix_tree_layout(
        tree,
        cp_size=1,
        layers=32,
        uses_gdn=True,
        refinement_work_budget=0,
        coefficients=DENSE_H2560_TABLE.coefficients_milli_us,
    )
    # Four 2,100-token sequences sharing a 2,000-token prefix: the selected
    # layout packs fewer tokens than the unshared 8,400 and more than the prefix.
    assert 2_000 < selected.layout.packed_tokens < 8_400


def test_moe_class_is_admitted_only_at_its_measured_shapes() -> None:
    from art.trainer_rank._planner_cost import (
        GDN_MOE_H2048_GEOMETRY,
        GDN_MOE_H2048_TABLE,
    )

    for shape in GDN_MOE_H2048_TABLE.shapes:
        selection = _selection(geometry=GDN_MOE_H2048_GEOMETRY, shape=shape)
        assert selection.table_id == GDN_MOE_H2048_TABLE.table_id, shape
    assert ParallelShape(tp=1, cp=4, ep=4) in GDN_MOE_H2048_TABLE.shapes
    # Unmeasured shapes (TP2 x CP2, EP1 at CP8, expert TP) and the dense table's
    # shapes with this geometry under a different dtype fall back.
    for shape in (
        ParallelShape(tp=2, cp=2),
        ParallelShape(tp=1, cp=8),
        ParallelShape(tp=2, cp=1, ep=2, etp=2),
    ):
        assert _selection(geometry=GDN_MOE_H2048_GEOMETRY, shape=shape).version == (
            COEFFICIENT_VERSION_FALLBACK
        ), shape
    assert (
        _selection(
            geometry=GDN_MOE_H2048_GEOMETRY,
            shape=ParallelShape(),
            param_dtype="torch.float16",
        ).version
        == COEFFICIENT_VERSION_FALLBACK
    )
    # The dense geometries never borrow the MoE table and vice versa.
    assert _selection(
        geometry=QWEN35_4B_GEOMETRY, shape=ParallelShape(tp=1, cp=4, ep=4)
    ).version == (COEFFICIENT_VERSION_FALLBACK)
