from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any, cast

import pytest
import torch
from torch import nn

from art.megatron.routing_replay import (
    MoeRoutingReplayBundle,
    MoeRoutingReplayController,
    ParallelTopology,
    RouterCallRoute,
    StepRouterRoutes,
    StepRoutes,
    TopologyAwareLocalTokenIndexer,
    build_router_key_from_module_name,
)
from art.megatron.training.trace import _routing_replay_token_uid_sets


def _make_route(
    rows: list[list[int]],
    *,
    sample_index: int | None = None,
    micro_slot: int | None = None,
) -> RouterCallRoute:
    indices = torch.tensor(rows, dtype=torch.int32)
    return RouterCallRoute(
        expert_indices=indices,
        expert_mask=torch.ones_like(indices, dtype=torch.bool),
        num_experts=3,
        sample_index=sample_index,
        micro_slot=micro_slot,
    )


def _make_bundle() -> tuple[MoeRoutingReplayBundle, RouterCallRoute]:
    router_key = "chunk_00.layer_0000.mlp.router"
    route = _make_route([[0, 2], [1, 0], [2, 1], [1, 0]], sample_index=0)
    bundle = MoeRoutingReplayBundle(
        topology=ParallelTopology(tp=1, ep=1, etp=1, dp=1, sp=False, cp=1, pp=1, vpp=1),
        num_steps=1,
        max_topk=2,
        router_keys=[router_key],
        steps={
            0: StepRoutes(
                routers={router_key: StepRouterRoutes(calls={0: route})},
                global_token_uids=torch.arange(4, dtype=torch.int64),
            )
        },
    )
    return bundle, route


def _make_sampled_bundle() -> MoeRoutingReplayBundle:
    router_key = "chunk_00.layer_0000.mlp.router"
    return MoeRoutingReplayBundle(
        topology=ParallelTopology(tp=1, ep=1, etp=1, dp=1, sp=False, cp=1, pp=1, vpp=1),
        num_steps=1,
        max_topk=2,
        router_keys=[router_key],
        steps={
            0: StepRoutes(
                routers={
                    router_key: StepRouterRoutes(
                        calls={
                            0: _make_route([[0, 2], [1, 0]], sample_index=0),
                            1: _make_route([[2, 1], [0, 1]], sample_index=1),
                        }
                    )
                },
                global_token_uids=torch.arange(2, dtype=torch.int64),
            )
        },
    )


def _make_multi_call_bundle() -> MoeRoutingReplayBundle:
    router_key = "chunk_00.layer_0000.mlp.router"
    return MoeRoutingReplayBundle(
        topology=ParallelTopology(tp=1, ep=1, etp=1, dp=1, sp=False, cp=1, pp=1, vpp=1),
        num_steps=1,
        max_topk=2,
        router_keys=[router_key],
        steps={
            0: StepRoutes(
                routers={
                    router_key: StepRouterRoutes(
                        calls={
                            0: _make_route([[0, 2]], sample_index=0),
                            1: _make_route([[1, 0]], sample_index=0),
                            2: _make_route([[2, 1]], sample_index=1),
                        }
                    )
                },
                global_token_uids=torch.arange(1, dtype=torch.int64),
            )
        },
    )


class _FakeParallelState:
    def __init__(
        self,
        *,
        tp_world_size: int = 1,
        tp_rank: int = 0,
        cp_world_size: int = 1,
    ) -> None:
        self._tp_world_size = tp_world_size
        self._tp_rank = tp_rank
        self._cp_world_size = cp_world_size

    def get_context_parallel_world_size(self) -> int:
        return self._cp_world_size

    def get_tensor_model_parallel_world_size(self) -> int:
        return self._tp_world_size

    def get_tensor_model_parallel_rank(self) -> int:
        return self._tp_rank


class _FakeGdnExecutionPlan:
    attention_token_indices = (0, 1, 2, 3)
    gdn_token_indices = (0, 2)


class _FakeAttentionState:
    gdn_execution_plan = _FakeGdnExecutionPlan()


class _FakeRouterReplay:
    def __init__(self) -> None:
        self.target_topk_idx: torch.Tensor | None = None
        self.action: Any = None
        self.targets_seen: list[torch.Tensor] = []

    def set_target_indices(self, topk_indices: torch.Tensor) -> None:
        self.target_topk_idx = topk_indices
        self.targets_seen.append(topk_indices.detach().cpu().clone())

    def set_router_replay_action(self, action: Any) -> None:
        self.action = action

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: int | None = None,
        group_topk: int | None = None,
        default_compute_topk: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del num_groups, group_topk
        if self.target_topk_idx is None:
            return default_compute_topk(scores, topk, None, None)
        indices = self.target_topk_idx.to(device=scores.device, dtype=torch.long)
        return scores.gather(1, indices), indices


class _FakeRouter(nn.Module):
    def __init__(self, *, topk: int = 2, router_replay: Any | None = None) -> None:
        super().__init__()
        self.topk = topk
        self.router_replay = (
            router_replay if router_replay is not None else _FakeRouterReplay()
        )
        self.config = type(
            "Config",
            (),
            {
                "sequence_parallel": False,
                "context_parallel_size": 1,
                "moe_router_fusion": False,
            },
        )()

    def routing(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.softmax(logits, dim=-1)

        def _default_topk(
            local_scores: torch.Tensor,
            topk: int,
            num_groups: int | None = None,
            group_topk: int | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del num_groups, group_topk
            return torch.topk(local_scores, k=topk, dim=1)

        selected_probs, selected_indices = self.router_replay.get_replay_topk(
            scores,
            self.topk,
            None,
            None,
            _default_topk,
        )
        probs = torch.zeros_like(scores)
        routing_map = torch.zeros_like(scores, dtype=torch.bool)
        rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
        probs[rows, selected_indices] = selected_probs
        routing_map[rows, selected_indices] = True
        return probs, routing_map


class _FakeMlp(nn.Module):
    def __init__(self, router: _FakeRouter | None = None) -> None:
        super().__init__()
        self.router = router if router is not None else _FakeRouter()


class _FakeLayer(nn.Module):
    def __init__(self, router: _FakeRouter | None = None) -> None:
        super().__init__()
        self.mlp = _FakeMlp(router)


class _FakeDecoder(nn.Module):
    def __init__(self, router: _FakeRouter | None = None) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(router)])


class _FakeChunk(nn.Module):
    def __init__(self, router: _FakeRouter | None = None) -> None:
        super().__init__()
        self.decoder = _FakeDecoder(router)


def _fake_chunk_router(chunk: _FakeChunk) -> _FakeRouter:
    layer = cast(_FakeLayer, chunk.decoder.layers[0])
    return cast(_FakeRouter, layer.mlp.router)


def _assert_target(
    replay: _FakeRouterReplay,
    expected: torch.Tensor,
    *,
    index: int = -1,
) -> None:
    assert torch.equal(replay.targets_seen[index], expected.to(torch.long))


def _expected_routing_map(route: RouterCallRoute) -> torch.Tensor:
    routing_map = torch.zeros(
        (route.num_global_tokens, route.num_experts), dtype=torch.bool
    )
    rows = torch.arange(route.num_global_tokens).unsqueeze(1)
    routing_map[rows, route.expert_indices.to(torch.long)] = True
    return routing_map


def test_routing_replay_token_uid_sets_keep_full_attention_layout_with_gdn_plan() -> (
    None
):
    token_uids = torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.int64)

    token_uid_sets = _routing_replay_token_uid_sets(
        token_uids,
        attention_state=_FakeAttentionState(),
    )
    attention_token_uids = token_uid_sets["attention"]
    gdn_token_uids = token_uid_sets["gdn"]

    assert attention_token_uids is not None
    assert torch.equal(
        attention_token_uids,
        torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64),
    )
    assert gdn_token_uids is not None
    assert torch.equal(
        gdn_token_uids,
        torch.tensor([0, 2], dtype=torch.int64),
    )


def test_build_router_key_from_compiled_module_name() -> None:
    assert (
        build_router_key_from_module_name(
            chunk_index=0,
            module_name="module.decoder.layers.0._orig_mod.mlp.router",
        )
        == "chunk_00.layer_0000.mlp.router"
    )


def test_build_router_key_from_nested_compiled_module_name() -> None:
    assert (
        build_router_key_from_module_name(
            chunk_index=3,
            module_name="module.decoder.layers.12.mlp._orig_mod.router",
        )
        == "chunk_03.layer_0012.mlp.router"
    )


def test_topology_aware_local_token_indexer_keeps_merged_rows_when_counts_match() -> (
    None
):
    indexer = TopologyAwareLocalTokenIndexer(
        parallel_state_module=_FakeParallelState(tp_world_size=2, tp_rank=1)
    )
    global_token_uids = torch.arange(256, dtype=torch.int64)

    local_uids = indexer.build_local_token_uids(
        global_token_uids=global_token_uids,
        num_local_tokens=256,
        sequence_parallel=True,
        context_parallel_size=1,
    )

    assert torch.equal(local_uids, global_token_uids)


def test_topology_aware_local_token_indexer_slices_sequence_parallel_rows() -> None:
    indexer = TopologyAwareLocalTokenIndexer(
        parallel_state_module=_FakeParallelState(tp_world_size=2, tp_rank=1)
    )
    global_token_uids = torch.arange(256, dtype=torch.int64)

    local_uids = indexer.build_local_token_uids(
        global_token_uids=global_token_uids,
        num_local_tokens=128,
        sequence_parallel=True,
        context_parallel_size=1,
    )

    assert torch.equal(local_uids, torch.arange(128, 256, dtype=torch.int64))


def test_bundle_roundtrip_disk() -> None:
    bundle, route = _make_bundle()
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = Path(tmp_dir)
        bundle.to_dir(bundle_path)
        loaded = MoeRoutingReplayBundle.from_dir(bundle_path)

    assert loaded.num_steps == 1
    assert loaded.max_topk == 2
    assert loaded.router_keys == bundle.router_keys
    loaded_route = loaded.steps[0].routers[bundle.router_keys[0]].calls[0]
    assert torch.equal(loaded_route.expert_indices, route.expert_indices)
    assert torch.equal(loaded_route.expert_mask, route.expert_mask)


def test_controller_uses_native_router_replay_target_indices() -> None:
    bundle, route = _make_bundle()
    controller = MoeRoutingReplayController(bundle=bundle, strict=True, device="cpu")
    chunk = _FakeChunk()
    router = _fake_chunk_router(chunk)
    replay = cast(_FakeRouterReplay, router.router_replay)

    controller.install_router_patches([chunk])
    controller.set_step(step_index=0, sample_index=[0])
    controller.begin_micro(0, 0)
    controller.set_local_input_token_uids(torch.arange(4, dtype=torch.int64))
    _probs, routing_map = router.routing(torch.randn((4, 3), dtype=torch.float32))

    expected_map = torch.zeros((4, 3), dtype=torch.bool)
    rows = torch.arange(4).unsqueeze(1)
    expected_map[rows, route.expert_indices.to(torch.long)] = True
    assert torch.equal(routing_map.cpu(), expected_map)
    _assert_target(replay, route.expert_indices)

    controller.finalize_step()
    controller.remove_router_patches()


def test_controller_explicit_token_uids_refresh_native_router_replay() -> None:
    bundle, route = _make_bundle()
    controller = MoeRoutingReplayController(bundle=bundle, strict=True, device="cpu")
    chunk = _FakeChunk()
    router = _fake_chunk_router(chunk)
    replay = cast(_FakeRouterReplay, router.router_replay)

    controller.install_router_patches([chunk])
    controller.set_step(step_index=0, sample_index=[0])
    controller.begin_micro(0, 0)
    controller.set_local_input_token_uids(torch.tensor([3, 1], dtype=torch.int64))
    assert replay.targets_seen == []
    _probs, routing_map = router.routing(torch.randn((2, 3), dtype=torch.float32))

    expected_indices = route.expert_indices.index_select(
        0, torch.tensor([3, 1], dtype=torch.long)
    )
    expected_map = torch.zeros((2, 3), dtype=torch.bool)
    rows = torch.arange(2).unsqueeze(1)
    expected_map[rows, expected_indices.to(torch.long)] = True
    _assert_target(replay, expected_indices, index=0)
    assert torch.equal(routing_map.cpu(), expected_map)

    controller.finalize_step()
    controller.remove_router_patches()


def test_controller_switches_prestaged_layout_targets() -> None:
    bundle, route = _make_bundle()
    controller = MoeRoutingReplayController(bundle=bundle, strict=True, device="cpu")
    chunk = _FakeChunk()
    router = _fake_chunk_router(chunk)
    replay = cast(_FakeRouterReplay, router.router_replay)

    controller.install_router_patches([chunk])
    controller.set_step(step_index=0, sample_index=[0])
    controller.begin_micro(0, 0)
    controller.prepare_micro_targets(
        {
            "attention": torch.tensor([0, 1], dtype=torch.int64),
            "gdn": torch.tensor([3, 1], dtype=torch.int64),
        }
    )
    assert replay.targets_seen == []

    _probs, attention_map = router.routing(torch.randn((2, 3), dtype=torch.float32))
    controller.set_active_token_uid_key("gdn")
    _probs, gdn_map = router.routing(torch.randn((2, 3), dtype=torch.float32))

    attention_indices = route.expert_indices.index_select(
        0, torch.tensor([0, 1], dtype=torch.long)
    )
    gdn_indices = route.expert_indices.index_select(
        0, torch.tensor([3, 1], dtype=torch.long)
    )
    _assert_target(replay, attention_indices, index=0)
    _assert_target(replay, gdn_indices, index=1)
    assert torch.equal(
        attention_map.cpu(), _expected_routing_map(_make_route([[0, 2], [1, 0]]))
    )
    assert torch.equal(
        gdn_map.cpu(), _expected_routing_map(_make_route([[1, 0], [1, 0]]))
    )

    controller.finalize_step()
    controller.remove_router_patches()


def test_controller_finalize_fails_when_unconsumed_calls_remain() -> None:
    bundle, _route = _make_bundle()
    controller = MoeRoutingReplayController(bundle=bundle, strict=True, device="cpu")
    chunk = _FakeChunk()
    controller.install_router_patches([chunk])
    controller.set_step(step_index=0, sample_index=[0])
    with pytest.raises(RuntimeError, match="consumption mismatch"):
        controller.finalize_step()


def test_controller_reuses_route_for_recompute_with_same_active_micro() -> None:
    bundle = _make_sampled_bundle()
    controller = MoeRoutingReplayController(bundle=bundle, strict=True, device="cpu")
    chunk = _FakeChunk()
    router = _fake_chunk_router(chunk)
    replay = cast(_FakeRouterReplay, router.router_replay)
    controller.install_router_patches([chunk])
    controller.set_step(step_index=0, sample_index=[0, 1])

    controller.begin_micro(0, 0)
    controller.set_local_input_token_uids(torch.arange(2, dtype=torch.int64))
    _probs, routing_map = router.routing(torch.randn((2, 3), dtype=torch.float32))
    _probs, recompute_routing_map = router.routing(
        torch.randn((2, 3), dtype=torch.float32)
    )
    controller.begin_micro(1, 1)
    controller.set_local_input_token_uids(torch.arange(2, dtype=torch.int64))
    _probs, next_routing_map = router.routing(torch.randn((2, 3), dtype=torch.float32))

    calls = bundle.steps[0].routers[bundle.router_keys[0]].calls
    _assert_target(replay, calls[0].expert_indices, index=0)
    _assert_target(replay, calls[1].expert_indices, index=1)
    assert torch.equal(routing_map.cpu(), _expected_routing_map(calls[0]))
    assert torch.equal(recompute_routing_map.cpu(), _expected_routing_map(calls[0]))
    assert torch.equal(next_routing_map.cpu(), _expected_routing_map(calls[1]))

    controller.finalize_step()
    controller.remove_router_patches()


def test_controller_consumes_multiple_captured_calls_before_recompute_reuse() -> None:
    bundle = _make_multi_call_bundle()
    controller = MoeRoutingReplayController(bundle=bundle, strict=True, device="cpu")
    chunk = _FakeChunk()
    router = _fake_chunk_router(chunk)
    replay = cast(_FakeRouterReplay, router.router_replay)
    controller.install_router_patches([chunk])
    controller.set_step(step_index=0, sample_index=[0, 1])

    with pytest.raises(RuntimeError, match="exactly one router call"):
        controller.begin_micro(0, 0)

    assert replay.targets_seen == []

    controller.remove_router_patches()


def test_controller_rejects_missing_native_router_replay() -> None:
    bundle, _route = _make_bundle()
    controller = MoeRoutingReplayController(bundle=bundle, strict=True, device="cpu")
    chunk = _FakeChunk(router=_FakeRouter(router_replay=None))
    _fake_chunk_router(chunk).router_replay = None

    with pytest.raises(RuntimeError, match="moe_enable_routing_replay=True"):
        controller.install_router_patches([chunk])


def test_controller_rejects_masked_slots() -> None:
    bundle, route = _make_bundle()
    route.expert_mask[0, 1] = False
    controller = MoeRoutingReplayController(bundle=bundle, strict=True, device="cpu")
    chunk = _FakeChunk()
    controller.install_router_patches([chunk])

    with pytest.raises(RuntimeError, match="masked slots are unsupported"):
        controller.set_step(step_index=0, sample_index=[0])


def test_controller_rejects_topk_mismatch() -> None:
    bundle, _route = _make_bundle()
    controller = MoeRoutingReplayController(bundle=bundle, strict=True, device="cpu")
    chunk = _FakeChunk(router=_FakeRouter(topk=1))
    controller.install_router_patches([chunk])

    with pytest.raises(RuntimeError, match="topk does not match"):
        controller.set_step(step_index=0, sample_index=[0])
