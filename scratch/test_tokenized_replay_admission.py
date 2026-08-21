from types import SimpleNamespace

import pytest

from art.distributed.art_runtime import DistributedPackingPlan
from art.distributed.rollout import RolloutModelSpec
from art.megatron.training.slot import MegatronTrainingSlot
from art.training.contracts import (
    ForwardBackwardRequest,
    LossConfig,
    OperationRef,
    TokenizedTrainingBatch,
)
from art.training.tokenized import TokenizedDatum, TokenizedMoeRoutes


def _batch(*, routed: bool) -> TokenizedTrainingBatch:
    return TokenizedTrainingBatch(
        datums=(
            TokenizedDatum(
                input_tokens=(1, 2),
                target_tokens=((2,), (3,)),
                weights=((0.0,), (1.0,)),
                moe_routes=(
                    TokenizedMoeRoutes(
                        num_experts=2,
                        dtype="uint8",
                        shape=(2, 1, 1),
                        data=(bytes((0, 1)),),
                    )
                    if routed
                    else None
                ),
            ),
        )
    )


def _request(batch: TokenizedTrainingBatch) -> ForwardBackwardRequest:
    return ForwardBackwardRequest(
        run_id="run",
        request_id="request",
        sequence_id=0,
        batch=batch,
        loss=LossConfig(name="cross_entropy"),
    )


def _slot(*, replay: bool):
    requests = []

    async def prepare_pack(request):
        requests.append(request)
        return DistributedPackingPlan(
            batch_id="batch",
            generation_id=request.generation_id,
            source_host="packing",
            trainer_hosts=("trainer",),
            storage_byte_count=1,
        )

    slot = MegatronTrainingSlot.__new__(MegatronTrainingSlot)
    slot._closed = False
    slot._batch_release_failures = []
    slot._runs = {
        "run": SimpleNamespace(model=RolloutModelSpec(payload={})),
    }
    slot.runtime = SimpleNamespace(prepare_pack=prepare_pack)
    slot.runtime_spec = SimpleNamespace(
        enable_moe_routing_replay=replay,
        packed_sequence_length=8,
    )
    return slot, requests


def _ref() -> OperationRef:
    return OperationRef(
        run_id="run",
        operation_id="operation",
        sequence_id=0,
        learner_parent_version=0,
        kind="forward_backward",
    )


@pytest.mark.asyncio
async def test_replay_rejects_route_free_tokenized_input_before_packing() -> None:
    slot, requests = _slot(replay=True)

    with pytest.raises(ValueError, match="replay requires routes"):
        await slot.prepare_forward_packing(_ref(), _request(_batch(routed=False)))

    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("replay,routed", [(True, True), (False, False)])
async def test_tokenized_replay_admission_preserves_packing_input(
    replay: bool, routed: bool
) -> None:
    slot, requests = _slot(replay=replay)
    batch = _batch(routed=routed)

    await slot.prepare_forward_packing(_ref(), _request(batch))

    assert len(requests) == 1
    assert requests[0].tokenized_batch is batch
