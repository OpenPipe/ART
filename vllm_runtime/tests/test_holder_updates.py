import asyncio

from art_vllm_runtime.holder_updates import (
    HolderAck,
    HolderMutation,
    HolderUpdateCoordinator,
    LoraMutation,
)
import pytest


class _Directory:
    def __init__(self) -> None:
        self.holders = ("replica-a", "replica-b")
        self.eligibility: dict[str, tuple[bool, int]] = {}

    async def current_holders(self, owner_id: str, lora_slot: str) -> tuple[str, ...]:
        assert (owner_id, lora_slot) == ("tenant", "slot")
        return self.holders

    async def set_eligible(
        self,
        owner_id: str,
        lora_slot: str,
        holder_id: str,
        *,
        eligible: bool,
        update_seq: int,
    ) -> None:
        assert (owner_id, lora_slot) == ("tenant", "slot")
        self.eligibility[holder_id] = (eligible, update_seq)


def _mutation(expected: int, generation: str) -> LoraMutation[str]:
    return LoraMutation(
        owner_id="tenant",
        request_id=f"request-{generation}",
        lora_slot="slot",
        expected_update_seq=expected,
        policy_version=expected + 1,
        generation_id=generation,
        source=f"s3://bucket/{generation}",
    )


@pytest.mark.asyncio
async def test_fans_out_without_collective_acceptance_and_recovers_failed_holder() -> (
    None
):
    directory = _Directory()
    release = asyncio.Event()
    fail_b = True

    async def apply(holder: str, command: HolderMutation) -> HolderAck:
        await release.wait()
        if holder == "replica-b" and fail_b:
            raise RuntimeError("holder failed")
        return HolderAck(
            holder_id=holder,
            lora_slot=command.lora_slot,
            update_seq=command.update_seq,
            generation_id=command.generation_id,
        )

    coordinator = HolderUpdateCoordinator(directory, apply)
    first = await coordinator.admit(_mutation(0, "generation-1"))
    assert first.targeted_holders == 2
    waiter = asyncio.create_task(coordinator.wait(first.operation_id))
    assert not waiter.done()
    release.set()
    result = await waiter
    assert result.succeeded_holders == ("replica-a",)
    assert result.failed_holders == ("replica-b",)
    assert directory.eligibility == {
        "replica-a": (True, 1),
        "replica-b": (False, 1),
    }

    fail_b = False
    second = await coordinator.admit(_mutation(1, "generation-2"))
    recovered = await coordinator.wait(second.operation_id)
    assert recovered.failed_holders == ()
    assert directory.eligibility["replica-b"] == (True, 2)


@pytest.mark.asyncio
async def test_rejects_sequence_conflicts_and_overlapping_slot_updates() -> None:
    directory = _Directory()
    release = asyncio.Event()

    async def apply(holder: str, command: HolderMutation) -> HolderAck:
        await release.wait()
        return HolderAck(
            holder, command.lora_slot, command.update_seq, command.generation_id
        )

    coordinator = HolderUpdateCoordinator(directory, apply)
    first = await coordinator.admit(_mutation(0, "generation-1"))
    assert await coordinator.admit(_mutation(0, "generation-1")) == first
    with pytest.raises(RuntimeError, match="already active"):
        await coordinator.admit(_mutation(1, "generation-2"))
    release.set()
    await coordinator.wait(first.operation_id)
    with pytest.raises(RuntimeError, match="precondition failed"):
        await coordinator.admit(_mutation(0, "generation-3"))
    changed = _mutation(1, "generation-1")
    with pytest.raises(RuntimeError, match="different content"):
        await coordinator.admit(changed)
