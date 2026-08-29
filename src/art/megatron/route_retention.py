from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from art.training import OperationRef, PackedInputCaptureRef
from art.vllm_route_transport import RetainedRouteBundleRef

type RouteBundleOwnershipHandle = object


@dataclass(frozen=True, slots=True)
class RouteBundleOwnershipTransfer:
    operation_id: str
    packed_input: PackedInputCaptureRef
    handle: RouteBundleOwnershipHandle


class RouteBundleOwnershipProvider(Protocol):
    """Service-owned durable leases for exact retained route objects.

    Handles are opaque to ART. Acquisition is keyed by the exact operation;
    transfer adds a target owner without invalidating the source; and the
    caller owns every returned handle until an idempotent release.
    """

    async def acquire(
        self,
        *,
        operation: OperationRef,
        bundles: tuple[RetainedRouteBundleRef, ...],
    ) -> RouteBundleOwnershipHandle:
        """Idempotently retain ``bundles`` for one exact operation."""

        ...

    async def transfer(
        self,
        handle: RouteBundleOwnershipHandle,
        *,
        transfer_id: str,
        target_owner_id: str,
    ) -> RouteBundleOwnershipHandle:
        """Idempotently add a target owner and return its owned handle."""

        ...

    async def release(self, handle: RouteBundleOwnershipHandle) -> None:
        """Idempotently release exactly the supplied ownership handle."""

        ...
