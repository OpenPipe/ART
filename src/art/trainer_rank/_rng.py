"""Model/caller RNG ownership and replay of recorded forward randomness."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import random
from typing import Any

import torch
import torch.distributed as dist


@dataclass
class RNGState:
    cpu: torch.Tensor
    cuda: dict[int, torch.Tensor]
    python: tuple[Any, ...] | None = None
    tracker: Any = None

    @classmethod
    def capture(
        cls,
        devices: Sequence[int],
        tracker: Any = None,
        *,
        torch_only: bool = False,
    ) -> "RNGState":
        return cls(
            torch.get_rng_state(),
            {device: torch.cuda.get_rng_state(device) for device in devices},
            None if torch_only else random.getstate(),
            None if tracker is None else deepcopy(tracker.get_states()),
        )

    def restore(self, tracker: Any = None) -> None:
        torch.set_rng_state(self.cpu)
        for device, state in self.cuda.items():
            torch.cuda.set_rng_state(state, device)
        if self.python is not None:
            random.setstate(self.python)
        if tracker is not None:
            tracker.set_states(deepcopy(self.tracker))

    @contextmanager
    def replay(self, tracker: Any = None) -> Iterator[None]:
        ambient = self.capture(
            tuple(self.cuda), tracker, torch_only=self.python is None
        )
        try:
            self.restore(tracker)
            yield
        finally:
            ambient.restore(tracker)


class TrainerRNG:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.devices = (
            (torch.cuda.current_device() if device.index is None else device.index,)
            if device.type == "cuda"
            else ()
        )
        initial = RNGState.capture(self.devices, torch_only=True)

        def derive(state: torch.Tensor, target: torch.device | str) -> torch.Tensor:
            digest = hashlib.sha256(b"art.trainer_rank.model" + state.numpy().tobytes())
            seed = int.from_bytes(digest.digest()[:8], "little")
            return torch.Generator(device=target).manual_seed(seed).get_state()

        # Capture before logical leaders run user code or initialize custom heads;
        # those draws must not change the first model stream relative to peers.
        self._model = RNGState(
            derive(initial.cpu, "cpu"),
            {index: derive(state, device) for index, state in initial.cuda.items()},
        )
        self._depth = 0

    @contextmanager
    def model(self) -> Iterator[None]:
        """Advance the private torch stream, restoring caller state even on error.

        Never span a public iterator yield. Python and Megatron's separate RNG
        tracker are untouched; activation checkpointing must preserve its RNG.
        """
        if self._depth:
            yield
            return
        with self._model.replay():
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                self._model = RNGState.capture(self.devices, torch_only=True)

    def synchronize(self, group: dist.ProcessGroup | None) -> None:
        """Continue the TP×CP leader's caller stream, never synchronizing DP.

        None means no model-parallel group, not WORLD. Explicit caller reseeding
        or restoration remains authoritative, and equal DP seeds remain equal.
        """
        if group is None or dist.get_world_size(group) == 1:
            return
        state = RNGState.capture(self.devices, torch_only=True)
        states = [state.cpu, *state.cuda.values()]
        payload = torch.cat(states).to(
            self.device if dist.get_backend(group) == "nccl" else "cpu"
        )
        dist.broadcast(payload, src=dist.get_global_rank(group, 0), group=group)
        received = payload.cpu().split([value.numel() for value in states])
        RNGState(
            received[0], dict(zip(state.cuda, received[1:], strict=True))
        ).restore()


def caller_group() -> dist.ProcessGroup | None:
    if not (dist.is_available() and dist.is_initialized()):
        return None
    try:
        from megatron.core import parallel_state as ps

        return ps.get_tensor_and_context_parallel_group(check_initialized=False)
    except (AssertionError, ImportError, RuntimeError, ValueError):
        return None
