"""Separate model RNG consumption from the replicated caller's PyTorch stream."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib

import torch
import torch.distributed as dist


@dataclass
class _State:
    cpu: torch.Tensor
    cuda: torch.Tensor | None

    @classmethod
    def capture(cls, device: torch.device) -> "_State":
        return cls(
            torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        )

    def restore(self, device: torch.device) -> None:
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda, device)

    def model_stream(self, device: torch.device) -> "_State":
        def derive(state: torch.Tensor, target: torch.device | str) -> torch.Tensor:
            digest = hashlib.sha256(b"art.trainer_rank.model" + state.numpy().tobytes())
            seed = int.from_bytes(digest.digest()[:8], "little")
            return torch.Generator(device=target).manual_seed(seed).get_state()

        return _State(
            derive(self.cpu, "cpu"),
            derive(self.cuda, device) if self.cuda is not None else None,
        )


class TrainerRNG:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._model: _State | None = None
        self._depth = 0

    @contextmanager
    def model(self) -> Iterator[None]:
        """Advance a private model stream and restore the caller even on error.

        Megatron's separate model-parallel RNG tracker is deliberately untouched.
        Native torch/Megatron checkpoints capture these forward states and restore
        the ambient caller state after backward recomputation themselves.
        Never keep this context open across a public iterator yield.
        """
        if self._depth:
            yield
            return
        caller = _State.capture(self.device)
        if self._model is None:
            # Copying the state would correlate model dropout with the first
            # caller draws. Derive a separate deterministic default stream.
            self._model = caller.model_stream(self.device)
        self._model.restore(self.device)
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            self._model = _State.capture(self.device)
            caller.restore(self.device)

    def synchronize(self, group: dist.ProcessGroup | None) -> None:
        """Continue the TP×CP leader's stream; never synchronize across DP.

        None means no model-parallel group, not the default WORLD group. The
        caller's live state is authoritative, including explicit manual seeding
        or RNG restoration between forwards. Identically seeded DP workers are
        allowed to stay identical; this method does not reseed them.
        """
        if group is None or dist.get_world_size(group) == 1:
            return
        state = _State.capture(self.device)
        sizes = [state.cpu.numel()]
        states = [state.cpu]
        if state.cuda is not None:
            sizes.append(state.cuda.numel())
            states.append(state.cuda)
        payload = torch.cat(states).to(
            self.device if dist.get_backend(group) == "nccl" else "cpu"
        )
        dist.broadcast(payload, src=dist.get_global_rank(group, 0), group=group)
        received = payload.cpu().split(sizes)
        _State(received[0], received[1] if len(received) == 2 else None).restore(
            self.device
        )


def caller_group() -> dist.ProcessGroup | None:
    if not (dist.is_available() and dist.is_initialized()):
        return None
    try:
        from megatron.core import parallel_state as ps

        return ps.get_tensor_and_context_parallel_group(check_initialized=False)
    except (AssertionError, ImportError, RuntimeError, ValueError):
        return None
