from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import random
from typing import Any, Iterator

from megatron.core import parallel_state as ps
import numpy as np
from pydantic import BaseModel, ConfigDict
import torch


class BaseMegatronSessionKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_model: str
    model_key: str
    num_layers: int
    precision: str
    allow_unvalidated_arch: bool


class BaseMegatronResetReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    process_group_reused: bool
    model_reused: bool
    parameter_count: int
    parameter_versions_unchanged: bool
    buffer_count: int
    buffers_restored: int
    gradient_tensors_cleared: int
    gradients_cleared: bool
    optimizer_released: bool
    moe_replay_was_active: bool
    moe_replay_cleared: bool
    rng_restored: bool


class BaseMegatronSession:
    def __init__(self) -> None:
        self.runtime: Any | None = None
        self.key: BaseMegatronSessionKey | None = None
        self.reset_report: BaseMegatronResetReport | None = None
        self._parameters: list[tuple[str, torch.nn.Parameter, int]] = []
        self._buffers: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        self._rng_state: tuple[Any, Any, torch.Tensor, list[torch.Tensor]] | None = None
        self._process_group: Any | None = None
        self._model_chunks: tuple[Any, ...] = ()

    def capture_runtime(self, runtime: Any, *, key: BaseMegatronSessionKey) -> None:
        if self.runtime is not None:
            raise RuntimeError("base Megatron session already owns a runtime")
        if not torch.distributed.is_initialized():  # type: ignore[possibly-missing-attribute]
            raise RuntimeError(
                "base Megatron session requires an initialized process group"
            )
        if runtime.model_support_spec.key != key.model_key:
            raise RuntimeError(
                f"base Megatron handler mismatch: {runtime.model_support_spec.key} != {key.model_key}"
            )
        self.runtime = runtime
        self.key = key
        self._parameters = [
            (f"{chunk_index}:{name}", parameter, parameter._version)
            for chunk_index, chunk in enumerate(runtime.model)
            for name, parameter in chunk.named_parameters()
        ]
        self._buffers = [
            (f"{chunk_index}:{name}", buffer, buffer.detach().clone())
            for chunk_index, chunk in enumerate(runtime.model)
            for name, buffer in chunk.named_buffers()
        ]
        self._rng_state = (
            random.getstate(),
            np.random.get_state(),
            torch.get_rng_state(),
            torch.cuda.get_rng_state_all(),
        )
        self._process_group = torch.distributed.group.WORLD  # type: ignore[possibly-missing-attribute]
        self._model_chunks = tuple(runtime.model)

    def owns_runtime(self, runtime: Any) -> bool:
        return self.runtime is runtime

    def reset_for_packing(self, *, key: BaseMegatronSessionKey) -> Any:
        from art.megatron import train as megatron_train

        runtime = self.runtime
        if runtime is None or self.key != key or self._rng_state is None:
            raise RuntimeError(
                f"base Megatron session is incompatible: retained={self.key}, requested={key}"
            )
        process_group_reused = (
            torch.distributed.is_initialized()  # type: ignore[possibly-missing-attribute]
            and torch.distributed.group.WORLD is self._process_group  # type: ignore[possibly-missing-attribute]
        )
        model_reused = len(runtime.model) == len(self._model_chunks) and all(
            current is retained
            for current, retained in zip(runtime.model, self._model_chunks, strict=True)
        )
        if not process_group_reused or not model_reused:
            raise RuntimeError("base Megatron process group or model was replaced")
        changed_parameters = [
            name
            for name, parameter, version in self._parameters
            if parameter._version != version
        ]
        if changed_parameters:
            raise RuntimeError(
                "HF parity mutated base parameters: "
                + ", ".join(changed_parameters[:8])
            )

        had_replay = runtime.moe_routing_replay_controller is not None
        megatron_train.configure_moe_routing_replay(runtime)
        if runtime.optimizer is not None:
            runtime.optimizer.zero_grad()
        megatron_train._zero_grad_buffers(runtime.model)
        gradient_tensors_cleared = 0
        for _name, parameter, _version in self._parameters:
            if parameter.grad is not None:
                parameter.grad = None
                gradient_tensors_cleared += 1
            main_grad = getattr(parameter, "main_grad", None)
            if isinstance(main_grad, torch.Tensor):
                main_grad.zero_()
                gradient_tensors_cleared += 1
        runtime.optimizer = None
        gradients_cleared = all(
            parameter.grad is None
            and (
                not isinstance(
                    main_grad := getattr(parameter, "main_grad", None), torch.Tensor
                )
                or not bool(torch.count_nonzero(main_grad).item())
            )
            for _name, parameter, _version in self._parameters
        )
        if not gradients_cleared:
            raise RuntimeError("base Megatron gradient reset was incomplete")

        buffers_restored = 0
        with torch.no_grad():
            for name, buffer, initial in self._buffers:
                if buffer.shape != initial.shape or buffer.dtype != initial.dtype:
                    raise RuntimeError(f"HF parity changed buffer metadata for {name}")
                if not torch.equal(buffer, initial):
                    buffer.copy_(initial)
                    buffers_restored += 1
        python_state, numpy_state, torch_state, cuda_states = self._rng_state
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state_all(cuda_states)
        for chunk in runtime.model:
            chunk.eval()
        self.reset_report = BaseMegatronResetReport(
            process_group_reused=True,
            model_reused=True,
            parameter_count=len(self._parameters),
            parameter_versions_unchanged=True,
            buffer_count=len(self._buffers),
            buffers_restored=buffers_restored,
            gradient_tensors_cleared=gradient_tensors_cleared,
            gradients_cleared=True,
            optimizer_released=runtime.optimizer is None,
            moe_replay_was_active=had_replay,
            moe_replay_cleared=runtime.moe_routing_replay_controller is None,
            rng_restored=True,
        )
        return runtime

    def close(self) -> None:
        runtime, self.runtime = self.runtime, None
        try:
            if runtime is not None:
                from art.megatron import train as megatron_train

                megatron_train.configure_moe_routing_replay(runtime)
            if getattr(ps, "model_parallel_is_initialized", lambda: False)():
                ps.destroy_model_parallel()
            if torch.distributed.is_initialized():  # type: ignore[possibly-missing-attribute]
                torch.distributed.destroy_process_group()  # type: ignore[possibly-missing-attribute]
        finally:
            self.key = None
            self._parameters.clear()
            self._buffers.clear()
            self._rng_state = None
            self._process_group = None
            self._model_chunks = ()
            del runtime
            torch.cuda.empty_cache()


_ACTIVE_SESSION: ContextVar[BaseMegatronSession | None] = ContextVar(
    "base_megatron_session", default=None
)


def active_base_megatron_session() -> BaseMegatronSession | None:
    return _ACTIVE_SESSION.get()


@contextmanager
def base_megatron_session() -> Iterator[BaseMegatronSession]:
    if _ACTIVE_SESSION.get() is not None:
        raise RuntimeError("nested base Megatron sessions are not supported")
    session = BaseMegatronSession()
    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
    finally:
        try:
            session.close()
        finally:
            _ACTIVE_SESSION.reset(token)
