"""Bind converted base tensors before wrapping, retaining the external owner."""

from collections.abc import Mapping

import torch


def bind_external_weights(
    chunks: list[torch.nn.Module], tensors: Mapping[str, torch.Tensor]
) -> list[torch.nn.Module]:
    """Keys are ``{chunk_index}:{parameter_or_buffer_name}`` in native layout.

    Validate every chunk before changing any model. Caller keeps the allocation
    owner alive for the runtime's lifetime. No weight copy or allocation occurs.
    Empty TE extra-state entries are the only non-parameter entries admitted.
    """
    bindings = []
    seen = set()
    module_chunk = {
        id(module): index
        for index, chunk in enumerate(chunks)
        for module in chunk.modules()
    }
    for index, chunk in enumerate(chunks):
        for name, module in chunk.named_modules():
            for parameters, values in (
                (True, module._parameters),
                (False, module._buffers),
            ):
                for attribute, target in values.items():
                    if target is None:
                        continue
                    key = f"{index}:" + ".".join(
                        part for part in (name, attribute) if part
                    )
                    source = tensors.get(key)
                    if source is None:
                        if parameters or target.is_meta:
                            raise ValueError("missing external weight: " + key)
                        continue
                    if (
                        source.shape != target.shape
                        or source.dtype != target.dtype
                        or source.is_meta
                    ):
                        raise ValueError("external weight layout mismatch: " + key)
                    if not source.is_contiguous():
                        raise ValueError("external weight must be contiguous: " + key)
                    seen.add(key)
                    bindings.append((module, attribute, target, source, parameters))
    extra = {
        key
        for key in tensors
        if key not in seen
        and not (key.endswith("._extra_state") and tensors[key].numel() == 0)
    }
    if extra:
        raise ValueError("unused external weights: " + str(sorted(extra)[:10]))
    for module, attribute, target, source, parameters in bindings:
        if parameters:
            replacement = torch.nn.Parameter(source, requires_grad=target.requires_grad)
            replacement.__dict__.update(target.__dict__)
            module._parameters[attribute] = replacement
        else:
            module._buffers[attribute] = source
    for index, chunk in enumerate(chunks):
        # Keep identity checks independent of names changed by LoRA/DDP wrappers.
        chunk._art_external_weight_aliases = [
            (
                module._parameters[attribute]
                if parameters
                else module._buffers[attribute],
                source.data_ptr(),
            )
            for module, attribute, _, source, parameters in bindings
            if module_chunk[id(module)] == index
        ]
    return chunks
