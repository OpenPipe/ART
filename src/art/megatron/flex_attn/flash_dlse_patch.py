"""Torch flex-flash compatibility patches for ART context parallel.

Remove the dLSE portion once torch upstream threads grad_logsumexp into the
flash flex backward path. Keep the block-sparse tile patch until FA4 exposes a
public autograd-compatible tile override for CUTE flex attention.
"""

from __future__ import annotations

import inspect
from typing import Any, cast

import torch

_PATCH_APPLIED = False
_TILE_PATCH_APPLIED = False


def _sm90_block_sparse_fwd_config(cute_interface: Any, head_dim: int, head_dim_v: int):
    del head_dim_v
    if int(head_dim) <= 128:
        return cute_interface.FwdConfig(128, 128, True, True)
    if int(head_dim) <= 192:
        return cute_interface.FwdConfig(128, 96, True, True)
    return cute_interface.FwdConfig(128, 64, True, True)


def _apply_flash_flex_block_sparse_tile_patch() -> None:
    global _TILE_PATCH_APPLIED
    if _TILE_PATCH_APPLIED:
        return

    try:
        import flash_attn.cute.interface as cute_interface  # ty: ignore[unresolved-import]
    except ModuleNotFoundError:
        _TILE_PATCH_APPLIED = True
        return

    cute_interface_any = cast(Any, cute_interface)
    original_tile_size_fwd_sm90 = cute_interface_any._tile_size_fwd_sm90

    def tile_size_fwd_sm90_art(
        head_dim,
        head_dim_v,
        is_causal,
        is_local,
        use_block_sparsity,
    ):
        if use_block_sparsity:
            return _sm90_block_sparse_fwd_config(
                cute_interface,
                int(head_dim),
                int(head_dim_v),
            )
        return original_tile_size_fwd_sm90(
            head_dim,
            head_dim_v,
            is_causal,
            is_local,
            use_block_sparsity,
        )

    cute_interface_any._tile_size_fwd_sm90 = tile_size_fwd_sm90_art
    _TILE_PATCH_APPLIED = True


def _patched_flash_backward_template_source(source: str) -> str:
    patched = source
    kernel_replacements = (
        (
            '{{def_kernel("Q", "K", "V", "OUT", "D_OUT", "LSE", "DK", "DV", "Q_NUM_BLKS", "Q_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}',
            '{{def_kernel("Q", "K", "V", "OUT", "D_OUT", "LSE", "DLSE", "DK", "DV", "Q_NUM_BLKS", "Q_IDX", "FULL_Q_NUM_BLKS", "FULL_Q_IDX")}}',
        ),
        (
            '{{def_kernel("Q", "K", "V", "OUT", "D_OUT", "LSE", "DK", "DV")}}',
            '{{def_kernel("Q", "K", "V", "OUT", "D_OUT", "LSE", "DLSE", "DK", "DV")}}',
        ),
        (
            'def_kernel("Q", "K", "V", "OUT", "D_OUT", "LSE", "DK", "DV")}}',
            'def_kernel("Q", "K", "V", "OUT", "D_OUT", "LSE", "DLSE", "DK", "DV")}}',
        ),
    )
    for before, after in kernel_replacements:
        if before in patched:
            patched = patched.replace(before, after, 1)
            break
    else:
        raise RuntimeError(
            "Unable to patch flash backward template: missing def_kernel signature"
        )
    lse_line = "        LSE,\n"
    if lse_line not in patched:
        raise RuntimeError(
            f"Unable to patch flash backward template: missing {lse_line!r}"
        )
    patched = patched.replace(
        lse_line,
        "        LSE,\n        dlse=DLSE,\n",
        1,
    )
    return patched


def apply_flash_flex_dlse_patch() -> None:
    global _PATCH_APPLIED
    _apply_flash_flex_block_sparse_tile_patch()
    if _PATCH_APPLIED:
        return

    from torch._inductor.codegen.cutedsl.cutedsl_template import CuteDSLTemplate
    import torch._inductor.kernel.flex.flex_attention as flex_attention_mod
    import torch._inductor.kernel.flex.flex_flash_attention as flex_flash_mod
    from torch._inductor.lowering import lowerings

    flex_attention_any = cast(Any, flex_attention_mod)
    flex_flash_any = cast(Any, flex_flash_mod)

    if (
        "grad_logsumexp"
        in inspect.signature(
            flex_flash_mod.create_flex_flash_attention_backward_kernel
        ).parameters
    ):
        _PATCH_APPLIED = True
        return

    patched_template = CuteDSLTemplate(
        name="flash_attention_backward_cutedsl_dlse",
        source=_patched_flash_backward_template_source(
            flex_flash_mod.flash_attention_backward_cutedsl_template.source
        ),
    )
    original_lowering = flex_attention_mod.flex_attention_backward
    original_flash_builder = flex_flash_any.create_flex_flash_attention_backward_kernel

    def create_flex_flash_attention_backward_kernel_with_dlse(
        query,
        key,
        value,
        out,
        logsumexp,
        grad_out,
        grad_logsumexp,
        scale,
        kernel_options,
        sparse_q_block_size,
        sparse_kv_block_size,
        fw_subgraph_buffer=None,
        joint_subgraph_buffer=None,
        score_mod_other_buffers=None,
        mask_graph_buffer=None,
        q_num_blocks=None,
        q_indices=None,
        full_q_num_blocks=None,
        full_q_indices=None,
    ):
        if grad_logsumexp is None:
            return original_flash_builder(
                query,
                key,
                value,
                out,
                logsumexp,
                grad_out,
                scale,
                kernel_options,
                sparse_q_block_size,
                sparse_kv_block_size,
                fw_subgraph_buffer=fw_subgraph_buffer,
                joint_subgraph_buffer=joint_subgraph_buffer,
                score_mod_other_buffers=score_mod_other_buffers,
                mask_graph_buffer=mask_graph_buffer,
                q_num_blocks=q_num_blocks,
                q_indices=q_indices,
                full_q_num_blocks=full_q_num_blocks,
                full_q_indices=full_q_indices,
            )

        if not flex_flash_mod.ensure_flash_available():
            raise RuntimeError("CUTE flash attention not available")

        batch_size, num_heads, seq_len_q, head_dim = query.get_size()
        _, num_heads_kv, seq_len_kv, v_head_dim = value.get_size()
        device = query.get_device()
        dtype = query.get_dtype()
        assert device is not None

        grad_query_strides = flex_flash_mod.infer_dense_strides(
            [batch_size, num_heads, seq_len_q, head_dim], query.get_stride()
        )
        grad_query = flex_flash_mod.empty_strided(
            size=[batch_size, num_heads, seq_len_q, head_dim],
            stride=grad_query_strides,
            dtype=dtype,
            device=device,
        )
        grad_key_strides = flex_flash_mod.infer_dense_strides(
            [batch_size, num_heads_kv, seq_len_kv, head_dim], key.get_stride()
        )
        grad_key = flex_flash_mod.empty_strided(
            size=[batch_size, num_heads_kv, seq_len_kv, head_dim],
            stride=grad_key_strides,
            dtype=dtype,
            device=device,
        )
        grad_value_strides = flex_flash_mod.infer_dense_strides(
            [batch_size, num_heads_kv, seq_len_kv, v_head_dim], value.get_stride()
        )
        grad_value = flex_flash_mod.empty_strided(
            size=[batch_size, num_heads_kv, seq_len_kv, v_head_dim],
            stride=grad_value_strides,
            dtype=dtype,
            device=device,
        )
        output_layout = flex_flash_mod.FixedLayout(
            device=device,
            dtype=dtype,
            size=[batch_size, num_heads, seq_len_q, head_dim],
            stride=[flex_flash_mod.sympy.sympify(s) for s in grad_query.get_stride()],
        )

        sparse_q_block_size = flex_flash_any.V.graph.sizevars.guard_int(
            sparse_q_block_size
        )
        sparse_kv_block_size = flex_flash_any.V.graph.sizevars.guard_int(
            sparse_kv_block_size
        )

        choices: list[Any] = []
        input_nodes = [
            query,
            key,
            value,
            out,
            grad_out,
            logsumexp,
            grad_logsumexp,
            grad_key,
            grad_value,
        ]

        has_block_mask = mask_graph_buffer is not None
        if has_block_mask:
            assert q_indices is not None
            assert full_q_num_blocks is not None
            assert full_q_indices is not None
            input_nodes.extend(
                [
                    q_num_blocks,
                    q_indices,
                    full_q_num_blocks,
                    full_q_indices,
                ]
            )

        has_score_mod = (
            fw_subgraph_buffer is not None and joint_subgraph_buffer is not None
        )
        subgraphs = []
        if has_score_mod:
            subgraphs.append(fw_subgraph_buffer)
            subgraphs.append(joint_subgraph_buffer)
        if has_block_mask:
            subgraphs.append(mask_graph_buffer)

        with flex_flash_mod.patch_fixed_layout_indexer_for_cutedsl():
            error = patched_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=output_layout,
                mutated_inputs=[grad_key, grad_value],
                subgraphs=subgraphs if subgraphs else None,
                SM_SCALE=scale,
                HAS_SCORE_MOD=has_score_mod,
                HAS_BLOCK_MASK=has_block_mask,
                SPARSE_Q_BLOCK_SIZE=sparse_q_block_size,
                SPARSE_KV_BLOCK_SIZE=sparse_kv_block_size,
            )

        for choice in choices:
            flex_flash_mod.wrap_choice_render_with_cutedsl_indexer(choice)

        if error or not choices:
            raise RuntimeError(f"CuteDSL template failed: {error}")

        template_output = choices[0].output_node()
        return (template_output, grad_key, grad_value, tuple())

    def flex_attention_backward_with_flash_dlse(*args, **kwargs):
        if kwargs:
            return original_lowering(*args, **kwargs)
        grad_logsumexp = args[6]
        if grad_logsumexp is None:
            return original_lowering(*args, **kwargs)

        (
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            fw_graph,
            joint_graph,
            block_mask,
            scale,
            kernel_options,
            score_mod_other_buffers,
            mask_mod_other_buffers,
        ) = args
        (
            _,
            _,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
            sparse_q_block_size,
            sparse_kv_block_size,
            mask_graph,
        ) = block_mask

        kernel_options, backend = (
            flex_attention_mod._sanitize_kernel_options_for_triton(kernel_options)
        )
        if backend != "FLASH":
            return original_lowering(*args, **kwargs)

        (
            query,
            key,
            value,
            logsumexp,
            grad_out,
            grad_logsumexp,
            kv_num_blocks,
            kv_indices,
            full_kv_num_blocks,
            full_kv_indices,
            q_num_blocks,
            q_indices,
            full_q_num_blocks,
            full_q_indices,
        ) = flex_attention_mod.maybe_realize(
            [
                query,
                key,
                value,
                logsumexp,
                grad_out,
                flex_attention_mod.ExternKernel.require_contiguous(grad_logsumexp),
                kv_num_blocks,
                kv_indices,
                full_kv_num_blocks,
                full_kv_indices,
                q_num_blocks,
                q_indices,
                full_q_num_blocks,
                full_q_indices,
            ]
        )

        device = query.get_device()
        dtype = query.get_dtype()
        bq, _, seq_len_q, _ = query.get_size()
        bkv, _, seq_len_kv, _ = value.get_size()
        assert flex_attention_mod.V.graph.sizevars.evaluate_expr(
            flex_flash_mod.sympy.Eq(bq, bkv) | flex_flash_mod.sympy.Eq(bkv, 1)
        ), f"Bq and Bkv must broadcastable. Got Bq={bq} and Bkv={bkv}"
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError(
                "Backward pass with mixed query, key, and value dtype is not supported, "
                f"got query.dtype={query.dtype}, key.dtype={key.dtype}, and value.dtype={value.dtype}"
            )

        kernel_options = {
            k: flex_attention_mod.V.graph.sizevars.guard_int(v)
            if isinstance(v, flex_flash_mod.sympy.Symbol)
            else v
            for k, v in kernel_options.items()
        }
        kernel_options.setdefault(
            "FLOAT32_PRECISION", flex_attention_mod.get_float32_precision()
        )
        kernel_options.setdefault(
            "IS_DIVISIBLE",
            flex_attention_mod.V.graph.sizevars.statically_known_true(
                seq_len_q % 128 == 0
            )
            and flex_attention_mod.V.graph.sizevars.statically_known_true(
                seq_len_kv % 128 == 0
            ),
        )

        fwd_placeholder_inps = [
            flex_attention_mod.create_placeholder(name, dtype, device)
            for name, dtype in [
                ("score", dtype),
                ("b", torch.int32),
                ("h", torch.int32),
                ("m", torch.int32),
                ("n", torch.int32),
            ]
        ]
        fw_subgraph_buffer = flex_attention_mod.build_subgraph_buffer(
            fwd_placeholder_inps + list(score_mod_other_buffers), fw_graph
        )
        flex_attention_mod.freeze_irnodes(fw_subgraph_buffer)

        joint_placeholder_inps = fwd_placeholder_inps + [
            flex_attention_mod.create_placeholder("grad_score_mod", dtype, device)
        ]
        joint_graph.graph_module.graph.eliminate_dead_code()
        flex_attention_mod.validate_joint_graph(joint_graph.graph_module.graph)
        all_joint_outputs = flex_attention_mod.build_subgraph_buffer(
            joint_placeholder_inps + list(score_mod_other_buffers), joint_graph
        )
        flex_attention_mod.freeze_irnodes(all_joint_outputs)
        joint_outputs = flex_attention_mod.process_joint_outputs(
            all_joint_outputs, len(joint_placeholder_inps)
        )

        mask_graph_placeholder_inps = [
            flex_attention_mod.create_placeholder(name, dtype, device)
            for name, dtype in [
                ("b", torch.int32),
                ("h", torch.int32),
                ("m", torch.int32),
                ("n", torch.int32),
            ]
        ]
        mask_graph_buffer = flex_attention_mod.build_subgraph_buffer(
            mask_graph_placeholder_inps + list(mask_mod_other_buffers), mask_graph
        )
        flex_attention_mod.freeze_irnodes(mask_graph_buffer)

        if not flex_flash_any._use_flex_flash_attention_backward(
            fw_graph,
            mask_graph,
            backend=backend,
            joint_outputs=joint_outputs,
            score_mod_other_buffers=score_mod_other_buffers,
        ):
            return original_lowering(*args, **kwargs)

        needs_block_mask = not flex_flash_mod.is_trivial_mask_graph(
            mask_graph.graph_module
        )
        if (
            torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled()
            and needs_block_mask
        ):
            raise NotImplementedError(
                "Deterministic backward for flex_attention with block_mask using the FLASH backend "
                "is not yet implemented. The TRITON backend supports deterministic backward."
            )
        if torch.is_deterministic_algorithms_warn_only_enabled() and needs_block_mask:
            flex_attention_any.warnings.warn(
                "Deterministic backward for flex_attention with block_mask using the FLASH backend "
                "is not yet implemented. Running non-deterministic backward.",
            )

        score_is_trivial = flex_flash_mod.is_trivial_score_graph(fw_graph.graph_module)
        return create_flex_flash_attention_backward_kernel_with_dlse(
            query,
            key,
            value,
            out,
            logsumexp,
            grad_out,
            grad_logsumexp,
            scale,
            kernel_options,
            sparse_q_block_size,
            sparse_kv_block_size,
            fw_subgraph_buffer=None if score_is_trivial else fw_subgraph_buffer,
            joint_subgraph_buffer=None
            if score_is_trivial
            else joint_outputs.grad_input,
            score_mod_other_buffers=list(score_mod_other_buffers),
            mask_graph_buffer=mask_graph_buffer if needs_block_mask else None,
            q_num_blocks=q_num_blocks if needs_block_mask else None,
            q_indices=q_indices if needs_block_mask else None,
            full_q_num_blocks=full_q_num_blocks if needs_block_mask else None,
            full_q_indices=full_q_indices if needs_block_mask else None,
        )

    flex_flash_any.create_flex_flash_attention_backward_kernel_with_dlse = (
        create_flex_flash_attention_backward_kernel_with_dlse
    )
    flex_attention_mod.flex_attention_backward = flex_attention_backward_with_flash_dlse
    lowerings[torch.ops.higher_order.flex_attention_backward] = (
        flex_attention_backward_with_flash_dlse
    )
    _PATCH_APPLIED = True
