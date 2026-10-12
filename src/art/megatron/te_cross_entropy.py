_GRADIENT = """        if reduce_loss:
            X_block = (tl.exp(X_block - m) / d - eps) / (n_non_ignore)
        else:
            X_block = tl.exp(X_block - m) / d - eps
"""

_FP32_GRADIENT = """        X_block = tl.exp(X_block - m) / d - eps
        X_block -= tl.where(X_offsets + rank * n_cols == y, 1 - label_smoothing, 0.0)
        if reduce_loss:
            X_block /= n_non_ignore
"""

_TARGET = """    # 6. Specially handle the i==y case where `dx_y = (softmax(x_y) - (1 - label_smoothing) / N`
    vocab_start_idx = rank * n_cols
    vocab_end_idx = (rank + 1) * n_cols
    if y >= vocab_start_idx:
        if y < vocab_end_idx:
            X_y = tl.load(X_ptr + y - vocab_start_idx)
            # Apply the same conditional scaling logic for the target token
            if reduce_loss:
                X_y += -(1 - label_smoothing) / (n_non_ignore)
            else:
                X_y += -(1 - label_smoothing)
            tl.store(X_ptr + y - vocab_start_idx, X_y)

"""


def patch_te_cross_entropy() -> None:
    from transformer_engine.common.triton.cross_entropy import cross_entropy_kernel

    if getattr(cross_entropy_kernel, "__art_fp32_target_grad__", False):
        return
    source = cross_entropy_kernel.src
    if source.count(_GRADIENT) != 1 or source.count(_TARGET) != 1:
        raise RuntimeError("Unsupported Transformer Engine cross-entropy kernel source")
    # TE 2.11/2.14 store gradients in the logits dtype. Combine the target term
    # in FP32 before that store, retaining each version's reduction and TP logic.
    source = source.replace(_GRADIENT, _FP32_GRADIENT).replace(_TARGET, "")
    cross_entropy_kernel._unsafe_update_src(source)
    # Triton's source update invalidates its hash but not compiled specializations.
    cross_entropy_kernel.device_caches.clear()
    setattr(cross_entropy_kernel, "__art_fp32_target_grad__", True)
