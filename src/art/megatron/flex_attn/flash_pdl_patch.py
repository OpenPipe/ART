"""Pinned FA4 SM90 block-sparse backward dependency workaround.

FA4 4.0.0b5 signals dependents before writing LSE/dPsum and clearing dQaccum,
but its sparse SM90 consumer lacks the dense consumer's dependency wait.
Disable PDL on both sides of that boundary. Suppressing only the producer's
signal is insufficient: a PDL consumer still needs a memory-visibility wait.
Remove after upgrading past upstream commit 4460ebbcb5ed.
"""

from __future__ import annotations

import ast
import importlib.metadata
import inspect
import linecache
import sys
import textwrap
from threading import RLock
from types import FunctionType, SimpleNamespace
from typing import Any


def _clone_function(function: Any, namespace: dict[str, Any]) -> Any:
    clone = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    return clone


def _preprocess_dispatch(interface: Any) -> Any:
    # Clone the Python compiler driver, not the DSL kernel. The factory changes
    # only the freshly constructed kernel instance, never the shared class.
    compiler = interface._compile_bwd_preprocess
    kernel_class = compiler.__globals__["FlashAttentionBackwardPreprocess"]

    def no_pdl_kernel(*args: Any, **kwargs: Any) -> Any:
        kernel = kernel_class(*args, **kwargs)
        kernel.use_pdl = False
        return kernel

    compile_no_pdl = _clone_function(
        compiler,
        {**compiler.__globals__, "FlashAttentionBackwardPreprocess": no_pdl_kernel},
    )
    original = interface._bwd_preprocess
    namespace = {
        **original.__globals__,
        "_compile_bwd_preprocess": compile_no_pdl,
    }
    no_pdl = _clone_function(original, namespace)
    namespace["_bwd_preprocess"] = no_pdl
    # Never reuse a previously compiled PDL preprocess (including a warm cache).
    no_pdl.compile_cache = interface.get_jit_cache("bwd_pre_art_no_pdl_v1")

    def dispatch(*args: Any, disable_pdl: bool, **kwargs: Any) -> Any:
        return (no_pdl if disable_pdl else original)(*args, **kwargs)

    dispatch.compile_cache = no_pdl.compile_cache
    return dispatch


def _patched_backward_code(function: Any) -> Any:
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_bwd_preprocess"
    ]
    if len(calls) != 1:
        raise RuntimeError("Unsupported FA4 backward preprocess call site")
    call = calls[0]
    call.func.id = "_art_bwd_preprocess_sm90_pdl"
    call.keywords.append(
        ast.keyword(
            arg="disable_pdl",
            value=ast.parse("arch // 10 == 9 and use_block_sparsity", mode="eval").body,
        )
    )
    # Only the sparse SM90 copy uses this cache. The ordinary entry's cache,
    # key tuples and persistent paths remain exactly as FA4 created them.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "compile_cache"
            and isinstance(node.value, ast.Name)
            and node.value.id == "_flash_attn_bwd"
        ):
            node.value.id = "_art_sparse_backward"
    ast.fix_missing_locations(tree)
    patched_source = ast.unparse(tree) + "\n"
    filename = f"{function.__code__.co_filename}.art_sm90_pdl"
    linecache.cache[filename] = (
        len(patched_source),
        None,
        patched_source.splitlines(keepends=True),
        filename,
    )
    namespace = dict(function.__globals__)
    exec(compile(patched_source, filename, "exec"), namespace)
    return namespace[function.__name__].__code__


def _patched_sm90_launch_code(function: Any) -> Any:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    launches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "launch"
    ]
    flags = [kw for call in launches for kw in call.keywords if kw.arg == "use_pdl"]
    if len(flags) != 1 or not isinstance(flags[0].value, ast.Constant):
        raise RuntimeError("Unsupported FA4 SM90 backward PDL launch")
    flags[0].value = ast.parse("not self.use_block_sparsity", mode="eval").body
    # CuTe's preprocessor requires @cute.jit in inspect's source. The identity
    # decorator here materializes the raw body without changing the JIT wrapper
    # or any of its previously imported references.
    tree.body[0].decorator_list = [ast.parse("cute.jit", mode="eval").body]
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    filename = f"{function.__code__.co_filename}.art_sm90_pdl_launch"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {**function.__globals__, "cute": SimpleNamespace(jit=lambda fn: fn)}
    exec(compile(source, filename, "exec"), namespace)
    return namespace[function.__name__].__code__


def installed_fa4_version() -> str | None:
    try:
        return importlib.metadata.version("flash-attn-4")
    except importlib.metadata.PackageNotFoundError:
        return None


def _assert_quiescent(interface: Any, state: dict[str, Any]) -> None:
    message = (
        "Install the FA4 sparse PDL patch from ART flex_attn import before any "
        "FA4 backward compilation or CUDA graph capture; compilation must be quiescent"
    )
    if state["active"]:
        raise RuntimeError(message + " (backward call active)")
    caches = list(state["caches"])
    for name in ("_flash_attn_bwd", "_bwd_preprocess", "_bwd_postprocess_convert"):
        function = getattr(interface, name, None)
        cache = getattr(function, "compile_cache", None)
        if cache is not None:
            caches.append(cache)
    for cache in caches:
        entries = cache if isinstance(cache, dict) else cache.cache
        if entries:
            raise RuntimeError(message + " (backward compile cache is populated)")
    # This query never initializes CUDA. A graph capture already in progress
    # is also too late, even when no backward kernel has yet populated a cache.
    torch = sys.modules.get("torch")
    if (
        torch is not None
        and torch.cuda.is_initialized()
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(message + " (CUDA graph capture active)")


def _entry(*args: Any, **kwargs: Any) -> Any:
    return globals()["_art_bwd_guard_dispatch"](*args, **kwargs)


def _guarded_dispatch(
    interface: Any, state: dict[str, Any], version: str, original: Any, sparse: Any
) -> Any:
    signature = inspect.signature(original)

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        with state["lock"]:
            state["active"] += 1
        try:
            arguments = signature.bind(*args, **kwargs).arguments
            arch = arguments.get("arch")
            if arch is None:
                arch = interface._get_device_arch()
            affected = (
                arch // 10 == 9 and arguments.get("block_sparse_tensors") is not None
            )
            if affected and version != "4.0.0b5":
                raise RuntimeError(
                    f"Recheck FA4 sparse PDL workaround for {version}: "
                    "refusing SM90 block-sparse backward before any launch"
                )
            return (sparse if affected else original)(*args, **kwargs)
        finally:
            with state["lock"]:
                state["active"] -= 1

    return dispatch


def apply_flash_sm90_sparse_pdl_patch(
    interface: Any = None, *, version: str | None = None
) -> bool:
    """Install while quiescent, before backward compilation or graph capture.

    Non-b5 versions install only a lazy guard; unaffected calls stay unpatched.
    An injected interface/version supports CPU contracts without importing FA4.
    State and locking live on the interface, surviving reloads of this module.
    """
    if interface is None:
        version = installed_fa4_version()
        if version is None:
            return False
        import flash_attn.cute.interface as interface  # ty: ignore[unresolved-import]
    elif version is None:
        version = "4.0.0b5"

    # setdefault publishes one lock even when callers loaded this patch module
    # under different names. importlib.reload preserves the interface dict.
    state = interface.__dict__.setdefault(
        "_art_sm90_pdl_state", {"lock": RLock(), "active": 0, "caches": []}
    )
    with state["lock"]:
        _assert_quiescent(interface, state)
        backward = interface._flash_attn_bwd
        entry_installed = getattr(backward, "_art_sm90_sparse_pdl", False)
        launches = []
        if version == "4.0.0b5":
            # The interface may retain an older class after the kernel module
            # reloads. Patch both references, independently of the entry marker.
            classes = [interface.FlashAttentionBackwardSm90]
            module = sys.modules.get(classes[0].__module__)
            current = getattr(module, "FlashAttentionBackwardSm90", classes[0])
            if current is not classes[0]:
                classes.append(current)
            for cls in classes:
                raw = inspect.unwrap(cls.__call__)
                if not getattr(raw, "_art_sm90_sparse_pdl_launch", False):
                    if getattr(raw, "_preprocessed", False):
                        raise RuntimeError(
                            "Install the FA4 sparse PDL patch before any FA4 "
                            "backward compilation (SM90 DSL preprocessing started)"
                        )
                    launches.append((raw, _patched_sm90_launch_code(raw)))

        if not entry_installed:
            signature = inspect.signature(backward)
            original = _clone_function(backward, backward.__globals__)
            sparse = None
            if version == "4.0.0b5":
                code = _patched_backward_code(backward)
                preprocess = _preprocess_dispatch(interface)
                sparse = _clone_function(backward, backward.__globals__)
                sparse.__code__ = code
                sparse.compile_cache = interface.get_jit_cache(
                    "bwd_art_sm90_sparse_no_pdl_v2"
                )
                state["caches"].extend([sparse.compile_cache, preprocess.compile_cache])
                backward.__globals__["_art_sparse_backward"] = sparse
                backward.__globals__["_art_bwd_preprocess_sm90_pdl"] = preprocess
            dispatch = _guarded_dispatch(interface, state, version, original, sparse)
            backward.__globals__["_art_bwd_guard_dispatch"] = dispatch
            # Keep the original function and compile_cache objects. Existing
            # imports and unaffected on-disk lookups remain effective.
            backward.__code__ = _entry.__code__
            backward.__defaults__ = None
            backward.__kwdefaults__ = None
            backward.__signature__ = signature
            backward._art_sm90_sparse_pdl = True
            state["caches"].append(backward.compile_cache)

        for raw, code in launches:
            raw.__code__ = code
            raw._art_sm90_sparse_pdl_launch = True
        return True
