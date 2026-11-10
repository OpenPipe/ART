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
import textwrap
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


def apply_flash_sm90_sparse_pdl_patch(interface: Any = None) -> bool:
    """Install before compilation; preserve references imported before install.

    The optional interface permits CPU contract tests without importing CUDA.
    Missing FA4 is retryable; a different installed version fails explicitly.
    """
    if interface is None:
        try:
            version = importlib.metadata.version("flash-attn-4")
        except importlib.metadata.PackageNotFoundError:
            return False
        if version != "4.0.0b5":
            raise RuntimeError(f"Recheck FA4 sparse PDL workaround for {version}")
        import flash_attn.cute.interface as interface  # ty: ignore[unresolved-import]

    backward = interface._flash_attn_bwd
    if getattr(backward, "_art_sm90_sparse_pdl", False):
        return True
    code = _patched_backward_code(backward)
    sm90_launch = inspect.unwrap(interface.FlashAttentionBackwardSm90.__call__)
    launch_code = _patched_sm90_launch_code(sm90_launch)
    dispatch = _preprocess_dispatch(interface)
    # Old compiled backward launchers embed use_pdl=True. Give the changed
    # launcher its own cache domain, including when disk caching is enabled.
    backward.compile_cache = interface.get_jit_cache("bwd_art_sparse_no_pdl_v1")
    # Keep the function objects and signatures intact: already imported
    # references and generated torch wrappers must see the patched bodies.
    sm90_launch.__code__ = launch_code
    backward.__globals__["_art_bwd_preprocess_sm90_pdl"] = dispatch
    backward.__code__ = code
    backward._art_sm90_sparse_pdl = True
    return True
