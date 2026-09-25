"""Conservative eligibility for invocation-local chat-render reuse."""

import inspect
import math
import sys
from types import CodeType
from typing import cast


def _render_context_key(value: object) -> object:
    """Snapshot plain JSON without losing mapping order or scalar types."""
    kind = type(value)
    if kind in (str, int, bool, type(None)):
        return kind, value
    if kind is float and math.isfinite(cast(float, value)):
        return kind, repr(value)
    if kind is list:
        return kind, tuple(_render_context_key(item) for item in cast(list, value))
    if kind is dict and all(type(key) is str for key in cast(dict, value)):
        return kind, tuple(
            (key, _render_context_key(item)) for key, item in cast(dict, value).items()
        )
    raise TypeError("Not a plain JSON rendering context")


def _code_contains(code, candidate, name):
    return (code is candidate and code.co_name == name) or any(
        _code_contains(child, candidate, name)
        for child in code.co_consts
        if isinstance(child, CodeType)
    )


def cacheable_chat_template(tokenizer, template, tools, kwargs, messages) -> bool:
    """Admit a finite, nonmutating Jinja subset, never arbitrary renderer purity.

    Callers must key all effective context and keep this cache local to one
    tokenization. Unsupported syntax, context or overrides retain normal rendering.
    No Transformers import or global eligibility memo is performed here.
    """
    try:
        base_module = sys.modules.get("transformers.tokenization_utils_base")
        chat = sys.modules.get("transformers.utils.chat_template_utils")
        if base_module is None or chat is None or type(template) is not str:
            return False
        base = base_module.PreTrainedTokenizerBase
        cls = type(tokenizer)
        module = sys.modules.get(cls.__module__)
        if (
            not isinstance(tokenizer, base)
            or not cls.__module__.startswith("transformers.")
            or getattr(module, cls.__name__, None) is not cls
            or type(tokenizer.chat_template) not in (str, type(None))
            or inspect.getattr_static(cls, "special_tokens_map")
            is not inspect.getattr_static(base, "special_tokens_map")
        ):
            return False
        for name in ("apply_chat_template", "get_chat_template"):
            method = getattr(tokenizer, name)
            if method.__self__ is not tokenizer or method.__func__ is not getattr(
                base, name
            ):
                return False
        # Check before the stock property calls str(): custom objects can hide
        # mutable state even if their resulting special-token strings look plain.
        added_token = sys.modules["tokenizers"].AddedToken
        special = tokenizer._special_tokens_map
        if type(special) is not dict or any(
            type(key) is not str or type(value) not in (str, type(None), added_token)
            for key, value in special.items()
        ):
            return False
        _render_context_key([messages, tools, kwargs, tokenizer.special_tokens_map])
        if type(kwargs) is not dict or kwargs.get("continue_final_message"):
            return False

        from jinja2 import defaults, nodes
        from jinja2.runtime import LoopContext
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from jinja2.utils import Namespace

        compiled = chat._compile_jinja_template(template)
        env = compiled.environment
        if type(env) is not ImmutableSandboxedEnvironment:
            return False
        tree = env.parse(template)  # HF's environment understands {% generation %}.
        parents = {
            child: node
            for node in (tree, *tree.find_all(nodes.Node))
            for child in node.iter_child_nodes()
        }
        macros = {node.name for node in tree.find_all(nodes.Macro)}
        filters = set("default length tojson trim items string safe".split())
        tests = set("string iterable mapping none undefined true false defined".split())
        methods = set(
            "get items keys values startswith endswith strip lstrip rstrip split rsplit replace lower upper join".split()
        )
        # A method object can expose an address when printed or aliased. Permit
        # only direct calls of the explicitly nonmutating methods above.
        method_names = {
            name
            for cls in (str, dict, list, tuple, int, float, LoopContext)
            for name in dir(cls)
            if callable(getattr(cls, name))
        }
        structural = set(
            "Template Output TemplateData Const Name Getattr Getitem Slice If For Assign AssignBlock NSRef Macro Call CallBlock Keyword Filter Test Compare Operand And Or Not Neg Pos Add Sub Mul Div FloorDiv Mod Pow Concat List Tuple Dict Pair CondExpr Break Continue ExtensionAttribute".split()
        )
        compiler = getattr(
            chat,
            "_cached_compile_jinja_template",
            chat._compile_jinja_template.__wrapped__,
        )
        for node in (tree, *tree.find_all(nodes.Node)):
            parent = parents.get(node)
            direct_call = isinstance(parent, nodes.Call) and parent.node is node
            if type(node).__name__ not in structural:
                return False
            if isinstance(node, nodes.Name):
                if node.name in {"self", "super", "caller"}:
                    return False
                if node.name in env.globals or node.name in macros:
                    if not direct_call:
                        return False
            if isinstance(node, nodes.Getattr):
                if node.attr.startswith("_") or (
                    node.attr in method_names and not direct_call
                ):
                    return False
            if isinstance(node, nodes.ExtensionAttribute) and not direct_call:
                return False
            if isinstance(node, nodes.Getitem):
                # Dynamic lookup can fetch a callable or renderer/context object.
                parts = (
                    (node.arg.start, node.arg.stop, node.arg.step)
                    if isinstance(node.arg, nodes.Slice)
                    else (node.arg,)
                )
                parts = tuple(
                    part.node if isinstance(part, (nodes.Neg, nodes.Pos)) else part
                    for part in parts
                )
                if any(
                    part is not None
                    and not (isinstance(part, nodes.Const) and type(part.value) is int)
                    for part in parts
                ):
                    return False
            if isinstance(node, nodes.Call):
                target = node.node
                if node.dyn_args is not None or node.dyn_kwargs is not None:
                    return False
                if isinstance(target, nodes.Name):
                    if target.name in macros:
                        continue
                    helper = env.globals.get(target.name)
                    if target.name == "namespace" and helper is Namespace:
                        continue
                    if target.name == "raise_exception" and _code_contains(
                        compiler.__code__,
                        getattr(helper, "__code__", None),
                        "raise_exception",
                    ):
                        continue
                    return False
                if isinstance(target, nodes.Getattr) and target.attr in methods:
                    continue
                if isinstance(target, nodes.ExtensionAttribute) and isinstance(
                    parent, nodes.CallBlock
                ):
                    extension = env.extensions.get(target.identifier)
                    helper = getattr(
                        getattr(extension, target.name, None), "__func__", None
                    )
                    if target.name == "_generation_support" and _code_contains(
                        compiler.__code__,
                        getattr(helper, "__code__", None),
                        "_generation_support",
                    ):
                        continue
                return False
            if isinstance(node, (nodes.Filter, nodes.Test)):
                if isinstance(node, nodes.Test):
                    if (
                        node.name not in tests
                        or env.tests.get(node.name)
                        is not defaults.DEFAULT_TESTS[node.name]
                    ):
                        return False
                elif node.name not in filters:
                    return False
                elif node.name == "tojson":
                    if not _code_contains(
                        compiler.__code__,
                        getattr(env.filters.get(node.name), "__code__", None),
                        "tojson",
                    ):
                        return False
                elif (
                    env.filters.get(node.name)
                    is not defaults.DEFAULT_FILTERS[node.name]
                ):
                    return False
                if node.name == "items" and not (
                    isinstance(parent, nodes.For) and parent.iter is node
                ):
                    return False  # Otherwise a generator's repr can expose identity.
        return True
    except Exception:
        return False
