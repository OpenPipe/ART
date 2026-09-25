"""Exercise the recovery proof with public primitive source fixtures.

Extract the production functions to keep protocol parsing and model imports out
of these focused tests. The existing tokenizer suite covers those integrations.
"""

from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import dataclass
from enum import IntFlag
from functools import lru_cache
import math
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest


@lru_cache
def _source():
    path = Path(__file__).resolve().parents[3] / "src/art/trajectories/_tokenize.py"
    return ast.parse(path.read_text())


def _function(name):
    return next(node for node in _source().body if getattr(node, "name", None) == name)


def _compile(nodes, namespace):
    nodes = [
        ast.ImportFrom("__future__", [ast.alias("annotations")], 0),
        *deepcopy(nodes),
    ]
    exec(
        compile(ast.fix_missing_locations(ast.Module(nodes, [])), __file__, "exec"),
        namespace,
    )


def _mask_entry():
    helper = _function("_translate_token_mask")
    # Deepest helper code identifies this one refusal, independently of line
    # offsets introduced by formatting or other upstream changes.
    raises = [n for n in ast.walk(helper) if isinstance(n, ast.Raise)]
    assert len(raises) == 1
    assert isinstance(raises[0].exc, ast.Call)
    assert isinstance(raises[0].exc.func, ast.Name)
    assert raises[0].exc.func.id == "ValueError"
    assert ast.literal_eval(raises[0].exc.args[0]) == (
        "Cannot preserve assistant boundaries across exact prompt token replacement"
    )
    block = next(
        n
        for n in _function("_tokenize_chat_view").body
        if isinstance(n, ast.Try)
        and any(
            isinstance(x, ast.Name) and x.id == "_translate_token_mask"
            for x in ast.walk(n)
        )
    )
    entry = ast.parse("def entry():\n    pass").body[0]
    assert isinstance(entry, ast.FunctionDef)
    entry.body = [block, ast.parse("return assistant_mask").body[0]]
    env = {}
    _compile([helper, entry], env)
    calls = []
    caller = NS(trace="untouched")

    def retry(history, **kwargs):
        calls.append(kwargs)
        assert kwargs["tokenizer"] is None
        assert kwargs["_exact_source_boundary_retry"] is True
        kwargs["_trace"].trace = NS(source_keys=[], sources={})
        return "recovered"

    class Trace:
        def __init__(self):
            self.trace = None

    env.update(
        canonical_rendered=[1, 1],
        rendered=[2, 1],
        canonical_assistant_mask=[False, True],
        canonical_output_mask=[False, True],
        canonical_stop_mask=[False, False],
        canonical_length_stop_mask=[False, False],
        _exact_source_boundary_retry=False,
        original_tokenizer=None,
        _projection_matches=True,
        chat_template=None,
        chat_template_kwargs=None,
        history=NS(model="wandb-artifact:///public/project/model"),
        base_model="public",
        _history_has_length_stop=lambda _: True,
        resolved_tokenizer=None,
        _TraceBuilder=Trace,
        _tokenize_chat_view=retry,
        _require_exact_chat_source_edges=lambda *args: None,
        _trace=None,
    )
    return env, calls, caller


def test_own_mask_refusal_retries_after_source_line_changes():
    env, calls, _ = _mask_entry()
    assert env["entry"]() == "recovered"
    assert len(calls) == 1


def test_decoder_same_text_error_is_not_a_mask_refusal():
    env, calls, _ = _mask_entry()
    error = ValueError(
        "Cannot preserve assistant boundaries across exact prompt token replacement"
    )

    class Decoder:
        def decode(self, *args, **kwargs):
            raise error

    env.update(
        canonical_rendered=[1],
        rendered=[2],
        canonical_assistant_mask=[True],
        resolved_tokenizer=Decoder(),
    )
    with pytest.raises(ValueError) as caught:
        env["entry"]()
    assert caught.value is error
    assert not calls


@pytest.mark.parametrize(
    "change",
    [
        {"_exact_source_boundary_retry": True},
        {"original_tokenizer": object()},
        {"_projection_matches": False},
        {"chat_template_kwargs": {}},
    ],
)
def test_ineligible_mask_refusal_propagates(change):
    env, calls, _ = _mask_entry()
    env.update(change)
    with pytest.raises(ValueError, match="Cannot preserve assistant"):
        env["entry"]()
    assert not calls


def test_successful_translation_does_not_retry():
    env, calls, _ = _mask_entry()
    env["rendered"] = [1, 1]
    assert env["entry"]() == [False, True]
    assert not calls


class Flag(IntFlag):
    EXACT = 1
    SAMPLED = 2
    ASSISTANT = 4
    STOP = 8
    OUTPUT = 16


def _builder_fixture():
    env: dict[str, Any] = dict(
        __name__=__name__,
        dataclass=dataclass,
        math=math,
        TokenFlag=Flag,
        TokenizedHistory=NS,
        _history_matches_projection=lambda h: h.projection,
        _source_signature=lambda s: s.key if s else None,
        _source_is_sampled=lambda s: s.sampled,
        _sampled_source_key=lambda s: s.key,
        _chat_source_prompt_tokens=lambda s: s.prompt,
        _chat_source_full_tokens=lambda s: (s.output, s.lp),
        _source_output_tokens=lambda s, k: s.output,
        _source_stop_evidence=lambda s, k: (s.kind,),
        _sampled_stop_suffix=lambda ids, **kw: int(kw["source"].kind == "stop"),
    )
    names = (
        "_RenderedLengthStopBoundary",
        "_HistoryTokenizationTrace",
        "_TraceBuilder",
        "_retained_output_suffix",
        "_mark_sampled_stops",
        "_require_exact_chat_source_edges",
        "_tokenize_exact_projected_chat_history",
    )
    _compile([_function(n) for n in names], env)
    a = NS(
        key="A",
        prompt=[10],
        output=[20] * 4096,
        lp=[-0.1] * 4096,
        kind="length",
        sampled=True,
    )
    b = NS(
        key="B",
        prompt=[10, *a.output, 90, 91, 70, 71],
        output=[30],
        lp=[-0.2],
        kind="stop",
        sampled=True,
    )
    history = NS(
        model="public",
        projection=True,
        messages=[{"role": "assistant"}] * 2,
        message_sources=[a, b],
    )
    boundaries = {"A": env["_RenderedLengthStopBoundary"]((90, 91), (70,))}
    return env, history, boundaries


def _build(env, history, boundaries, native=True):
    trace = env["_TraceBuilder"]()
    value = env["_tokenize_exact_projected_chat_history"](
        history,
        tokenizer=None,
        projection_validated=True,
        length_stop_boundaries=boundaries,
        _native_prompt_context=native,
        _trace=trace,
    )
    return value, trace.trace


def test_native_context_requires_complete_tail_and_conditioning():
    env, history, boundaries = _builder_fixture()
    assert _build(env, history, boundaries, native=False)[0] is None
    value, trace = _build(env, history, boundaries)
    env["_require_exact_chat_source_edges"](history, value, trace, None)
    assert value.tokens == [*history.message_sources[1].prompt, 30]
    assert sum(bool(f & Flag.SAMPLED) for f in value.flags) == 4097
    assert not any(f & Flag.STOP for f in value.flags[1:4097])
    assert value.flags[4098] == Flag.EXACT | Flag.STOP
    assert all(math.isnan(value.logprobs[i]) for i in range(4097, 4101))
    assert boundaries["A"].following == (70,)


@pytest.mark.parametrize("case", ["tail", "prompt", "logprobs", "suffix_only"])
def test_native_context_refuses_incomplete_authority(case):
    env, history, boundaries = _builder_fixture()
    a, b = history.message_sources
    if case == "tail":
        b.prompt[4098] = 92
    elif case == "prompt":
        b.prompt[0] = 11
    elif case == "logprobs":
        a.lp.pop()
    else:
        b.prompt.pop(1)
    assert _build(env, history, boundaries)[0] is None


@pytest.mark.parametrize(
    "case", ["prefix", "ids", "logprobs", "stop", "missing_output"]
)
def test_full_source_guard_rejects_changed_training_edges(case):
    env, history, boundaries = _builder_fixture()
    value, trace = _build(env, history, boundaries)
    if case == "prefix":
        history.message_sources[0].prompt = [11]
    elif case == "ids":
        value.tokens[1] = 22
    elif case == "logprobs":
        value.logprobs[1] = -0.3
    elif case == "missing_output":
        history.message_sources[0].output = None
    else:
        value.flags[1] |= Flag.STOP
    with pytest.raises(ValueError, match="conditioned source proof"):
        env["_require_exact_chat_source_edges"](history, value, trace, None)
