from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import copy
import copyreg
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
import gc
import io
import math
import pickle
from typing import Any, cast
import weakref

from openai.types.chat import ChatCompletion, ChatCompletionUserMessageParam
from pydantic import BaseModel
import pytest

import art.trajectories as tr
from art.trajectories import _parallel as p


def unpack_test_payload(payload):
    """Inspect either raw pickle or the private framed result in assertions."""
    return pickle.loads(payload[1:] if payload.startswith(b"\0") else payload)


@pytest.fixture
def without_cyclic_gc():
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()
        gc.collect()


def fixture(multi=False):
    response = ChatCompletion.model_validate(
        {
            "id": "public",
            "object": "chat.completion",
            "created": 0,
            "model": "toy",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hello"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "x",
                                "bytes": [120],
                                "logprob": -0.5,
                                "top_logprobs": [],
                            }
                            for _ in range(64)
                        ]
                    },
                }
            ],
        }
    )
    exchange = tr.ChatCompletionsExchange(
        request=tr.ChatCompletionsRequest(
            model="toy",
            messages=[
                ChatCompletionUserMessageParam(role="user", content="Public fixture")
            ],
        ),
        response=response,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    parent = tr.Trajectory(
        exchanges=tr.TrajectoryExchanges(chat_completions=[exchange]),
        reward=2.0,
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    worker = unpack_test_payload(pickle.dumps(parent))
    e = worker.exchanges.chat_completions[0]
    history = tr.ChatCompletionsHistory(
        model="toy",
        messages=[e.request["messages"][0]],
        message_sources=[tr.ChatCompletionsMessageSource(exchange=e, request_index=0)],
    )
    values: dict[str, Any] = dict(
        history=history,
        model="toy",
        tokens=[1, 2, 3],
        logprobs=[math.nan, -0.5, -0.25],
        flags=[
            tr.TokenFlag.EXACT,
            tr.TokenFlag.EXACT | tr.TokenFlag.SAMPLED,
            tr.TokenFlag.EXACT
            | tr.TokenFlag.SAMPLED
            | tr.TokenFlag.STOP
            | tr.TokenFlag.ASSISTANT,
        ],
    )
    if multi:
        h = tr.TokenizedHistory(**values)
        result = tr.TokenizedMultiHistoryTrajectory(trajectory=worker, histories=[h, h])
    else:
        result = tr.TokenizedTrajectory(trajectory=worker, **values)
    return parent, result


def history(result):
    return (
        result.histories[0].history
        if isinstance(result, tr.TokenizedMultiHistoryTrajectory)
        else result.history
    )


_fallback_callbacks: list[str] = []


class FallbackTag:
    def __init__(self, behavior):
        self.behavior = behavior

    def __eq__(self, other):
        _fallback_callbacks.append("tag equality")
        if self.behavior == "runtime":
            raise RuntimeError("ordinary tuple equality callback")
        if self.behavior == "interrupt":
            raise KeyboardInterrupt("ordinary tuple equality callback")
        return False


class FallbackBytes(bytes):
    def __eq__(self, other):
        _fallback_callbacks.append("bytes equality")
        return bytes.__eq__(self, other)


class FallbackTuple(tuple):
    def __len__(self):
        _fallback_callbacks.append("tuple length")
        return super().__len__()


def restore_fallback(value):
    return value


class FallbackResult(tr.TokenizedTrajectory):
    def __reduce_ex__(self, protocol):
        return restore_fallback, (self.__dict__["fallback_value"],)


@pytest.mark.parametrize(
    "kind",
    [
        "false",
        "runtime",
        "interrupt",
        "matching_invalid",
        "matching_valid",
        "bytes_subclass",
        "tuple_subclass",
        "frame_bytes",
    ],
)
def test_ordinary_fallback_result_cannot_be_mistaken_for_optimized_frame(
    monkeypatch, kind
):
    outcomes = []
    for optimized in (False, True):
        parent, result = fixture()
        if kind in {"false", "runtime", "interrupt"}:
            value = (FallbackTag(kind), 0, b"")
        elif kind == "matching_invalid":
            value = (b"art-process-sources-v1", 0, b"")
        elif kind == "matching_valid":
            value = (
                b"art-process-sources-v1",
                tuple(type(x) for x in p._process_sources(parent)),
                pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL),
            )
        elif kind == "bytes_subclass":
            value = (FallbackBytes(b"art-process-sources-v1"), 0, b"")
        elif kind == "tuple_subclass":
            value = FallbackTuple((0, 0, b""))
        else:
            value = b"\0" + pickle.dumps((b"art-process-sources-v1", 0, b""))
        result = FallbackResult.model_construct(**result.__dict__)
        result.__dict__["fallback_value"] = value
        _fallback_callbacks.clear()
        payload = p._serialize_process_result(result)
        assert payload.startswith(pickle.PROTO)
        with monkeypatch.context() as patch:
            if not optimized:
                # The original receiver differs only at this loading boundary.
                patch.setattr(
                    p, "_load_process_result", lambda data, parent: pickle.loads(data)
                )
            try:
                restored = p._deserialize_process_result(payload, parent)
                outcome = ("success", type(restored).__name__)
            except BaseException as error:
                outcome = (type(error).__name__, str(error))
        outcomes.append((outcome, list(_fallback_callbacks)))
    assert outcomes[0][0][0] == "_ProcessTransferError"
    assert "unexpected process result" in outcomes[0][0][1]
    assert outcomes[0][1] == []
    assert outcomes[1] == outcomes[0]


def test_optimized_frame_is_distinct_from_ordinary_pickle():
    parent, result = fixture()
    payload = p._serialize_process_result(result)
    assert payload.startswith(b"\0")
    assert unpack_test_payload(payload)[0] == b"art-process-sources-v1"
    with pytest.raises(pickle.UnpicklingError):
        pickle.loads(payload)
    restored = p._deserialize_process_result(payload, parent)
    assert isinstance(restored, tr.TokenizedTrajectory)
    assert restored.trajectory is parent
    assert restored.tokens == result.tokens


_result_equality_armed = False
_result_equality_raises = False
_result_equality_calls: list[str] = []


class ResultEquality(type(tr.TokenizedTrajectory)):
    def __eq__(cls, other):
        if _result_equality_armed:
            _result_equality_calls.append("result equality")
            if _result_equality_raises:
                raise RuntimeError("custom result class equality")
        return cls is other

    __hash__ = type.__hash__


class CustomResult(tr.TokenizedTrajectory, metaclass=ResultEquality):
    pass


class CustomMultiResult(tr.TokenizedMultiHistoryTrajectory, metaclass=ResultEquality):
    pass


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_result_type_admission_does_not_call_custom_equality(multi, raises):
    global _result_equality_armed, _result_equality_raises
    _result_equality_raises = raises
    outcomes = []
    for optimized in (False, True):
        parent, result = fixture(multi)
        result = (CustomMultiResult if multi else CustomResult).model_construct(
            **result.__dict__
        )
        _result_equality_calls.clear()
        _result_equality_armed = True
        phase = "serialize"
        try:
            payload = (
                p._serialize_process_result(result)
                if optimized
                else pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            )
            phase = "restore"
            restored = p._deserialize_process_result(payload, parent)
            outcome = (
                "success",
                restored.model_dump_json(),
                restored.trajectory is parent,
            )
        except Exception as error:
            outcome = (phase, type(error).__name__, str(error))
        finally:
            _result_equality_armed = False
        outcomes.append((outcome, list(_result_equality_calls)))
    assert outcomes[0][0][0] == "success"
    assert outcomes[0][1] == []
    assert outcomes[1] == outcomes[0]


@pytest.mark.parametrize("owner", ["result", "history", "trajectory"])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_instance_method_shadow_preserves_callbacks_and_failure_phase(
    monkeypatch, owner, alias, raises
):
    def count(source):
        object.__setattr__(source, "prompt_index", source.prompt_index + 1)
        if raises:
            raise RuntimeError("instance interning callback")

    outcomes = []
    eligibility = []
    for optimized in (False, True):
        parent, result = fixture()
        monkeypatch.setattr(tr.CompletionsSource, "__call__", count, raising=False)
        probe = tr.CompletionsSource.model_construct(exchange=None, prompt_index=0)
        target = {
            "result": result,
            "history": history(result),
            "trajectory": result.trajectory,
        }[owner]
        target.__dict__["_mark_pickle_strings_interned"] = probe
        history(result).chat_template_kwargs = {"probe": probe}
        if alias:
            history(result).chat_template_kwargs["alias"] = result.trajectory
        eligibility.append(p._process_plain_models(result))
        assert probe.prompt_index == 0
        phase = "serialize"
        try:
            payload = (
                p._serialize_process_result(result)
                if optimized
                else pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            )
            phase = "restore"
            restored = p._deserialize_process_result(payload, parent)
            outcome = (
                "success",
                history(restored).chat_template_kwargs["probe"].prompt_index,
                restored.trajectory is parent,
                type(unpack_test_payload(payload)) is tr.TokenizedTrajectory,
            )
        except Exception as error:
            outcome = (phase, type(error).__name__, str(error))
        outcomes.append((outcome, probe.prompt_index))
    assert outcomes[0][0][0] == ("serialize" if raises else "success")
    assert outcomes[1] == outcomes[0]
    assert eligibility == [None, None]


@pytest.mark.parametrize(
    "name",
    [
        "model_dump",
        "model_copy",
        "__private_attributes__",
        "__getnewargs__",
        "__getnewargs_ex__",
    ],
)
def test_instance_state_shadow_of_model_api_uses_ordinary_pickle(name):
    # Some shadowed APIs are consulted by pickle itself: preserve its failure too.
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        _, result = fixture()
        result.__dict__[name] = None
        assert p._process_plain_models(result) is None
        try:
            payload = encode(result)
            outcomes.append(("success", type(unpack_test_payload(payload))))
        except Exception as error:
            outcomes.append((type(error), str(error)))
    assert outcomes[1] == outcomes[0]


@pytest.mark.parametrize("owner", ["result", "history", "trajectory"])
def test_unshadowed_instance_data_keeps_source_elision(owner):
    parent, result = fixture()
    target = {
        "result": result,
        "history": history(result),
        "trajectory": result.trajectory,
    }[owner]
    target.__dict__["public_fixture_data"] = {"value": [1, 2, 3]}
    assert p._process_plain_models(result) is not None
    payload = p._serialize_process_result(result)
    assert unpack_test_payload(payload)[0] == b"art-process-sources-v1"
    restored = p._deserialize_process_result(payload, parent)
    assert restored.trajectory is parent
    assert history(restored).messages == history(result).messages


_registry_target = type(None)
_registry_behavior = "raise"
_registry_calls: list[str] = []


class RegistryCollision(type):
    def __hash__(cls):
        return hash(_registry_target)

    def __eq__(cls, other):
        _registry_calls.append(other.__name__)
        if _registry_behavior == "raise":
            raise RuntimeError("unrelated reducer registry equality")
        return _registry_behavior == "true"


class RegisteredCollision(metaclass=RegistryCollision):
    pass


def _restore_registered_trajectory():
    return tr.Trajectory(
        metadata={"registered_reducer": True},
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _registered_reduce(value):
    return _restore_registered_trajectory, ()


@pytest.mark.parametrize("target", [type(None), tr.Trajectory])
@pytest.mark.parametrize("behavior", ["raise", "false", "true"])
@pytest.mark.parametrize("boundary", ["input", "result"])
def test_reducer_registry_callbacks_keep_ordinary_order(target, behavior, boundary):
    global _registry_target, _registry_behavior
    _registry_target, _registry_behavior = target, behavior
    outcomes = []
    for optimized in (False, True):
        # Fixture creation itself pickles; install the registry only afterwards.
        parent, result = fixture()
        copyreg.pickle(RegisteredCollision, _registered_reduce)
        try:
            _registry_calls.clear()
            phase = "serialize"
            try:
                if boundary == "input":
                    options = p._ProcessOptions(False, False, None, None, None, None)
                    if optimized:
                        payload = p._process_payloads([parent], options)[0]
                    else:
                        with p._without_pickle_string_interning():
                            payload = pickle.dumps(
                                (parent, options), protocol=pickle.HIGHEST_PROTOCOL
                            )
                    phase = "restore"
                    actual, actual_options = unpack_test_payload(payload)
                    outcome = (
                        "success",
                        actual.model_dump_json(),
                        actual_options.source_refs_allowed,
                    )
                else:
                    payload = (
                        p._serialize_process_result(result)
                        if optimized
                        else pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                    )
                    phase = "restore"
                    actual = p._deserialize_process_result(payload, parent)
                    outcome = ("success", actual.model_dump_json())
            except Exception as error:
                # The process-input wrapper adds transfer context to ordinary
                # pickle errors. Compare its preserved cause and failure phase.
                message = str(error)
                if boundary == "input" and isinstance(error, p._ProcessTransferError):
                    message = message.removeprefix(
                        "could not serialize process input: RuntimeError: "
                    )
                outcome = (phase, message)
            outcomes.append((outcome, list(_registry_calls)))
        finally:
            copyreg.dispatch_table.pop(RegisteredCollision)
    assert outcomes[0] == outcomes[1]


def test_registry_eligibility_does_not_observe_custom_key_equality():
    global _registry_target, _registry_behavior
    _registry_target, _registry_behavior = type(None), "raise"
    copyreg.pickle(RegisteredCollision, _registered_reduce)
    try:
        _registry_calls.clear()
        assert p._process_plain_models(None, check_schema=True) is None
        assert _registry_calls == []
    finally:
        copyreg.dispatch_table.pop(RegisteredCollision)


def test_registry_mapping_subclass_uses_ordinary_pickle(monkeypatch):
    calls = []

    class Registry(dict):
        def __contains__(self, key):
            calls.append(key.__name__)
            raise RuntimeError("registry membership callback")

    monkeypatch.setattr(copyreg, "dispatch_table", Registry(copyreg.dispatch_table))
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result = fixture()
        calls.clear()
        payload = encode(result)
        restored = p._deserialize_process_result(payload, parent)
        outcomes.append((restored.model_dump_json(), list(calls)))
    assert outcomes[0] == outcomes[1]
    assert type(unpack_test_payload(payload)) is tr.TokenizedTrajectory


class OrdinaryRegistryKey:
    pass


class OrdinaryRegistryModel(BaseModel):
    value: int


class OrdinaryRegistryEnum(Enum):
    VALUE = 1


@pytest.mark.parametrize(
    "key", [OrdinaryRegistryKey, OrdinaryRegistryModel, OrdinaryRegistryEnum]
)
def test_unrelated_passive_registry_key_keeps_source_elision(key):
    copyreg.pickle(key, _registered_reduce)
    try:
        parent, result = fixture()
        payload = p._serialize_process_result(result)
        assert type(unpack_test_payload(payload)) is tuple
        restored = p._deserialize_process_result(payload, parent)
        assert (
            history(restored).message_sources[0].exchange
            is parent.exchanges.chat_completions[0]
        )
    finally:
        copyreg.dispatch_table.pop(key)


class ReconstructedInventory(list):
    def __init__(self, values):
        super().__init__(values)
        self.iterations = 0

    def __reduce__(self):
        return list, (self[:],)

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise RuntimeError("second parent inventory iteration")
        return super().__iter__()


@pytest.mark.parametrize(
    "name", [None, "chat_completions", "completions", "responses", "messages"]
)
def test_actual_parent_inventory_is_checked_before_spawn(name):
    parents = [fixture()[0] for _ in range(2)]
    inventories = []
    if name is not None:
        for parent in parents:
            values = ReconstructedInventory(getattr(parent.exchanges, name))
            setattr(parent.exchanges, name, values)
            inventories.append(values)
    options = p._ProcessOptions(False, False, None, None, None, None)
    payload = p._process_payloads([parents[0]], options)[0]
    assert all(values.iterations == 0 for values in inventories)
    worker, worker_options = unpack_test_payload(payload)
    if name is not None:
        assert type(getattr(worker.exchanges, name)) is list
    with ProcessPoolExecutor(max_workers=1, mp_context=p._process_context()) as pool:
        encoded, ordinary, worker_plain = pool.submit(
            _spawn_result_pair, payload
        ).result(timeout=30)
    assert worker_plain
    expected = p._deserialize_process_result(ordinary, parents[1])
    restored = p._deserialize_process_result(encoded, parents[0])
    assert restored.model_dump_json() == expected.model_dump_json()
    assert worker_options.source_refs_allowed is (name is None)
    assert type(unpack_test_payload(encoded)) is (
        tuple if name is None else tr.TokenizedTrajectory
    )
    assert all(values.iterations == 1 for values in inventories)


@pytest.mark.parametrize("slot", [None, False, True])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("storage", ["extra", "private"])
def test_marker_attribute_fallback_preserves_fresh_graph_effects(
    monkeypatch, slot, fallback, storage
):
    marker_name = "_art_pickle_strings_interned"
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result = fixture(True)
        h = history(result)
        key = "".join(["public-key-", "z" * 100])
        seed = key.encode().decode()
        assert seed == key and seed is not key
        h.chat_template_kwargs = {key: 1, "second-key": 2}
        result.trajectory.metadata = {"seed": seed, "retained": h}
        if slot is not None:
            object.__setattr__(result, marker_name, slot)
        with monkeypatch.context() as patch:
            if storage == "private":
                patch.setattr(
                    type(result), "__private_attributes__", {marker_name: object()}
                )
                object.__setattr__(
                    result, "__pydantic_private__", {marker_name: fallback}
                )
            else:
                object.__setattr__(
                    result, "__pydantic_extra__", {marker_name: fallback}
                )
            payload = encode(result)
            restored = p._deserialize_process_result(payload, parent)
        outcomes.append(
            (
                list(h.chat_template_kwargs),
                any(x is seed for x in h.chat_template_kwargs),
                list(history(restored).chat_template_kwargs),
            )
        )
    assert outcomes[0] == outcomes[1]
    if slot is None:
        assert type(unpack_test_payload(payload)) is tr.TokenizedMultiHistoryTrajectory


@pytest.mark.parametrize("slot", [None, False, True])
@pytest.mark.parametrize("storage", ["extra", "private", "shadowed_private"])
@pytest.mark.parametrize("raises", [False, True])
def test_fallback_marker_callbacks_are_not_used_for_eligibility(
    monkeypatch, slot, storage, raises
):
    marker_name = "_art_pickle_strings_interned"
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result = fixture(True)
        h = history(result)
        h.chat_template_kwargs = {"first": 1, "second": 2}
        marker = EffectfulInterningMarker(h.chat_template_kwargs, raises)
        if slot is not None:
            object.__setattr__(result, marker_name, slot)
        with monkeypatch.context() as patch:
            if storage == "extra":
                object.__setattr__(result, "__pydantic_extra__", {marker_name: marker})
            else:
                if storage == "private":
                    patch.setattr(
                        type(result), "__private_attributes__", {marker_name: False}
                    )
                else:
                    result.__dict__["__private_attributes__"] = {marker_name: False}
                object.__setattr__(
                    result, "__pydantic_private__", {marker_name: marker}
                )
            p._process_plain_models(result)
            assert marker.calls == 0
            phase = "serialize"
            try:
                payload = encode(result)
                phase = "restore"
                restored = p._deserialize_process_result(payload, parent)
                outcome = ("success", list(history(restored).chat_template_kwargs))
            except Exception as error:
                outcome = (phase, type(error), str(error))
        outcomes.append((outcome, marker.calls, list(h.chat_template_kwargs)))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][1] == (1 if slot is None else 0)


def test_instance_private_marker_table_uses_ordinary_attribute_resolution():
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result = fixture(True)
        h = history(result)
        key = "".join(["public-key-", "z" * 100])
        seed = key.encode().decode()
        h.chat_template_kwargs = {key: 1, "second-key": 2}
        result.trajectory.metadata = {"seed": seed, "retained": h}
        marker_name = "_art_pickle_strings_interned"
        result.__dict__["__private_attributes__"] = {marker_name: False}
        object.__setattr__(result, "__pydantic_private__", {marker_name: True})
        payload = encode(result)
        restored = p._deserialize_process_result(payload, parent)
        outcomes.append(
            (list(h.chat_template_kwargs), list(history(restored).chat_template_kwargs))
        )
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_undeclared_rebound_field_descriptor_matches_ordinary_pickle(
    monkeypatch, inherited, raises
):
    def set_source(owner, value):
        owner.chat_template_kwargs["setter_calls"] += 1
        if raises:
            raise RuntimeError("public undeclared source setter")
        owner.__dict__["instructions_source"] = value

    owner_class = (
        tr.ChatCompletionsHistory.__base__ if inherited else tr.ChatCompletionsHistory
    )
    monkeypatch.setattr(
        owner_class,
        "instructions_source",
        property(lambda owner: owner.__dict__.get("instructions_source"), set_source),
        raising=False,
    )
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result = fixture()
        result.history = history(result).model_copy(
            update={
                "instructions_source": result.trajectory.exchanges.chat_completions[0],
                "chat_template_kwargs": {"setter_calls": 0},
            }
        )
        payload = encode(result)
        try:
            restored = p._deserialize_process_result(payload, parent)
            outcome = (
                "success",
                history(restored).chat_template_kwargs["setter_calls"],
            )
        except Exception as error:
            outcome = (type(error), str(error))
        outcomes.append(outcome)
        assert history(result).chat_template_kwargs["setter_calls"] == 0
    assert outcomes[0] == outcomes[1]
    if not raises:
        assert outcomes[0] == ("success", 1)


@pytest.mark.parametrize("inherited", [False, True])
@pytest.mark.parametrize("raises", [False, True])
def test_descriptor_type_resolution_keeps_ordinary_callbacks(
    monkeypatch, inherited, raises
):
    calls = []

    class DescriptorType(type):
        def __getattribute__(cls, name):
            if name == "__mro__":
                calls.append("descriptor type lookup")
                if raises:
                    raise RuntimeError("public descriptor type lookup")
            return super().__getattribute__(name)

    class SourceDescriptor(metaclass=DescriptorType):
        def __get__(self, owner, cls=None):
            return self if owner is None else owner.__dict__["instructions_source"]

        def __set__(self, owner, value):
            calls.append("source setter")
            owner.__dict__["instructions_source"] = value

    inputs = [fixture(), fixture()]
    owner_class = (
        tr.ChatCompletionsHistory.__base__ if inherited else tr.ChatCompletionsHistory
    )
    monkeypatch.setattr(
        owner_class, "instructions_source", SourceDescriptor(), raising=False
    )
    outcomes = []
    for (parent, result), encode in zip(
        inputs, (pickle.dumps, p._serialize_process_result)
    ):
        result.history = history(result).model_copy(
            update={
                "instructions_source": result.trajectory.exchanges.chat_completions[0]
            }
        )
        calls.clear()
        phase = "serialize"
        try:
            payload = encode(result)
            phase = "restore"
            restored = p._deserialize_process_result(payload, parent)
            outcome = (
                "success",
                history(restored).__dict__["instructions_source"]
                is parent.exchanges.chat_completions[0],
            )
        except Exception as error:
            outcome = (phase, type(error), str(error))
        outcomes.append((outcome, list(calls)))
    assert outcomes[0] == (("success", True), ["source setter"])
    assert outcomes[1] == outcomes[0]


@pytest.mark.parametrize("inherited", [False, True])
def test_parent_only_undeclared_descriptor_survives_fresh_spawn(monkeypatch, inherited):
    calls = []

    def set_source(owner, value):
        calls.append(owner)
        owner.chat_template_kwargs["setter_calls"] = 1
        owner.__dict__["instructions_source"] = value

    owner_class = (
        tr.ChatCompletionsHistory.__base__ if inherited else tr.ChatCompletionsHistory
    )
    monkeypatch.setattr(
        owner_class,
        "instructions_source",
        property(lambda owner: owner.__dict__.get("instructions_source"), set_source),
        raising=False,
    )
    parent, _ = fixture()
    payload = p._process_payloads(
        [parent], p._ProcessOptions(False, False, None, None, None, None)
    )[0]
    assert calls == []
    with ProcessPoolExecutor(max_workers=1, mp_context=p._process_context()) as pool:
        encoded, ordinary, worker_plain = pool.submit(
            _spawn_result_pair, payload, True
        ).result(timeout=30)
    assert worker_plain
    expected = p._deserialize_process_result(ordinary, parent)
    restored = p._deserialize_process_result(encoded, parent)
    assert (
        history(restored).chat_template_kwargs
        == history(expected).chat_template_kwargs
        == {"setter_calls": 1}
    )
    assert len(calls) == 2
    assert type(unpack_test_payload(encoded)) is tr.TokenizedTrajectory


@pytest.mark.parametrize("mismatch", [False, True])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("alias_first", [False, True])
def test_noninventory_exchange_equality_keeps_original_alias_order(
    mismatch, nested, alias_first
):
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result = fixture()
        pa = parent.exchanges.chat_completions[0]
        wa = result.trajectory.exchanges.chat_completions[0]
        pa.request["temperature"] = 0.1
        wa.request["temperature"] = 0.2 if mismatch else 0.1
        ps = tr.ChatCompletionsMessageSource(exchange=pa, request_index=0)
        ws = tr.ChatCompletionsMessageSource(exchange=wa, request_index=0)
        pb = pa.model_copy(deep=True, update={"start_time": None, "end_time": None})
        wb = wa.model_copy(deep=True, update={"start_time": None, "end_time": None})
        for exchange, source in ((pb, ps), (wb, ws)):
            exchange.request["chat_template_kwargs"] = {"source": source}
            exchange.request["temperature"] = 0.3
        parent.exchanges.chat_completions.append(pb)
        result.trajectory.exchanges.chat_completions.append(wb)
        source = tr.ChatCompletionsMessageSource(
            exchange=wb.model_copy(), request_index=0
        )
        h = history(result)
        h.messages = [[source]] if nested else [source]
        h.message_sources = [ws]
        if alias_first:
            h.__dict__["messages"] = h.__dict__.pop("messages")
        payload = encode(result)
        try:
            restored = p._deserialize_process_result(payload, parent)
            restored_source = history(restored).messages[0]
            if nested:
                restored_source = restored_source[0]
            outcome = ("success", restored_source.exchange is pb)
        except Exception as error:
            outcome = (type(error), str(error))
        outcomes.append(outcome)
    assert outcomes[0] == outcomes[1]
    assert type(unpack_test_payload(payload)) is tr.TokenizedTrajectory


def test_passive_undeclared_source_and_inventory_alias_keep_fast_path():
    parent, result = fixture(True)
    h = history(result)
    source = h.message_sources[0]
    h.__dict__["instructions_source"] = source.exchange
    h.chat_template_kwargs = {"source_alias": source}
    payload = p._serialize_process_result(result)
    assert type(unpack_test_payload(payload)) is tuple
    restored = p._deserialize_process_result(payload, parent)
    restored_h = history(restored)
    assert (
        restored_h.chat_template_kwargs["source_alias"] is restored_h.message_sources[0]
    )
    assert restored_h.instructions_source is parent.exchanges.chat_completions[0]


def roundtrip(parent, result):
    old = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    new = p._serialize_process_result(result)
    return (
        p._deserialize_process_result(old, parent),
        p._deserialize_process_result(new, parent),
        old,
        new,
    )


@pytest.mark.parametrize("multi", [False, True])
def test_values_types_identity_and_input_nonmutation(multi):
    parent, result = fixture(multi)
    before = parent.model_dump_json(), result.model_dump_json()
    expected, restored, old, new = roundtrip(parent, result)
    assert type(restored) is type(expected)
    assert restored.model_dump_json() == expected.model_dump_json()
    assert restored.trajectory is parent
    assert (
        history(restored).message_sources[0].exchange
        is parent.exchanges.chat_completions[0]
    )
    assert (
        history(restored).messages[0]
        is not parent.exchanges.chat_completions[0].request["messages"][0]
    )
    if multi:
        assert restored.histories[0] is restored.histories[1]
    assert before == (parent.model_dump_json(), result.model_dump_json())
    assert len(new) < len(old) / 2


@pytest.mark.parametrize("kind", ["exchange", "trajectory", "source", "shared_source"])
def test_kwargs_source_aliases_preserve_existing_rebinding(kind):
    parent, result = fixture()
    h = history(result)
    source = h.message_sources[0]
    values = {
        "exchange": result.trajectory.exchanges.chat_completions[0],
        "trajectory": result.trajectory,
        "source": source.model_copy(),
        "shared_source": source,
    }
    h.chat_template_kwargs = {"value": values[kind]}
    expected, restored, _, _ = roundtrip(parent, result)
    for output in [expected, restored]:
        h = history(output)
        value = h.chat_template_kwargs["value"]
        if kind == "exchange":
            assert value is not parent.exchanges.chat_completions[0]
        elif kind == "trajectory":
            assert value is not parent
        elif kind == "source":
            assert value.exchange is not parent.exchanges.chat_completions[0]
        else:
            assert value is h.message_sources[0]
            assert value.exchange is parent.exchanges.chat_completions[0]
    assert expected.model_dump_json() == restored.model_dump_json()


@pytest.mark.parametrize("owner", ["result", "source"])
def test_aliased_replaced_state_uses_legacy_pickle(owner):
    parent, result = fixture()
    h = history(result)
    value = result if owner == "result" else h.message_sources[0]
    h.chat_template_kwargs = {"state": value.__dict__}
    payload = p._serialize_process_result(result)
    assert isinstance(unpack_test_payload(payload), tr.TokenizedTrajectory)
    restored = p._deserialize_process_result(payload, parent)
    target = restored if owner == "result" else history(restored).message_sources[0]
    assert history(restored).chat_template_kwargs["state"] is target.__dict__


@pytest.mark.parametrize(
    "role", ["message", "input", "instructions", "system", "token", "string"]
)
@pytest.mark.parametrize("target", ["trajectory", "response"])
def test_unvalidated_source_roles_preserve_detached_objects(role, target):
    parent, result = fixture()
    parent_value, worker_value = parent, result.trajectory
    if target == "response":
        parent_value = parent.exchanges.chat_completions[0].response
        worker_value = result.trajectory.exchanges.chat_completions[0].response
        # Inventory membership alone must not confer an exchange's receiver role.
        cast(Any, parent.exchanges.chat_completions).append(parent_value)
        cast(Any, result.trajectory.exchanges.chat_completions).append(worker_value)
    if role == "message":
        h = result.history.model_copy(
            update={
                "message_sources": [
                    history(result)
                    .message_sources[0]
                    .model_copy(update={"exchange": worker_value})
                ]
            }
        )
    elif role in ("input", "instructions"):
        h = tr.ResponsesHistory.model_construct(
            model="toy",
            input=[],
            input_sources=[
                tr.ResponsesItemSource.model_construct(exchange=worker_value)
            ]
            if role == "input"
            else [],
            instructions_source=worker_value if role == "instructions" else None,
        )
    elif role == "system":
        h = tr.AnthropicMessagesHistory.model_construct(
            model="toy",
            messages=[],
            message_sources=[],
            system_source=worker_value,
        )
    else:
        cls = (
            tr.CompletionsTokenHistory
            if role == "token"
            else tr.CompletionsStringHistory
        )
        span = (
            tr.CompletionsTokenSourceSpan
            if role == "token"
            else tr.CompletionsStringSourceSpan
        )
        h = cls.model_construct(
            model="toy",
            prompt=[1] if role == "token" else "x",
            sampled_spans=[],
            prompt_sources=[
                span.model_construct(
                    start=0,
                    end=1,
                    source=tr.CompletionsSource.model_construct(
                        exchange=worker_value, prompt_index=0
                    ),
                )
            ],
        )
    result = result.model_copy(update={"history": h})
    expected, restored, _, payload = roundtrip(parent, result)
    assert type(unpack_test_payload(payload)) is tr.TokenizedTrajectory
    for output in (expected, restored):
        h = history(output)
        if role == "message":
            value = h.message_sources[0].exchange
        elif role == "input":
            value = h.input_sources[0].exchange
        elif role == "instructions":
            value = h.instructions_source
        elif role == "system":
            value = h.system_source
        else:
            value = h.prompt_sources[0].source.exchange
        assert value is not parent_value
        value.__dict__["public_probe"] = True
    assert "public_probe" not in parent_value.__dict__
    assert "public_probe" not in worker_value.__dict__


@pytest.mark.parametrize(
    "path", ["models", "dict", "list", "state_cycle", "extra", "container_cycle"]
)
def test_shared_model_state_preserves_all_alias_paths(path):
    parent, result = fixture()
    h = history(result)
    source = h.message_sources[0]
    second = source.model_copy()
    object.__setattr__(second, "__dict__", source.__dict__)
    h.message_sources.append(second)
    if path == "dict":
        h.chat_template_kwargs = {"state": source.__dict__}
    elif path == "list":
        h.chat_template_kwargs = {"state": [source.__dict__]}
    elif path == "state_cycle":
        source.__dict__["cycle"] = source.__dict__
    elif path == "extra":
        object.__setattr__(h, "__pydantic_extra__", source.__dict__)
    elif path == "container_cycle":
        value = {"source": source}
        value["cycle"] = value
        h.chat_template_kwargs = {"value": value}
    expected, restored, _, payload = roundtrip(parent, result)
    assert type(unpack_test_payload(payload)) is tr.TokenizedTrajectory
    for output in (expected, restored):
        actual = history(output)
        first, second = actual.message_sources
        assert first is not second
        assert first.__dict__ is second.__dict__
        assert first.exchange is parent.exchanges.chat_completions[0]
        if path == "dict":
            assert actual.chat_template_kwargs["state"] is first.__dict__
        elif path == "list":
            assert actual.chat_template_kwargs["state"][0] is first.__dict__
        elif path == "state_cycle":
            assert first.__dict__["cycle"] is first.__dict__
        elif path == "extra":
            assert actual.__pydantic_extra__ is first.__dict__
        elif path == "container_cycle":
            value = actual.chat_template_kwargs["value"]
            assert value["cycle"] is value
            assert value["source"] is first
        object.__setattr__(first, "request_index", 5)
        assert second.request_index == 5
    assert source.request_index == 0


def _spawn_result_pair(payload, undeclared=False, metadata_flag=None):
    captured = []

    def tokenize(trajectory, **kwargs):
        _, result = fixture()
        result.trajectory = trajectory
        if metadata_flag is not None:
            trajectory.metadata["flag"] = tr.TokenFlag(metadata_flag)
        h = history(result)
        exchange = trajectory.exchanges.chat_completions[0]
        h.messages = [exchange.request["messages"][0]]
        object.__setattr__(h.message_sources[0], "exchange", exchange)
        if undeclared:
            h.__dict__["instructions_source"] = exchange
            h.chat_template_kwargs = {}
        captured.append(result)
        return result

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(tr.Trajectory, "tokenize", tokenize)
        encoded = p._tokenize_process_payload(payload)
        return (
            encoded,
            pickle.dumps(captured[0], protocol=pickle.HIGHEST_PROTOCOL),
            p._process_plain_models(captured[0]) is not None,
        )


@pytest.mark.parametrize("shadow", [False, True])
def test_parent_instance_resolution_contract_survives_spawn(shadow):
    parent, _ = fixture()
    parent.__dict__["model_copy" if shadow else "public_fixture_data"] = 7
    payload = p._process_payloads(
        [parent], p._ProcessOptions(False, False, None, None, None, None)
    )[0]
    assert unpack_test_payload(payload)[1].source_refs_allowed is not shadow
    with ProcessPoolExecutor(max_workers=1, mp_context=p._process_context()) as pool:
        encoded, ordinary, worker_plain = pool.submit(
            _spawn_result_pair, payload
        ).result(timeout=30)
    assert worker_plain is not shadow
    expected = p._deserialize_process_result(ordinary, parent)
    restored = p._deserialize_process_result(encoded, parent)
    assert type(unpack_test_payload(encoded)) is (
        tr.TokenizedTrajectory if shadow else tuple
    )
    assert restored.model_dump_json() == expected.model_dump_json()
    assert restored.trajectory is parent


@pytest.mark.parametrize("hook", ["new", "missing", "numeric_repr"])
@pytest.mark.parametrize("raises", [False, True])
def test_parent_flag_construction_keeps_ordinary_callbacks(monkeypatch, hook, raises):
    parent, _ = fixture()
    sentinel = 1 << (23 if raises else 24)
    flag_value = sentinel | 1 if hook == "numeric_repr" else sentinel
    if hook == "new":
        parent.metadata["flag"] = tr.TokenFlag(flag_value)
    else:
        # The worker creates a value not yet reconstructed in this parent.
        # Prior parametrizations may have cached it; restore that cache on exit.
        monkeypatch.delitem(tr.TokenFlag._value2member_map_, flag_value, raising=False)
    calls = []
    ordinary_new = tr.TokenFlag.__new__
    ordinary_missing = tr.TokenFlag._missing_

    def construct(cls, value):
        calls.append(value)
        if value == sentinel and raises:
            raise RuntimeError("public flag construction")
        return ordinary_new(cls, value) if hook == "new" else ordinary_missing(value)

    def numeric_repr(value):
        calls.append(value)
        if raises:
            raise RuntimeError("public flag construction")
        return repr(value)

    monkeypatch.setattr(
        tr.TokenFlag,
        {"new": "__new__", "missing": "_missing_", "numeric_repr": "_numeric_repr_"}[
            hook
        ],
        staticmethod(numeric_repr)
        if hook == "numeric_repr"
        else staticmethod(construct)
        if hook == "new"
        else classmethod(construct),
    )
    payload = p._process_payloads(
        [parent], p._ProcessOptions(False, False, None, None, None, None)
    )[0]
    assert calls == []
    with ProcessPoolExecutor(max_workers=1, mp_context=p._process_context()) as pool:
        encoded, ordinary, worker_plain = pool.submit(
            _spawn_result_pair, payload, False, flag_value if hook != "new" else None
        ).result(timeout=30)
    assert worker_plain
    outcomes = []
    for data in (ordinary, encoded):
        if hook != "new":
            tr.TokenFlag._value2member_map_.pop(flag_value, None)
        calls.clear()
        try:
            restored = p._deserialize_process_result(data, parent)
            assert isinstance(restored, tr.TokenizedTrajectory)
            outcome = (
                "success",
                restored.tokens,
                history(restored).messages,
                restored.trajectory is parent,
            )
        except Exception as error:
            outcome = (type(error), str(error))
        outcomes.append((outcome, list(calls)))
    assert sentinel in outcomes[0][1]
    assert outcomes[1] == outcomes[0]


@pytest.mark.parametrize("value", [0, 3, (1 << 25) | 1])
def test_standard_flag_construction_keeps_source_elision(value):
    parent, result = fixture()
    parent.metadata["flag"] = tr.TokenFlag(value)
    result.trajectory.metadata["flag"] = tr.TokenFlag(value)
    payload = p._process_payloads(
        [parent], p._ProcessOptions(False, False, None, None, None, None)
    )[0]
    assert unpack_test_payload(payload)[1].source_refs_allowed
    encoded = p._serialize_process_result(result)
    assert type(unpack_test_payload(encoded)) is tuple
    restored = p._deserialize_process_result(encoded, parent)
    assert restored.trajectory is parent
    assert restored.trajectory.metadata["flag"] == value
    assert (
        history(restored).message_sources[0].exchange
        is parent.exchanges.chat_completions[0]
    )


@pytest.mark.parametrize("callback", [None, "assignment", "reconstruction"])
def test_spawn_checks_parent_before_eliding_sources(monkeypatch, callback):
    parent, _ = fixture()
    if callback == "assignment":
        original_assign = tr.TokenizedTrajectory.__setattr__

        def assign(self, name, value):
            if name == "trajectory" and self.trajectory is value:
                value.metadata["touched"] = True
            original_assign(self, name, value)

        monkeypatch.setattr(tr.TokenizedTrajectory, "__setattr__", assign)
    elif callback == "reconstruction":
        original_restore = tr.ChatCompletionsExchange.__setstate__

        def restore(self, state):
            original_restore(self, state)
            self.request["messages"][0]["content"] += "!"

        monkeypatch.setattr(tr.ChatCompletionsExchange, "__setstate__", restore)
    options = p._ProcessOptions(False, False, None, None, None, None)
    payload = p._process_payloads([parent], options)[0]
    with ProcessPoolExecutor(max_workers=1, mp_context=p._process_context()) as pool:
        encoded, ordinary, worker_plain = pool.submit(
            _spawn_result_pair, payload
        ).result(timeout=30)
    assert worker_plain  # A fresh spawn does not inherit the parent's callback.
    expected = p._deserialize_process_result(ordinary, parent)
    restored = p._deserialize_process_result(encoded, parent)
    assert parent.metadata == {}
    assert restored.model_dump_json() == expected.model_dump_json()
    assert type(unpack_test_payload(encoded)) is (
        tuple if callback is None else tr.TokenizedTrajectory
    )
    assert history(restored).messages[0]["content"] == (
        "Public fixture!" if callback == "reconstruction" else "Public fixture"
    )


def test_shared_arrays_and_equal_distinct_exchange_positions():
    parent, result = fixture(True)
    parent.exchanges.chat_completions.append(
        parent.exchanges.chat_completions[0].model_copy(deep=True)
    )
    result.trajectory.exchanges.chat_completions.append(
        result.trajectory.exchanges.chat_completions[0].model_copy(deep=True)
    )
    h = history(result)
    h.message_sources.append(
        tr.ChatCompletionsMessageSource(
            exchange=result.trajectory.exchanges.chat_completions[1], request_index=0
        )
    )
    second = result.histories[0].model_copy()
    result.histories.append(second)
    restored = p._deserialize_process_result(
        p._serialize_process_result(result), parent
    )
    assert isinstance(restored, tr.TokenizedMultiHistoryTrajectory)
    assert restored.histories[0].tokens is restored.histories[2].tokens
    assert restored.histories[0].logprobs is restored.histories[2].logprobs
    assert restored.histories[0].flags is restored.histories[2].flags
    assert [id(s.exchange) for s in history(restored).message_sources] == [
        id(e) for e in parent.exchanges.chat_completions
    ]


def test_inventory_change_is_transfer_error():
    parent, result = fixture()
    payload = p._serialize_process_result(result)
    parent.exchanges.chat_completions.clear()
    with pytest.raises(p._ProcessTransferError, match="structure has changed"):
        p._deserialize_process_result(payload, parent)


@pytest.mark.parametrize("index", [-1, 100, True, "0"])
def test_malformed_reference_is_transfer_error(index):
    parent, result = fixture()
    stream = io.BytesIO()

    class Writer(pickle.Pickler):
        def persistent_id(self, value):
            return index if value is result else None

    Writer(stream).dump(result)
    payload = b"\0" + pickle.dumps(
        (
            b"art-process-sources-v1",
            tuple(type(x) for x in p._process_sources(parent)),
            stream.getvalue(),
        )
    )
    with pytest.raises(
        p._ProcessTransferError, match="Invalid process source reference"
    ):
        p._deserialize_process_result(payload, parent)


def test_actual_worker_input_and_parent_receiver(monkeypatch):
    parent, _ = fixture()

    def tokenize(trajectory, **kwargs):
        _, result = fixture()
        result.trajectory = trajectory
        object.__setattr__(
            history(result).message_sources[0],
            "exchange",
            trajectory.exchanges.chat_completions[0],
        )
        return result

    monkeypatch.setattr(tr.Trajectory, "tokenize", tokenize)
    options = p._ProcessOptions(False, False, None, None, None, None)
    payload = p._process_payloads([parent], options)[0]
    restored = p._deserialize_process_result(
        p._tokenize_process_payload(payload), parent
    )
    assert restored.trajectory is parent
    assert (
        history(restored).message_sources[0].exchange
        is parent.exchanges.chat_completions[0]
    )


def test_unmapped_canonical_reference_uses_legacy_graph():
    parent, result = fixture()
    parent.exchanges.chat_completions.append(
        parent.exchanges.chat_completions[0].model_copy(deep=True)
    )
    result.trajectory.exchanges.chat_completions.append(
        result.trajectory.exchanges.chat_completions[0].model_copy(deep=True)
    )
    source = tr.ChatCompletionsMessageSource(
        exchange=result.trajectory.exchanges.chat_completions[1], request_index=0
    )
    # The receiver traverses models/lists outside the normal source fields too.
    history(result).messages.append(source)
    payload = p._serialize_process_result(result)
    assert isinstance(unpack_test_payload(payload), tr.TokenizedTrajectory)
    restored = p._deserialize_process_result(payload, parent)
    assert (
        history(restored).messages[-1].exchange is parent.exchanges.chat_completions[1]
    )


@pytest.mark.parametrize(
    "protocol", ["responses", "messages", "completions_token", "completions_string"]
)
def test_protocol_history_source_fields(protocol):
    from anthropic.types import Message
    from openai.types import Completion
    from openai.types.responses import Response

    now = datetime(2026, 1, 1, tzinfo=UTC)
    if protocol == "responses":
        e = tr.ResponsesExchange(
            request=tr.ResponsesRequest(model="toy", input=[]),
            response=Response.model_validate(
                {
                    "id": "r",
                    "object": "response",
                    "created_at": 0,
                    "model": "toy",
                    "output": [],
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                }
            ),
            start_time=now,
            end_time=now,
        )
        exchanges = tr.TrajectoryExchanges(responses=[e])
        h = tr.ResponsesHistory(
            model="toy",
            input=[],
            input_sources=[tr.ResponsesItemSource(exchange=e, request_index=0)],
            instructions="Public",
            instructions_source=e,
        )
    elif protocol == "messages":
        e = tr.MessagesExchange(
            request=tr.MessagesRequest(model="toy", messages=[], max_tokens=3),
            response=Message.model_validate(
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "toy",
                    "content": [],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ),
            start_time=now,
            end_time=now,
        )
        exchanges = tr.TrajectoryExchanges(messages=[e])
        h = tr.AnthropicMessagesHistory(
            model="toy",
            messages=[],
            message_sources=[tr.AnthropicMessageSource(exchange=e, request_index=0)],
            system="Public",
            system_source=e,
        )
    else:
        e = tr.CompletionsExchange(
            request=tr.CompletionsRequest(model="toy", prompt=[1]),
            response=Completion.model_validate(
                {
                    "id": "c",
                    "object": "text_completion",
                    "created": 0,
                    "model": "toy",
                    "choices": [],
                }
            ),
            start_time=now,
            end_time=now,
        )
        exchanges = tr.TrajectoryExchanges(completions=[e])
        source = tr.CompletionsSource(exchange=e, prompt_index=0)
        if protocol == "completions_token":
            h = tr.CompletionsTokenHistory(
                model="toy",
                prompt=[1],
                sampled_spans=[],
                prompt_sources=[
                    tr.CompletionsTokenSourceSpan(start=0, end=1, source=source)
                ],
            )
        else:
            h = tr.CompletionsStringHistory(
                model="toy",
                prompt="x",
                sampled_spans=[],
                prompt_sources=[
                    tr.CompletionsStringSourceSpan(start=0, end=1, source=source)
                ],
            )
    parent = tr.Trajectory(exchanges=exchanges, reward=0.0)
    result = tr.TokenizedTrajectory(
        trajectory=parent,
        history=h,
        model="toy",
        tokens=[1],
        logprobs=[math.nan],
        flags=[tr.TokenFlag.EXACT],
    )
    worker = unpack_test_payload(pickle.dumps(result))
    expected, restored, _, new = roundtrip(parent, worker)
    assert unpack_test_payload(new)[0] == b"art-process-sources-v1"
    assert expected.model_dump_json() == restored.model_dump_json()
    if protocol == "responses":
        assert restored.history.instructions_source is e
        assert restored.history.input_sources[0].exchange is e
    elif protocol == "messages":
        assert restored.history.system_source is e
        assert restored.history.message_sources[0].exchange is e
    else:
        assert restored.history.prompt_sources[0].source.exchange is e


class SpecializedResult(tr.TokenizedTrajectory):
    pass


def _malformed_transfer_case(position, value):
    value = copy.deepcopy(value)
    parent, result = fixture(position in ("histories", "multi_history"))
    parent.start_time = result.trajectory.start_time = datetime(2026, 1, 1, tzinfo=UTC)
    if position == "trajectory":
        result = result.model_copy(update={"trajectory": value})
    elif position == "exchanges":
        result = result.model_copy(
            update={
                "trajectory": tr.Trajectory.model_construct(
                    **{**result.trajectory.__dict__, "exchanges": value}
                )
            }
        )
    elif position.startswith("inventory."):
        exchanges = result.trajectory.exchanges.model_copy(
            update={position.split(".")[1]: value}
        )
        result = result.model_copy(
            update={
                "trajectory": tr.Trajectory.model_construct(
                    **{**result.trajectory.__dict__, "exchanges": exchanges}
                )
            }
        )
    elif position == "histories":
        result = result.model_copy(update={"histories": value})
    elif position == "multi_history":
        result.histories[0] = result.histories[0].model_copy(update={"history": value})
    elif position == "history":
        result = result.model_copy(update={"history": value})
    else:
        if position == "responses":
            h = tr.ResponsesHistory(model="toy", input=[], input_sources=[])
            field = "input_sources"
        elif position == "messages":
            h = tr.AnthropicMessagesHistory(
                model="toy", messages=[], message_sources=[]
            )
            field = "message_sources"
        elif position in ("token", "string"):
            cls = (
                tr.CompletionsTokenHistory
                if position == "token"
                else tr.CompletionsStringHistory
            )
            h = cast(Any, cls)(
                model="toy",
                prompt=[1] if position == "token" else "x",
                prompt_sources=[],
                sampled_spans=[],
            )
            field = "prompt_sources"
        else:
            h = history(result)
            field = "message_sources" if position == "chat" else "input_sources"
        result = result.model_copy(
            update={"history": h.model_copy(update={field: value})}
        )
    return parent, result


@pytest.mark.parametrize(
    "position",
    [
        "trajectory",
        "exchanges",
        "inventory.chat_completions",
        "inventory.completions",
        "inventory.responses",
        "inventory.messages",
        "histories",
        "history",
        "multi_history",
        "chat",
        "responses",
        "messages",
        "token",
        "string",
        "undeclared",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        None,
        1,
        "text",
        b"bytes",
        {},
        {"key": [1]},
        set(),
        frozenset(),
        [None],
        (None,),
        [{}],
        ({"key": 1},),
    ],
)
def test_malformed_layout_preserves_ordinary_values_and_exception_phase(
    position, value
):
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        # Do not serialize the same graph twice: reducers can mutate it.
        parent, result = _malformed_transfer_case(position, value)
        before = repr(parent)
        phase = "serialize"
        try:
            payload = encode(result)
            phase = "restore"
            restored = p._deserialize_process_result(payload, parent)
            outcomes.append(("success", repr(restored)))
        except Exception as error:
            outcomes.append((phase, type(error), str(error)))
        assert repr(parent) == before
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize(
    "position",
    [
        "trajectory",
        "exchanges",
        "inventory.chat_completions",
        "inventory.completions",
        "inventory.responses",
        "inventory.messages",
        "histories",
        "history",
        "multi_history",
    ],
)
def test_missing_layout_fields_preserve_ordinary_exception_phase(position):
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result = _malformed_transfer_case(position, None)
        if position.startswith("inventory."):
            owner, field = result.trajectory.exchanges, position.split(".")[1]
        elif position == "exchanges":
            owner, field = result.trajectory, "exchanges"
        elif position == "multi_history":
            owner, field = result.histories[0], "history"
        else:
            owner, field = result, position
        owner.__dict__.pop(field)
        phase = "serialize"
        try:
            payload = encode(result)
            phase = "restore"
            restored = p._deserialize_process_result(payload, parent)
            outcomes.append(("success", repr(restored)))
        except Exception as error:
            outcomes.append((phase, type(error), str(error)))
    assert outcomes[0] == outcomes[1]


def _private_interning_case(path, slot="private", public_alias=False):
    parent, result = fixture(True)
    h = history(result)
    key = "".join(["public-key-", "z" * 100])
    seed = key.encode().decode()
    assert seed == key and seed is not key
    h.chat_template_kwargs = {key: 1, "second-key": 2}
    hidden = tr.Trajectory.model_construct(metadata={"seed": seed, "retained": h})
    value = hidden
    if path == "list":
        value = [hidden]
    elif path == "tuple":
        value = (hidden,)
    elif path == "dict":
        value = {"hidden": hidden}
    elif path == "cycle":
        value = {"hidden": hidden}
        value["cycle"] = value
    elif path == "plain_model":
        value = tr.ChatCompletionsMessageSource.model_construct(exchange=hidden)
    if path == "public":
        state: dict[str, Any] = {
            **result.trajectory.__dict__,
            "metadata": {"hidden": hidden},
        }
        worker = tr.Trajectory.model_construct(**state)
    else:
        worker = tr.Trajectory.model_construct(**result.trajectory.__dict__)
        if slot == "private":
            worker._policy_token_counts = cast(Any, value)
        else:
            object.__setattr__(worker, "__pydantic_fields_set__", value)
        if public_alias:
            worker.metadata["public_alias"] = value
    result = result.model_copy(update={"trajectory": worker})
    return parent, result, key, seed


@pytest.mark.parametrize(
    "path", ["direct", "list", "tuple", "dict", "cycle", "plain_model"]
)
@pytest.mark.parametrize("slot", ["private", "fields_set"])
@pytest.mark.parametrize("public_alias", [False, True])
def test_nonpublic_interning_effects_match_on_independent_fresh_graphs(
    path, slot, public_alias
):
    outcomes = []
    payload_types = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result, key, seed = _private_interning_case(path, slot, public_alias)
        payload = encode(result)
        worker_keys = list(history(result).chat_template_kwargs)
        restored = p._deserialize_process_result(payload, parent)
        outcomes.append(
            (
                ["K" if x == key else "L" for x in worker_keys],
                any(x is seed for x in worker_keys),
                [
                    "K" if x == key else "L"
                    for x in history(restored).chat_template_kwargs
                ],
            )
        )
        payload_types.append(type(unpack_test_payload(payload)))
    assert outcomes[0] == outcomes[1] == (["L", "K"], True, ["L", "K"])
    if not public_alias:
        assert payload_types == [tr.TokenizedMultiHistoryTrajectory] * 2


@pytest.mark.parametrize("private", [None, {1: 2}, {"nested": [1, 2, ("x",)]}])
def test_passive_private_state_keeps_source_elision(private):
    parent, result = fixture()
    result.trajectory._policy_token_counts = private
    assert type(unpack_test_payload(p._serialize_process_result(result))) is tuple


@pytest.mark.parametrize(
    "slot", ["__pydantic_private__", "__pydantic_extra__", "__pydantic_fields_set__"]
)
@pytest.mark.parametrize("primed", [False, True])
@pytest.mark.parametrize("owner", ["result", "trajectory"])
@pytest.mark.parametrize(
    "value", [None, 0, 1, "", "x", [], [1], (), {}, {"key": 1}, set(), {"key"}]
)
def test_model_state_shapes_preserve_values_and_exception_phase(
    slot, primed, owner, value
):
    _assert_model_state_transfer(slot, primed, owner, value, missing=False)


@pytest.mark.parametrize(
    "slot", ["__pydantic_private__", "__pydantic_extra__", "__pydantic_fields_set__"]
)
@pytest.mark.parametrize("primed", [False, True])
@pytest.mark.parametrize("owner", ["result", "trajectory"])
def test_missing_model_state_slots_preserve_exception_phase(slot, primed, owner):
    _assert_model_state_transfer(slot, primed, owner, None, missing=True)


def _assert_model_state_transfer(slot, primed, owner, value, *, missing):
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result = fixture(True)
        if primed:
            pickle.dumps(result)
        target = result if owner == "result" else result.trajectory
        if missing:
            object.__delattr__(target, slot)
        else:
            object.__setattr__(target, slot, copy.deepcopy(value))
        before = repr(parent)
        phase = "serialize"
        try:
            payload = encode(result)
            phase = "restore"
            restored = p._deserialize_process_result(payload, parent)
            outcomes.append(("success", repr(restored)))
        except Exception as error:
            outcomes.append((phase, type(error), str(error)))
        assert repr(parent) == before
    assert outcomes[0] == outcomes[1]


def test_public_interning_effects_are_preserved_without_private_fallback():
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result, key, seed = _private_interning_case("public")
        payload = encode(result)
        restored = p._deserialize_process_result(payload, parent)
        outcomes.append(list(history(restored).chat_template_kwargs))
        assert type(unpack_test_payload(payload)) is (
            tuple
            if encode is p._serialize_process_result
            else tr.TokenizedMultiHistoryTrajectory
        )
    assert outcomes[0] == outcomes[1] == ["second-key", key]


def _marked_interning_case(root_marker, child_marker):
    parent, result = fixture(True)
    pickle.dumps(result)
    h = history(result)
    key = "".join(["public-key-", "z" * 100])
    seed = key.encode().decode()
    h.chat_template_kwargs = {key: 1, "second-key": 2}
    child = tr.Trajectory.model_construct(metadata={"seed": seed, "retained": h})
    pickle.dumps(child)
    result.trajectory.metadata["hidden"] = child
    h.chat_template_kwargs = {key: 1, "second-key": 2}
    for model, marker in ((result, root_marker), (child, child_marker)):
        if marker is None:
            object.__delattr__(model, "_art_pickle_strings_interned")
        else:
            object.__setattr__(model, "_art_pickle_strings_interned", marker)
    return parent, result, key


@pytest.mark.parametrize("root_marker", [None, False, True])
@pytest.mark.parametrize("child_marker", [None, False, True])
@pytest.mark.parametrize("skip", [False, True])
def test_interning_marker_matrix_uses_independent_graphs(
    root_marker, child_marker, skip
):
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result, key = _marked_interning_case(root_marker, child_marker)
        context = p._without_pickle_string_interning() if skip else nullcontext()
        with context:
            payload = encode(result)
        restored = p._deserialize_process_result(payload, parent)
        outcomes.append(
            (
                list(history(result).chat_template_kwargs),
                list(history(restored).chat_template_kwargs),
            )
        )
        if encode is p._serialize_process_result and (
            root_marker is not True or child_marker is True
        ):
            assert type(unpack_test_payload(payload)) is tuple
    order = (
        [key, "second-key"]
        if skip or (root_marker is True and child_marker is True)
        else ["second-key", key]
    )
    assert outcomes[0] == outcomes[1] == (order, order)


class EffectfulInterningMarker:
    def __init__(self, mapping, raises):
        self.mapping = mapping
        self.raises = raises
        self.calls = 0

    def __bool__(self):
        self.calls += 1
        key = next(iter(self.mapping))
        self.mapping[key] = self.mapping.pop(key)
        if self.raises:
            raise RuntimeError("public marker error")
        return True


@pytest.mark.parametrize("owner", ["result", "trajectory", "history"])
@pytest.mark.parametrize("raises", [False, True])
def test_nonboolean_markers_preserve_effects_without_preflight_callbacks(owner, raises):
    outcomes = []
    for encode in (pickle.dumps, p._serialize_process_result):
        parent, result = fixture(True)
        pickle.dumps(result)
        h = history(result)
        h.chat_template_kwargs = {"first": 1, "second": 2}
        marker = EffectfulInterningMarker(h.chat_template_kwargs, raises)
        target = {"result": result, "trajectory": result.trajectory, "history": h}[
            owner
        ]
        object.__setattr__(target, "_art_pickle_strings_interned", marker)
        p._process_plain_models(result)
        assert marker.calls == 0
        try:
            payload = encode(result)
            restored = p._deserialize_process_result(payload, parent)
            outcome = ("success", list(history(restored).chat_template_kwargs))
        except Exception as error:
            outcome = (type(error), str(error))
        outcomes.append((outcome, marker.calls, list(h.chat_template_kwargs)))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][1:] == (1, ["second", "first"])


def test_marker_descriptor_is_rejected_before_reading_instance(monkeypatch):
    _, result = fixture(True)
    calls = []
    monkeypatch.setattr(
        tr.TokenizedMultiHistoryTrajectory,
        "_art_pickle_strings_interned",
        property(lambda value: calls.append(value)),
        raising=False,
    )
    assert p._process_plain_models(result) is None
    assert calls == []


def test_specialized_result_retains_legacy_pickle():
    parent, result = fixture()
    special = SpecializedResult.model_validate(result.model_dump())
    payload = p._serialize_process_result(special)
    assert type(unpack_test_payload(payload)) is SpecializedResult
    restored = p._deserialize_process_result(payload, parent)
    assert type(restored) is SpecializedResult
    assert restored.trajectory is parent


def test_inventory_type_change_is_transfer_error():
    parent, result = fixture()
    payload = p._serialize_process_result(result)
    parent.exchanges.chat_completions[0] = cast(Any, object())
    with pytest.raises(p._ProcessTransferError, match="structure has changed"):
        p._deserialize_process_result(payload, parent)


class ReconstructedSourceAlias:
    def __init__(self, source):
        self.source = source

    def __reduce__(self):
        return getattr, (self.source, "exchange")


class ReconstructedModelAlias(BaseModel):
    source: Any

    def __reduce__(self):
        return getattr, (self.source, "exchange")


@pytest.mark.parametrize("factory", [ReconstructedSourceAlias, ReconstructedModelAlias])
def test_reconstructed_kwargs_alias_stays_detached(factory):
    parent, result = fixture()
    history(result).chat_template_kwargs = {
        "alias": factory(source=history(result).message_sources[0])
    }
    expected, restored, _, _ = roundtrip(parent, result)
    assert (
        history(expected).chat_template_kwargs["alias"]
        is not parent.exchanges.chat_completions[0]
    )
    assert (
        history(restored).chat_template_kwargs["alias"]
        is not parent.exchanges.chat_completions[0]
    )


class RegisteredSource(tr.ChatCompletionsMessageSource):
    marker: str = "registered"


def restore_registered_source(state):
    return RegisteredSource.model_construct(**state)


def reduce_registered_source(source):
    return restore_registered_source, (source.__dict__,)


def test_registered_source_reducer_is_preserved(monkeypatch):
    import copyreg

    parent, result = fixture()
    monkeypatch.setitem(
        copyreg.dispatch_table,
        tr.ChatCompletionsMessageSource,
        reduce_registered_source,
    )
    expected, restored, _, _ = roundtrip(parent, result)
    for value in [expected, restored]:
        source = history(value).message_sources[0]
        assert type(source) is RegisteredSource
        assert source.marker == "registered"
        assert source.exchange is parent.exchanges.chat_completions[0]


class RestoringExchange(tr.ChatCompletionsExchange):
    def __setstate__(self, state):
        super().__setstate__(state)
        message = cast(dict[str, Any], self.request["messages"][0])
        content = message["content"]
        assert isinstance(content, str)
        message["content"] = content + "!"


def test_omitted_exchange_restore_hook_changes_retained_shared_message():
    parent, result = fixture()
    for trajectory in [parent, result.trajectory]:
        old = trajectory.exchanges.chat_completions[0]
        trajectory.exchanges.chat_completions[0] = RestoringExchange.model_construct(
            **old.__dict__
        )
    e = result.trajectory.exchanges.chat_completions[0]
    object.__setattr__(history(result).message_sources[0], "exchange", e)
    history(result).messages[0] = e.request["messages"][0]
    expected = p._deserialize_process_result(pickle.dumps(result), parent)
    restored = p._deserialize_process_result(
        p._serialize_process_result(result), parent
    )
    assert history(expected).messages[0]["content"] == "Public fixture!"
    assert (
        history(restored).messages[0]["content"]
        == history(expected).messages[0]["content"]
    )


class NewArgsCounter(BaseModel):
    count: int = 0

    def __getnewargs__(self):
        self.count += 1
        return ()


class NewArgsExCounter(BaseModel):
    count: int = 0

    def __getnewargs_ex__(self):
        self.count += 1
        return (), {}


@pytest.mark.parametrize("factory", [NewArgsCounter, NewArgsExCounter])
def test_fallback_must_not_execute_newargs_twice(factory):
    parent, result = fixture()
    counter = factory()
    history(result).chat_template_kwargs = {
        "counter": counter,
        "later": datetime(2026, 1, 1, tzinfo=UTC),
    }
    expected = p._deserialize_process_result(pickle.dumps(result), parent)
    assert counter.count == 1
    counter.count = 0
    restored = p._deserialize_process_result(
        p._serialize_process_result(result), parent
    )
    assert history(expected).chat_template_kwargs["counter"].count == 1
    assert history(restored).chat_template_kwargs["counter"].count == 1
    assert counter.count == 1


class RestoringCompletion(ChatCompletion):
    def __setstate__(self, state):
        super().__setstate__(state)
        message = self.choices[0].message
        assert isinstance(message.content, str)
        message.content += "!"


def test_nested_omitted_sdk_hook_retains_shared_kwargs_effect():
    parent, result = fixture()
    exchange = result.trajectory.exchanges.chat_completions[0]
    exchange.response = RestoringCompletion.model_construct(
        **exchange.response.__dict__
    )
    history(result).chat_template_kwargs = {
        "message": exchange.response.choices[0].message
    }
    payload = p._serialize_process_result(result)
    assert isinstance(unpack_test_payload(payload), tr.TokenizedTrajectory)
    restored = p._deserialize_process_result(payload, parent)
    assert history(restored).chat_template_kwargs["message"].content == "Hello!"
    assert (
        parent.exchanges.chat_completions[0].response.choices[0].message.content
        == "Hello"
    )


def test_plain_cycle_uses_memoized_eligibility_and_preserves_alias():
    parent, result = fixture()
    cycle = {}
    cycle["self"] = cycle
    history(result).chat_template_kwargs = {"cycle": cycle}
    assert p._process_plain_models(result) is not None
    restored = p._deserialize_process_result(
        p._serialize_process_result(result), parent
    )
    actual = history(restored).chat_template_kwargs["cycle"]
    assert actual["self"] is actual


def test_custom_timezone_is_declined_without_invoking_its_reducer():
    from datetime import timedelta, timezone, tzinfo

    calls = []

    class CustomTimezone(tzinfo):
        def __reduce__(self):
            calls.append("reduce")
            return timezone, (timedelta(0),)

    parent, result = fixture()
    history(result).chat_template_kwargs = {
        "date": datetime(2026, 1, 1, tzinfo=CustomTimezone())
    }
    assert p._process_plain_models(result) is None
    assert calls == []
    restored = p._deserialize_process_result(
        p._serialize_process_result(result), parent
    )
    assert calls == ["reduce"]
    assert history(restored).chat_template_kwargs["date"].tzinfo is UTC


class MutatingHashKey(BaseModel):
    source: Any

    def __hash__(self):
        self.source.exchange.request["messages"][0]["content"] += "!"
        return 1


@pytest.mark.parametrize(
    "container", ["dict", "set", "frozenset", "tuple_key", "frozen_key"]
)
@pytest.mark.parametrize("value_alias_first", [False, True])
def test_model_hash_reconstruction_preserves_detached_effect(
    container, value_alias_first
):
    parent, result = fixture()
    key = MutatingHashKey(source=history(result).message_sources[0])
    if container == "dict":
        value = {key: None}
    elif container == "set":
        value = {key}
    elif container == "frozenset":
        value = frozenset([key])
    elif container == "tuple_key":
        value = {(key,): None}
    else:
        value = {frozenset([key]): None}
    history(result).chat_template_kwargs = {"value": value}
    if value_alias_first:
        # This tuple/key can be visited as an ordinary value before its hashed
        # position. Eligibility memoization must distinguish those contexts.
        history(result).chat_template_kwargs["alias"] = next(iter(value))
    before = parent.model_dump_json()
    worker_content = history(result).messages[0]["content"]
    # Isolate reconstruction from the existing string-interning traversal,
    # which itself rebuilds containers and can invoke user hashing.
    with p._without_pickle_string_interning():
        expected, restored, _, _ = roundtrip(parent, result)
    assert history(expected).messages[0]["content"] == worker_content + "!"
    assert history(restored).messages[0] == history(expected).messages[0]
    assert parent.model_dump_json() == before
    assert history(result).messages[0]["content"] == worker_content


def restore_offset(message):
    message["content"] += "!"
    return RestoringOffset(hours=1)


class RestoringOffset(timedelta):
    message: dict[str, Any]

    def __reduce__(self):
        return restore_offset, (self.message,)


def restore_timezone_name(message):
    message["content"] += "!"
    return RestoringTimezoneName("custom")


class RestoringTimezoneName(str):
    message: dict[str, Any]

    def __reduce__(self):
        return restore_timezone_name, (self.message,)


@pytest.mark.parametrize("part", ["offset", "name"])
def test_omitted_timezone_reconstruction_changes_retained_shared_message(part):
    parent, result = fixture()
    if part == "offset":
        value = RestoringOffset(hours=1)
        zone = timezone(value)
    else:
        value = RestoringTimezoneName("custom")
        zone = timezone(timedelta(hours=1), value)
    value.message = history(result).messages[0]
    result.trajectory.metadata["zone"] = zone
    before = parent.model_dump_json()
    expected, restored, _, _ = roundtrip(parent, result)
    assert history(expected).messages[0]["content"] == "Public fixture!"
    assert history(restored).messages[0] == history(expected).messages[0]
    assert parent.model_dump_json() == before
    assert history(result).messages[0]["content"] == "Public fixture"


@pytest.mark.parametrize(
    "zone", [UTC, timezone(timedelta(hours=1)), timezone(timedelta(hours=-2), "plain")]
)
def test_builtin_key_containers_and_timezones_keep_source_elision(zone):
    parent, result = fixture()
    result.trajectory.metadata["zone"] = zone
    history(result).chat_template_kwargs = {
        "keys": {(1, "plain", frozenset([2, 3])): {"set": {4, 5}}}
    }
    assert p._process_plain_models(result) is not None
    expected, restored, _, new = roundtrip(parent, result)
    assert unpack_test_payload(new)[0] == b"art-process-sources-v1"
    assert (
        history(restored).chat_template_kwargs == history(expected).chat_template_kwargs
    )
    assert history(restored).messages == history(expected).messages


def test_user_model_as_value_retains_ordinary_pickle_contract():
    parent, result = fixture()
    history(result).chat_template_kwargs = {
        "value": MutatingHashKey(source=history(result).message_sources[0])
    }
    before = parent.model_dump_json()
    expected, restored, _, new = roundtrip(parent, result)
    assert type(unpack_test_payload(new)) is tr.TokenizedTrajectory
    assert history(restored).messages == history(expected).messages
    assert parent.model_dump_json() == before


class FinalizingModel(BaseModel):
    message: Any

    def __del__(self):
        self.message["content"] += "!"


def test_omitted_model_retirement_changes_retained_shared_message():
    parent, result = fixture()
    finalizer = FinalizingModel(message=history(result).messages[0])
    result.trajectory.metadata["finalizer"] = finalizer
    before = parent.model_dump_json()
    expected, restored, _, _ = roundtrip(parent, result)
    assert history(expected).messages[0]["content"] == "Public fixture!"
    assert history(restored).messages == history(expected).messages
    assert parent.model_dump_json() == before
    assert history(result).messages[0]["content"] == "Public fixture"


class ExchangeDescriptorMixin:
    @property
    def exchange(self):
        return self.__dict__["exchange"]

    @exchange.setter
    def exchange(self, value):
        self.__dict__["exchange"] = value
        self.__dict__["request_index"] += 1


class DescriptorSource(tr.ChatCompletionsMessageSource, ExchangeDescriptorMixin):
    pass


def test_inherited_descriptor_rebinding_effect_is_preserved():
    parent, result = fixture()
    source = history(result).message_sources[0]
    history(result).message_sources[0] = DescriptorSource.model_construct(
        **source.__dict__
    )
    before = parent.model_dump_json()
    expected, restored, _, _ = roundtrip(parent, result)
    assert history(expected).message_sources[0].request_index == 1
    assert history(restored).message_sources[0].request_index == 1
    assert history(result).message_sources[0].request_index == 0
    assert parent.model_dump_json() == before


def test_schema_model_with_added_finalizer_retains_retirement_effect(monkeypatch):
    parent, result = fixture()

    def finalize(exchange):
        exchange.request["messages"][0]["content"] += "!"

    monkeypatch.setattr(tr.ChatCompletionsExchange, "__del__", finalize, raising=False)
    before = parent.model_dump_json()
    expected, restored, _, _ = roundtrip(parent, result)
    assert history(expected).messages[0]["content"] == "Public fixture!"
    assert history(restored).messages == history(expected).messages
    assert parent.model_dump_json() == before
    assert history(result).messages[0]["content"] == "Public fixture"


def test_schema_model_with_inherited_field_descriptor_preserves_rebinding(monkeypatch):
    parent, result = fixture()
    monkeypatch.setattr(
        tr._HistorySource, "exchange", ExchangeDescriptorMixin.exchange, raising=False
    )
    expected, restored, _, _ = roundtrip(parent, result)
    assert history(expected).message_sources[0].request_index == 1
    assert history(restored).message_sources[0].request_index == 1
    assert history(result).message_sources[0].request_index == 0


class DeleteOnlyExchange:
    def __get__(self, instance, owner=None):
        return instance.__dict__["exchange"]

    def __delete__(self, instance):
        pass


def test_schema_model_delete_only_descriptor_preserves_rebinding_error(monkeypatch):
    parent, result = fixture()
    monkeypatch.setattr(
        tr._HistorySource, "exchange", DeleteOnlyExchange(), raising=False
    )
    ordinary = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    candidate = p._serialize_process_result(result)
    for payload in (ordinary, candidate):
        with pytest.raises(p._ProcessTransferError, match="__set__"):
            p._deserialize_process_result(payload, parent)


class UserMetadata(BaseModel):
    value: int


def test_plain_user_model_outside_schema_also_uses_ordinary_pickle():
    parent, result = fixture()
    result.trajectory.metadata["value"] = UserMetadata(value=1)
    expected, restored, _, new = roundtrip(parent, result)
    assert type(unpack_test_payload(new)) is tr.TokenizedTrajectory
    assert history(restored).messages == history(expected).messages


def test_schema_result_custom_assignment_preserves_rebinding_effect(monkeypatch):
    parent, result = fixture()
    ordinary_setattr = tr.TokenizedTrajectory.__setattr__

    def setattr_with_identity_effect(self, name, value):
        if name == "trajectory" and self.trajectory is value:
            history(self).messages[0]["content"] += "!"
        ordinary_setattr(self, name, value)

    monkeypatch.setattr(
        tr.TokenizedTrajectory, "__setattr__", setattr_with_identity_effect
    )
    expected, restored, _, _ = roundtrip(parent, result)
    assert history(expected).messages[0]["content"] == "Public fixture"
    assert history(restored).messages == history(expected).messages


def test_cached_assignment_handler_preserves_rebinding_effect(monkeypatch):
    parent, result = fixture()

    def setter(model, name, value):
        if model.trajectory is value:
            history(model).messages[0]["content"] += "!"
        model.__dict__[name] = value
        model.__pydantic_fields_set__.add(name)

    monkeypatch.setitem(
        tr.TokenizedTrajectory.__pydantic_setattr_handlers__, "trajectory", setter
    )
    expected, restored, _, _ = roundtrip(parent, result)
    assert history(expected).messages[0]["content"] == "Public fixture"
    assert history(restored).messages == history(expected).messages


def test_unknown_standard_assignment_authority_uses_ordinary_pickle(monkeypatch):
    parent, result = fixture()
    # A future Pydantic version may move this private authority. Safe fallback
    # must remain available instead of assuming an unknown handler is passive.
    monkeypatch.setattr(
        p.pydantic_main,
        "_SIMPLE_SETATTR_HANDLERS",
        {
            **p.pydantic_main._SIMPLE_SETATTR_HANDLERS,
            "model_field": None,
        },
    )
    payload = p._serialize_process_result(result)
    assert type(unpack_test_payload(payload)) is tr.TokenizedTrajectory


def reader_lifetime_trial(multi, optimized, hold, invalid=False):
    parent, result = fixture(multi)
    references = [
        weakref.ref(parent),
        weakref.ref(parent.exchanges.chat_completions[0]),
    ]
    payload = (
        p._serialize_process_result(result)
        if optimized
        else pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
    )
    assert payload.startswith(b"\0") is optimized
    if invalid:
        tag, kinds, _ = unpack_test_payload(payload)
        # Protocol 5, integer 255, persistent reference, STOP.
        payload = b"\0" + pickle.dumps((tag, kinds, b"\x80\x05K\xffQ."))
        with pytest.raises(pickle.UnpicklingError, match="Invalid process source"):
            p._load_process_result(payload, parent)
        return references, []
    restored = p._deserialize_process_result(payload, parent)
    assert restored.trajectory is parent
    holders = [parent] if hold == "input" else [restored] if hold == "output" else []
    return references, holders


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("hold", ["none", "input", "output"])
def test_process_reader_releases_sources_without_cyclic_gc(
    without_cyclic_gc, multi, optimized, hold
):
    references, holders = reader_lifetime_trial(multi, optimized, hold)
    assert all(
        (reference() is not None) == (hold != "none") for reference in references
    )
    holders.clear()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("multi", [False, True])
def test_process_reader_failure_releases_sources_without_cyclic_gc(
    without_cyclic_gc, multi
):
    references, _ = reader_lifetime_trial(multi, True, "none", invalid=True)
    assert all(reference() is None for reference in references)


def test_process_reader_repeated_success_does_not_retain_sources(without_cyclic_gc):
    references = [
        reference
        for _ in range(16)
        for reference in reader_lifetime_trial(False, True, "none")[0]
    ]
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_process_writer_releases_sources_without_cyclic_gc(
    without_cyclic_gc, multi, optimized
):
    def serialize():
        _, result = fixture(multi)
        reference = weakref.ref(result.trajectory)
        payload = (
            p._serialize_process_result(result)
            if optimized
            else pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
        )
        assert payload.startswith(b"\0") is optimized
        return reference, payload

    reference, payload = serialize()
    assert payload
    assert reference() is None
