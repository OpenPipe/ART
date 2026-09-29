import math
from typing import Any, cast

import pytest
import torch

import art.trajectories as tr
from art.trajectories import tensors


def history(tokens, flags, model="policy"):
    return tr.TokenizedHistory.model_construct(
        history=tr.LegacyHistory(messages_and_choices=[]),
        model=model,
        tokens=tokens,
        flags=flags,
        logprobs=[math.nan] * len(tokens) if isinstance(tokens, list) else [],
    )


@pytest.mark.parametrize("tokens", [[], [4], [4, 4, 4], [4, 5, 4]])
@pytest.mark.parametrize(
    "where",
    [
        None,
        tr.TokenFlag(0),
        tr.TokenFlag.SAMPLED,
        tr.TokenFlag.OUTPUT | tr.TokenFlag.STOP,
    ],
)
def test_single_history_needs_no_prefix_nodes(monkeypatch, tokens, where):
    flags = [0, 2, 24][: len(tokens)]
    value = history(tokens, flags)
    expected = tr._FirstOccurrenceTrie().mask(value.model, tokens, flags, where=where)

    def unexpected_node():
        pytest.fail("one history must not allocate a prefix tree")

    monkeypatch.setattr(tr, "_TokenPrefixNode", unexpected_node)
    assert tr.first_occurrence_masks(iter([value]), where=where) == [expected]
    assert value.tokens is tokens and value.flags is flags


@pytest.mark.parametrize("failure", [None, "token", "flag"])
@pytest.mark.parametrize("lengths", [(2, 2), (1, 2), (2, 1)])
def test_single_history_preserves_conversion_and_iterator_errors(failure, lengths):
    marker = RuntimeError("conversion failure")

    def run(public):
        events = []

        class Number:
            def __init__(self, name, index, value):
                self.name, self.index, self.value = name, index, value

            def __int__(self):
                events.append((self.name, self.index))
                if self.name == failure:
                    raise marker
                return self.value

        def numbers(name, count, value):
            for index in range(count):
                events.append(("yield_" + name, index))
                yield Number(name, index, value)

        tokens, flags = numbers("token", lengths[0], 7), numbers("flag", lengths[1], 2)
        where = Number("where", 0, 2)
        try:
            result = (
                tr.first_occurrence_masks(
                    [history(tokens, flags)], where=cast(Any, where)
                )[0]
                if public
                else tr._FirstOccurrenceTrie().mask(
                    "policy", tokens, flags, where=cast(Any, where)
                )
            )
            return result, events
        except RuntimeError as error:
            assert error is marker
            return (type(error), error.args), events
        except ValueError as error:
            return (type(error), error.args), events

    assert run(True) == run(False)


@pytest.mark.parametrize("raise_on_hash", [False, True])
def test_custom_model_hash_stays_on_ordinary_path(raise_on_hash):
    events = []
    marker = RuntimeError("hash callback")

    class Model(str):
        def __hash__(self):
            events.append("hash")
            if raise_on_hash:
                raise marker
            return super().__hash__()

    model = Model("policy")
    value = history([7], [2], model)

    def run(public):
        events.clear()
        try:
            result = (
                tr.first_occurrence_masks([value], where=tr.TokenFlag.SAMPLED)[0]
                if public
                else tr._FirstOccurrenceTrie().mask(
                    model, [7], [2], where=tr.TokenFlag.SAMPLED
                )
            )
        except RuntimeError as error:
            assert error is marker
            result = "raised"
        return result, list(events)

    assert run(True) == run(False)


def test_unhashable_model_keeps_ordinary_error():
    with pytest.raises(TypeError, match="unhashable"):
        tr.first_occurrence_masks([history([7], [2], [])])


@pytest.mark.parametrize("tensor", [False, True])
def test_history_subclasses_keep_ordinary_path(monkeypatch, tensor):
    class CustomTokenized(tr.TokenizedHistory):
        pass

    class CustomTensorized(tensors.TensorizedHistory):
        pass

    cls = CustomTensorized if tensor else CustomTokenized
    value = history([7], [3])
    if tensor:
        value = value.tensorize()
    custom = cls.model_construct(**vars(value))
    original = tr._TokenPrefixNode
    created = []

    def node():
        result = original()
        created.append(result)
        return result

    monkeypatch.setattr(tr, "_TokenPrefixNode", node)
    masks = tr.first_occurrence_masks(cast(Any, [custom]))
    assert len(created) == 2
    assert (masks[0].tolist() if tensor else masks[0]) == [True]


def test_repeated_history_still_claims_prefixes_and_models_are_separate():
    value = history([7, 7], [2, 2])
    assert tr.first_occurrence_masks([value, value]) == [[True, True], [False, False]]
    other = history(value.tokens, value.flags, "other")
    assert tr.first_occurrence_masks([value, other]) == [[True, True], [True, True]]
    assert tr.first_occurrence_masks([]) == []


@pytest.mark.parametrize("where", [None, tr.TokenFlag(0), tr.TokenFlag.SAMPLED])
@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_single_tensor_history_preserves_dtype_device_aliases(
    monkeypatch, where, dtype
):
    token_values = torch.tensor([7, 7, 9], dtype=dtype)
    flag_values = torch.tensor([0, 2, 24], dtype=dtype)
    value = tensors.TensorizedHistory.model_construct(
        history=tr.LegacyHistory(messages_and_choices=[]),
        model="policy",
        tokens=token_values,
        flags=flag_values,
        logprobs=torch.zeros(3),
    )
    expected = tr._FirstOccurrenceTrie().mask(
        "policy", [7, 7, 9], [0, 2, 24], where=where
    )

    def unexpected_node():
        pytest.fail("one tensor history must not allocate a prefix tree")

    monkeypatch.setattr(tr, "_TokenPrefixNode", unexpected_node)
    actual = tr.first_occurrence_masks([value], where=where)[0]
    assert actual.tolist() == expected
    assert actual.dtype is torch.bool and actual.device == token_values.device
    assert actual.data_ptr() not in {token_values.data_ptr(), flag_values.data_ptr()}
    assert value.tokens is token_values and value.flags is flag_values
    assert token_values.tolist() == [7, 7, 9] and flag_values.tolist() == [0, 2, 24]


def test_tensor_direct_generator_retains_ordinary_behavior():
    value = history([7], [3]).tensorize()
    assert tensors.first_occurrence_masks(cast(Any, iter([value])))[0].tolist() == [
        True
    ]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_tensor_token_conversion_errors_are_not_skipped(invalid):
    value = tensors.TensorizedHistory.model_construct(
        history=tr.LegacyHistory(messages_and_choices=[]),
        model="policy",
        tokens=torch.tensor([invalid]),
        flags=torch.tensor([2]),
        logprobs=torch.zeros(1),
    )
    error = ValueError if math.isnan(invalid) else OverflowError
    with pytest.raises(error):
        tr.first_occurrence_masks([value])


def test_tensor_where_callback_can_append_history():
    value = history([7], [3]).tensorize()
    rows = [value]

    class Where:
        def __int__(self):
            if len(rows) == 1:
                rows.append(value)
            return 2

    masks = tensors.first_occurrence_masks(rows, where=cast(Any, Where()))
    assert [mask.tolist() for mask in masks] == [[True], [False]]


def test_tensor_model_getter_can_append_history(monkeypatch):
    value = history([7], [3]).tensorize()
    rows = [value]

    def model(self):
        if len(rows) == 1:
            rows.append(value)
        return "policy"

    monkeypatch.setattr(
        tensors.TensorizedHistory, "model", property(model), raising=False
    )
    masks = tensors.first_occurrence_masks(rows, where=tr.TokenFlag.SAMPLED)
    assert [mask.tolist() for mask in masks] == [[True], [False]]


def test_custom_tensor_conversion_can_append_history(monkeypatch):
    value = history([7], [3]).tensorize()
    rows = [value]

    class Number:
        def __int__(self):
            if len(rows) == 1:
                rows.append(value)
            return 7

    class Converted:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [[Number()], [3]]

    monkeypatch.setattr(torch, "stack", lambda values: Converted())
    masks = tensors.first_occurrence_masks(rows, where=tr.TokenFlag.SAMPLED)
    assert [mask.tolist() for mask in masks] == [[True], [False]]


@pytest.mark.parametrize("where", [None, tr.TokenFlag.SAMPLED])
@pytest.mark.parametrize("initial_flags", [[0, 2], [2, 2]])
def test_late_tensor_device_callback_preserves_prior_claims(where, initial_flags):
    rows = []
    armed = False

    class DeviceTensor(torch.Tensor):
        @property
        def device(self):
            if armed and len(rows) == 1:
                rows.append(rows[0])
                rows[0].flags[:] = 2
            return super().device

    value = tensors.TensorizedHistory.model_construct(
        history=tr.LegacyHistory(messages_and_choices=[]),
        model="policy",
        tokens=torch.tensor([7, 7]).as_subclass(DeviceTensor),
        flags=torch.tensor(initial_flags),
        logprobs=torch.zeros(2),
    )
    rows.append(value)
    armed = True
    masks = tensors.first_occurrence_masks(rows, where=where)
    assert [mask.tolist() for mask in masks] == (
        [[True, True], [False, False]]
        if where is None
        else [
            [bool(v & 2) for v in initial_flags],
            [not bool(v & 2) for v in initial_flags],
        ]
    )
    assert rows[0] is rows[1]
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("change_tokens", [False, True])
def test_output_callback_cannot_rewrite_deferred_prefix_or_claims(
    monkeypatch, change_tokens
):
    value = history([7, 7], [3, 3]).tensorize()
    rows = [value]
    converted_tokens, converted_flags = [7, 7], [3, 3]
    original_tensor = torch.tensor

    class Converted:
        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return [converted_tokens, converted_flags]

    def tensor(data, **kwargs):
        if len(rows) == 1:
            rows.append(value)
            if change_tokens:
                converted_tokens[:] = [8, 8]
            converted_flags[:] = [0, 0]
            data[:] = [False, False]
        return original_tensor(data, **kwargs)

    monkeypatch.setattr(torch, "stack", lambda values: Converted())
    monkeypatch.setattr(torch, "tensor", tensor)
    masks = tensors.first_occurrence_masks(rows)
    # Constructor mutation changes its own output, but not the first prefix claims.
    assert [mask.tolist() for mask in masks] == [
        [False, False],
        [change_tokens, change_tokens],
    ]
