from typing import Any, cast

from pydantic import BaseModel
import pytest

from art.trajectories import (
    ChatCompletionsExchange,
    CompletionsExchange,
    MessagesExchange,
    ResponsesExchange,
    _tokenize,
)

_EXCHANGES = (
    ChatCompletionsExchange,
    CompletionsExchange,
    MessagesExchange,
    ResponsesExchange,
)


@pytest.mark.parametrize("target", [*_EXCHANGES, BaseModel])
@pytest.mark.parametrize("nested", [False, True])
def test_spoofed_model_class_cannot_admit_unobserved_state(target, nested):
    class Fake:
        __pydantic_decorators__ = True
        # Even a complete model-shaped facade has no actual model authority.
        model_fields = {}
        model_extra = None

        def __init__(self):
            self.model = "public-model"
            self.request = {"messages": []}
            self.revision = [0]

        @property
        def __class__(self):
            return target

    value = Fake()
    assert isinstance(value, target)
    assert not issubclass(type(value), target)
    context = [value, value] if nested else value
    with pytest.raises(TypeError, match="Unsupported mutable"):
        _tokenize._tokenization_context(context)
    validate = _tokenize._tokenization_context_validator(context)
    validate(False)
    value.revision[0] = 1
    with pytest.raises(ValueError, match="cannot be checked"):
        validate(True)


@pytest.mark.parametrize("target", _EXCHANGES)
@pytest.mark.parametrize("subclass", [False, True])
def test_actual_exchange_classes_preserve_request_authority(target, subclass):
    kind = type("ActualExchange", (target,), {}) if subclass else target
    value = cast(
        Any, kind.model_construct(request={"model": "public-model", "revision": [0]})
    )
    context = [value, value]
    expected = _tokenize._tokenization_context(context)
    validate = _tokenize._tokenization_context_validator(context)
    value.request["revision"][0] = 1
    assert _tokenize._tokenization_context(context) != expected
    with pytest.raises(ValueError, match="context changed"):
        validate(True)
    value.request["revision"][0] = 0
    assert _tokenize._tokenization_context(context) == expected
    validate(True)


def test_actual_model_retains_field_and_physical_state_validation():
    class Model(BaseModel):
        revision: list[int]

    value = Model(revision=[0])
    context = [value, value]
    expected = _tokenize._tokenization_context(context)
    validate = _tokenize._tokenization_context_validator(context)
    value.revision[0] = 1
    assert _tokenize._tokenization_context(context) != expected
    with pytest.raises(ValueError, match="context changed"):
        validate(True)
    value.revision[0] = 0
    assert _tokenize._tokenization_context(context) == expected
    validate(True)
