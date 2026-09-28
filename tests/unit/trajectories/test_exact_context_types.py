from datetime import datetime
from typing import Any, cast

import pytest
from test_tokenize import _character_template_history

from art.trajectories import _tokenize


@pytest.mark.parametrize(
    "target",
    [str, int, bool, bytes, type(None), datetime, float, list, tuple, set, frozenset],
)
@pytest.mark.parametrize("shape", ["opaque", "mapping", "scalar"])
def test_metaclass_equality_cannot_grant_exact_type_admission(target, shape):
    class Meta(type):
        def __eq__(cls, other):
            return other is target

        __hash__ = type.__hash__

    class Opaque(metaclass=Meta):
        revision: list[int]

        def __iter__(self):
            return iter(())

    class Mapping(dict, metaclass=Meta):
        revision: list[int]

    class Scalar(str, metaclass=Meta):
        revision: list[int]

    value = {"opaque": Opaque, "mapping": Mapping, "scalar": Scalar}[shape]()
    value.revision = [0]
    if shape == "opaque":
        with pytest.raises(TypeError, match="Unsupported mutable"):
            _tokenize._tokenization_context(value)
        validate = _tokenize._tokenization_context_validator(value)
        validate(False)
        with pytest.raises(ValueError, match="cannot be checked"):
            validate(True)
    else:
        expected = _tokenize._tokenization_context(value)
        validate = _tokenize._tokenization_context_validator(value)
        value.revision[0] = 1
        assert _tokenize._tokenization_context(value) != expected
        with pytest.raises(ValueError, match="context changed"):
            validate(True)
        value.revision[0] = 0
        assert _tokenize._tokenization_context(value) == expected
        validate(True)


@pytest.mark.parametrize("location", ["tokenizer", "explicit"])
def test_rich_template_normalization_remains_guarded(monkeypatch, location):
    class Meta(type):
        def __eq__(cls, other):
            return other is str

        __hash__ = type.__hash__

    class Template(str, metaclass=Meta):
        pass

    history, tokenizer, _ = _character_template_history()
    calls = []
    original = _tokenize._TraceBuilder.checked

    def checked(self, function, *args, **kwargs):
        if function is _tokenize._resolved_chat_template:
            calls.append("guarded normalization")
        return original(self, function, *args, **kwargs)

    monkeypatch.setattr(_tokenize._TraceBuilder, "checked", checked)
    if location == "tokenizer":
        setattr(tokenizer, "chat_template", Template("public template"))
        result = history.tokenize(tokenizer=tokenizer)
    else:
        result = history.tokenize(
            tokenizer=tokenizer, chat_template=cast(Any, Template("public template"))
        )
    assert result.tokens and calls


@pytest.mark.parametrize("base", [str, dict])
def test_metaclass_equality_cannot_hide_renderer_visible_type_replacement(base):
    class Meta(type):
        def __eq__(cls, other):
            return other is Left or other is Right

        __hash__ = type.__hash__

    Left = Meta("Left", (base,), {})
    Right = Meta("Right", (base,), {})
    left, right = Left(), Right()
    left.revision = right.revision = [0]
    context = [left]
    expected = _tokenize._tokenization_context(context)
    validate = _tokenize._tokenization_context_validator(context)
    context[0] = right
    assert _tokenize._tokenization_context(context) != expected
    with pytest.raises(ValueError, match="context changed"):
        validate(True)
    context[0] = left
    assert _tokenize._tokenization_context(context) == expected
    validate(True)


def test_custom_type_tag_preserves_class_lifetime_without_instance_retention():
    import gc
    import weakref

    class Meta(type):
        pass

    class Value(str, metaclass=Meta):
        pass

    value = Value("public")
    kind_ref, value_ref = weakref.ref(Value), weakref.ref(value)
    expected = _tokenize._tokenization_context(value)
    del value, Value
    gc.collect()
    assert kind_ref() is not None and value_ref() is None
    del expected
    gc.collect()
    assert kind_ref() is None
