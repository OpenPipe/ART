from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Trajectory


@dataclass(frozen=True, slots=True)
class ModelSelector:
    """Private model selection policy used at training boundaries."""

    value: str
    automatic_family: tuple[str, str] | None = None

    def matches(self, candidate: str) -> bool:
        if self.automatic_family is not None:
            prefix, separator = self.automatic_family
            return bool(
                re.fullmatch(
                    f"{re.escape(prefix)}{re.escape(separator)}[0-9]+",
                    candidate,
                )
            )
        return fnmatchcase(candidate, self.value)


def public_model_selector(value: str) -> ModelSelector:
    return ModelSelector(value)


def automatic_training_model_selector(value: str) -> ModelSelector:
    if match := re.fullmatch(r"(.*)@([0-9]+)", value):
        return ModelSelector(value, (match.group(1), "@"))
    if match := re.fullmatch(r"(.*):step([0-9]+)", value):
        return ModelSelector(value, (match.group(1), ":step"))
    return ModelSelector(value)


def resolve_training_model(
    trajectory: Trajectory,
    selector: ModelSelector | str | None,
) -> str:
    """Resolve one concrete, single-protocol captured model for training."""

    exchanges_by_protocol = {
        "Chat Completions": trajectory.exchanges.chat_completions,
        "Completions": trajectory.exchanges.completions,
        "Responses": trajectory.exchanges.responses,
        "Anthropic Messages": trajectory.exchanges.messages,
    }
    exchanges = [
        (protocol, exchange)
        for protocol, protocol_exchanges in exchanges_by_protocol.items()
        for exchange in protocol_exchanges
    ]
    if not exchanges:
        raise ValueError("Exchange training requires at least one captured exchange")
    if any(exchange.model is None for _, exchange in exchanges):
        raise ValueError("Every training exchange must identify its model")

    concrete_models = {exchange.model for _, exchange in exchanges}
    if selector is None:
        matches = concrete_models
    else:
        selector = (
            public_model_selector(selector) if isinstance(selector, str) else selector
        )
        exact = (
            {candidate for candidate in concrete_models if candidate == selector.value}
            if selector.automatic_family is None
            else set()
        )
        matches = exact or {
            candidate
            for candidate in concrete_models
            if candidate is not None and selector.matches(candidate)
        }
    if not matches:
        value = selector.value if isinstance(selector, ModelSelector) else selector
        raise ValueError(f"Trajectory contains no exchanges for model {value!r}")
    if len(matches) != 1:
        raise ValueError(
            "Exchange training requires exactly one concrete model; matched "
            f"{sorted(matches)}"
        )
    selected_model = next(iter(matches))
    if selected_model is None:
        raise AssertionError("model identity was checked above")
    protocols = {
        protocol for protocol, exchange in exchanges if exchange.model == selected_model
    }
    if len(protocols) != 1:
        raise ValueError(
            "Exchange training does not support mixed protocols for one model; found "
            f"{sorted(protocols)}"
        )
    return selected_model


__all__ = [
    "ModelSelector",
    "automatic_training_model_selector",
    "public_model_selector",
    "resolve_training_model",
]
