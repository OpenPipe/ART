from pydantic import BaseModel
from abc import ABC, abstractmethod
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from typing import TypeVar, Generic, get_args, ClassVar, cast


ParamsType = TypeVar("ParamsType", bound=BaseModel)


class Tool(Generic[ParamsType], ABC):
    name: str
    description: str
    parameters: ClassVar[type[BaseModel]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Find the generic base Tool[ParamsType] in the class's bases
        generic_base = None
        if hasattr(cls, "__orig_bases__"):  # Check if it exists first
            for base in cls.__orig_bases__:  # type: ignore[attr-defined]
                origin = getattr(base, "__origin__", None)
                if origin is Tool:
                    generic_base = base
                    break

        if generic_base:
            params_type_args = get_args(generic_base)
            if params_type_args and issubclass(params_type_args[0], BaseModel):
                cls.parameters = params_type_args[0]
            else:
                raise TypeError(
                    f"Class {cls.__name__} inherited Tool[...] but the type parameter "
                    f"is missing or not a Pydantic BaseModel: {params_type_args}"
                )
        elif (
            cls is not Tool
        ):  # Only raise if it's a subclass but not a proper generic Tool subclass
            raise TypeError(
                f"Class {cls.__name__} must inherit from Tool[YourParamsModel], "
                f"not Tool directly (unless it's an intermediate abstract class)."
            )

    def call(self, call: ChatCompletionMessageToolCall) -> str:
        if call.function.name != self.name:
            raise ValueError(
                f"Tool {self.name} called with wrong name: {call.function.name}"
            )

        validated_args_model = self.parameters.model_validate_json(
            call.function.arguments
        )

        args: ParamsType = cast(ParamsType, validated_args_model)

        return self._implementation(args)

    @abstractmethod
    def _implementation(self, args: ParamsType) -> str:
        raise NotImplementedError

    def schema(self) -> ChatCompletionToolParam:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_json_schema(),
            },
        }
