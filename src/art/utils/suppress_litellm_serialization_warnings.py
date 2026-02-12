import warnings


def suppress_litellm_serialization_warnings():
    """
    Suppress litellm's internal Pydantic serialization warnings.

    These warnings occur due to a known litellm issue (#11759) where response
    types (Message, StreamingChoices) have mismatched field counts during
    internal serialization. The warnings don't affect functionality.

    Scoped to only silence:
    - UserWarning category
    - From pydantic.main module
    - Matching "Pydantic serializer warnings: PydanticSerializationUnexpectedValue"
    """
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"pydantic\.main",
        message=r"Pydantic serializer warnings:\s+PydanticSerializationUnexpectedValue",
    )
