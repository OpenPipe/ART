from typing import TypedDict
from typing_extensions import Required


class TorchtuneArgs(TypedDict, total=False):
    model: Required[str]
