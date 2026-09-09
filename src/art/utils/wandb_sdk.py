from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wandb.apis.public import Api
    from wandb.sdk.artifacts.artifact import Artifact
    from wandb.sdk.wandb_run import Run
    from wandb.sdk.wandb_settings import Settings


def api(*args: Any, **kwargs: Any) -> Api:
    from wandb.apis.public import Api

    return Api(*args, **kwargs)


def artifact(*args: Any, **kwargs: Any) -> Artifact:
    from wandb.sdk.artifacts.artifact import Artifact

    return Artifact(*args, **kwargs)


def init(*args: Any, **kwargs: Any) -> Run:
    from wandb.sdk.wandb_init import init as wandb_init

    return wandb_init(*args, **kwargs)


def login(*args: Any, **kwargs: Any) -> Any:
    from wandb.sdk.wandb_login import login as wandb_login

    return wandb_login(*args, **kwargs)


def settings(*args: Any, **kwargs: Any) -> Settings:
    from wandb.sdk.wandb_settings import Settings

    return Settings(*args, **kwargs)


def comm_error_type() -> type[Exception]:
    from wandb.errors import CommError

    return CommError
