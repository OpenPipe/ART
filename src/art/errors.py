"""
This file contains errors that are returned by LocalBackend. They are normal exceptions
with status_code and detail attributes that can be handled by FastAPI exception handlers
to return them as JSON responses with appropriate HTTP status codes.
"""


class ARTError(Exception):
    """Base class for ART exceptions that should be converted to HTTP responses."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.detail = message
        self.status_code = status_code


class ForbiddenBucketCreationError(ARTError):
    """An error raised when the user receives a 403 Forbidden error when trying to create a bucket.

    This can occur if the bucket already exists and belongs to another user, or if the user's credentials
    do not have permission to create a bucket.

    Status code: 403
    """

    def __init__(self, message: str):
        super().__init__(message, status_code=403)


class UnsupportedLoRADeploymentProviderError(ARTError):
    """An error raised when the user attempts to deploy a model to a provider that does not support
    serverless LoRA deployment.

    Status code: 400
    """

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class UnsupportedBaseModelDeploymentError(ARTError):
    """An error raised when the user attempts to deploy a model to a provider that does not support
    it for serverless LoRA deployment.

    Status code: 400
    """

    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class LoRADeploymentTimedOutError(ARTError):
    """An error raised when deployment of a LoRA times out.

    Status code: 504
    """

    def __init__(self, message: str):
        super().__init__(message, status_code=504)
