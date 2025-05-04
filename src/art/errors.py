from fastapi import HTTPException

"""
This file contains errors that are returned by LocalBackend. They extend HTTPException
so that FastAPI can return them as JSON responses, but have descriptive names to aid with
debugging when running LocalBackend without a remote connection.
"""


class ForbiddenBucketCreationError(HTTPException):
    """An error raised when the user receives a 403 Forbidden error when trying to create a bucket.

    This can occur if the bucket already exists and belongs to another user, or if the user's credentials
    do not have permission to create a bucket.
    """

    def __init__(self, message: str):
        super().__init__(
            status_code=403,
            detail=message,
        )
