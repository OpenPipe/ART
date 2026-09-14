import httpx
import pytest

from art.serverless.client import Client


@pytest.mark.parametrize(
    "base_url, expected_base",
    [
        (None, "https://forge.coreweave.com/api/training/v1"),
        ("https://training.example/custom/v1", "https://training.example/custom/v1"),
    ],
)
async def test_training_requests_preserve_base_path(
    base_url: str | None, expected_base: str
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "job-id"})

    async with Client(api_key="test-key", base_url=base_url) as client:
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        await client.training_jobs.create(model_id="model-id", trajectory_groups=[])
        await client.sft_training_jobs.create(
            model_id="model-id", training_data_url="https://data.example/train.jsonl"
        )

    assert [str(request.url) for request in requests] == [
        f"{expected_base}/preview/training-jobs",
        f"{expected_base}/preview/sft-training-jobs",
    ]
    assert all(
        request.headers["authorization"] == "Bearer test-key" for request in requests
    )
