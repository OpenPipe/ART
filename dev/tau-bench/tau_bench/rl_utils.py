import os
from typing import Any, Dict, List
import art
from openpipe.client import AsyncOpenPipe, UpdateLogTagsRequestFiltersItem
from langfuse import Langfuse
from art.trajectories import MetadataValue
import uuid
import time


def log_trajectory_to_langfuse(
    traj: art.Trajectory, messages: List[Dict[str, Any]]
) -> None:
    """
    Push one trajectory to Langfuse with task_idx and step for comparison.
    """
    # Initialize langfuse
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST"),
    )
    phase = traj.metadata.get("phase", "unknown")
    step = traj.metadata.get("training_step", 0)
    task_idx = traj.metadata.get("task_index", 0)
    env = traj.metadata.get("env", "unknown")

    trace_name = f"rl-{phase}-step-{step}-task-{task_idx}"

    # Create trace with trajectory data
    trace = langfuse.trace(
        name=trace_name,
        input={
            "task_idx": task_idx,
            "step": step,
            "phase": phase,
            "metadata": traj.metadata,
        },
        output={"messages": messages, "reward": traj.reward, "metadata": traj.metadata},
        metadata={
            "task_idx": task_idx,
            "training_step": step,
            "phase": phase,
            "env": env,
        },
    )

    # Add reward as a score
    trace.score(name="reward", value=traj.reward)


def string_to_string_dict(metadata: dict[str, Any]) -> dict[str, str]:
    string_dict = {}
    for key, value in metadata.items():
        if isinstance(value, MetadataValue):
            string_dict[key] = str(value)
    return string_dict


def create_response_payload() -> dict[str, Any]:
    return {
        "id": f"chatcmpl-dummy-{str(uuid.uuid4())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "dummy-model",
        "choices": [
            {
                "index": 0,
                "message": {"content": "Dummy Response", "role": "assistant"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def log_trajectory_to_openpipe(
    traj: art.Trajectory, messages: List[Dict[str, Any]]
) -> None:
    """
    Push one trajectory to Langfuse with task_idx and step for comparison.
    """
    # Initialize langfuse
    op_client = AsyncOpenPipe(api_key=os.environ["OPENPIPE_API_KEY"])
    report_payload_metrics = string_to_string_dict(traj.metadata)
    resp_payload = create_response_payload()
    traj.metadata["completion_id"] = resp_payload["id"]

    await op_client.report(
        req_payload={
            "model": traj.metadata["model"],
            "messages": messages,
            "tools": traj.tools,
            "metadata": report_payload_metrics,
        },
        resp_payload=resp_payload,
        status_code=200,
    )
    await op_client.base_client._client_wrapper.httpx_client.aclose()


async def update_openpipe_log(traj: art.Trajectory):
    op_client = AsyncOpenPipe(api_key=os.environ["OPENPIPE_API_KEY"])

    update_log_metadata_response = await op_client.update_log_metadata(
        filters=[
            UpdateLogTagsRequestFiltersItem(
                field="completionId",
                equals=traj.metadata["completion_id"],  # type: ignore
            ),
        ],
        metadata={
            "reward": str(traj.reward),
            "judge_explanation": traj.metadata["judge_explanation"],
        },
    )
    print(f"update_log_metadata_response: {update_log_metadata_response}")
    await op_client.base_client._client_wrapper.httpx_client.aclose()


async def update_steps_for_openpipe_logs(
    trajectory_groups: List[art.TrajectoryGroup], global_step: int
):
    op_client = AsyncOpenPipe(api_key=os.environ["OPENPIPE_API_KEY"])
    for trajectory_group in trajectory_groups:
        for trajectory in trajectory_group:
            completion_id = trajectory.metadata["completion_id"]
            await op_client.update_log_metadata(
                filters=[
                    UpdateLogTagsRequestFiltersItem(
                        field="completionId",
                        equals=completion_id,  # type: ignore
                    ),
                ],
                metadata={
                    "training_step": str(global_step),
                },
            )
    print(f"Updated {len(trajectory_groups)} trajectory groups with step {global_step}")
    await op_client.base_client._client_wrapper.httpx_client.aclose()
