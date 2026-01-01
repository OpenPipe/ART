import argparse
import asyncio
import json
import os
import random
from typing import List

from dotenv import load_dotenv
from servers.python.mcp_alphavantage.server_params import (
    server_params as alphavantage_server_params,
)
from servers.python.mcp_balldontlie.server_params import (
    server_params as balldontlie_server_params,
)
import weave

import art
from art.local import LocalBackend
from art.rewards.ruler import ruler_score_group

from ..rollout import McpScenario, rollout

load_dotenv()

random.seed(42)

# Initialize the server
backend = LocalBackend()


async def generate_val_groups(
    model: art.Model, val_scenarios: List[McpScenario]
) -> list[art.TrajectoryGroup]:
    groups = await art.gather_trajectory_groups(
        (
            art.TrajectoryGroup(rollout(model, val_scenarios[i]) for _ in range(4))
            for i in range(len(val_scenarios))
        ),
        pbar_desc=f"gather {model.name}",
        max_exceptions=1,
    )

    return groups


async def calculate_beat_comp(
    groups: list[art.TrajectoryGroup],
    control_groups: list[art.TrajectoryGroup],
    control_first: bool = True,
):
    promises = []

    if control_groups is not None:
        for i in range(len(groups)):
            for j in range(len(groups[i].trajectories)):
                trajectories = [
                    control_groups[i].trajectories[j],
                    groups[i].trajectories[j],
                ]
                group = art.TrajectoryGroup(
                    trajectories if control_first else reversed(trajectories)
                )

                async def score_group(group_idx: int, trajectory_idx: int):
                    scored_group = await ruler_score_group(
                        group,
                        judge_model="openai/o4-mini",
                        debug=True,
                    )

                    if control_first:
                        control_score = scored_group.trajectories[0].reward
                        benchmark_score = scored_group.trajectories[1].reward
                    else:
                        benchmark_score = scored_group.trajectories[0].reward
                        control_score = scored_group.trajectories[1].reward

                    reward_diff = benchmark_score - control_score

                    metric_name = (
                        "beat_comp" if control_first else "beat_comp_control_last"
                    )

                    if reward_diff > 0.1:
                        groups[group_idx].trajectories[trajectory_idx].metrics[
                            metric_name
                        ] = 1
                    elif reward_diff < -0.1:
                        groups[group_idx].trajectories[trajectory_idx].metrics[
                            metric_name
                        ] = 0
                    else:
                        groups[group_idx].trajectories[trajectory_idx].metrics[
                            metric_name
                        ] = 0.5

                promises.append(score_group(i, j))

    await asyncio.gather(*promises)


async def log_comparison_model(
    comparison_model: art.Model,
    val_scenarios: List[McpScenario],
    control_groups: list[art.TrajectoryGroup] | None = None,
) -> list[art.TrajectoryGroup]:
    groups = await generate_val_groups(comparison_model, val_scenarios)

    if control_groups is not None:
        await calculate_beat_comp(groups, control_groups, control_first=True)
        await calculate_beat_comp(groups, control_groups, control_first=False)

    await comparison_model.log(
        groups,
        split="val",
    )
    await backend._experimental_push_to_s3(
        comparison_model,
    )

    return groups


async def run_benchmarks(server: str = "mcp_alphavantage"):
    if server == "mcp_alphavantage":
        scenarios_path = "servers/python/mcp_alphavantage/scenarios/val.jsonl"
        server_params = alphavantage_server_params
    elif server == "mcp_balldontlie":
        scenarios_path = "servers/python/mcp_balldontlie/scenarios/val.jsonl"
        server_params = balldontlie_server_params
    else:
        raise ValueError(
            f"Unsupported server: {server}. Use 'mcp_alphavantage' or 'mcp_balldontlie'"
        )

    weave.init(server)

    # comparison models
    gpt_4o_mini = art.Model(
        name="gpt-4o-mini",
        project=server,
        inference_model_name="openai/gpt-4o-mini",
        inference_base_url="https://openrouter.ai/api/v1",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    gpt_4o = art.Model(
        name="gpt-4o",
        project=server,
        inference_model_name="openai/gpt-4o",
        inference_base_url="https://openrouter.ai/api/v1",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    gpt_41 = art.Model(
        name="gpt-4.1",
        project=server,
        inference_model_name="openai/gpt-4.1",
        inference_base_url="https://openrouter.ai/api/v1",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    o3 = art.Model(
        name="o3",
        project=server,
        inference_model_name="o3",
        inference_base_url="https://api.openai.com/v1",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
    )
    o4_mini = art.Model(
        name="o4-mini",
        project=server,
        inference_model_name="o4-mini",
        inference_base_url="https://api.openai.com/v1",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
    )
    sonnet_4 = art.Model(
        name="sonnet-4",
        project=server,
        inference_model_name="anthropic/claude-sonnet-4",
        inference_base_url="https://openrouter.ai/api/v1",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    with open(scenarios_path) as f:
        raw_val_scenarios = [json.loads(line.strip()) for line in f if line.strip()]
    val_scenarios = [
        McpScenario(
            task_description=scenario["task"],
            server_params=server_params,
            max_turns=5,
        )
        for scenario in raw_val_scenarios
    ]
    await gpt_4o_mini.register(backend)
    await gpt_4o.register(backend)
    await gpt_41.register(backend)
    await o3.register(backend)
    await o4_mini.register(backend)
    await sonnet_4.register(backend)

    control_groups = await generate_val_groups(gpt_41, val_scenarios)

    for comparison_model in [
        gpt_4o_mini,
        gpt_4o,
        gpt_41,
        o3,
        o4_mini,
        sonnet_4,
    ]:
        await log_comparison_model(comparison_model, val_scenarios, control_groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmarks for MCP servers")
    parser.add_argument(
        "--server",
        choices=["mcp_alphavantage", "mcp_balldontlie"],
        default="mcp_alphavantage",
        help="MCP server to benchmark (default: mcp_alphavantage)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmarks(args.server))
