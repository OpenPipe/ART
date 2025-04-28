import random
import asyncio

import art
from dotenv import load_dotenv
import random
from pydantic import BaseModel

from openpipe.client import OpenPipe

import openai
import time
import math


from art.skypilot.api import SkyPilotAPI
from utils import (
    generate_game,
    get_opponent_move,
    apply_agent_move,
    check_winner,
    render_board,
)


load_dotenv()

op_client = OpenPipe()
print("OpenPipe client initialized")

random.seed(42)


class CustomConfig(BaseModel):
    litellm_model_name: str | None = None


@art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(
    model: art.Model, iteration: int, is_validation: bool
) -> art.Trajectory:
    game = generate_game()

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": f"You are a tic-tac-toe player. You are playing against an opponent. Always choose the move most likely to lead to an eventual win. Return your move as an XML object with a single property 'move', like so: <move>A1</move>. Optional moves are 'A1', 'B3', 'C2', etc. You are the {game['agent_symbol']} symbol.",
            }
        ],
        reward=0,
    )

    move_number = 0

    if game["agent_symbol"] == "o":
        starting_opponent_move = get_opponent_move(game)
        game["board"][starting_opponent_move[0]][starting_opponent_move[1]] = game[
            "opponent_symbol"
        ]

    while check_winner(game["board"]) is None:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )

        requested_at = int(time.time() * 1000)
        messages = trajectory.messages()

        try:
            client = model.openai_client()
            chat_completion = await client.chat.completions.create(
                model=model.name,
                messages=messages,
                max_completion_tokens=128,
            )
            last_completion = chat_completion
        except openai.LengthFinishReasonError as e:
            raise e
        except Exception as e:
            print("caught exception generating chat completion")
            print(e)
            global failing_trajectory
            failing_trajectory = trajectory
            raise e

        try:
            op_client.report(
                requested_at=requested_at,
                received_at=int(time.time() * 1000),
                req_payload={
                    "messages": messages,
                    "metadata": {
                        "notebook-id": "tic-tac-toe",
                        "iteration": str(iteration),
                        "validation": str(is_validation),
                        "move_number": str(move_number),
                    },
                },
                resp_payload=chat_completion,
                status_code=200,
            )
        except Exception as e:
            print(f"Error reporting to OpenPipe: {e}")

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        try:
            apply_agent_move(game, content)
        except ValueError as e:
            trajectory.reward = -1 + (math.log(move_number + 1) / math.log(100))
            break

        if check_winner(game["board"]) is not None:
            break
        move_number += 1

        opponent_move = get_opponent_move(game)
        game["board"][opponent_move[0]][opponent_move[1]] = game["opponent_symbol"]

    winner = check_winner(game["board"])

    if winner == game["agent_symbol"]:
        trajectory.reward = 1
    elif winner == game["opponent_symbol"]:
        trajectory.reward = 0
    elif winner == "draw":
        trajectory.reward = 0.5

    try:
        op_client.update_log_metadata(
            filters=[
                {
                    "field": "completionId",
                    "equals": last_completion.id,
                }
            ],
            metadata={
                "reward": str(trajectory.reward),
                "reward_assigned": "true",
            },
        )
    except Exception as e:
        print(f"Error updating log metadata: {e}")

        print(trajectory.reward)

    return trajectory


DESTROY_AFTER_RUN = False


async def main():
    # run from the root of the repo
    api = await SkyPilotAPI.initialize_cluster(
        cluster_name="art6", art_version=".", env_path=".env", gpu="H100"
    )

    model = art.TrainableModel(
        name="005", project="tic-tac-toe", base_model="Qwen/Qwen2.5-3B-Instruct"
    )
    await model.register(api)

    for i in range(await model.get_step(), 3):
        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    rollout(model, i, is_validation=False) for _ in range(48)
                )
                for _ in range(1)
            ),
            pbar_desc="gather",
        )
        await model.delete_checkpoints()
        await model.train(train_groups, config=art.TrainConfig(learning_rate=1e-4))

    if DESTROY_AFTER_RUN:
        await api.down()


if __name__ == "__main__":
    asyncio.run(main())
