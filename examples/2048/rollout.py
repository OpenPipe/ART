import os
import openai
import time
import math
import requests
import weave
from openpipe.client import AsyncOpenPipe
from dotenv import load_dotenv

import art
from utils import (
    apply_agent_move,
    check_game_finished,
    generate_game,
    max_cell_value,
    total_board_value,
    render_board,
    WINNING_VALUE,
)


load_dotenv()

op_client = AsyncOpenPipe(os.getenv("OPENPIPE_API_KEY"))


@weave.op
@art.retry(exceptions=(openai.LengthFinishReasonError, requests.ReadTimeout))
async def rollout(model: art.Model, step: int, is_validation: bool) -> art.Trajectory:
    game = generate_game()

    move_number = 0

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are an excellent 2048 player. Always choose the move most likely to lead to combine cells to eventually reach the number 2048. Optional moves are 'left', 'right', 'up', 'down'. Return your move as an XML object with a single property 'move', like so: <move>left</move>",
            }
        ],
        reward=0,
    )

    while True:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )

        requested_at = int(time.time() * 1000)
        messages = trajectory.messages()

        async def get_completion():
            client = model.openai_client()
            return await client.chat.completions.create(
                max_completion_tokens=128,
                messages=messages,
                model=model.name,
            )

        try:
            chat_completion = await get_completion()
            last_completion = chat_completion
        except openai.LengthFinishReasonError as e:
            raise e
        except Exception as e:
            print("caught exception generating chat completion", e)
            raise e

        try:
            if op_client.api_key:
                await op_client.report(
                    requested_at=requested_at,
                    received_at=int(time.time() * 1000),
                    req_payload={
                        "model": model.name,
                        "messages": messages,
                        "metadata": {
                            "game_id": game["id"],
                            "notebook-id": "2048",
                            "step": str(step),
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
            move_number += 1
        except ValueError:
            trajectory.reward = -1
            break

        if check_game_finished(game):
            max_value = max_cell_value(game)
            board_value = total_board_value(game)
            trajectory.metrics["max_value"] = max_value
            trajectory.metrics["board_value"] = board_value
            trajectory.metrics["num_moves"] = move_number

            if max_value < WINNING_VALUE:
                # scale max value logarithmically between 0 for 2 and 1 for WINNING_VALUE
                max_value_reward = (math.log(max_value, 2) - 1) / (
                    math.log(WINNING_VALUE, 2) - 1
                )
                # scale board value logarithmically between 0 for 2 * 16 and 1 for WINNING_VALUE * 16
                board_value_reward = (math.log(board_value, 2) - 1) / (
                    math.log(WINNING_VALUE * 16, 2) - 1
                )
                # combine the two rewards, with max value having a higher weight
                trajectory.reward = max_value_reward + (board_value_reward * 0.2)
                trajectory.metrics["win"] = 0
            else:
                # double reward if the agent wins
                trajectory.reward = 2
                trajectory.metrics["win"] = 1
            break

    try:
        if op_client.api_key:
            await op_client.update_log_metadata(
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

    return trajectory
