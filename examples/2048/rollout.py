import asyncio
import math
import os

from dotenv import load_dotenv
import openai
import requests
from utils import (
    WINNING_VALUE,
    apply_agent_move,
    check_game_finished,
    generate_game,
    max_cell_value,
    render_board,
    total_board_value,
)
import weave

import art

load_dotenv()


@weave.op
@art.retry(exceptions=(openai.LengthFinishReasonError, requests.ReadTimeout))
async def rollout(
    model: art.Model, step: int, is_validation: bool, verbose: bool = False
) -> art.Trajectory:
    game = generate_game()

    move_number = 0

    trajectory = art.Trajectory(
        messages_and_choices=[
            {
                "role": "system",
                "content": "You are an excellent 2048 player. Always choose the move most likely to lead to combine cells to eventually reach the number 2048. Optional moves are 'left', 'right', 'up', 'down'. Return your move as an XML object with a single property 'move', like so: <move>left</move>",
            },
        ],
        metadata={
            "game_id": game["id"],
            "step": step,
            "validation": is_validation,
        },
        reward=0,
    )

    while True:
        trajectory.messages_and_choices.append(
            {"role": "user", "content": render_board(game)}
        )
        if verbose:
            print(render_board(game))

        async def get_completion():
            client = model.openai_client()
            return await client.chat.completions.create(
                max_completion_tokens=128,
                messages=trajectory.messages(),
                model=model.name,
            )

        try:
            chat_completion = await get_completion()
        except openai.LengthFinishReasonError as e:
            raise e
        except Exception as e:
            print("caught exception generating chat completion", e)
            raise e

        choice = chat_completion.choices[0]
        content = choice.message.content
        assert isinstance(content, str)
        trajectory.messages_and_choices.append(choice)

        try:
            apply_agent_move(game, content)
            if verbose:
                print(content)
            move_number += 1
        except ValueError:
            trajectory.metrics["invalid_move"] = 1
            trajectory.reward = -1
            break

        if check_game_finished(game):
            trajectory.metrics["invalid_move"] = 0
            break

    max_value = max_cell_value(game)
    board_value = total_board_value(game)
    agent_won = max_value == WINNING_VALUE
    trajectory.metrics["max_value"] = max_value
    trajectory.metrics["board_value"] = board_value
    trajectory.metrics["num_moves"] = move_number
    trajectory.metrics["win"] = agent_won

    # try to get as close to the winning value as possible
    # otherwise, try to maximize number of high cells on board
    # but above all else: WIN THE GAME!
    if agent_won:
        # double reward if the agent wins
        trajectory.reward = 2
    else:
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

    return trajectory


if __name__ == "__main__":
    gpt_4o_mini = art.Model(
        name="gpt-4o-mini",
        project="2048",
        inference_model_name="openai/gpt-4o-mini",
        inference_base_url="https://openrouter.ai/api/v1",
        inference_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    async def main():
        trajectory = await rollout(gpt_4o_mini, 0, True, True)
        print("================" * 3)
        print("METRICS\n")
        print(trajectory.metrics)
        print("================" * 3)
        print("REWARD\n")
        print(trajectory.reward)
        print("================" * 3)

    asyncio.run(main())
