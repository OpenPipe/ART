import art
import openai
from openai.types.chat import ChatCompletion
import time
import math
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openpipe.client import OpenPipe

from game_utils import (
    TicTacToeGame,
    generate_game,
    apply_agent_move,
    check_winner,
    render_board,
)

load_dotenv()

op_client = OpenPipe()
print("OpenPipe client initialized")


op_client = OpenPipe(api_key=os.getenv("OPENPIPE_API_KEY"))


class PlayerState(BaseModel):
    trajectory: art.Trajectory
    last_completion: ChatCompletion | None
    invalid_move: bool


async def get_agent_move(
    game: TicTacToeGame,
    model: art.Model,
    player_state: PlayerState,
) -> str:
    player_state.trajectory.messages_and_choices.append(
        {"role": "user", "content": render_board(game)}
    )

    messages = player_state.trajectory.messages()
    try:
        client = model.openai_client()
        completion = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=messages,
            max_completion_tokens=128,
            temperature=1.0,
        )
    except openai.LengthFinishReasonError as e:
        raise e
    except Exception as e:
        print("caught exception generating chat completion")
        print(e)
        raise e

    choice = completion.choices[0]
    move = choice.message.content
    if move is None:
        raise ValueError("No move returned")

    player_state.trajectory.messages_and_choices.append(choice)
    player_state.last_completion = completion
    return move


class TicTacToeScenario(BaseModel):
    step: int


@art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(
    x_model: art.Model, y_model: art.Model, scenario: TicTacToeScenario
) -> list[art.Trajectory]:
    game = generate_game()

    player_states = {
        "x": PlayerState(
            trajectory=art.Trajectory(messages_and_choices=[], reward=0),
            last_completion=None,
            invalid_move=False,
        ),
        "o": PlayerState(
            trajectory=art.Trajectory(messages_and_choices=[], reward=0),
            last_completion=None,
            invalid_move=False,
        ),
    }

    for symbol in ["x", "o"]:
        player_states[symbol].trajectory.messages_and_choices.append(
            {
                "role": "system",
                "content": f"You are a tic-tac-toe player. You are playing against an opponent. Always choose the move most likely to lead to an eventual win. Return your move as an XML object with a single property 'move', like so: <move>A1</move>. Optional moves are 'A1', 'B3', 'C2', etc. You are the {symbol} symbol.",
            }
        )

    move_number = 0

    start_time = int(time.time() * 1000)

    while (
        check_winner(game["board"]) is None
        and not player_states["x"].invalid_move
        and not player_states["o"].invalid_move
    ):
        for symbol in ["x", "o"]:
            player_state = player_states[symbol]

            move = await get_agent_move(
                game=game, model=x_model, player_state=player_state
            )

            try:
                apply_agent_move(game=game, move=move, symbol=symbol)
            except ValueError:
                player_state.invalid_move = True
                player_state.trajectory.reward = -10 + (
                    math.log(move_number + 1) / math.log(10)
                )
                break

            move_number += 1
            if check_winner(game["board"]) is not None:
                break

    winner = check_winner(game["board"])

    if winner == "x" or winner == "o":
        winner_state = player_states[winner]
        loser_state = player_states["x" if winner == "o" else "o"]

        winner_state.trajectory.reward = 1
        winner_state.trajectory.metrics["win"] = 1
        loser_state.trajectory.reward = 0
        loser_state.trajectory.metrics["win"] = 0
    elif winner == "draw":
        for symbol in ["x", "o"]:
            player_states[symbol].trajectory.reward = 0.5
            player_states[symbol].trajectory.metrics["win"] = 0.5

    for symbol in ["x", "o"]:
        player_state = player_states[symbol]
        player_state.trajectory.metrics["num_moves"] = move_number
        player_state.trajectory.metrics["invalid_move"] = (
            1 if player_state.invalid_move else 0
        )

    if op_client.api_key:
        for symbol in ["x", "o"]:
            player_state = player_states[symbol]
            trajectory = player_state.trajectory
            messages = trajectory.messages()
            # avoid double-reporting the last assistant completion message
            if messages[-1]["role"] == "assistant":
                messages = messages[:-1]

            model = x_model if symbol == "x" else y_model
            try:
                reported_win = (
                    trajectory.metrics["win"] if "win" in trajectory.metrics else -1
                )
                op_client.report(
                    requested_at=start_time,
                    received_at=int(time.time() * 1000),
                    req_payload={
                        "model": model.name,
                        "messages": messages,
                        "metadata": {
                            "notebook-id": "tic-tac-toe",
                            "step": str(scenario.step),
                            "num_moves": str(move_number),
                            "win": str(reported_win),
                            "reward": str(trajectory.reward),
                            "invalid_move": str(player_state.invalid_move),
                            "symbol": symbol,
                        },
                    },
                    resp_payload=player_state.last_completion,
                    status_code=200,
                )
            except Exception as e:
                print(f"Error reporting to OpenPipe: {e}")

    return player_states["x"].trajectory, player_states["o"].trajectory
