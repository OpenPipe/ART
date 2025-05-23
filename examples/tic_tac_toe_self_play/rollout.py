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
    unwrap_move,
)

load_dotenv()

op_client = OpenPipe(api_key=os.getenv("OPENPIPE_API_KEY"))


class PlayerState(BaseModel):
    trajectory: art.Trajectory
    last_completion: ChatCompletion | None
    invalid_move: bool


class ModelConfig(BaseModel):
    requires_reasoning: bool = False


async def get_agent_move(
    game: TicTacToeGame,
    player_state: PlayerState,
    model: art.Model,
    shadowmaster: art.Model | None = None,
    predestined_move: str | None = None,
) -> str:
    assert isinstance(model.config, ModelConfig)
    player_state.trajectory.messages_and_choices.append(
        {"role": "user", "content": render_board(game)}
    )

    messages = player_state.trajectory.messages()
    try:
        if shadowmaster and not predestined_move:
            assert isinstance(shadowmaster.config, ModelConfig)
            shadowmaster_client = shadowmaster.openai_client()
            shadowmaster_completion = await shadowmaster_client.chat.completions.create(
                model=shadowmaster.get_inference_name(),
                messages=messages,
                max_completion_tokens=2000
                if shadowmaster.config.requires_reasoning
                else 100,
                reasoning_effort="low"
                if shadowmaster.config.requires_reasoning
                else None,
                temperature=1.0,
            )
            predestined_move = shadowmaster_completion.choices[0].message.content

        client = model.openai_client()
        completion = await client.chat.completions.create(
            model=model.get_inference_name(),
            messages=messages,
            max_completion_tokens=2000 if model.config.requires_reasoning else 100,
            reasoning_effort="low" if model.config.requires_reasoning else None,
            temperature=1.0,
            extra_body={"guided_choice": [predestined_move]}
            if predestined_move and model.trainable
            else None,
        )
    except openai.LengthFinishReasonError as e:
        raise e
    except Exception as e:
        print("caught exception generating chat completion")
        print(e)
        raise e

    player_state.last_completion = completion

    choice = completion.choices[0]
    move_xml = choice.message.content
    if move_xml is None:
        raise ValueError("No move returned")

    player_state.trajectory.messages_and_choices.append(choice)

    return unwrap_move(move_xml)


def record_first_move_metrics(trajectory: art.Trajectory, square: str) -> None:
    for row in ["A", "B", "C"]:
        for col in ["1", "2", "3"]:
            board_square = f"{row}{col}"
            trajectory.metrics[board_square] = 1 if board_square == square else 0


class TicTacToeScenario(BaseModel):
    step: int
    split: str
    x_shadowmaster: art.Model | None = None
    o_shadowmaster: art.Model | None = None
    initial_move: str | None = None


@art.retry(exceptions=(openai.LengthFinishReasonError,))
async def rollout(
    x_model: art.Model, o_model: art.Model, scenario: TicTacToeScenario
) -> list[art.Trajectory]:
    game = generate_game()

    player_states = {
        "x": PlayerState(
            trajectory=art.Trajectory(
                messages_and_choices=[],
                reward=0,
                metadata={"model_name": x_model.name},
            ),
            last_completion=None,
            invalid_move=False,
        ),
        "o": PlayerState(
            trajectory=art.Trajectory(
                messages_and_choices=[],
                reward=0,
                metadata={"model_name": o_model.name},
            ),
            last_completion=None,
            invalid_move=False,
        ),
    }

    for symbol in ["x", "o"]:
        player_states[symbol].trajectory.messages_and_choices.append(
            {
                "role": "system",
                "content": f"You are a tic-tac-toe player. You are playing against an opponent. Always choose the move most likely to lead to an eventual win. Return your move as an XML object with a single property 'move', like so: <move>A1</move>. Optional moves are 'A1', 'B2', 'C3', etc. You are the {symbol} symbol.",
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
            model = x_model if symbol == "x" else o_model
            player_state = player_states[symbol]
            shadowmaster = (
                scenario.x_shadowmaster if symbol == "x" else scenario.o_shadowmaster
            )

            try:
                square = await get_agent_move(
                    game=game,
                    player_state=player_state,
                    model=model,
                    shadowmaster=shadowmaster,
                    predestined_move=scenario.initial_move
                    if move_number == 0
                    else None,
                )
                if move_number == 0:
                    record_first_move_metrics(player_state.trajectory, square)
                apply_agent_move(game=game, square=square, symbol=symbol)
            except ValueError:
                player_state.invalid_move = True
                player_state.trajectory.reward = -2 + (
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

        winner_state.trajectory.reward = 1 - move_number / 40
        winner_state.trajectory.metrics["win"] = 1
        loser_state.trajectory.reward = 0 + move_number / 40
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

            model = x_model if symbol == "x" else o_model
            shadowmaster = (
                scenario.x_shadowmaster if symbol == "x" else scenario.o_shadowmaster
            )
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
                            "project": "tic-tac-toe",
                            "split": scenario.split,
                            "step": str(scenario.step),
                            "num_moves": str(move_number),
                            "win": str(reported_win),
                            "reward": str(trajectory.reward),
                            "invalid_move": str(player_state.invalid_move),
                            "symbol": symbol,
                            "shadowmaster": shadowmaster.name if shadowmaster else "",
                            "initial_move": unwrap_move(scenario.initial_move)
                            if scenario.initial_move
                            else "",
                        },
                    },
                    resp_payload=player_state.last_completion,
                    status_code=200,
                )
            except Exception as e:
                print(f"Error reporting to OpenPipe: {e}")

    return player_states["x"].trajectory, player_states["o"].trajectory
