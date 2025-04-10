from dotenv import load_dotenv
import random
from typing import TypedDict
from typing import Literal
from pydantic import BaseModel
import string
import xml.etree.ElementTree as ET

load_dotenv()

WINNING_VALUE = 512


class TwentyFortyEightGame(TypedDict):
    id: str
    board: list[list[int | None]]


def populate_random_cell(game: TwentyFortyEightGame) -> None:
    all_clear_coordinates = [
        (i, j)
        for i in range(len(game["board"]))
        for j in range(len(game["board"][i]))
        if game["board"][i][j] is None
    ]
    random_clear_coordinates = random.choice(all_clear_coordinates)
    # 90% chance to populate a 2, 10% chance to populate a 4
    game["board"][random_clear_coordinates[0]][random_clear_coordinates[1]] = (
        2 if random.random() < 0.9 else 4
    )


def generate_game(board_length: int = 4) -> TwentyFortyEightGame:
    # random 6 character string
    id = "".join(random.choices(string.ascii_letters + string.digits, k=6))
    game = {
        "id": id,
        "board": [[None for _ in range(board_length)] for _ in range(board_length)],
    }

    # populate two random cells
    populate_random_cell(game)
    populate_random_cell(game)

    return game


def render_board(game: TwentyFortyEightGame) -> str:
    board = game["board"]
    # print something like this:
    # _    | 2    | _    | 4
    # 4    | 8    | 2    | 16
    # 16   | 32   | 64   | 128
    # _    | 2    | 2    | 4
    # where _ is an empty cell

    max_cell_width = max(
        [len(str(cell)) for row in board for cell in row if cell is not None]
    )

    board_str = ""
    for row in board:
        # pad the cells with spaces to make them the same width
        board_str += "|".join(
            [
                str(cell).rjust(max_cell_width)
                if cell is not None
                else "_".rjust(max_cell_width)
                for cell in row
            ]
        )
        board_str += "\n"
    return board_str


# condense, privileging matches at the start of the sequence
# sequences should be passed starting with cells that are the furthest in the direction in which the board is being condensed
def condense_sequence(sequence: list[int | None]) -> list[int | None]:
    condensed_sequence = []

    gapless_sequence = [cell for cell in sequence if cell is not None]

    i = 0
    while i < len(gapless_sequence):
        if (
            i + 1 < len(gapless_sequence)
            and gapless_sequence[i] == gapless_sequence[i + 1]
        ):
            condensed_sequence.append(gapless_sequence[i] * 2)
            i += 2
        else:
            condensed_sequence.append(gapless_sequence[i])
            i += 1

    # pad the sequence with None at the end
    return condensed_sequence + [None] * (4 - len(condensed_sequence))


def condense_board(
    game: TwentyFortyEightGame, direction: Literal["left", "right", "up", "down"]
) -> None:
    if direction == "left":
        for row in game["board"]:
            condensed_row = condense_sequence(row)
            for i in range(len(row)):
                row[i] = condensed_row[i]

    if direction == "right":
        for row in game["board"]:
            reversed_row = row[::-1]
            # reverse the row before and after condensing
            condensed_row = condense_sequence(reversed_row)[::-1]
            for i in range(len(row)):
                row[i] = condensed_row[i]

    if direction == "up":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]

            condensed_column = condense_sequence(column)
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]

    if direction == "down":
        for col_index in range(len(game["board"][0])):
            column = [row[col_index] for row in game["board"]]
            reversed_column = column[::-1]
            condensed_column = condense_sequence(reversed_column)[::-1]
            for row_index in range(len(column)):
                game["board"][row_index][col_index] = condensed_column[row_index]


class AgentMove(BaseModel):
    direction: Literal["left", "right", "up", "down"]


def apply_agent_move(game: TwentyFortyEightGame, move_xml: str) -> None:
    direction = None
    # parse the move
    try:
        root = ET.fromstring(move_xml)
        direction = root.text
    except Exception as e:
        raise ValueError("Invalid xml")

    if direction not in ["left", "right", "up", "down"]:
        raise ValueError("Invalid direction")

    condense_board(game, direction)

    populate_random_cell(game)


def max_cell_value(game: TwentyFortyEightGame) -> int:
    return max([cell for row in game["board"] for cell in row if cell is not None])


def check_game_finished(game: TwentyFortyEightGame) -> bool:
    if max_cell_value(game) >= WINNING_VALUE:
        return True

    # check if any cell is empty
    if any(cell is None for row in game["board"] for cell in row):
        return False

    return True
