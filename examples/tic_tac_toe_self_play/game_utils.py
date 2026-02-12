from typing import Literal, TypedDict
import xml.etree.ElementTree as ET


class TicTacToeGame(TypedDict):
    board: list[list[str]]


def generate_game(board_length: int = 3) -> TicTacToeGame:
    board = [["_" for _ in range(board_length)] for _ in range(board_length)]
    return TicTacToeGame(
        board=board,
    )


possible_moves = [
    "<move>A1</move>",
    "<move>A2</move>",
    "<move>A3</move>",
    "<move>B1</move>",
    "<move>B2</move>",
    "<move>B3</move>",
    "<move>C1</move>",
    "<move>C2</move>",
    "<move>C3</move>",
]


def render_board(game: TicTacToeGame) -> str:
    board = game["board"]
    board_length = len(board)
    # print something like this:
    #    1   2   3
    # A  _ | x | x
    # B  o | _ | _
    # C  _ | o | _
    # where _ is an empty cell

    board_str = "   " + "   ".join([str(i + 1) for i in range(board_length)]) + "\n"
    for i in range(board_length):
        board_str += f"{chr(65 + i)}  {board[i][0]} | {board[i][1]} | {board[i][2]}\n"
    return board_str


def unwrap_move(move: str) -> str:
    try:
        root = ET.fromstring(move)
        return root.text
    except Exception:
        raise ValueError("Invalid xml")


def apply_agent_move(
    game: TicTacToeGame, square: str, symbol: Literal["x", "o"]
) -> None:
    board_length = len(game["board"])

    try:
        row_index = ord(square[0]) - 65
        col_index = int(square[1]) - 1
    except Exception as e:
        print(e)
        raise ValueError("Unable to parse square")

    if (
        row_index < 0
        or row_index >= board_length
        or col_index < 0
        or col_index >= board_length
    ):
        raise ValueError(
            f"Invalid move, row or column out of bounds: {row_index}, {col_index}"
        )

    # check if the move is valid
    if game["board"][row_index][col_index] != "_":
        raise ValueError("Square already occupied")

    game["board"][row_index][col_index] = symbol


def check_winner(board: list[list[str]]) -> Literal["x", "o", "draw", None]:
    board_length = len(board)
    # check rows
    for row in board:
        if row.count(row[0]) == board_length and row[0] != "_":
            return row[0]
    # check columns
    for col in range(board_length):
        if [board[row][col] for row in range(board_length)].count(
            board[0][col]
        ) == board_length and board[0][col] != "_":
            return board[0][col]

    # top right to bottom left
    upward_diagonal = [board[i][board_length - i - 1] for i in range(board_length)]
    if (
        upward_diagonal.count(upward_diagonal[0]) == board_length
        and upward_diagonal[0] != "_"
    ):
        return upward_diagonal[0]

    # top left to bottom right
    downward_diagonal = [board[i][i] for i in range(board_length)]
    if (
        downward_diagonal.count(downward_diagonal[0]) == board_length
        and downward_diagonal[0] != "_"
    ):
        return downward_diagonal[0]

    # check for draw
    if all(cell != "_" for row in board for cell in row):
        return "draw"
    return None
