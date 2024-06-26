import chess
import chess.engine
import random
from reconchess.utilities import *
from typing import Optional
from reconchess import *


class MyAgent(Player):
    def __init__(self):
        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        self.engine = chess.engine.SimpleEngine.popen_uci('./stockfish', setpgrp=True)

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[chess.Square], move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Square]:
        # Filter out the sense actions that are not valid.
        valid_sense_actions = [square for square in sense_actions if not self.is_edge_or_own_piece(square)]

        # Return None if there are no valid sense actions.
        if not valid_sense_actions:
            return None

        # Randomly select a sense action from the valid list.
        return random.choice(valid_sense_actions)

    def is_edge_or_own_piece(self, square: chess.Square) -> bool:
        # Checks if the square is on the edge or has own piece.
        if chess.square_rank(square) in (0, 7) or chess.square_file(square) in (0, 7):
            return True
        piece = self.board.piece_at(square)
        if piece and piece.color == self.color:
            return True
        return False

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        try:
            # if the engine is already terminated then this call will throw an exception
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass