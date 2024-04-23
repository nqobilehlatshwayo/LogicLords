import random

from reconchess import *


class RandomBot(Player):
    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        pass

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        pass

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:

        valid_sense_actions = []
        for square, piece in self.board.piece_map().items():
            # Ignore squares on the edges of the board
            if (chess.square_rank(square) in (0, 7) and chess.square_file(square) in (
            0, 7)) or piece.color == self.color:
                sense_actions.remove(square)
            return random.choice(sense_actions)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        pass

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        pass

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        pass
