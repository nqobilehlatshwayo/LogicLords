import os
from copy import copy, deepcopy
from keras.models import load_model

import chess
import chess.engine
import random
from reconchess.utilities import *
from typing import Optional, Dict
from reconchess import *
from chess import *

import numpy as np

# Static evaluation tables
QUEEN_SCORE = 80
ROOK_SCORE = 40
BISHOP_SCORE = 41
PAWN_SCORE = 15
KNIGHT_SCORE = 45

piece_to_value = {'-': 0, 'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
                  'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12}

game_states = []


def save_game_states(game_states, filename):
    # Check if the filename already exists
    if os.path.exists(filename):
        # If the filename exists, append a suffix until a unique filename is found
        i = 1
        while True:
            new_filename = f"Data/{filename}_{i}.json"
            if not os.path.exists(new_filename):
                filename = new_filename
                break
            i += 1

    # Write the game states to the new filename
    with open(filename, 'w') as f:
        json.dump(game_states, f)


class Neural_Network(Player):
    def __init__(self):
        self.model = load_model("Data/trained_model.keras")
        self.game_history = []
        self.future_opponent_move = None
        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        self.engine = chess.engine.SimpleEngine.popen_uci('./stockfish', setpgrp=True)

    def order_moves(self, move_actions, board):
        # MVV/LVA heuristic: sort captures by the value of the captured piece and the value of the capturing piece
        ordered_moves = sorted(move_actions, key=lambda move: piece_to_value.get(board.piece_at(move.to_square), 0),
                               reverse=True)
        return ordered_moves

    def encode_board(self, fen):
        board_state = fen.split()[0]  # Extracting the board state part from FEN
        encoded_board = np.zeros((8, 8), dtype=int)  # 8x8 grid for pieces
        row, col = 0, 0
        for char in board_state:
            if char == '/':  # Move to the next row
                row += 1
                col = 0
            elif char.isdigit():  # Empty squares
                col += int(char)
            else:
                piece_value = 0
                if char.islower():  # Black piece
                    piece_value = -1 * piece_to_value[char.lower()]
                else:  # White piece
                    piece_value = piece_to_value[char.lower()]
                encoded_board[row, col] = piece_value
                col += 1
        return encoded_board

    def encode_additional_features(self, fen):
        parts = fen.split()
        current_player = 0 if parts[1] == 'w' else 1  # 0 for white, 1 for black
        castling_rights = [1 if c in parts[2] else 0 for c in
                           'KQkq']  # King and queen side castling rights for both players
        en_passant_square = -1 if parts[3] == '-' else (
            (ord(parts[3][0]) - ord('a')), (int(parts[3][1]) - 1))  # En passant square or -1 if none
        total_moves = int(parts[5])  # Total moves
        return current_player, castling_rights, en_passant_square, total_moves

    def preprocess_board_state(self, fen):
        encoded_board = self.encode_board(fen)
        encoded_features = self.encode_additional_features(fen)
        # Flatten the board and concatenate with additional features
        flattened_board = encoded_board.flatten()
        en_passant_square_file = encoded_features[2][0] if encoded_features[2] != -1 else -1
        en_passant_square_rank = encoded_features[2][1] if encoded_features[2] != -1 else -1
        preprocessed_state = np.concatenate((flattened_board, [encoded_features[0]], encoded_features[1],
                                             [en_passant_square_file, en_passant_square_rank], [encoded_features[3]]))
        return np.array(preprocessed_state)

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color
        self.game_history.append(board.fen())

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)
            self.game_history.append(self.board.fen())

    def choose_sense(self, sense_actions: List[chess.Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[chess.Square]:
        # if our piece was just captured, sense where it was captured
        if self.my_piece_captured_square:
            return self.my_piece_captured_square

        # if we might capture a piece when we move, sense where the capture will occur
        future_move = self.choose_move(move_actions, seconds_left)
        if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
            return future_move.to_square

        # sense where we predict our opponent moved during their previous turn
        if self.future_opponent_move is not None:
            return self.future_opponent_move.to_square

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

    def count_pieces(self, board) -> Dict[Color, int]:
        """
        Count the number of pieces for each color on the board.
        """
        piece_counts = {chess.WHITE: 0, chess.BLACK: 0}
        for piece in board.piece_map().values():
            piece_counts[piece.color] += 1
        return piece_counts

    def evaluate_board(self, board):
        # Initialize variables
        player = -1

        # Determine player's color
        if self.color:
            player = 1

        input_sample = self.preprocess_board_state(board.fen())
        input_sample = input_sample.reshape(1, -1, 72)  # Reshape to match the input shape of the model
        prediction = self.model.predict(input_sample)
        return float(prediction[0][0])

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        enemy_king_square = self.board.king(not self.color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                # assume/check if this is a possible move from the list parameter:
                return chess.Move(attacker_square, enemy_king_square)

        best_move = None
        best_score = -math.inf
        ordered_moves = self.order_moves(move_actions, self.board)
        for move in ordered_moves:
            if move not in self.board.legal_moves:
                continue
            self.board.push(move)
            score = self.evaluate_board(self.board)
            if score > best_score:
                best_score = score
                best_move = move
            #alpha = max(alpha, best_score)
            self.board.pop()
        if best_move:
            return best_move

        # otherwise, try to move with the stockfish chess engine
        try:
            self.board.turn = self.color
            self.board.clear_stack()
            result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
            return result.move
        except chess.engine.EngineTerminatedError:
            print('Stockfish Engine died')
        except chess.engine.EngineError:
            print('Stockfish Engine bad state at "{}"'.format(self.board.fen()))

        # if all else fails, pass
        return None

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)
        self.game_history.append(self.board.fen())

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        try:
            # if the engine is already terminated then this call will throw an exception
            game_states.append(self.game_history)
            # game_states.append(serialized_game_history_moves)
            winner_value = []
            if winner_color:
                winner_value = [1, 0]

            else:
                winner_value = [0, 1]
            game_state = {
                'board_states': game_states,
                # 'game_moves': serialized_game_history_moves,
                'winner': winner_value
            }
            filename = "game_states"
            save_game_states(game_state, filename)
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass
