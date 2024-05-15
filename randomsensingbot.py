# Nqobile Hlatshwayo
# Ayanda Thwala
# Mahloromela Seabi

import chess
import chess.engine
import random
from reconchess.utilities import *
from typing import Optional
from reconchess import *
import numpy as np
from collections import defaultdict


class RandomSensing(Player):
    def __init__(self):
        self.board = None
        self.color = None
        self.moveNum = 0
        self.my_piece_captured_square = None
        self.possible_states = set()  # Set to store possible board states
        self.opponent_piece_captured_square = None
        self.future_opponent_move = None
        self.engine = chess.engine.SimpleEngine.popen_uci('./stockfish', setpgrp=True)

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color

        if self.color == chess.WHITE:
            self.opponent_color = chess.BLACK
        else:
            self.opponent_color = chess.WHITE

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

    def choose_sense(self, sense_actions: List[chess.Square], move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Square]:
        for square, piece in self.board.piece_map().items():
            if piece.color == self.color:
                sense_actions.remove(square)
        # don't sense on a square along the edge
        edges = np.array([0, 1, 2, 3, 4, 5, 6, 7,
                          8, 15, 16, 23, 24, 31, 32,
                          39, 40, 47, 48, 55, 56, 57,
                          58, 59, 60, 61, 62, 63])
        sense_actions = np.setdiff1d(sense_actions, edges)
        sense_actions = sense_actions.tolist()

        return random.choice(sense_actions)


    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        for square, piece in sense_result:
            if piece is None:
                self.board.remove_piece_at(square)
            else:
                self.board.set_piece_at(square, piece)

    def generate_move(self, board, timelimit):
        

        # Rule 1: Check if the opposing king is in attack by one of our pieces and capture it
        enemy_color = not board.turn
        enemy_king_square = board.king(enemy_color)
        if enemy_king_square is not None:
            attackers = board.attackers(board.turn, enemy_king_square)
            if attackers:
                for square in attackers:
                    move = chess.Move(square, enemy_king_square)
                    if move in board.legal_moves:
                        return move

        # Rule 2: Use Stockfish to generate a move
        try:
            board.clear_stack()
            result = self.engine.play(board, chess.engine.Limit(time=timelimit))
            return result.move
        except chess.engine.EngineTerminatedError:
                print('Stockfish Engine died')
        except chess.engine.EngineError:
                print('Stockfish Engine bad state at "{}"'.format(board.fen()))

        return chess.Move.null()  # Return null move if no other move is found

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # if we might be able to take the king, try to
        # enemy_king_square = self.board.king(not self.color)
        # if enemy_king_square:
        #     # if there are any ally pieces that can take king, execute one of those moves
        #     enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
        #     if enemy_king_attackers:
        #         attacker_square = enemy_king_attackers.pop()
        #         # assume/check if this is a possible move from the list parameter:
        #         return chess.Move(attacker_square, enemy_king_square)
                    
        # try:
        #     self.board.turn = self.color
        #     self.board.clear_stack()
        #     result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
        #     return result.move
        # except chess.engine.EngineTerminatedError:
        #     print('Stockfish Engine died')
        # except chess.engine.EngineError:
        #     print('Stockfish Engine bad state at "{}"'.format(self.board.fen()))
        
        # Retaliation move
        # if self.my_piece_captured_square:
        #         attackers = self.board.attackers(self.color, self.my_piece_captured_square)
        #         if attackers:
        #             attack = attackers.pop()
        #             mv = chess.Move(attack, self.my_piece_captured_square)
        #             if mv in move_actions:
        #                 return mv
        move_frequencies = defaultdict(int)

        no_opps_board = without_opponent_pieces(self.board)

        all_moves = set()
        all_moves.update(self.board.generate_pseudo_legal_moves()) 

        for move in no_opps_board.generate_castling_moves():
            if not is_illegal_castle(self.board, move) and move not in all_moves:
                all_moves.add(move)
                # this would be a valid castling move in RBC

        for move in all_moves:
            temp_board = self.board.copy()
            temp_board.push(move)
            self.possible_states.add(temp_board.fen())  # Add FEN strings to the set

        num_boards = len(self.possible_states)
        if num_boards > 10000:
            self.possible_states = set(random.sample(self.possible_states, 10000))
            num_boards = 10000
        time_limit = 10 / num_boards
        for fen in self.possible_states:
            state = chess.Board(fen)
            best_move = self.generate_move(state, time_limit)
            if best_move != chess.Move.null():
                move_frequencies[best_move] += 1

        # print(move_frequencies)
        sorted_moves = sorted(move_frequencies.keys(), key=lambda m: (-move_frequencies[m], m.uci()))
        # print()
        # print("Sorted moves: ")
        # print(sorted_moves)
        for move in sorted_moves:
            if move in move_actions:
                # print(move)
                return move
        
        # print("Random move :(")
        return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # if a move was executed, apply it to our board
        if taken_move is not None:
            self.board.push(taken_move)

        # predict opponent's next move and sense there during our next turn
        # try:
        #     self.board.turn = not self.color
        #     self.board.clear_stack()
        #     result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
        #     self.future_opponent_move = result.move
        # except chess.engine.EngineTerminatedError:
        #     print('Stockfish Engine died')
        # except chess.engine.EngineError:
        #     print('Stockfish Engine bad state at "{}"'.format(self.board.fen()))

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        try:
            # if the engine is already terminated then this call will throw an exception
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass