import random
from reconchess import *
import time
import chess
import chess.engine

class KingSlayer2(Player):
    def __init__(self):
        self.board = None
        self.color = None
        self.opponent_color = None
        self.move_num = 0
        self.my_piece_captured_square = None
        self.opp_king_captured = False
        self.capture_allowed = False
        self.depth = 3
        self.engine = chess.engine.SimpleEngine.popen_uci('./stockfish', setpgrp=True)
        self.game_history = []

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.board = board
        self.color = color
        self.capture_allowed = False
        self.depth = 3

        if self.color == chess.WHITE:
            self.opponent_color = chess.BLACK
        else:
            self.opponent_color = chess.WHITE

        self.game_history.append(board.fen())

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # if the opponent captured our piece, remove it from our board.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)
            self.game_history.append(self.board.fen())

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> Square:
        coin = random.randint(0, 100)

        # if the opponent king is captured, and the game is still going on,
        # we've made a huge mistake and need to look for the king
        if self.opp_king_captured:
            square_to_return = self.opp_king_captured
            if coin > 80:
                square_to_return = square_to_return + 1
            else:
                square_to_return = square_to_return - 1
            self.opp_king_captured = False
            return square_to_return

        # sense if our piece was captured
        if self.my_piece_captured_square:
            return self.my_piece_captured_square

        # Filter out the sense actions that are not valid.
        valid_sense_actions = [square for square in sense_actions if not self.is_edge_or_own_piece(square)]

        # Return None if there are no valid sense actions.
        if not valid_sense_actions:
            return None

        # Randomly select a sense action from the valid list.
        return random.choice(valid_sense_actions)

    def is_edge_or_own_piece(self, square: chess.Square) -> bool:
        # Checks if the square is on the edge line of the chessboard.
        if chess.square_rank(square) in [0, 7] and chess.square_file(square) in [0, 7]:
            return True
        piece = self.board.piece_at(square)
        if piece and piece.color == self.color:
            return True
        return False

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # add changes to board, if any
        for square, piece in sense_result:
            self.board.set_piece_at(square, piece)

    # handle the situation if our king is attacked
    def handle_king_attacked(self, board):
        for move in board.pseudo_legal_moves:
            if not board.is_into_check(move):
                return move

        return None

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        now = time.time()
        self.move_num += 1

        # fix the turns if that's an issue on our board
        if self.board.turn == self.opponent_color:
            self.board.push(chess.Move.null())

        # check if we can capture enemy king
        enemy_king_square = self.board.king(self.opponent_color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = self.board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                self.opp_king_captured = enemy_king_square
                return chess.Move(attacker_square, enemy_king_square)

        # check if our king is in danger
        my_king_square = self.board.king(self.color)
        if my_king_square:
            my_king_attackers = self.board.attackers(self.opponent_color, my_king_square)
            if my_king_attackers:
                move = self.handle_king_attacked(self.board.copy())
                if move is not None:
                    return move

        # apply our king path search if none of the above options happen
        found, move = self.king_path(self.board.copy(), 0, self.depth, seconds_left - (time.time() - now))

        # if we didn't find a silent path, retry with capture allowed
        if move is None:
            self.capture_allowed = True
            found, move = self.king_path(self.board.copy(), 0, self.depth, seconds_left - (time.time() - now))
            self.capture_allowed = False

        if found and move in move_actions:
            return move

        # otherwise, try the Stockfish engine move
        try:
            self.board.turn = self.color
            self.board.clear_stack()
            result = self.engine.play(self.board, chess.engine.Limit(time=0.5))
            if result.move in self.board.legal_moves:
                return result.move
        except chess.engine.EngineTerminatedError:
            print(chess.engine.EngineTerminatedError)
        except chess.engine.EngineError:
            print('Stockfish Engine bad state at "{}"'.format(self.board.fen()))

        # if all else fails, return a random move
        return random.choice(move_actions + [None])

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        if self.board.turn == self.opponent_color:
            self.board.push(chess.Move.null())
        if taken_move is not None:
            self.board.push(taken_move)
            self.game_history.append(self.board.fen())
        elif requested_move is not None:
            self.board.push(requested_move)
            self.game_history.append(self.board.fen())

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        self.engine.quit()

    def king_path(self, board, depth, max_depth, time_remaining):
        now = time.time()

        if time_remaining < 0.1:
            return None, 999999

        if board.turn != self.color:
            board.push(chess.Move.null())
        '''
        Check if opponent's king is attacked
        '''
        enemy_king_square = board.king(self.opponent_color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()

                return True, depth

        if depth == max_depth:
            return False, depth

        '''
        if opponent's king is not attacked right now, go deeper in search
        '''
        path_found = False
        best_move = None
        smallest_depth = 9999999
        capture = False
        '''
        Look at every move in the move list
        '''
        for move in board.generate_pseudo_legal_moves():
            if (not board.is_capture(move) or self.capture_allowed):

                board.push(move)

                found, candidate_depth = self.king_path(board.copy(), depth + 1, max_depth,
                                                        time_remaining - (time.time() - now))
                if found:

                    path_found = True

                    coin = random.randint(0, 100)
                    if (candidate_depth < smallest_depth or (candidate_depth == smallest_depth and coin > 30)):
                        smallest_depth = candidate_depth
                        best_move = move
                board.pop()
                if (board.turn != self.color):
                    board.pop()
        if depth > 0:
            return path_found, smallest_depth
        else:
            return path_found, best_move
