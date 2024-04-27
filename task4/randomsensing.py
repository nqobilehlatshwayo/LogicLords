import chess
import chess.engine
import random
from reconchess import *
from reconchess.utilities import *
from typing import Optional
from collections import defaultdict



class RandomSensing(Player):
    def __init__(self):
        self.board = None
        self.color = None
        self.my_piece_captured_square = None
        self.possible_states = set()
        self.engine_process = None
        self.states_in_window = set()
        self.consistent_states = set()

        # Terminate any existing Stockfish engine process
        # self.terminate_engine_process()

        # Create a new Stockfish engine instance
        self.engine = chess.engine.SimpleEngine.popen_uci('./stockfish', setpgrp=True)

    def handle_game_start(self, color, board, opponent_name):
        # Your initialization logic, such as storing your color and the starting position.
        self.color = color
        self.board = board
        self.opponent_name = opponent_name
        print(f"My Agent color: %s" % self.color)
        # print(self.board)
    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        '''
        whether or not your opponent has made a capture provides you with information. 
        Use this to narrow down the list of moves that could have
        been executed, and hence the possible list of states the agent is currently in. Note that if
        your agent plays as white, then this function is called at the beginning of the game, even
        though your opponent has not previously moved
        '''

        # This function gets called whether your opponent has made a capture.
        # Update your board state based on the information.
        self.my_piece_captured_square = capture_square
        if captured_my_piece:
            self.board.remove_piece_at(capture_square)

        # Generate all possible states at the start of my turn. This is a list of FEN strings.
        self.possible_states = self.generate_all_possible_states(self.board)
        # print("Start of my turn")

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        '''
        choose_sense: this is where you will choose where to sense on the board. Your baseline
        agent should select a square uniformly at random, but should ignore squares on the edges
        of the board (since this would make the window smaller than it needs to be)
        '''

        # if our piece was just captured, sense where it was captured
        # if self.my_piece_captured_square:
        #     return self.my_piece_captured_square

        # if we might capture a piece when we move, sense where the capture will occur

        # future_move = self.choose_move(move_actions, seconds_left)
        # if future_move is not None and self.board.piece_at(future_move.to_square) is not None:
        #     return future_move.to_square

        # otherwise, choose a sense square that's not on the edge of the board or does not have our own pieces
        non_edge_own_piece_squares = [square for square in sense_actions if not self.is_edge_or_own_piece(square)]
        if non_edge_own_piece_squares:
            return random.choice(non_edge_own_piece_squares)
        else:
            # In the rare case that all squares are either on the edge or have our pieces, we can still choose randomly
            return random.choice(sense_actions)
    
    def handle_sense_result(self, sense_result):
        ''''
        handle_sense_result: this returns the window of the board around the sensing point.
        Use this to narrow down the list of possible states
        '''
        # Use this result to update your board state.
        
        # Note that the sense result is a list of (square, piece) tuples, where piece is None
        # if the square is empty.

        self.states_in_window = set()
        # inconsistent_states = self.possible_states.difference()
        # print(f"Possible states: {len(self.possible_states)}")
        
        # Add the states that are consistent with the sense result to the states_in_window set
        for state in self.possible_states:
            consistent = True
            non_empty_symbol = 0
            board = chess.Board(state)
            for square, piece in sense_result:
                # print(f"Pieces on {square}: {board.piece_at(square)} and {piece}")
                if board.piece_at(square) == piece:
                    pass
                else:
                    consistent = False
                    break
            if consistent:
                self.states_in_window.add(state)
            
        inconsistent_states = self.possible_states.difference(self.states_in_window)
        self.consistent_states = self.possible_states.difference(inconsistent_states)

        for square, piece in sense_result:
            if piece is None:
                self.board.remove_piece_at(square)
            else:
                self.board.set_piece_at(square, piece)

        # self.possible_states = self.generate_all_possible_states(self.board)


    def multiple_moves(self, fen_strings):
        move_frequencies = defaultdict(int)
        for fen_string in fen_strings:
            move = self.generate_move(fen_string, len(fen_strings))
            # print(chess.Board(fen_string))
            # print(f"Move: {move}")
            # print()
            if move:
                move_frequencies[move] += 1
        return move_frequencies

    def generate_move(self, fen, size):
        board = chess.Board(fen)

        # Rule 1: Check if the opposing king is attacked and capture it
       
        enemy_king_square = board.king(not self.color)
        if enemy_king_square:
            # if there are any ally pieces that can take king, execute one of those moves
            enemy_king_attackers = board.attackers(self.color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                return chess.Move(attacker_square, enemy_king_square)


        # Rule 2:
        # set the time limit for Stockfish to be 10/N, where N is the number of boards. Should the number of boards exceed 10000,
        # randomly remove states to reduce the number to 10000

        N = size 
        if N > 10000:
            self.consistent_states = random.sample(self.consistent_states, 10000)
            N = len(self.consistent_states)
        
        t = 10/N
        # print(f"Size of possible states: {N}, Time limit: {t}")

        try:
            board.turn = self.color
            board.clear_stack()
            result = self.engine.play(board, chess.engine.Limit(time=t))
            return result.move
        except chess.engine.EngineTerminatedError:
            print('Stockfish Engine died')
        except chess.engine.EngineError:
            print(f'Stockfish Engine bad state at "{board.fen()}"')

        # If no legal move found, return a null move
        return None

    def choose_move(self,  move_actions, seconds_left):
        '''
        choose_move: select a move to play using the majority voting strategy described previously. 
        Since there will be many boards stored, you should set the time limit for Stockfish
        to be 10/N, where N is the number of boards. Should the number of boards exceed 10000,
        randomly remove states to reduce the number to 10000. This will ensure the number of
        states being tracked doesn’t expand exponentially.
        '''

        # Based on the updated board state, choose a move.
        # For now, we just use the multiple_moves method.
        if self.consistent_states is not None:
            posible_moves = self.multiple_moves(self.consistent_states)
        else:
            posible_moves = self.multiple_moves(self.possible_states)
        
        sorted_moves = sorted(posible_moves.keys(), key=lambda m: (-posible_moves[m], m.uci()))
        for move in sorted_moves:
            if (move != None) and move in move_actions:
                return move
            
        return None  # You may opt to make no move.

    def is_edge_or_own_piece(self, square: chess.Square) -> bool:
        # Checks if the square is on the edge or has own piece.
        if chess.square_rank(square) in (0, 7) or chess.square_file(square) in (0, 7):
            return True
        piece = self.board.piece_at(square)
        if piece and piece.color == self.color:
            return True
        return False
    
    def generate_all_possible_states(self, board):
        all_moves = {chess.Move.null()}

        no_opps_board = without_opponent_pieces(board)

        all_moves.update(move for move in board.generate_pseudo_legal_moves())

        for move in no_opps_board.generate_castling_moves():
            if not is_illegal_castle(board, move) and move not in all_moves:
                # this would be a valid castling move in RBC
                # print(move)
                all_moves.add(move)

        posible_positions = set()

        for move in all_moves:
            temp_board = board.copy()
            temp_board.push(move)
            posible_positions.add(temp_board.fen())

        # posible_positions = sorted(posible_positions, key=lambda x: str(x))
        return posible_positions
    
    def handle_move_result(self, requested_move, taken_move, captured_opponent_piece, capture_square):
        '''
        handle_move_result: this is where you can update your states based on the outcome of
        the move, if the move was taken. Note that the move requested and the move actually
        executed may differ (for example, you wanted to move a piece from one square to another,
        but there was another piece in the way, and so the moving piece ended up at a different
        square). You could therefore use this information to rule out certain states.
        '''

        # this function is called after your move is executed.
        if taken_move is not None and self.board.is_pseudo_legal(taken_move):
            self.board.push(taken_move)

    def handle_game_end(self, winner_color, win_reason, game_history):
        # Terminate the Stockfish engine process
        # self.terminate_engine_process()

        # Close the engine instance
        try:
            self.engine.quit()
        except chess.engine.EngineTerminatedError:
            pass

# You may use the strategies adopted by TroutBot or any other improvements.
# Your bot will have to be tested in a round-robin tournament against other agents like RandomBot, TroutBot, etc. [[5]], [[6]], [[7]].

# Be sure to assemble the elements of the game loop and interaction with the Stockfish engine
# according to the instructions provided to ensure the agent operates correctly.

