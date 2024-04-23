from reconchess.utilities import *
import chess

fen_string = input()
capture_square = input()

board = chess.Board(fen_string)

all_moves = [chess.Move.null()]     # Initialize the list of all possible moves with the null move (0000)

no_opps_board = without_opponent_pieces(board)

all_moves.extend(move for move in board.generate_pseudo_legal_moves())

for move in no_opps_board.generate_castling_moves():
    if not is_illegal_castle(board, move) and move not in all_moves:
        # this would be a valid castling move in RBC
        # print(move)
        all_moves.append(move)

all_legal_captures = []
for move in all_moves:
    if board.is_capture(move):
        all_legal_captures.append(move)

capture_moves = []
for move in all_legal_captures:
    if move.to_square == chess.SQUARE_NAMES.index(capture_square):
        capture_moves.append(move)

# Generate the possible positions after the capture moves
possible_positions = []
for move in capture_moves:
    temp_board = board.copy(stack=False)
    temp_board.push(move)
    possible_positions.append(temp_board.fen())

possible_positions.sort()  # Sort the positions in alphabetical order

# Output the possible positions
for pos in possible_positions:
    print(pos)

