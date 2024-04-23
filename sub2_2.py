from reconchess.utilities import *
import chess

fen_string = input()
board = chess.Board(fen_string)

all_moves = [chess.Move.null()]
# print('0000')

no_opps_board = without_opponent_pieces(board)

all_moves.extend(move for move in board.generate_pseudo_legal_moves())

for move in no_opps_board.generate_castling_moves():
    if not is_illegal_castle(board, move) and move not in all_moves:
        # this would be a valid castling move in RBC
        # print(move)
        all_moves.append(move)

posible_positions = []

for move in all_moves:
    temp_board = board.copy()
    temp_board.push(move)
    posible_positions.append(temp_board.fen())

posible_positions.sort(key=lambda x: str(x))  # Sort the positions in alphabetical order

for pos in posible_positions:
    print(pos)    