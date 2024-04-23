from reconchess.utilities import *
import chess


fen_string = input()
board = chess.Board(fen_string)

all_moves = ['0000']
# print('0000')

no_opps_board = without_opponent_pieces(board)

all_moves.extend(move.uci() for move in board.generate_pseudo_legal_moves())

for move in no_opps_board.generate_castling_moves():
    if not is_illegal_castle(board, move) and str(move) not in all_moves:
        # this would be a valid castling move in RBC
        # print(move)
        all_moves.append(str(move))

all_moves.sort()

for move in all_moves:
    print(move)
    