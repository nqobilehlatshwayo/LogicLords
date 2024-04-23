import chess

fen_string = input()

board = chess.Board(fen_string)

print(board)