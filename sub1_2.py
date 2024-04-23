import chess

fen_string = input()
board = chess.Board(fen_string)

move_uci = input()
move = chess.Move.from_uci(move_uci)

board.push(move)

print(board.fen())