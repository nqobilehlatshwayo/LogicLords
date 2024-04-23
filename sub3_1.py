import chess.engine
engine = chess.engine.SimpleEngine.popen_uci('/opt/stockfish/stockfish', setpgrp=True)

def generate_move(fen):
    board = chess.Board(fen)

    # Rule 1: Check if the opposing king is in attack by one of our pieces and capture it
    enemy_color = not board.turn
    enemy_king_square = board.king(enemy_color)
    if enemy_king_square is not None:
        attackers = board.attackers(board.turn, enemy_king_square)
        if attackers:
            for square in attackers:
                move = chess.Move(square, enemy_king_square)
                if move in board.legal_moves:
                    return move.uci()

    # Rule 2: Use Stockfish to generate a move
    try:
        board.clear_stack()
        result = engine.play(board, chess.engine.Limit(time=0.5))
        return result.move.uci()
    except chess.engine.EngineTerminatedError:
            print('Stockfish Engine died')
    except chess.engine.EngineError:
            print('Stockfish Engine bad state at "{}"'.format(board.fen()))

    return "0000"  # Return null move if no other move is found

position = input()
move = generate_move(position)

print(move)
engine.quit()