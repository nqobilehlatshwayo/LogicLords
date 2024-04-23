import chess.engine
engine = chess.engine.SimpleEngine.popen_uci('/opt/stockfish/stockfish', setpgrp=True)
from collections import defaultdict

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

def multiple_moves(fen_strings):
    
    # Generate a dictionary of moves and their frequencies
    move_frequencies = defaultdict(int)
    for fen_string in fen_strings:
        move = generate_move(fen_string)
        if move:
            move_frequencies[move] += 1

    return move_frequencies

# Accept a list of possible positions/states as input
num_pos = int(input())
posible_positions = []
for _ in range(num_pos):
    posible_positions.append(input())

# Generate the moves for each posible position/state
posible_moves = multiple_moves(posible_positions)

# Sort the moves by their frequencies and alphabetically
sorted_moves = sorted(posible_moves.keys(), key=lambda m: (-posible_moves[m], m))

# Output the most frequent move (recommended move)
print(sorted_moves[0])

engine.quit()