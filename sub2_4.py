from reconchess.utilities import *
import chess

# Function to get the piece on a given square
def get_piece(board, square):
    
    piece = board.piece_at(chess.parse_square(square))

    # If the piece is None (no piece on the square), return '?'
    return piece.symbol() if piece else '?'

# Function to check if a board's window matches the sensed window
def is_consistent(board, window):

    # Check if the piece on each square in the window matches the sensed window
    for square, piece in window.items():
        # print("State: Piece on " + square + " is " + get_piece(board, square))
        # print("Window: Piece on " + square + " is " + piece)
        if get_piece(board, square) != piece:
            return False
    return True


# Function to filter the states based on the window
def filter_states(states, window):
    consistent_states = []

    # Check if each state is consistent with the window
    for state in states:
        board = chess.Board(state)
        if is_consistent(board, window):
            consistent_states.append(state)

    # Sort the consistent states in alphabetical order
    consistent_states.sort()

    return consistent_states

# Read the input
num_states = int(input())
states = []
for _ in range(num_states):
    states.append(input())
window_string = input()

# Create a dictionary to store the window
window = {}
for line in window_string.split(';'):
    if line:
        key, value = line.split(':')
        window[key] = value

# Filter the states based on the window
result = filter_states(states, window)

# Print the filtered states
for state in result:
    print(state)