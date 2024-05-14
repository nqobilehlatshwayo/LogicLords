import numpy as np
import json
import os
import numpy as np
from keras.src.layers import RNN
from keras.src.saving import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define constants for piece types
piece_to_value = {'-': 0, 'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
                  'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12}


# Define function to encode board state from FEN string
def encode_board(fen):
    board_state = fen.split()[0]  # Extracting the board state part from FEN
    encoded_board = np.zeros((8, 8), dtype=int)  # 8x8 grid for pieces
    row, col = 0, 0
    for char in board_state:
        if char == '/':  # Move to the next row
            row += 1
            col = 0
        elif char.isdigit():  # Empty squares
            col += int(char)
        else:
            piece_value = 0
            if char.islower():  # Black piece
                piece_value = -1 * piece_to_value[char.lower()]
            else:  # White piece
                piece_value = piece_to_value[char.lower()]
            encoded_board[row, col] = piece_value
            col += 1
    return encoded_board


# Define function to encode additional features
def encode_additional_features(fen):
    parts = fen.split()
    current_player = 0 if parts[1] == 'w' else 1  # 0 for white, 1 for black
    castling_rights = [1 if c in parts[2] else 0 for c in
                       'KQkq']  # King and queen side castling rights for both players
    en_passant_square = -1 if parts[3] == '-' else (
        (ord(parts[3][0]) - ord('a')), (int(parts[3][1]) - 1))  # En passant square or -1 if none
    total_moves = int(parts[5])  # Total moves
    return current_player, castling_rights, en_passant_square, total_moves


# Define function to preprocess board state
def preprocess_board_state(fen):
    # encode the board
    encoded_board = encode_board(fen)
    # encode the additional features
    encoded_features = encode_additional_features(fen)
    # Flatten the board and concatenate with additional features
    flattened_board = encoded_board.flatten()
    # handling the en_passant square
    en_passant_square_file = encoded_features[2][0] if encoded_features[2] != -1 else -1
    en_passant_square_rank = encoded_features[2][1] if encoded_features[2] != -1 else -1
    # Conatenating the encoded additional features with the encoded board
    preprocessed_state = np.concatenate((flattened_board, [encoded_features[0]], encoded_features[1],
                                         [en_passant_square_file, en_passant_square_rank], [encoded_features[3]]))
    return preprocessed_state


# Initialize lists to store preprocessed states and winners
all_preprocessed_states = []
all_winners = []


# Define function to load and preprocess data from a JSON file
def process_json_file(file_name):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
        board_states = data["board_states"][0]

        winner = data["winner"]
        preprocessed_states = [preprocess_board_state(fen) for fen in board_states]
        return preprocessed_states, winner
    except (json.decoder.JSONDecodeError, KeyError) as e:
        print(f"Error processing file {file_name}: {e}")
        return [], None


# Iterate over each JSON file in the working directory

for file_name in os.listdir():
    if file_name.endswith(".json"):
        preprocessed_states, winner = process_json_file(file_name)
        winners = []
        if winner is not None:
            for i in range(len(preprocessed_states)):
                if preprocessed_states[i][64] == winner[0]:
                    winners.append(winner[0])
                else:
                    winners.append(winner[1])

        all_preprocessed_states.extend(np.array(preprocessed_states))
        all_winners.extend(winners)

# Convert lists to numpy arrays
X = all_preprocessed_states
y = all_winners

# shuffle the data without misplacing the results
data = [(x, y) for x, y in zip(X, y)]
np.random.shuffle(data)

X = np.array([x for x, _ in data])
y = np.array([y for _, y in data])

#adjust this ratios to split the data
train_ratio = 0.40
val_ratio = 0.30

# Calculate the number of samples for each split
num_samples = len(X)
num_train_samples = int(train_ratio * num_samples)
num_val_samples = int(val_ratio * num_samples)
num_test_samples = num_samples - num_train_samples - num_val_samples

# Split the data
X_train = X[:num_train_samples]
y_train = y[:num_train_samples]
X_val = X[num_train_samples:num_train_samples + num_val_samples]
y_val = y[num_train_samples:num_train_samples + num_val_samples]
X_test = X[-num_test_samples:]
y_test = y[-num_test_samples:]


# Define the model
def create_model(input_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape),  # LSTM layer with 64 units
        Dropout(0.2),  # Dropout layer to reduce overfitting
        Dense(32, activation='relu'),  # Fully connected layer with 32 units and ReLU activation
        Dense(1, activation='sigmoid')  # Output layer with sigmoid activation for binary classification
    ])
    return model


# Assuming you have X_train and y_train for training data
X_train = X_train.reshape(-1, 1, 72)
model = create_model(input_shape=(1, 72))  # Adjust input_shape to match the reshaped X_train
# model = load_model("trained_model.keras") #uncomment this line if you want to re-use the trained model

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_val = X_val.reshape(-1, 1, 72)

# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_val, y_val))

X_test = X[-num_test_samples:]
y_test = y[-num_test_samples:]

X_test = X_test.reshape(-1, 1, 72)
# Evaluate the model on the validation set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Testing Loss:", test_loss)
print("Testing Accuracy:", test_accuracy)
# Save the model
model.save("trained_model3.keras")
