import subprocess

# Define the command to execute
for i in range(10):
    command2 = ["python", "-m", "reconchess.scripts.rc_bot_match", "Stockfish.py", 'rs2.py']
    B = subprocess.run(command2)
    print(B)
