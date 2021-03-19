from pathlib import Path

from recognitionMethods import recognition

command_recordings_dir = Path("C:\\Users\Samuele\Desktop\Database resgistrazioni\(64721,30)_ComandiDatasetRinominati")

print("Let's try to recognise something!")
while True:
    recognition(command_recordings_dir)
    if input("Repeat the program? (Y/N)").strip().upper() != 'Y':
        break
