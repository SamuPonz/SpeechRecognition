from pathlib import Path

from recognitionMethods import recognise_audiofile

command_recordings_dir = Path("C:\\Users\Samuele\Desktop\Database_registrazioni\(64721,30)_ComandiDatasetRinominati")

print("Let's try to recognise something!")
while True:
    recognise_audiofile(command_recordings_dir)
    if input("Repeat the program? (Y/N)").strip().upper() != 'Y':
        break
