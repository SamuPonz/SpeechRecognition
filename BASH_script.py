from pathlib import Path
import wave
from recognitionMethods import recognise_after_record


print("Let's try to recognise something!")
while True:
    recognise_after_record()
    if input("Repeat the program? (Y/N)").strip().upper() != 'Y':
        break

