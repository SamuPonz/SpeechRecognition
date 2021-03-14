from recognitionMethods import recognition

print("Let's try to recognise something!")
while True:
    recognition()
    if input("Repeat the program? (Y/N)").strip().upper() != 'Y':
        break
