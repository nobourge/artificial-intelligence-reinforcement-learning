# given file character quantifier.py

from os import name
import os
import sys


def get_file_name(file_path: str) -> str:
    return os.path.basename(file_path)

def get_file_extension(file_path: str) -> str:
    return os.path.splitext(file_path)[1]

def get_file_name_without_extension(file_path: str) -> str:

    return os.path.splitext(get_file_name(file_path))[0]

# print quantity of characters in a file
def get_characters_quantity(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as file:
        return len(file.read())
    
if __name__ == "__main__":
    #get arg as file path
    file_path = sys.argv[1]
    print("character quantity: ")
    file_path = "character_quantifier.py"
    print(get_characters_quantity(file_path))