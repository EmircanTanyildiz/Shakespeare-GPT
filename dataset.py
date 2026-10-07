import os
import torch
import urllib.request
SHAKESPEARE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt"
DATA_PATH = "data/shakespeare.txt"
def download_shakespeare():
    if os.path.isfile(DATA_PATH):
        print("Data file exists, skipping")
        return

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    urllib.request.urlretrieve(SHAKESPEARE_URL, DATA_PATH)

class CharacterTokenizer:
    def __init__(self, text: str):
        self.characters=sorted(list(set(text)))
        self.vocab_size=len(self.characters)
        self.char_to_id = {}
        for index, char in enumerate(self.characters):
            self.char_to_id[char] = index
        self.id_to_char = {}
        for index, char in enumerate(self.characters):
            self.id_to_char[index] = char
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Characters: {self.characters}")

    def encode(self, text: str) -> list:
        ids = []
        for char in text:
            ids.append(self.char_to_id[char])
        return ids

if __name__ == "__main__":
    download_shakespeare()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharacterTokenizer(text)
    print(tokenizer.encode("hello"))
    #[46, 43, 50, 50, 53]

