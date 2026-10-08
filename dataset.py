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
    def decode(self, ids: list) -> str:
        text = []
        for index in ids:
            text.append(self.id_to_char[index])
        return ''.join(text)

def load_data(train_split: float = 0.9):
    download_shakespeare()
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharacterTokenizer(text)
    all_ids = tokenizer.encode(text)
    data = torch.tensor(all_ids, dtype=torch.long)
    split_index=int(train_split * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    print(f"Train Size: {len(train_data)}")
    print(f"Test Size: {len(test_data)}")
    return train_data, test_data, tokenizer





    #print(f"Data: {data.size()}")
    #print(data)
    #print(tokenizer.encode("hello"))
    # [46, 43, 50, 50, 53]
    print(tokenizer.decode([46, 43, 50, 50, 53]))

def getBatch(data: torch.Tensor, block_size: int, batch_size):
    # "to be or not to be"
    # block size-> sequence lenght
    # batch size -> number of sequence
    maxStart = len(data) - block_size - 1

    positions = torch.randint(maxStart, (batch_size, ))
    #print(f"positions: {positions}")
    x_list = []
    y_list = []
    for pos in positions:
        x_list.append(data[pos : pos + block_size])
        y_list.append(data[pos + 1 : pos + block_size + 1])

    x = torch.stack(x_list)
    y = torch.stack(y_list)
    print(x)
    print(y)
    return x, y


if __name__ == "__main__":
    trainData, testData, tokenizer = load_data()
    x,y = getBatch(trainData, block_size=256, batch_size=5)
    print(x.shape, y.shape)





