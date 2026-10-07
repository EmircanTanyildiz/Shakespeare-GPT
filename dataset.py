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

if __name__ == "__main__":
    download_shakespeare()