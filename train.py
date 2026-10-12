import os
import time
import torch
from model import GPT
from dataset import load_data, getBatch
EMBEDDING_DIM = 384
num_of_heads=12
num_of_layers=6
block_size=256
DROPOUT=0.1
BATCH_SIZE=64
MAX_ITERS=5000
EVAL_INTERVAL=500
LEARNING_RATE=3e-4
WARMUP_ITERS=100
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints/model.pt"


def get_learning_rate(iteration):
    if iteration <WARMUP_ITERS:
        return LEARNING_RATE * (iteration / WARMUP_ITERS)
    else:
        return LEARNING_RATE

