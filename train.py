import os
import time
import torch
from model import GPT
from dataset import load_data, getBatch, CharacterTokenizer
EMBEDDING_DIM = 384
num_of_heads=6
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
@torch.no_grad()
def evaluate(model, train_data, test_data):
    model.eval()
    result={}
    for name, data in [("train",train_data), ("test",test_data)]:
        total_loss = 0.0
        for _ in range(100):
            x, y = getBatch(data, block_size=block_size, batch_size=BATCH_SIZE)
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, loss = model(x,y)
            total_loss+=loss.item()
        result[name]=total_loss/100
    model.train()
    return result
@torch.no_grad()
def generate_sample(model, tokenizer):
    model.eval()
    prompt="ROMEO: "
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    output_ids = model.generate(input_ids,200, 0.7)
    model.train()
    return tokenizer.decode(output_ids[0].tolist())

def train():
    train_data, test_data, tokenizer = load_data()
    vocab_size = tokenizer.vocab_size

    model = GPT(vocab_size=vocab_size,
                 embedding_dim= EMBEDDING_DIM,
                 num_heads= num_of_heads,
                 num_layers= num_of_layers,
                 dropout= DROPOUT,
                 block_size = block_size)
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    os.makedirs("checkpoints", exist_ok=True)
    start_time = time.time()
    for iteration in range(MAX_ITERS):
        lr = get_learning_rate(iteration)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            x, y = getBatch(train_data, block_size=block_size, batch_size=BATCH_SIZE)
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, loss = model(x,y)
            optimizer.zero_grad()
            loss.backward()
            #gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS -1:
                losses = evaluate(model, train_data, test_data)
                elapsed = time.time() - start_time
                print(f"Iter: {iteration}, Train Loss: {losses['train']}, Test Loss: {losses['test']}, learning rate: {lr} Elapsed: {elapsed}")
                if iteration > 0:
                    print(generate_sample(model, tokenizer))
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "iteration": MAX_ITERS,
        "test_loss" : losses["test"],
        "config": {
            "vocab_size": vocab_size,
            "embedding_dim": EMBEDDING_DIM,
            "num_heads": num_of_heads,
            "num_layers": num_of_layers,
            "dropout": DROPOUT,
            "block_size": block_size
        }

    }

    torch.save(checkpoint, CHECKPOINT_DIR)
    total_time = time.time() - start_time
    print(f"Total time: {total_time}")
    print(f"Final Test loss: {losses['test']}")
    print(f"Model saved to: {CHECKPOINT_DIR}")





if __name__ == "__main__":
    train()
