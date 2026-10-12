

import torch
from model import GPT
from dataset import download_shakespeare, CharacterTokenizer, DATA_PATH



def load_model(checkpoint_path: str = "checkpoints/model.pt"):

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']

    download_shakespeare()
    with open(DATA_PATH, 'r', encoding='utf-8') as file:
        text = file.read()
    tokenizer = CharacterTokenizer(text)

    model = GPT(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        block_size=config['block_size'],
        dropout=0.0  # No dropout during generation
    )

    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    print(f"Model loaded!")
    print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  Training iterations: {checkpoint['iteration']}")

    return model, tokenizer, device


@torch.no_grad()
def generate(model, tokenizer, device, prompt: str, max_tokens: int = 500, temperature: float = 0.8):
    """
    Generate text given a starting prompt.

    -------------------------------------------------------------------------
    HOW TEXT GENERATION WORKS (Autoregressive)
    -------------------------------------------------------------------------
    The model generates one character at a time:

    1. Start with prompt:           "ROMEO:"
    2. Model predicts next char:    "ROMEO:" → 'O'
    3. Append and repeat:           "ROMEO:O" → ' '
    4. Continue:                    "ROMEO:O " → 'w'
    5. And so on...                 "ROMEO:O w" → 'h'

    This continues until we reach max_tokens.

    TEMPERATURE controls randomness:
    - Low (0.5):  Very predictable, "safe" choices
    - Medium (0.8): Balanced creativity and coherence (default)
    - High (1.5):  Very creative but may be nonsensical
    -------------------------------------------------------------------------
    """
    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = input_ids.unsqueeze(0)  # Add batch dimension: shape becomes (1, seq_len)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature
    )

    generated_text = tokenizer.decode(output_ids[0])

    return generated_text




if __name__ == "__main__":

    print("=" * 60)
    print("Shakespeare GPT - Text Generation")
    print("=" * 60)

    model, tokenizer, device = load_model("checkpoints/model.pt")

    prompts = [
        "ROMEO:",
        "To be, or not to be",
        "JULIET:\nO Romeo, ",
    ]

    print("\n" + "=" * 60)
    print("Generating Text...")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n--- Prompt: {repr(prompt)} ---\n")

        generated_text = generate(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_tokens=300,
            temperature=0.8
        )

        print(generated_text)
        print("\n" + "-" * 60)

    print("\n" + "=" * 60)
    print("Try Your Own Prompts!")
    print("=" * 60)
    print("Enter a prompt to generate text, or 'quit' to exit.")
    print("Tip: Try character names like 'HAMLET:', 'KING:', 'First Citizen:'")

    while True:
        try:
            prompt = input("\nYour prompt: ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Farewell!")
                break

            if not prompt:
                continue

            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_tokens=500,
                temperature=0.8
            )

            print("\n" + generated_text)

        except KeyboardInterrupt:
            print("\n\nFarewell!")
            break