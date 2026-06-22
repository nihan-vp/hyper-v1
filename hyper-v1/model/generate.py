import json
import torch
from hyper_model import HyperV1

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 256

with open("tokenizer/tokenizer.json", "r", encoding="utf-8") as f:
    tok = json.load(f)

stoi = tok["stoi"]
itos = {int(k): v for k, v in tok["itos"].items()}
vocab_size = tok["vocab_size"]

model = HyperV1(vocab_size=vocab_size, block_size=block_size).to(device)
model.load_state_dict(torch.load("checkpoints/hyper-v1-tiny.pt", map_location=device))
model.eval()

def encode(text):
    return [stoi[c] for c in text if c in stoi]

def decode(tokens):
    return "".join([itos[i] for i in tokens])

@torch.no_grad()
def generate(prompt, max_new_tokens=300):
    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return decode(idx[0].tolist())

prompt = input("You: ")
print(generate("User: " + prompt + "\nAssistant: "))