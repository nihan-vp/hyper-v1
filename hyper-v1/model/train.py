import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from hyper_model import HyperV1

device = "cuda" if torch.cuda.is_available() else "cpu"

block_size = 256
batch_size = 8
epochs = 50
lr = 3e-4

with open("tokenizer/tokenizer.json", "r", encoding="utf-8") as f:
    tok = json.load(f)

stoi = tok["stoi"]
itos = {int(k): v for k, v in tok["itos"].items()}
vocab_size = tok["vocab_size"]

text = open("data/train.txt", "r", encoding="utf-8").read()
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

class TextDataset(Dataset):
    def __len__(self):
        return len(data) - block_size

    def __getitem__(self, idx):
        x = data[idx:idx + block_size]
        y = data[idx + 1:idx + block_size + 1]
        return x, y

loader = DataLoader(TextDataset(), batch_size=batch_size, shuffle=True)

model = HyperV1(
    vocab_size=vocab_size,
    n_embd=256,
    n_head=4,
    n_layer=4,
    block_size=block_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for x, y in tqdm(loader):
        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1} Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "checkpoints/hyper-v1-tiny.pt")

print("Training complete.")