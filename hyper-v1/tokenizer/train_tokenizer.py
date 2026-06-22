import json

text = open("data/train.txt", "r", encoding="utf-8").read()

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

tokenizer = {
    "stoi": stoi,
    "itos": itos,
    "vocab_size": len(chars)
}

with open("tokenizer/tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer, f, ensure_ascii=False, indent=2)

print("Vocab size:", len(chars))