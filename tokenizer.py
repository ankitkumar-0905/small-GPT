import torch

# 1. Training data read karo
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Saare unique characters nikalo
chars = sorted(list(set(text)))

# 3. Character → ID
stoi = {ch: i for i, ch in enumerate(chars)}

# 4. ID → Character
itos = {i: ch for i, ch in enumerate(chars)}

# 5. Encode function
def encode(text):
    return [stoi[ch] for ch in text]

# 6. Decode function
def decode(ids):
    return "".join(itos[i] for i in ids)

# Test
encoded = encode("Hello")
print("Encoded:", encoded)

decoded = decode(encoded)
print("Decoded:", decoded)