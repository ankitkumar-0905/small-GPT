
import torch
import torch.nn as nn
import torch.optim as optim

from tokenizer import encode
from model import SmallGPT


# =========================
# 1. Dataset
# =========================

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

data = torch.tensor(
    encode(text),
    dtype=torch.long
)


# =========================
# 2. Train / Validation Split
# =========================

split = int(0.9 * len(data))

train_data = data[:split]
val_data = data[split:]


# =========================
# 3. Configuration
# =========================

block_size = 64
batch_size = 4


# =========================
# 4. Get Batch
# =========================

def get_batch(data):

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size]
        for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y


# =========================
# 5. Model
# =========================

model = SmallGPT()


# =========================
# 6. Loss + Optimizer
# =========================

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001
)


# =========================
# 7. Estimate Loss
# =========================

@torch.no_grad()
def estimate_loss(data, eval_iters=10):

    model.eval()

    total_loss = 0

    for _ in range(eval_iters):

        x, y = get_batch(data)

        logits = model(x)

        loss = loss_fn(
            logits.view(-1, logits.size(-1)),
            y.view(-1)
        )

        total_loss += loss.item()

    model.train()

    return total_loss / eval_iters


# =========================
# 8. Training
# =========================

for step in range(10000):

    x, y = get_batch(train_data)

    logits = model(x)

    loss = loss_fn(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()


    # =========================
    # 9. Loss Check
    # =========================

    if step % 200 == 0:

        train_loss = estimate_loss(train_data)
        val_loss = estimate_loss(val_data)

        print(
            f"Step: {step} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )


# =========================
# 10. Save Model
# =========================

torch.save(
    model.state_dict(),
    "small_gpt.pth"
)

print("Model saved successfully!")

