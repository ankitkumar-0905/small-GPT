import torch

from tokenizer import encode, decode
from model import SmallGPT


# =========================
# 1. Model load
# =========================

model = SmallGPT()

model.load_state_dict(
    torch.load("small_gpt.pth")
)

model.eval()


# =========================
# 2. Starting text
# =========================

start_text = input("Enter your prompt: ")

context = torch.tensor(
    [encode(start_text)],
    dtype=torch.long
)


# =========================
# 3. Text Generation
# =========================

for _ in range(100):

    # Last 8 tokens hi model ko denge
    context_input = context[:, -64:]

    # Prediction
    logits = model(context_input)

    # Sirf last token ki prediction
    logits = logits[:, -1, :]

    # Probabilities
    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    # Next token select
    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )

    # Context me add
    context = torch.cat(
        [context, next_token],
        dim=1
    )


# =========================
# 4. Decode
# =========================

generated_text = decode(
    context[0].tolist()
)

print("\n-------------------Generated Text:---------------\n")
print(generated_text)