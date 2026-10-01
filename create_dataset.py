from datasets import load_dataset
from pathlib import Path

TARGET_SIZE = 10 * 1024 * 1024  # 10 MB

output_file = Path("data/input.txt")
output_file.parent.mkdir(exist_ok=True)

dataset = load_dataset(
    "common-pile/project_gutenberg_filtered",
    split="train",
    streaming=True
)

total = 0

with open(output_file, "w", encoding="utf-8") as f:
    for row in dataset:
        text = row.get("text", "")

        if not text:
            continue

        remaining = TARGET_SIZE - total

        if len(text.encode("utf-8")) >= remaining:
            text = text.encode("utf-8")[:remaining].decode(
                "utf-8",
                errors="ignore"
            )

        f.write(text)
        f.write("\n")

        total = output_file.stat().st_size

        print(f"Dataset size: {total / (1024 * 1024):.2f} MB")

        if total >= TARGET_SIZE:
            break

print("\nDone!")
print(f"Final size: {total / (1024 * 1024):.2f} MB")
print(f"Saved to: {output_file}")