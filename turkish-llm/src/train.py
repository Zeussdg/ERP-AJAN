"""
Faz 0 prototipi icin egitim dongusu.

Akis: tokenizer ile korpusu token id'lerine cevirir, train/val olarak
ayirir, GPT modelini (model.py) bu veri uzerinde egitir ve checkpoint
kaydeder. Veri kucuk oldugu icin (yaklasik birkac on bin token) tamamen
bellekte tutuluyor - buyuk corpus'larla (Faz 2/3) calisirken bu kisim
disk tabanli, parca parca okuyan bir versiyona donusecek.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

PROJECT_ROOT = Path(__file__).parent.parent
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Egitim ayarlari
BATCH_SIZE = 64
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 50          # her degerlendirmede ortalamasi alinacak batch sayisi
LEARNING_RATE = 3e-4
VAL_SPLIT = 0.1
SEED = 1337


def load_corpus_ids(tokenizer):
    files = sorted(DATA_DIR.glob("*.txt"))

    if not files:
        raise SystemExit(f"{DATA_DIR} altinda .txt dosyasi bulunamadi.")

    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    ids = tokenizer.encode(text).ids

    return torch.tensor(ids, dtype=torch.long)


def get_batch(data, block_size, batch_size, device):
    # Veri uzunlugu icinde rastgele block_size+1 uzunlugunda pencereler
    # secilir; x girdi, y bir sonraki token'i tahmin etmesi gereken hedef
    # (x'in sagdan bir kaydirilmis hali).
    max_start = len(data) - block_size - 1
    start_indices = torch.randint(0, max_start, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in start_indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in start_indices])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device):
    model.eval()
    results = {}

    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)

        for i in range(EVAL_ITERS):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()

        results[split_name] = losses.mean().item()

    model.train()
    return results


def main():
    torch.manual_seed(SEED)

    print(f"Cihaz: {DEVICE}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    data = load_corpus_ids(tokenizer)
    print(f"Toplam token sayisi: {len(data):,}")

    split_index = int(len(data) * (1 - VAL_SPLIT))
    train_data = data[:split_index]
    val_data = data[split_index:]
    print(f"Train: {len(train_data):,} token | Val: {len(val_data):,} token")

    config = GPTConfig(vocab_size=tokenizer.get_vocab_size())
    model = GPT(config).to(DEVICE)
    print(f"Parametre sayisi: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for step in range(1, MAX_ITERS + 1):
        x, y = get_batch(train_data, config.block_size, BATCH_SIZE, DEVICE)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data, config.block_size, BATCH_SIZE, DEVICE)
            print(
                f"adim {step:5d}/{MAX_ITERS} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "step": step,
                        "val_loss": best_val_loss,
                    },
                    CHECKPOINT_PATH,
                )

    print(f"\nEgitim bitti. En iyi val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
