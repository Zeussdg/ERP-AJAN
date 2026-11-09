"""
Faz 4: gorev/diyalog fine-tuning.

Faz 3'te egitilen base modeli (model_base_best.pt) alir, data/dialogue/
altindaki sentetik "soru -> tool + argumanlar" veri setiyle devam
egitimi yapar. Amac: modelin serbest metin uretmek yerine, dogal dilden
yapisal bir {tool, arguments} JSON'u uretmeyi ogrenmesi.

Format: "Soru: {metin}\nCevap: {json_hedef}<eos>" - kayip (loss) SADECE
"Cevap:" sonrasindaki (hedef) kisimda hesaplanir, prompt kismi maskelenir
(ignore_index=-100) - model soruyu tekrar uretmeyi degil, dogru cevabi
uretmeyi ogrensin diye.

Veri kucuk oldugu icin (Faz 3'un aksine) rastgele pencereleme yerine
klasik epoch tabanli egitim kullanildi - her epoch tum veriyi bir kez
gorur.

SADECE Colab'da, Drive mount edilmis haldeyken calisir.
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT

DRIVE_ROOT = Path("/content/drive/MyDrive/turkish-llm")
TOKENIZER_PATH = DRIVE_ROOT / "tokenizer" / "tokenizer.json"
BASE_CHECKPOINT_PATH = DRIVE_ROOT / "checkpoints" / "model_base_best.pt"
TRAIN_PATH = DRIVE_ROOT / "dialogue" / "train.jsonl"
VAL_PATH = DRIVE_ROOT / "dialogue" / "val.jsonl"
LATEST_CHECKPOINT_PATH = DRIVE_ROOT / "checkpoints" / "model_finetuned_latest.pt"
BEST_CHECKPOINT_PATH = DRIVE_ROOT / "checkpoints" / "model_finetuned_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# Faz 3'un 256'sindan kucuk - bizim soru+cevap ciftlerimiz cok daha kisa
# (max 111 token olcduk, 128 rahat pay birakiyor). Ayni pozisyon
# gomme tablosunun bir alt kumesi kullanilir, sorun cikarmaz.
BLOCK_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 5e-5  # base pretraining'deki (3e-4) LR'den kasten kucuk
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
IGNORE_INDEX = -100
SEED = 1337


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def encode_example(tokenizer, row, pad_id, eos_id):
    prompt = f"Soru: {row['text']}\nCevap: "
    target = json.dumps({"tool": row["tool"], "arguments": row["arguments"]}, ensure_ascii=False)

    prompt_ids = tokenizer.encode(prompt).ids
    target_ids = tokenizer.encode(target).ids + [eos_id]
    full_ids = prompt_ids + target_ids

    # ONEMLI: causal LM'de label_ids[i], input_ids[i]'nin AYNISI degil,
    # BIR SONRAKI token'i (input_ids[i+1]) olmali - model.py/train_base.py
    # ile ayni "shift" konvansiyonu. Bu kaydirma yapilmazsa model "gordugun
    # son token neyse onu tekrar et" gibi anlamsiz ama loss'u yapay olarak
    # sifira indiren bir kisayol ogrenir (ilk denemede tam da bu oldu).
    input_ids = full_ids[:-1]
    label_ids = full_ids[1:]

    # Prompt kismindaki "bir sonraki token" tahminleri maskelenir - model
    # soruyu tekrar uretmeyi degil, sadece "Cevap:"dan sonrasini (hedefi)
    # uretmeyi ogrensin.
    num_prompt_positions = len(prompt_ids) - 1
    label_ids = [IGNORE_INDEX] * num_prompt_positions + label_ids[num_prompt_positions:]

    input_ids = input_ids[:BLOCK_SIZE]
    label_ids = label_ids[:BLOCK_SIZE]

    pad_len = BLOCK_SIZE - len(input_ids)
    input_ids = input_ids + [pad_id] * pad_len
    label_ids = label_ids + [IGNORE_INDEX] * pad_len

    return input_ids, label_ids


def build_tensors(tokenizer, rows, pad_id, eos_id):
    all_inputs = []
    all_labels = []

    for row in rows:
        input_ids, label_ids = encode_example(tokenizer, row, pad_id, eos_id)
        all_inputs.append(input_ids)
        all_labels.append(label_ids)

    return torch.tensor(all_inputs, dtype=torch.long), torch.tensor(all_labels, dtype=torch.long)


def get_batches(inputs, labels, batch_size, shuffle=True):
    n = inputs.size(0)
    indices = torch.randperm(n) if shuffle else torch.arange(n)

    for start in range(0, n, batch_size):
        batch_idx = indices[start:start + batch_size]
        yield inputs[batch_idx], labels[batch_idx]


def compute_loss(model, x, y, device):
    x = x.to(device)
    y = y.to(device)

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    return loss


@torch.no_grad()
def evaluate(model, inputs, labels, device):
    model.eval()
    total_loss = 0.0
    count = 0

    for x, y in get_batches(inputs, labels, BATCH_SIZE, shuffle=False):
        loss = compute_loss(model, x, y, device)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


def main():
    torch.manual_seed(SEED)
    print(f"Cihaz: {DEVICE}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")

    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = load_jsonl(VAL_PATH)
    print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")

    train_inputs, train_labels = build_tensors(tokenizer, train_rows, pad_id, eos_id)
    val_inputs, val_labels = build_tensors(tokenizer, val_rows, pad_id, eos_id)

    resume = LATEST_CHECKPOINT_PATH.exists()

    if resume:
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        config = checkpoint["config"]
    else:
        base_checkpoint = torch.load(BASE_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        config = base_checkpoint["config"]

    model = GPT(config).to(DEVICE)

    if resume:
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"Fine-tuning checkpoint'inden devam ediliyor: epoch {checkpoint['epoch']}, val loss {best_val_loss:.4f}")
    else:
        model.load_state_dict(base_checkpoint["model_state_dict"])
        start_epoch = 1
        best_val_loss = float("inf")
        print(f"Base model yuklendi (val loss {base_checkpoint['val_loss']:.4f}), fine-tuning basliyor.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    if resume:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    LATEST_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        for x, y in get_batches(train_inputs, train_labels, BATCH_SIZE, shuffle=True):
            loss = compute_loss(model, x, y, DEVICE)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

        val_loss = evaluate(model, val_inputs, val_labels, DEVICE)
        elapsed_min = (time.time() - start_time) / 60

        print(f"epoch {epoch:3d}/{EPOCHS} | val loss {val_loss:.4f} | {elapsed_min:.1f} dk")

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
        }

        torch.save(checkpoint_payload, LATEST_CHECKPOINT_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_payload, BEST_CHECKPOINT_PATH)
            print(f"  -> yeni en iyi checkpoint kaydedildi ({BEST_CHECKPOINT_PATH})")

    print(f"\nFine-tuning bitti. En iyi val loss: {best_val_loss:.4f}")
    print(f"En iyi checkpoint: {BEST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
