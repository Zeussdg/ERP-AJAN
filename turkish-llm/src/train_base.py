"""
Faz 3: gercek boyutlu base pretraining.

src/train.py'nin (Faz 0 prototipi, GTX 1650/4GB icin 4.2M parametre)
buyutulmus hali. Bu script SADECE Google Colab'da, Drive mount edilmis
haldeyken calisacak sekilde yazildi (DRIVE_ROOT asagida) - T4'un 15GB
VRAM'i ve 520MB'lik gercek Wikipedia+ERP korpusu icin.

Faz 0'a gore farklar:
  - Model: ~29M parametre (4.2M yerine), context 256 (128 yerine)
  - Karisik hassasiyet (fp16 autocast + GradScaler) - T4'te hem daha
    hizli hem daha az VRAM kullanir
  - Warmup + cosine learning rate cizelgesi (sabit LR yerine)
  - Gradient clipping (buyuk ölçekte egitim kararliligi icin)
  - Checkpoint dogrudan Drive'a kaydedilir - Colab runtime sifirlansa
    bile egitilen agirliklar kaybolmaz
  - Kesinti sonrasi kaldigi yerden devam edebilir: model_base_latest.pt
    her eval'da (val loss iyilesmese bile) yazilir ve script yeniden
    calistirildiginda otomatik yuklenir; model_base_best.pt ise sadece
    val loss iyilestiginde guncellenen, Faz 4'te kullanilacak asil model
"""

import math
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model import GPT, GPTConfig

# Drive'da "Upload to Colab" sirasinda data/erp/ -> duz "erp", D:'deki
# wikipedia/processed -> "wikipedia/processed" olarak yuklendi; yollar
# buna gore ayarlandi. Kendi Drive duzenin farkliysa burayi guncelle.
DRIVE_ROOT = Path("/content/drive/MyDrive/turkish-llm")
TOKENIZER_PATH = DRIVE_ROOT / "tokenizer" / "tokenizer.json"
WIKI_DIR = DRIVE_ROOT / "wikipedia" / "processed"
ERP_DIR = DRIVE_ROOT / "erp"
# Ikisi ayri tutulur: LATEST her eval'da (kesintiden sonra devam etmek
# icin, val loss iyilesmese bile), BEST sadece val loss iyilestiginde
# (Faz 4'te kullanilacak asil model) kaydedilir. Sadece BEST'e guvenmek,
# val loss uzun sure iyilesmezse resume'un cok geriden baslamasina yol
# acabilirdi.
LATEST_CHECKPOINT_PATH = DRIVE_ROOT / "checkpoints" / "model_base_latest.pt"
BEST_CHECKPOINT_PATH = DRIVE_ROOT / "checkpoints" / "model_base_best.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"

# Model (Faz 0'daki 4.2M yerine ~29M parametre)
MODEL_CONFIG_KWARGS = dict(
    block_size=256,
    n_layer=8,
    n_head=8,
    n_embd=512,
    dropout=0.1,
)

# Egitim ayarlari
BATCH_SIZE = 128
MAX_ITERS = 20000
WARMUP_ITERS = 500
EVAL_INTERVAL = 500
EVAL_ITERS = 50
MAX_LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
VAL_SPLIT = 0.02
# Tokenizer egitimindeki 25x'ten kasten daha dusuk - burada amac ERP
# terimlerinin gomme (embedding) vektorlerine hafif bir agirlik katmak,
# base modelin davranisini ERP'ye kilitlemek degil (o Faz 4'un isi).
ERP_REPEAT_IN_TRAINING = 5
SEED = 1337


def load_corpus_ids(tokenizer):
    wiki_files = sorted(WIKI_DIR.glob("*.txt"))
    erp_files = sorted(ERP_DIR.glob("*.txt"))

    if not wiki_files:
        raise SystemExit(f"{WIKI_DIR} altinda .txt dosyasi bulunamadi.")
    if not erp_files:
        raise SystemExit(f"{ERP_DIR} altinda .txt dosyasi bulunamadi.")

    # uint16 kullanilir (vocab_size 8000, 65536'nin altinda) - 130M+
    # token'lik bir diziyi duz Python int listesi olarak tutmak Colab'in
    # bellegini gereksiz yere sisirir, numpy array cok daha verimli.
    arrays = []

    for path in wiki_files:
        print(f"Tokenize ediliyor: {path.name}")
        text = path.read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        arrays.append(np.array(ids, dtype=np.uint16))

    erp_text = "\n".join(path.read_text(encoding="utf-8") for path in erp_files)
    erp_ids = np.array(tokenizer.encode(erp_text).ids, dtype=np.uint16)

    for _ in range(ERP_REPEAT_IN_TRAINING):
        arrays.append(erp_ids)

    all_ids = np.concatenate(arrays)
    return torch.from_numpy(all_ids.astype(np.int64))


def get_batch(data, block_size, batch_size, device):
    max_start = len(data) - block_size - 1
    start_indices = torch.randint(0, max_start, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in start_indices])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in start_indices])

    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def get_lr(step):
    if step < WARMUP_ITERS:
        return MAX_LR * (step + 1) / WARMUP_ITERS

    if step >= MAX_ITERS:
        return MIN_LR

    decay_ratio = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return MIN_LR + coeff * (MAX_LR - MIN_LR)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device):
    model.eval()
    results = {}

    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(EVAL_ITERS)

        for i in range(EVAL_ITERS):
            x, y = get_batch(data, block_size, batch_size, device)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
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

    # Daha once kaydedilmis bir checkpoint varsa (kesinti/yeniden baslatma
    # sonrasi), sifirdan yeni bir model kurmak yerine oradan devam edilir.
    # Checkpoint'teki config kullanilir (MODEL_CONFIG_KWARGS sonradan
    # degismis olsa bile mimari uyumsuzlugu onlemek icin).
    resume = LATEST_CHECKPOINT_PATH.exists()

    if resume:
        checkpoint = torch.load(LATEST_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        config = checkpoint["config"]
    else:
        config = GPTConfig(vocab_size=tokenizer.get_vocab_size(), **MODEL_CONFIG_KWARGS)

    model = GPT(config).to(DEVICE)
    print(f"Parametre sayisi: {model.num_parameters():,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    if resume:
        model.load_state_dict(checkpoint["model_state_dict"])

        # Eski (optimizer/scaler durumu kaydetmeyen) checkpoint'lerle de
        # geriye donuk uyumlu olmak icin .get() kullanilir - yoksa
        # optimizer/scaler sifirdan baslar, sadece model agirliklari ve
        # adim sayaci devam eder.
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print("Uyari: checkpoint'te optimizer durumu yok, optimizer sifirdan basliyor.")

        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_step = checkpoint["step"] + 1
        best_val_loss = checkpoint["val_loss"]
        print(f"Checkpoint'ten devam ediliyor: adim {checkpoint['step']}, val loss {best_val_loss:.4f}")
    else:
        start_step = 1
        best_val_loss = float("inf")

    LATEST_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    for step in range(start_step, MAX_ITERS + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, config.block_size, BATCH_SIZE, DEVICE)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS:
            losses = estimate_loss(model, train_data, val_data, config.block_size, BATCH_SIZE, DEVICE)
            elapsed_min = (time.time() - start_time) / 60

            print(
                f"adim {step:6d}/{MAX_ITERS} | lr {lr:.2e} | "
                f"train loss {losses['train']:.4f} | val loss {losses['val']:.4f} | "
                f"{elapsed_min:.1f} dk"
            )

            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": config,
                "step": step,
                "val_loss": losses["val"],
            }

            # LATEST her eval'da yazilir - kesinti sonrasi resume icin.
            torch.save(checkpoint_payload, LATEST_CHECKPOINT_PATH)

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                checkpoint_payload["val_loss"] = best_val_loss
                torch.save(checkpoint_payload, BEST_CHECKPOINT_PATH)
                print(f"  -> yeni en iyi checkpoint kaydedildi ({BEST_CHECKPOINT_PATH})")

    print(f"\nEgitim bitti. En iyi val loss: {best_val_loss:.4f}")
    print(f"En iyi checkpoint: {BEST_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
