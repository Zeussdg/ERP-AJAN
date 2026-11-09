"""
Faz 0 prototipi icin orneklem/inference scripti.

checkpoints/model.pt'yi yukler, verilen bir baslangic metninden (prompt)
itibaren modelin otomatik olarak token uretmesini saglar ve sonucu
insan tarafindan okunabilir metne geri cevirir.

Bu asamada anlamli/dogru Turkce cumleler beklenmiyor - amac, egitilmis
agirliklarin gercekten yuklenip modelin uctan uca (tokenizer -> model ->
tokenizer) calisip calismadigini gormek.
"""

from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import GPT

PROJECT_ROOT = Path(__file__).parent.parent
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "model.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = "Türkiye"
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 40


def main():
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    config = checkpoint["config"]

    model = GPT(config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Cihaz: {DEVICE}")
    print(f"Checkpoint adimi: {checkpoint['step']} | val loss: {checkpoint['val_loss']:.4f}")

    prompt_ids = tokenizer.encode(PROMPT).ids
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_k=TOP_K,
    )

    generated_text = tokenizer.decode(output_ids[0].tolist())

    print("\n--- Uretilen metin ---")
    print(generated_text)


if __name__ == "__main__":
    main()
