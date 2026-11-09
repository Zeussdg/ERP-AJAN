"""
Faz 4 icin sentetik "soru -> tool + argumanlar" veri setini uretir.

tools.py'deki TOOLS kataloguna gore her tool icin ornekler toplanir,
karistirilir, train/val olarak ayrilir ve JSONL olarak kaydedilir:
    {"text": "...", "tool": "get_customer_balance" | null, "arguments": {...}}

Cikti: data/dialogue/train.jsonl, data/dialogue/val.jsonl
"""

import json
import random
from pathlib import Path

from tools import TOOLS

OUTPUT_DIR = Path(__file__).parent
VAL_SPLIT = 0.1
SEED = 1337

# Her tool'un nihai ornek sayisi bu sayiyi gecmeyecek sekilde kirpilir -
# az parametreli tool'lar (get_low_stock gibi) asagida cogaltma ile bu
# sayiya yaklasir, cok parametreli tool'lar (create_invoice gibi) zaten
# dogal olarak bu civarda kaliyor. Amac tool'lar arasi dengesizligi
# azaltmak, hepsini birbirine esitlemek degil.
MAX_PER_TOOL_FINAL = 200


def _augment(text):
    """Ayni niyeti farkli yuzeysel ifadelerle cogaltir - anlam/argumanlar
    degismez, sadece soyleyis bicimi degisir. Modelin tek bir kalip
    cumleyi ezberlemek yerine niyeti genellemesine yardimci olur."""

    stripped = text.rstrip("?").rstrip()

    # Onemli: set() KULLANMA - Python'da string hash'i process basina
    # rastgele (PYTHONHASHSEED), set sirasi calistirmadan calistirmaya
    # degisir. Bu, ayni SEED'le bile farkli makinelerde (or. yerel vs
    # Colab) farkli veri seti uretilmesine yol acardi. Sira korunan,
    # elle dedup edilen bir liste kullanilir - tam deterministik.
    candidates = [
        text,
        f"Lütfen {stripped}?",
        f"{stripped} bakar mısın?",
        f"{stripped}?",
    ]

    seen = set()
    unique_variants = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_variants.append(candidate)

    return unique_variants


def build_examples():
    rng = random.Random(SEED)
    examples = []

    for tool_key, generator in TOOLS.items():
        tool_name = None if tool_key == "none" else tool_key

        tool_examples = []
        for text, arguments in generator():
            for variant_text in _augment(text):
                tool_examples.append(
                    {
                        "text": variant_text,
                        "tool": tool_name,
                        "arguments": arguments or {},
                    }
                )

        if len(tool_examples) > MAX_PER_TOOL_FINAL:
            tool_examples = rng.sample(tool_examples, MAX_PER_TOOL_FINAL)

        examples.extend(tool_examples)

    return examples


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)

    examples = build_examples()
    rng.shuffle(examples)

    split_index = int(len(examples) * (1 - VAL_SPLIT))
    train_rows = examples[:split_index]
    val_rows = examples[split_index:]

    write_jsonl(OUTPUT_DIR / "train.jsonl", train_rows)
    write_jsonl(OUTPUT_DIR / "val.jsonl", val_rows)

    print(f"Toplam ornek: {len(examples):,}")
    print(f"Train: {len(train_rows):,} | Val: {len(val_rows):,}")

    print("\nTool basina ornek sayisi:")
    counts = {}
    for row in examples:
        key = row["tool"] or "none"
        counts[key] = counts.get(key, 0) + 1
    for key, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {key:28s} {count:5d}")

    print("\n--- Ornek satirlar ---")
    for row in examples[:5]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
