"""
Faz 1: gercek boyutlu byte-level BPE tokenizer egitimi.

Iki kaynak birlikte kullanilir:
  1) Genel Turkce Wikipedia korpusu (D:/turkish-llm-data/wikipedia/processed) -
     modelin dogru Turkce dilbilgisini/kelime dagarcigini ogrenmesi icin.
  2) data/erp/ altindaki kucuk, elle hazirlanmis ERP/is asistani alani metni -
     hedef kullanim (ERP asistani) icin onemli terimlerin (fatura, stok,
     cari, siparis, KDV, vade...) tokenizer'da yeterince temsil edilmesini
     saglamak icin ERP_REPEAT kadar tekrarlanarak agirligi artirilir.

Vocab boyutu bilincli olarak mutevazi tutuldu (8000) - amac genel amacli
dev bir tokenizer degil, hedeflenen ERP/is asistani kullanim alanina
uygun, gereksiz buyutulmemis bir tokenizer.
"""

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PROJECT_ROOT = Path(__file__).parent.parent
WIKI_DIR = Path("D:/turkish-llm-data/wikipedia/processed")
ERP_DIR = PROJECT_ROOT / "data" / "erp"
OUTPUT_PATH = Path(__file__).parent / "tokenizer.json"

VOCAB_SIZE = 8000
MIN_FREQUENCY = 2
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]

# ERP metni Wikipedia'ya kiyasla cok kucuk (KB'lar vs yuzlerce MB) oldugu
# icin, tekrarlanmadan trainer'a verilirse ERP terimleri yeterince sik
# gecmeyip diger her sey gibi genel alt-parcalara bolunur. Ayni dosyalari
# birden fazla kez listeye eklemek (gercek veri kopyalamadan) bu terimlerin
# ogrenilen BPE birlestirmelerinde daha fazla agirlik kazanmasini saglar.
ERP_REPEAT = 25

# ERP terminolojisinin gercekten "tek/az token'a" sikisip sikismadigini
# gormek icin egitim sonunda test edilecek ornek kelimeler/ifadeler.
ERP_CHECK_SAMPLES = [
    "fatura", "stok", "cari hesap", "sipariş", "tedarikçi",
    "vade tarihi", "KDV oranı", "birim fiyat", "kritik stok seviyesi",
]


def collect_training_files():
    wiki_files = sorted(str(path) for path in WIKI_DIR.glob("*.txt"))
    erp_files = sorted(str(path) for path in ERP_DIR.glob("*.txt"))

    if not wiki_files:
        raise SystemExit(f"{WIKI_DIR} altinda .txt dosyasi bulunamadi.")
    if not erp_files:
        raise SystemExit(f"{ERP_DIR} altinda .txt dosyasi bulunamadi.")

    return wiki_files + erp_files * ERP_REPEAT


def main():
    files = collect_training_files()
    print(f"Toplam egitim dosyasi (ERP tekrarlari dahil): {len(files)}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    tokenizer.train(files=files, trainer=trainer)
    tokenizer.save(str(OUTPUT_PATH))

    print(f"Tokenizer kaydedildi: {OUTPUT_PATH}")
    print(f"Gercek vocab boyutu: {tokenizer.get_vocab_size()}")

    print("\n--- Genel round-trip testi ---")
    sample = "Merhaba, bu Türkçe bir cümledir. Yapay zeka ile ilgileniyorum."
    encoded = tokenizer.encode(sample)
    print(f"Orijinal : {sample}")
    print(f"Token adet: {len(encoded.ids)}")
    print(f"Decode eslesme: {tokenizer.decode(encoded.ids) == sample}")

    print("\n--- ERP terim kapsama testi (kac token'a bolunuyor) ---")
    for term in ERP_CHECK_SAMPLES:
        ids = tokenizer.encode(term).ids
        pieces = [tokenizer.decode([i]) for i in ids]
        print(f"{term!r:30s} -> {len(ids)} token {pieces}")


if __name__ == "__main__":
    main()
