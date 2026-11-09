"""
wikiextractor'in urettigi ham ciktiyi (<doc id="" url="" title="">...</doc>
bloklari icinde dagitilmis cok sayida kucuk dosya) temizleyip, egitim
scriptlerinin (tokenizer/train_tokenizer.py, src/train.py) dogrudan
okuyabilecegi birkac buyuk .txt dosyasina (shard) donusturur.

Kullanim:
    python data/clean_wikipedia_extract.py <wikiextractor_cikti_klasoru> <hedef_klasor>

Cikti formati fetch_sample.py ile tutarli tutuldu ("# Baslik\n\nmetin\n\n"),
boylece iki kaynak (kucuk prototip ornegi + tam Wikipedia) ayni sekilde
islenebiliyor.
"""

import re
import sys
from pathlib import Path

DOC_OPEN_RE = re.compile(r'<doc id="\d+" url="[^"]*" title="([^"]*)">')
DOC_CLOSE = "</doc>"
SHARD_SIZE_BYTES = 50_000_000  # ~50MB'lik parcalar halinde kaydet
MIN_ARTICLE_CHARS = 200


def iter_articles(extracted_dir: Path):
    for file_path in sorted(extracted_dir.rglob("wiki_*")):
        text = file_path.read_text(encoding="utf-8")

        title = None
        buffer_lines = []

        for line in text.split("\n"):
            match = DOC_OPEN_RE.match(line)

            if match:
                title = match.group(1)
                buffer_lines = []
                continue

            if line.strip() == DOC_CLOSE:
                body = "\n".join(buffer_lines).strip()

                if title and len(body) >= MIN_ARTICLE_CHARS:
                    yield title, body

                title = None
                continue

            if title is not None:
                buffer_lines.append(line)


def main():
    if len(sys.argv) != 3:
        raise SystemExit(
            "Kullanim: python data/clean_wikipedia_extract.py <wikiextractor_cikti> <hedef_klasor>"
        )

    extracted_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    shard_index = 0
    shard_bytes = 0
    article_count = 0

    out_file = open(output_dir / f"wiki_tr_{shard_index:03d}.txt", "w", encoding="utf-8")

    try:
        for title, body in iter_articles(extracted_dir):
            block = f"# {title}\n\n{body}\n\n"
            block_bytes = len(block.encode("utf-8"))

            if shard_bytes + block_bytes > SHARD_SIZE_BYTES:
                out_file.close()
                shard_index += 1
                shard_bytes = 0
                out_file = open(output_dir / f"wiki_tr_{shard_index:03d}.txt", "w", encoding="utf-8")

            out_file.write(block)
            shard_bytes += block_bytes
            article_count += 1

            if article_count % 20000 == 0:
                print(f"{article_count:,} makale islendi...")
    finally:
        out_file.close()

    print(f"Bitti: {article_count:,} makale, {shard_index + 1} shard dosyasi -> {output_dir}")


if __name__ == "__main__":
    main()
