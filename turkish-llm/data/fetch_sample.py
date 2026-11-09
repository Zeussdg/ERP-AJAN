"""
Faz 0 prototipi icin kucuk bir Turkce metin korpusu hazirlar.

Turkce Wikipedia'dan rastgele makalelerin duz metin ozetlerini (extract)
MediaWiki API'si uzerinden ceker ve data/raw/wiki_tr_sample.txt dosyasina
yazar. Amac: tokenizer + model + egitim donguisunun uctan uca calistigini
dogrulamak icin yeterli hacimde, gercek Turkce metin saglamak.

Faz 2'de (asil base pretraining verisi) ayni mantik çok daha buyuk
olcekte, tam Wikipedia dump'i uzerinden tekrarlanacak.
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://tr.wikipedia.org/w/api.php"
OUTPUT_PATH = Path(__file__).parent / "raw" / "wiki_tr_sample.txt"
TARGET_BYTES = 250_000  # Faz 0 prototipi icin yeterli; asil corpus Faz 2'de
BATCH_SIZE = 20
MIN_ARTICLE_CHARS = 200  # cok kisa (taslak) maddeleri ele
REQUEST_DELAY_SECONDS = 2.0
MAX_RETRIES = 5


def fetch_random_batch():
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(BATCH_SIZE),
        "prop": "extracts",
        "explaintext": "1",
        "exlimit": "max",
        "format": "json",
        "formatversion": "2",
    }

    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "turkish-llm-dataprep/0.1 (local research project)"},
    )

    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload.get("query", {}).get("pages", [])
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < MAX_RETRIES - 1:
                wait_seconds = 5 * (attempt + 1)
                print(f"429 alindi, {wait_seconds}s bekleniyor...")
                time.sleep(wait_seconds)
                continue
            raise

    return []


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_titles = set()
    total_bytes = 0
    article_count = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_file:
        while total_bytes < TARGET_BYTES:
            pages = fetch_random_batch()

            for page in pages:
                title = page.get("title", "")
                extract = page.get("extract", "").strip()

                if title in seen_titles or len(extract) < MIN_ARTICLE_CHARS:
                    continue

                seen_titles.add(title)

                block = f"# {title}\n\n{extract}\n\n"
                out_file.write(block)

                total_bytes += len(block.encode("utf-8"))
                article_count += 1

                if total_bytes >= TARGET_BYTES:
                    break

            print(f"{article_count} makale, {total_bytes / 1000:.0f} KB toplandi...")
            time.sleep(REQUEST_DELAY_SECONDS)

    print(f"Bitti: {article_count} makale, {total_bytes / 1000:.0f} KB -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
