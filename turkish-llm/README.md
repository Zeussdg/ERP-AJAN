# Turkish LLM (R&D)

Sıfırdan (mimari + ağırlıklar kendi kodumuz ve kendi eğitimimizle, hiçbir dış AI API'si kullanmadan) inşa edilen, küçük-orta ölçekli, Türkçe ve ERP/iş asistanı alanına özel bir dil modeli.

klovo-ai projesinden tamamen bağımsızdır.

## Donanım

- Yerel: NVIDIA GTX 1650 (4GB VRAM) — mimari geliştirme, hata ayıklama ve küçük prototip eğitimleri için.
- Bulut: gerektiğinde kiralık GPU (RunPod/Vast.ai/Lambda vb.) — asıl temel model (base) pretraining'i için.

## Yaklaşım: iki aşamalı

1. **Base pretraining** — genel bir Türkçe metin korpusuyla (örn. Türkçe Wikipedia + ek açık kaynaklar), küçük-orta ölçekli (~30-125M parametre) bir GPT tarzı transformer'ı sıfırdan eğitmek. Amaç modele Türkçe dilbilgisini ve genel dünya bilgisini öğretmek. Bu aşama bulut GPU gerektirir.
2. **Fine-tuning (alan özelleştirme)** — base model Türkçeyi öğrendikten sonra, ERP/iş asistanı alanına özel veri (cari, stok, fatura senaryoları, örnek diyaloglar) ile özelleştirmek. Bu aşama GTX 1650'de bile yapılabilir.

## Veri konumu

Büyük veri C sürücüsünde yer olmadığı için **D sürücüsünde** tutuluyor (proje reposunun dışında, git'e girmiyor):
- `D:\turkish-llm-data\wikipedia\raw\` — ham Wikipedia dump'ı (.xml.bz2)
- `D:\turkish-llm-data\wikipedia\extracted\` — wikiextractor çıktısı (ham, `<doc>` etiketli)
- `D:\turkish-llm-data\wikipedia\processed\` — temizlenmiş, eğitime hazır shard dosyaları (`wiki_tr_000.txt` ... `wiki_tr_010.txt`, ~520MB, 260.095 makale)

Not: Extraction işlemi 11. shard'da (wikiextractor'ın bilinen bir sorunu — muhtemelen dev bir şablon üzerinde regex patlaması) tıkandı ve durduruldu; tamamlanan 11 shard (~700MB ham, 520MB temizlenmiş) kullanılıyor. Bu, Türkçe Wikipedia'nın büyük çoğunluğunu kapsıyor ve amacımız için yeterli.

## Kapsam (önemli karar)

Bu bir genel amaçlı sohbet modeli değil. Hedef: **ERP'ye özel bir "niyet anlama + görev yürütme" asistanı**, klovo-ai'deki `GeminiAIProvider.decide_tool()`'un API'siz/yerel karşılığı. Model gerçek ERP verilerini (cari bakiye, stok miktarı, fatura tutarları) ağırlıklarında ezberlemez — bu veriler her zaman değişken olduğu için deterministik bir Python katmanında (klovo-ai'deki `tool_executor` gibi) kalır. Modelin işi: doğal Türkçe girdiyi anlayıp doğru tool + doğru parametreleri üretmek. Wikipedia korpusu bu yüzden hâlâ gerekli — amaç "genel kültür" değil, modelin akıcı/doğru Türkçe üretebilmesi için gereken dil çeşitliliği.

## Yol haritası

- [x] **Faz 0 — Prototip (yerel, GTX 1650, maliyetsiz)**: nanoGPT tarzı basit bir transformer mimarisi (PyTorch), kendi BPE tokenizer'ımız (yerel eğitilir), küçük bir Türkçe metin örneğiyle uçtan uca eğitim/örnekleme döngüsünü çalışır hale getirmek.
- [x] **Faz 2 — Veri toplama (base corpus)**: Türkçe Wikipedia dump indirildi, wikiextractor ile temizlendi (~520MB, 260K makale, D sürücüsünde).
- [x] **Faz 1 — Tokenizer**: Wikipedia + ERP-Türkçe terminolojisi (`data/erp/erp_domain_tr.txt`, 25x ağırlıklı) birlikte kullanılarak vocab=8000 BPE tokenizer eğitildi. ERP terimleri makul şekilde sıkışıyor (`fatura`→2 token, `stok`→2 token).
- [x] **Faz 3 — Base pretraining (Colab T4 GPU, ücretsiz)**: **Tamamlandı.** 20.000 adım, final val loss **2.9573** (perplexity ≈ 19.2, başlangıç 8.99'du). En iyi checkpoint: `/content/drive/MyDrive/turkish-llm/checkpoints/model_base_best.pt`.
- [x] **Faz 4 — Görev/diyalog fine-tuning**: **Tamamlandı ve Colab'da doğrulandı.** Sentetik veri seti (`data/dialogue/`, 1.876 örnek, 16 tool + "tool gerektirmeyen soru") ile base model fine-tune edildi. Yolda kritik bir hata bulundu ve düzeltildi (label shift eksikliği — model önce anlamsız bir kısayol öğrenmişti, gerçek üretim testiyle yakalandı, sadece loss'a bakılsaydı kaçırılırdı). Düzeltme + veri zenginleştirme (uzun özel isimler, tekrarlı rakamlı fatura no'ları, eksik `get_customer_sales` ifade kalıpları) sonrası **görülmemiş müşteri isimleri/ifadelerle genelleme testi** tekrarlandı: tool seçimi doğru, uzun isimler ve tekrarlı rakamlar artık doğru kopyalanıyor, her zaman geçerli JSON üretiyor. Kalan tek küçük kusur: bazı isimlerin başında gereksiz boşluk (kozmetik, Faz 5'te `.strip()` ile çözülür).
- [ ] **Faz 5 — Sisteme entegrasyon**: eğitilmiş model, klovo-ai `AIService` sözleşmesine uyan bir `LocalAIProvider` olarak sarmalanır (Gemini/Mock'un yerini alır).
- [ ] **Faz 6 — Değerlendirme ve iyileştirme**: klovo-ai'nin test senaryolarıyla doğru tool/parametre seçim oranı ölçülür, hatalı durumlar Faz 4 verisine eklenerek döngüsel iyileştirilir.

## Proje yapısı

```
turkish-llm/
  data/          # ham ve işlenmiş metin verisi (git'e girmez)
  tokenizer/     # tokenizer eğitim scripti + eğitilmiş tokenizer dosyaları
  src/           # model mimarisi, eğitim döngüsü, örnekleme scriptleri
  checkpoints/   # eğitilmiş model ağırlıkları (git'e girmez)
```

## Şu an neredeyiz

Faz 0 içinde:
- [x] Ortam kurulumu (venv, PyTorch cu126, GTX 1650 doğrulandı)
- [x] Örnek Türkçe metin (Türkçe Wikipedia'dan ~250KB, `data/fetch_sample.py`)
- [x] Tokenizer eğitimi (byte-level BPE, vocab=4000, `tokenizer/train_tokenizer.py`)
- [x] Model mimarisi (`src/model.py` — GPT tarzı, 4.2M parametre, GTX 1650'de test edildi: ~620MB VRAM)
- [x] Eğitim döngüsü (`src/train.py` — GTX 1650'de çalıştı, en iyi val loss 5.71 @ adım 400; sonrası overfitting, veri çok küçük olduğu için beklenen bir durum)
- [x] Örnekleme/inference (`src/generate.py` — pipeline uçtan uca doğrulandı: model Wikipedia'nın yüzeysel yapısını [başlıklar, yıl parantezleri] öğrenmiş ama gerçek dilbilgisi yok, beklenen sonuç: 72K token ve 4.2M parametre bunun için yetersiz)

**Faz 0 tamamlandı.** Tüm boru hattı (ortam → veri → tokenizer → model → eğitim → üretim) GTX 1650'de uçtan uca çalışır durumda.
