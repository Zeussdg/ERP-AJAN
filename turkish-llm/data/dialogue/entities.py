"""
Faz 4 sentetik veri uretimi icin ornek ERP varliklari (musteri, urun,
sehir vb.). klovo-ai'nin mock verisiyle ayni tarzda ama bu projeye ozel,
bagimsiz bir kopya - klovo-ai'ye bir bagimlilik olusturmamak icin.

Gercek is verisi degil, sadece sablon cumlelerini doldurmak icin
kullanilan cesitlilik kaynagi.
"""

CUSTOMERS = [
    "ABC Elektronik",
    "XYZ Teknoloji",
    "Duran Bilişim",
    "Mega Sistemler",
    "Net Bilgisayar",
    "Yıldız Ticaret",
    "Akın Elektronik",
    "Deniz Yazılım",
    "Kartal Otomasyon",
    "Pusula Teknoloji",
    "Güven Ofis Malzemeleri",
    "Vega Bilişim",
    # Daha uzun/coklu heceli isimler bilincli olarak eklendi - model
    # kisa isimleri kolay kopyalarken uzun ozel isimlerde (test sirasinda
    # "Boğaziçi Mühendislik" gibi) hece dusurme hatasi yapmisti. Bu
    # (farkli) uzun isimlerle egitimde daha fazla pratik yapmasi icin.
    "Akdeniz Gıda Sanayi",
    "Marmara Lojistik Hizmetleri",
    "Karadeniz Maden İşletmeleri",
    "Trakya Tarım Ürünleri",
]

CITIES = ["İstanbul", "Ankara", "İzmir", "Kayseri", "Bursa", "Antalya"]

PRODUCTS = [
    ("Laptop Pro 15", "Bilgisayar"),
    ("Business Laptop 14", "Bilgisayar"),
    ("Monitor 27 Inc", "Monitör"),
    ("Mekanik Klavye", "Aksesuar"),
    ("Wireless Mouse", "Aksesuar"),
    ("Ofis Sandalyesi", "Mobilya"),
    ("A4 Fotokopi Kağıdı", "Sarf Malzemesi"),
    ("Yazıcı Kartuşu", "Sarf Malzemesi"),
    ("USB Bellek 64GB", "Aksesuar"),
    ("Harici Disk 1TB", "Aksesuar"),
]

CATEGORIES = sorted({category for _, category in PRODUCTS})

RISK_LEVELS = ["CRITICAL", "WARNING", "NORMAL"]

ORDER_PHRASES_DESC = ["en çok", "en fazla", "en iyi"]
ORDER_PHRASES_ASC = ["en az", "en düşük"]

LIMITS = [3, 5, 10]

DATE_RANGES = [
    ("2026-08-01", "2026-08-31"),
    ("2026-09-01", "2026-09-15"),
    (None, None),
]

# Tekrarli/ardisik rakam iceren numaralar bilincli eklendi - model
# testte "FTR-2026-9999"i "FTR-2026-99" olarak kopyalamis, tekrarli
# rakamlarda pratik eksikligi vardi.
INVOICE_NUMBERS = [
    "FTR-2026-0142", "FTR-2026-0198", "FTR-2026-0231", "FTR-2026-0305",
    "FTR-2026-0099", "FTR-2026-1122", "FTR-2026-7777",
]

KDV_ORANLARI = [1, 8, 20]
