"""
Faz 4 icin tool katalogu: her tool'un (1) hangi parametreleri aldigini
ve (2) o tool'u tetikleyen dogal dil sablonlarini tanimlar.

Onemli bir tasarim karari: klovo-ai'deki bazi tool'lar customer_id gibi
sayisal ID parametreleri aliyordu (Gemini'nin once find_customer ile ID
bulup sonra ikinci bir tool'u cagirdigi cok-adimli bir akisla). Bizim
kucuk modelimiz icin bu gereksiz karmasiklik: burada tum musteri/urun
parametreleri dogrudan isim olarak tanimlandi (customer_name,
product_name). Isim -> ID cozumlemesi Faz 5'te calistirilacak
deterministik Python katmaninin (tool_executor) isi olacak, modelin
degil - model sadece metinden doğru ismi cikarmayi ogrenecek.

Fatura (create/get/update/cancel) tool'lari klovo-ai'de hic yoktu, bu
projeye ozel eklendi.
"""

import random

from entities import (
    CATEGORIES,
    CITIES,
    CUSTOMERS,
    DATE_RANGES,
    INVOICE_NUMBERS,
    KDV_ORANLARI,
    LIMITS,
    ORDER_PHRASES_ASC,
    ORDER_PHRASES_DESC,
    PRODUCTS,
    RISK_LEVELS,
)

RNG = random.Random(1337)

# Bir sablon x varlik kombinasyonu asiri buyudugunde (orn. musteri sayisi
# x sablon sayisi x sehir sayisi...), rastgele bu kadarina kirpiyoruz -
# amac cesitliligi korumak, veri setini gereksiz sismek degil.
MAX_PER_TOOL = 400


def _cap(examples):
    if len(examples) > MAX_PER_TOOL:
        return RNG.sample(examples, MAX_PER_TOOL)
    return examples


def _find_customer():
    templates = [
        "{c} hakkında bilgi ver",
        "{c} firmasının bilgilerini getir",
        "{c} müşterisini bul",
        "{c} kimdir",
        "{c} hangi şehirde",
        "{c} diye bir müşterimiz var mı",
    ]
    examples = []
    for customer in CUSTOMERS:
        for template in templates:
            examples.append((template.format(c=customer), {"name": customer}))
    return _cap(examples)


def _get_customer_balance():
    templates = [
        "{c}'ın bakiyesi ne kadar",
        "{c} ne kadar borçlu",
        "{c}'ın cari hesabı nasıl",
        "{c} firmasının kredi limiti ne kadar",
        "{c}'ın güncel bakiyesini göster",
        "{c} bize borçlu mu",
    ]
    examples = []
    for customer in CUSTOMERS:
        for template in templates:
            examples.append((template.format(c=customer), {"customer_name": customer}))
    return _cap(examples)


def _get_customer_transactions():
    templates = [
        "{c}'ın cari hareketlerini listele",
        "{c} için borç ve alacak hareketlerini göster",
        "{c}'ın hesap ekstresini getir",
        "{c}'ın son ödemelerini göster",
    ]
    examples = []
    for customer in CUSTOMERS:
        for template in templates:
            examples.append((template.format(c=customer), {"customer_name": customer}))
    return _cap(examples)


def _get_customer_sales():
    templates = [
        "{c} toplam ne kadar satış yapmış",
        "{c}'ın siparişlerini listele",
        "{c} ile yaptığımız satışları göster",
        "{c} kaç kere sipariş vermiş",
        "{c} ile geçmişte ne kadar iş yapmışız",
        "{c} ile ne kadar iş yapmışız",
        "{c} ile olan satış geçmişimiz nasıl",
        "{c} ile ticaret geçmişimiz nasıl",
        "Geçmişte {c}'a ne kadar satış yapmışız",
        "{c} ile şimdiye kadar ne kadar alışveriş yapmışız",
    ]
    examples = []
    for customer in CUSTOMERS:
        for template in templates:
            examples.append((template.format(c=customer), {"customer_name": customer}))
    return _cap(examples)


def _get_top_customers():
    templates_generic = [
        "{order} satış yapan müşteri kim",
        "{order} kazandıran müşterimiz kim",
    ]
    templates_city = [
        "{city}'daki {order} satış yapan müşteri kim",
        "{city} bölgesinde {order} kazandıran müşteri hangisi",
    ]
    templates_limit = [
        "İlk {limit} müşteriyi satışa göre sırala",
    ]

    examples = []

    for order_word, order_value in [(w, "desc") for w in ORDER_PHRASES_DESC] + [(w, "asc") for w in ORDER_PHRASES_ASC]:
        for template in templates_generic:
            examples.append((template.format(order=order_word), {"order": order_value}))

        for city in CITIES:
            for template in templates_city:
                examples.append(
                    (template.format(order=order_word, city=city), {"order": order_value, "city": city})
                )

    for limit in LIMITS:
        for template in templates_limit:
            examples.append((template.format(limit=limit), {"order": "desc", "limit": limit}))

    return _cap(examples)


def _get_top_products():
    templates_generic = [
        "{order} satan ürün hangisi",
        "{order} satılan ürünümüz ne",
    ]
    templates_category = [
        "{category} kategorisinde {order} satan ürün hangisi",
    ]
    templates_limit = [
        "İlk {limit} ürünü satışa göre listele",
    ]

    examples = []

    for order_word, order_value in [(w, "desc") for w in ORDER_PHRASES_DESC] + [(w, "asc") for w in ORDER_PHRASES_ASC]:
        for template in templates_generic:
            examples.append((template.format(order=order_word), {"order": order_value}))

        for category in CATEGORIES:
            for template in templates_category:
                examples.append(
                    (template.format(order=order_word, category=category), {"order": order_value, "category": category})
                )

    for limit in LIMITS:
        for template in templates_limit:
            examples.append((template.format(limit=limit), {"order": "desc", "limit": limit}))

    return _cap(examples)


def _get_low_stock():
    texts = [
        "Kritik stoktaki ürünleri listele",
        "Hangi ürünler stokta azaldı",
        "Yeniden sipariş vermemiz gereken ürünler neler",
        "Stoğu biten ürünler var mı",
        "Acil stok girişi yapmamız gereken ürünler neler",
        "Stok seviyesi düşük ürünleri göster",
        "Kritik seviyedeki ürünleri raporla",
        "Hangi ürünlerden acil sipariş vermeliyim",
        "Envanterde azalan ürünler hangileri",
        "Stok uyarısı olan ürünleri listele",
        "Depoda azalan ürünleri göster",
        "Tükenmek üzere olan ürünler var mı",
    ]
    return [(text, {}) for text in texts]


def _get_stock_status():
    examples = [
        ("Genel stok durumunu özetle", {}),
        ("Tüm ürünlerin stok raporu", {}),
        ("Stok durumumuzu göster", {}),
    ]
    for risk in RISK_LEVELS:
        examples.append((f"{risk} risk seviyesindeki ürünleri göster", {"risk": risk}))
        examples.append((f"Hangi ürünler {risk} durumda", {"risk": risk}))
    return examples


def _get_sales_summary():
    examples = [
        ("Toplam satışımız ne kadar", {}),
        ("Genel satış özetini ver", {}),
        ("Şu ana kadar ne kadar satış yaptık", {}),
        ("Toplam kaç satış işlemi yaptık", {}),
        ("Satış toplamımızı özetle", {}),
        ("Genel ciromuz ne kadar", {}),
        ("Kaç farklı müşteriye satış yaptık", {}),
    ]
    for start, end in DATE_RANGES:
        if start is None:
            continue
        examples.append((f"{start} ile {end} arası satış özetini ver", {"start_date": start, "end_date": end}))
        examples.append((f"{start} - {end} döneminde ne kadar satış yaptık", {"start_date": start, "end_date": end}))
    return examples


def _get_monthly_sales():
    texts = [
        "Aylık satışları göster",
        "Satışlarımı aylara göre listele",
        "Ay ay satış performansımız nasıl",
        "Aylara göre satış grafiğini göster",
        "Hangi ay en çok satış yapmışız",
        "Aylık satış raporunu ver",
        "Son ayların satış performansı nasıl",
        "Ay bazında ciromuzu göster",
    ]
    return [(text, {}) for text in texts]


def _get_sales_growth():
    texts = [
        "Geçen aya göre satışlarımız nasıl",
        "Satış büyümemizi göster",
        "Satışlarımız arttı mı azaldı mı",
        "Önceki aya göre performansımız ne durumda",
        "Satışlarımızda büyüme var mı",
        "Bu ay geçen aya göre ne kadar değişti",
        "Satış trendimiz yükseliyor mu düşüyor mu",
        "Aylık büyüme oranımız kaç",
    ]
    return [(text, {}) for text in texts]


def _get_overdue_customers():
    texts = [
        "Vadesi geçen müşterileri listele",
        "Kimlerin ödemesi gecikti",
        "Gecikmiş borcu olan müşteriler kim",
        "Vadesi geçmiş alacaklarımızı göster",
        "Ödemesi geciken firmalar hangileri",
        "Hangi müşterilerin vadesi geçti",
        "Tahsilatı gecikmiş müşterileri raporla",
        "Vadesi geçen faturalarımız var mı",
    ]
    return [(text, {}) for text in texts]


def _create_invoice():
    templates = [
        "{c} için {p} ürününden {q} adet fatura kes",
        "{c} adına {q} adet {p} faturası oluştur",
        "{c} firmasına {q} adet {p} için fatura düzenle",
    ]
    examples = []
    for customer in CUSTOMERS:
        for product, _category in PRODUCTS:
            quantity = RNG.choice([1, 2, 3, 5, 10])
            template = RNG.choice(templates)
            text = template.format(c=customer, p=product, q=quantity)
            examples.append(
                (text, {"customer_name": customer, "product_name": product, "quantity": quantity})
            )
    return _cap(examples)


def _get_invoice():
    templates = [
        "{i} numaralı faturayı getir",
        "{i} faturasının detayını göster",
        "{i} numaralı faturayı görüntüle",
    ]
    examples = []
    for invoice_no in INVOICE_NUMBERS:
        for template in templates:
            examples.append((template.format(i=invoice_no), {"invoice_no": invoice_no}))
    return examples


def _update_invoice():
    templates = [
        "{i} numaralı faturanın KDV oranını yüzde {k} yap",
        "{i} faturasını güncelle, KDV yüzde {k} olsun",
    ]
    examples = []
    for invoice_no in INVOICE_NUMBERS:
        for kdv in KDV_ORANLARI:
            template = RNG.choice(templates)
            examples.append(
                (template.format(i=invoice_no, k=kdv), {"invoice_no": invoice_no, "kdv_orani": kdv})
            )
    return examples


def _cancel_invoice():
    templates = [
        "{i} numaralı faturayı iptal et",
        "{i} faturasını iptal etmek istiyorum",
        "{i} numaralı fatura yanlış kesildi, iptal et",
    ]
    examples = []
    for invoice_no in INVOICE_NUMBERS:
        for template in templates:
            examples.append((template.format(i=invoice_no), {"invoice_no": invoice_no}))
    return examples


def _no_tool():
    # ERP kapsamina girmeyen sorular - model her soruya zorla bir tool
    # uydurmamayi, gerektiginde "bu benim isim degil" demeyi ogrenmeli.
    texts = [
        "Bugün hava nasıl",
        "Merhaba, nasılsın",
        "Türkiye'nin başkenti neresi",
        "Bana bir şaka anlat",
        "En sevdiğin renk ne",
        "Saat kaç",
        "İstanbul'da hava durumu nasıl",
        "Bir şiir yaz",
        "Python nedir",
        "Dünyanın en yüksek dağı hangisi",
        "Bugün günlerden ne",
        "Nasıl bir yapay zekasın",
    ]
    return [(text, {}) for text in texts]


# Tool adi -> (natural language, arguments) uretici fonksiyon. "none",
# ERP disi/tool gerektirmeyen sorular icin ozel bir anahtar - generate.py
# bunu tool=null olarak yazar.
TOOLS = {
    "none": _no_tool,
    "find_customer": _find_customer,
    "get_customer_balance": _get_customer_balance,
    "get_customer_transactions": _get_customer_transactions,
    "get_customer_sales": _get_customer_sales,
    "get_top_customers": _get_top_customers,
    "get_top_products": _get_top_products,
    "get_low_stock": _get_low_stock,
    "get_stock_status": _get_stock_status,
    "get_sales_summary": _get_sales_summary,
    "get_monthly_sales": _get_monthly_sales,
    "get_sales_growth": _get_sales_growth,
    "get_overdue_customers": _get_overdue_customers,
    "create_invoice": _create_invoice,
    "get_invoice": _get_invoice,
    "update_invoice": _update_invoice,
    "cancel_invoice": _cancel_invoice,
}
