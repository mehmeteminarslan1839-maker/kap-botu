from datetime import date, timedelta
import html
import json
import os
import time
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]

WATCHLIST = []  # Örn: ["THYAO", "ASELS"] — boşsa tüm BIST hisseleri
MIN_SCORE_TO_NOTIFY = 7
SEEN_FILE = "seen_kap.json"
KAP_WARMUP_URL = "https://www.kap.org.tr/tr/bildirim-sorgu"
KAP_LIST_URL = "https://www.kap.org.tr/tr/api/disclosure/members/byCriteria"
GEMINI_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-2.5-flash-lite"]

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": KAP_WARMUP_URL,
})
client = genai.Client(api_key=GEMINI_API_KEY)


def load_seen_ids() -> set:
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()


def save_seen_ids(seen_ids: set) -> None:
    with open(SEEN_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(seen_ids)[-500:], f)


def send_telegram(text: str) -> None:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    session.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        },
        timeout=20,
    ).raise_for_status()


def fetch_latest_disclosures() -> list:
    session.get(KAP_WARMUP_URL, timeout=30)
    today = date.today()
    yesterday = today - timedelta(days=1)
    payload = {
        "fromDate": yesterday.isoformat(),
        "toDate": today.isoformat(),
        "mkkMemberOidList": [],
        "subjectList": [],
    }
    resp = session.post(
        KAP_LIST_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return sorted(data, key=lambda x: x.get("disclosureIndex") or 0)


def fetch_disclosure_text(idx: int) -> str:
    try:
        resp = session.get(
            f"https://www.kap.org.tr/tr/api/notification/attachment-detail/{idx}",
            headers={"Referer": f"https://www.kap.org.tr/tr/Bildirim/{idx}"},
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            bodies = data[0].get("disclosureBody") or []
            if bodies:
                soup = BeautifulSoup(bodies[0], "html.parser")
                return " ".join(soup.get_text(separator=" ").split())[:4500]
    except Exception:
        pass
    return ""


def analyze_with_gemini(stock_code: str, title: str, content: str) -> dict:
    prompt = f"""
Sen Borsa İstanbul'da uzman bir finansal analistsin. Aşağıdaki KAP bildirimini puanla.
Hisse: {stock_code}
Başlık: {title}
İçerik: {content if content else "Başlığa göre değerlendir."}

8-10: Dev anlaşma/ihale, yüksek bedelsiz/temettü, rekor kâr VEYA büyük zarar/ceza.
6-7: Anlamlı iş ilişkisi, geri alım, finansman haberi.
1-5: Rutin prosedür, devre kesici, olağan genel kurul.

Etiket: "YÜKSEK POZİTİF", "OLUMLU", "NÖTR / RUTİN", "OLUMSUZ"
SADECE JSON döndür:
{{"puan": 7, "etiket": "OLUMLU", "ozet": "1-2 cümle özet.", "yorum": "1-2 cümle fiyat etkisi nedeni."}}
"""
    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.15,
                ),
            )
            return json.loads(response.text)
        except Exception as exc:
            print(f"[MODEL DENENDİ: {model_name}] {exc}")
            continue

    return {
        "puan": 7,
        "etiket": "NÖTR / RUTİN",
        "ozet": "Yapay zeka sunucusu şu an yoğun olduğu için otomatik özet çıkarılamadı.",
        "yorum": "Detayları aşağıdaki KAP bağlantısından inceleyebilirsiniz.",
    }


def process_disclosure(item: dict, force_send: bool = False) -> None:
    idx = item.get("disclosureIndex")
    if not idx:
        return

    raw_stock = str(
        item.get("stockCodes")
        or item.get("relatedStocks")
        or item.get("kapTitle")
        or "BİLGİ"
    )
    raw_title = str(item.get("subject") or item.get("summary") or "KAP Bildirimi")
    raw_date = str(item.get("publishDate") or "")

    if WATCHLIST and not force_send:
        codes = [c.strip().upper() for c in raw_stock.split(",")]
        if not any(c in WATCHLIST for c in codes):
            return

    analysis = analyze_with_gemini(raw_stock, raw_title, fetch_disclosure_text(idx))
    score = int(analysis.get("puan", 5))
    if score < MIN_SCORE_TO_NOTIFY and not force_send:
        return

    badges = {
        "YÜKSEK POZİTİF": "🔥 YÜKSEK POZİTİF",
        "OLUMLU": "🟢 OLUMLU",
        "NÖTR / RUTİN": "⚪ NÖTR / RUTİN",
        "OLUMSUZ": "🔴 OLUMSUZ",
    }
    etiket = str(analysis.get("etiket", "NÖTR / RUTİN")).upper()
    badge = badges.get(etiket, f"ℹ️ {html.escape(etiket)}")
    kap_url = f"https://www.kap.org.tr/tr/Bildirim/{idx}"

    msg = (
        f"<b>{badge} | Etki: {score}/10</b>\n"
        f"📌 <b>Hisse:</b> {html.escape(raw_stock)}\n"
        f"📰 <b>Başlık:</b> {html.escape(raw_title)}\n"
        f"🕒 <b>Tarih:</b> {html.escape(raw_date)}\n\n"
        f"📝 <b>Özet:</b> {html.escape(str(analysis.get('ozet', '')))}\n"
        f"💡 <b>Yorum:</b> {html.escape(str(analysis.get('yorum', '')))}\n\n"
        f"🔗 <a href='{kap_url}'>KAP Bildirimini Aç</a>"
    )
    send_telegram(msg)


def main() -> None:
    seen_ids = load_seen_ids()
    disclosures = fetch_latest_disclosures()
    if not disclosures:
        print("KAP verisi alınamadı, sonraki çalışmada tekrar denenecek.")
        return

    if not seen_ids:
        for item in disclosures:
            idx = item.get("disclosureIndex")
            if idx:
                seen_ids.add(idx)
        process_disclosure(disclosures[-1], force_send=True)
        save_seen_ids(seen_ids)
        return

    for item in disclosures:
        idx = item.get("disclosureIndex")
        if idx and idx not in seen_ids:
            seen_ids.add(idx)
            process_disclosure(item)
            time.sleep(2)

    save_seen_ids(seen_ids)


if __name__ == "__main__":
    main()
