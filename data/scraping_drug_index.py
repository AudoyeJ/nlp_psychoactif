import requests
from bs4 import BeautifulSoup
import json
import time
import os
from urllib.parse import urljoin

# ====================
# CONFIG
# ====================

BASE_URL = "https://www.effectindex.com"
REPORTS_URL = f"{BASE_URL}/reports/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}
JSON_FILE = "rapports.json"


# ====================
# EXTRACTIONS
# ====================

def extract_context_info(soup):
    context = {
        "age": "",
        "gender": "",
        "setting": ""
    }

    for box in soup.select(".report__infoBox"):
        header = box.select_one(".report__infoBoxHeader")
        if not header:
            continue

        if "context" not in header.get_text(strip=True).lower():
            continue

        for item in box.select(".report__infoBoxItem"):
            label = item.select_one(".label")
            value = item.select_one(".value")

            if not label or not value:
                continue

            key = label.get_text(strip=True).lower().rstrip(":")
            if key in context:
                context[key] = value.get_text(strip=True)

        break

    return context


def extract_dosages(soup):
    dosages = {}

    for row in soup.select(".report__infoBoxTable tbody tr"):
        name = row.select_one(".report__infoBoxTableName")
        dose = row.select_one(".report__infoBoxTableDose")
        roa = row.select_one(".report__infoBoxTableRoA")

        if not name:
            continue

        substance = name.get_text(strip=True)
        dosages[substance] = {
            "dosage": dose.get_text(strip=True) if dose else "",
            "route": roa.get_text(strip=True) if roa else ""
        }

    return dosages


def extract_related_effects(soup):
    effects = {}

    container = soup.select_one(".reportRelatedEffects")
    if not container:
        return effects

    for category in container.select(".category"):
        title = category.select_one("h2")
        if not title:
            continue

        cat_name = title.get_text(strip=True).lower()
        effect_list = [
            a.get_text(strip=True)
            for a in category.select("a.effect")
            if a.get_text(strip=True)
        ]

        if effect_list:
            effects[cat_name] = effect_list

    return effects


def extract_report_content(soup):
    intro = ""
    conclusion = ""
    logs = {}

    # ---- Intro / Conclusion
    for box in soup.select(".report__textBox"):
        header = box.select_one(".report__textBoxHeader")
        text = box.select_one(".report__textBoxText")

        if not header or not text:
            continue

        title = header.get_text(strip=True).lower()
        content = text.get_text(separator=" ", strip=True)

        if "introduction" in title:
            intro = content
        elif "conclusion" in title or "aftermath" in title:
            conclusion = content

    # ---- Logs
    for logbox in soup.select(".report__logBox"):
        header = logbox.select_one(".report__logBoxHeader")
        if not header:
            continue

        phase = header.get_text(strip=True).lower()
        descriptions = logbox.select(".logDescription")

        text = " ".join(
            d.get_text(separator=" ", strip=True)
            for d in descriptions
            if d.get_text(strip=True)
        )

        if text:
            logs[phase] = text

    return {
        "introduction": intro,
        "logs": logs,
        "conclusion": conclusion,
        "context": extract_context_info(soup),
        "dosages": extract_dosages(soup),
        "related_effects": extract_related_effects(soup)
    }


# ====================
# CHARGEMENT JSON
# ====================

if os.path.exists(JSON_FILE):
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("📂 JSON existant chargé")
else:
    data = {}
    print("📂 Nouveau JSON")

existing_urls = {
    r["url"]
    for reports in data.values()
    for r in reports
    if "url" in r
}

print(f"🔁 {len(existing_urls)} rapports déjà présents")


# ====================
# SCRAPING
# ====================

page = 1

while True:
    print(f"\n📄 Page {page}")
    resp = requests.get(f"{REPORTS_URL}?page={page}", headers=HEADERS)

    if resp.status_code != 200:
        break

    soup = BeautifulSoup(resp.text, "html.parser")
    reports = soup.select("a.reportList__item")

    if not reports:
        print("🛑 Plus aucun rapport")
        break

    new_reports_found = 0

    for report in reports:
        href = report.get("href")
        if not href:
            continue

        report_url = urljoin(BASE_URL, href)

        if report_url in existing_urls:
            continue

        new_reports_found += 1

        title_tag = report.select_one("h4")
        titre = title_tag.get_text(strip=True) if title_tag else ""

        substances = [
            s.get_text(strip=True)
            for s in report.select(".substanceName")
            if s.get_text(strip=True)
        ]

        print(f"⬇ {titre} | {substances}")

        try:
            r = requests.get(report_url, headers=HEADERS, timeout=10)
            if r.status_code != 200:
                continue

            soup_r = BeautifulSoup(r.text, "html.parser")
            content = extract_report_content(soup_r)

            for substance in substances:
                data.setdefault(substance, []).append({
                    "url": report_url,
                    "titre": titre,
                    **content
                })

            with open(JSON_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            existing_urls.add(report_url)
            print("💾 Sauvegardé")

            time.sleep(0.5)

        except requests.RequestException as e:
            print("❌ Erreur réseau :", e)

    if new_reports_found == 0:
        print("🛑 Tous les rapports déjà scrapés — arrêt")
        break

    page += 1
    time.sleep(1)

print("\n✅ Scraping terminé proprement")
