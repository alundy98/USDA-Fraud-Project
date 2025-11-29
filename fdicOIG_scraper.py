import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time

BASE_URL = "https://www.fdicoig.gov/news/all?page={page}"
MAX_AGE_YEARS = 6      # Stops at ~2019 (based on current year = 2025)
CUTOFF_YEAR = datetime.now().year - MAX_AGE_YEARS


def extract_article(url):
    """Extract title, date, and full text from a FDIC-OIG article page."""

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"[ERROR] Failed to fetch article {url}: {e}")
        return None

    # --------------------------
    # Extract Title
    # --------------------------
    title_tag = soup.select_one("main div h1 span")
    title = title_tag.get_text(strip=True) if title_tag else None

    # --------------------------
    # Extract Date
    # --------------------------
    date_tag = soup.select_one("main div time")

    raw_date = date_tag.get_text(strip=True) if date_tag else None

    year = None
    if raw_date:
        try:
            parsed = datetime.strptime(raw_date, "%B %d, %Y")
            year = parsed.year
        except:
            pass

    # --------------------------
    # Extract Article Text (all paragraphs)
    # --------------------------
    paragraphs = soup.select("main div article div div div div div p")
    text = "\n".join([p.get_text(" ", strip=True) for p in paragraphs])

    return {
        "title": title,
        "url": url,
        "full_date": raw_date,
        "published": year,
        "text": text,
        "scraped_at": datetime.utcnow().isoformat()
    }


def scrape_page(page_num):
    """Scrape 1 FDIC-OIG index page."""
    url = BASE_URL.format(page=page_num)
    print(f"[FETCH] {url}")

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"[ERROR] Could not fetch page {page_num}: {e}")
        return [], True

    # ---------------------------------------
    # Article links appear as <a> elements inside <h2> or <span>
    # ---------------------------------------
    links = soup.select("a")

    results = []
    stop_scraping = False

    for link in links:
        href = link.get("href", "")
        if not href:
            continue

        # Only accept Investigation Press Releases
        if "/news/investigations-press-releases/" not in href:
            continue

        # Skip summary announcements
        if "/news/summary-announcements/" in href:
            continue

        # Fix relative links
        if href.startswith("/"):
            href = "https://www.fdicoig.gov" + href

        print(f" -> Article found: {href}")

        data = extract_article(href)
        if not data:
            continue

        # Stop when we reach older than cutoff
        if data["published"] and data["published"] < CUTOFF_YEAR:
            print(f"Reached year {data['published']} < cutoff {CUTOFF_YEAR}. Stopping.")
            stop_scraping = True
            continue

        results.append(data)

        time.sleep(1)

    return results, stop_scraping


def main():
    page = 0
    all_articles = []

    print("Start scraping FDIC OIG Investigations...")

    while True:
        page_results, stop = scrape_page(page)
        if not page_results and page == 0:
            print("No articles found — check selectors.")
            break

        all_articles.extend(page_results)

        if stop:
            break

        page += 1
        time.sleep(2)

    df = pd.DataFrame(all_articles)
    df.to_csv("fdic_oig_investigations_dataset.csv", index=False)

    print(f"Done! Saved {len(all_articles)} articles to fdic_oig_investigations_dataset.csv")


if __name__ == "__main__":
    main()
