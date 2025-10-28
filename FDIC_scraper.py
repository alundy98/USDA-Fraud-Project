import requests
from bs4 import BeautifulSoup
import pdfplumber
import pandas as pd
from datetime import datetime
from io import BytesIO
import time
import re
#start directly from the full FDIC search URL(can change query params as needed)
#double check that the css/ xpath selectors match exactly what you need, its the only way to access it
BASE_URL = "https://www.fdic.gov/fdic-search?query=theft%20OR%20wrongful%20OR%20unauthorized%20OR%20unfair%20OR%20deceptive%20OR%20fraud%20OR%20scam%20OR%20fraudulent%20OR%20abusive%20-inactive&site=&orderby=date&pg={page}"
MAX_AGE_YEARS = 10

def extract_pdf_text(url):
    #Extracts text from pdf files, NOT SCANNED IMAGES< NO OCR
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with pdfplumber.open(BytesIO(resp.content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        print(f"[PDF ERROR] {url}: {e}")
        return ""

def extract_html_text(url):
    #Scrape article text from HTML page
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"[HTML ERROR] {url}: {e}")
        return ""

def scrape_page(page_num):
    #the main scraping function for a single search results page
    url = BASE_URL.format(page=page_num)
    print(f"[FETCH] {url}")
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        print(f"[STOP] Page {page_num} returned {resp.status_code}")
        return [], True

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    #direct access hyperlink content
    articles = soup.select("#block-fdic-theme-content div p a")
    if not articles:
        print(f"[STOP] No articles found on page {page_num}.")
        return [], True
    #stopper for old articles
    cutoff_year = datetime.now().year - MAX_AGE_YEARS
    #BOOL VALUE to stop it when reaches point
    stop_scraping = False

    for link in articles:
        href = link.get("href", "")
        title = link.get_text(strip=True)
        if not href or not title:
            continue
        #standardize the urls, the og hyperlink isnt directly accessible otherwise
        if not href.startswith("http"):
            href = f"https://www.fdic.gov{href}"

        #This gets publish date, in two ways:
        #published = the 4 digit yr value used to check if were done scraping
        #full_date_str = the full date string for storage/display
        published = None
        full_date_str = None
        parent_p = link.find_parent("p")
        if parent_p:
            text = parent_p.get_text(" ", strip=True)
            #they thankfully all follow this pattern
            date_match = re.search(r'([A-Za-z]+\.? \d{1,2}, \d{4})', text)
            if date_match:
                full_date_str = date_match.group(1)
                try:
                    #try full month name first
                    published = datetime.strptime(full_date_str, "%B %d, %Y").year
                except ValueError:
                    try:
                        #try abbreviated month
                        published = datetime.strptime(full_date_str, "%b %d, %Y").year
                    except ValueError:
                        published = None

                #stop scraping if article is older than cutoff
                if published and published < cutoff_year:
                    stop_scraping = True
                    continue

        print(f"  → {title[:70]}... ({full_date_str})")

        #Extract from either pdf or html
        text_content = ""
        if href.lower().endswith(".pdf"):
            text_content = extract_pdf_text(href)
        else:
            text_content = extract_html_text(href)
            if not text_content.strip():
                text_content = extract_pdf_text(href)

        results.append({
            "title": title,
            "url": href,
            "published": published,
            "full_date": full_date_str,
            "text": text_content,
            "scraped_at": datetime.utcnow().isoformat()
        })
        time.sleep(1)

    return results, stop_scraping

#boiler plate main method
def main():
    page = 1
    all_articles = []
    print("[START] Scraping FDIC site...")

    while True:
        page_results, stop = scrape_page(page)
        if not page_results:
            break
        all_articles.extend(page_results)
        if stop:
            print(f"[STOP] Reached articles older than {MAX_AGE_YEARS} years.")
            break
        page += 1
        time.sleep(2)

    df = pd.DataFrame(all_articles)
    df.to_csv("final.csv", index=False)
    print(f"[DONE] Saved {len(all_articles)} articles to fdic_articles.csv")

if __name__ == "__main__":
    main()
