from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
import pdfplumber
from io import BytesIO
import requests
from datetime import datetime, timezone
import pandas as pd
import json
import re
#fdic scraper class, uses selenium to load js pages instead of api calls
class FDICScraperSelenium:
    pdf_count = 0
    
    def __init__(self, base_url, max_pages=5, headless=True):
        self.base_url = base_url
        self.max_pages = max_pages
        options = Options()
        options.headless = headless
        self.driver = webdriver.Chrome(options=options)

    def get_page_html(self, page):
        url = self.base_url.replace("pg=1", f"pg={page}")
        self.driver.get(url)
        time.sleep(3)  #this wait time is rlly important for js to load, it throws tantrums otherwise
        return self.driver.page_source

    def parse_search_results(self, html):
        soup = BeautifulSoup(html, "html.parser")
        results =[]

        for a in soup.find_all("a", href=True):
            #url is in anchor tags, pulls the actual url out of link
            href = a["href"]
            #get visible text title
            title = a.get_text(strip=True)
            if not title:
                continue
            #filter for relevant FDIC article URLs
            if any(x in href.lower() for x in ["/news/", "/press/", "/consumers/", "/regulations/"]):
                if not href.startswith("http"):
                    #this makes relative URLs absolute to go back to
                    href = "https://www.fdic.gov" + href
                #Hopefully find a date of publication
                date_text = None
                parent = a.find_parent()
                if parent:
                    #if parent tag is found looks for time or span elements that contain date
                    date_span = parent.find("time") or parent.find("span", class_=re.compile("date", re.I))
                    if date_span:
                        date_text = date_span.get_text(strip=True)
                #adding result dict to list
                results.append({"title": title, "url": href, "date": date_text})
        return results

    def extract_article_text(self, url, retries=3):
        is_pdf = url.lower().endswith(".pdf")
        for attempt in range(retries):#retry logic so it can stave off any rate limit/time issues
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 429:#rate limit code handling
                    time.sleep(60 * (attempt + 1))#exponential wait in case it solves issue
                    continue
                resp.raise_for_status()
                #this is mostly from a bulkier pdf handler online, simplified but seems to be working
                #full thing I saw could be rewritten and possibly get the pdfs that are just scanned images,
                #but thats a whole thing rlly
                if is_pdf or "application/pdf" in resp.headers.get("Content-Type", ""):
                    FDICScraperSelenium.pdf_count += 1
                    with pdfplumber.open(BytesIO(resp.content)) as pdf_file:
                        pages = [p.extract_text() for p in pdf_file.pages if p.extract_text()]#gets text per page
                    text = "\n".join(pages).strip()#combine pages
                    return text if text else None

                soup = BeautifulSoup(resp.text, "html.parser")
                #likely tags for where the meat of the article is located
                selectors = ["main", ".main-content", ".article-body", ".entry-content", "article"]
                for sel in selectors:
                    #routine below goes through to find the content section, if none clearly marked it falls back on getting all text, headers and all
                    section = soup.select_one(sel)
                    if section:
                        paragraphs = [p.get_text(" ", strip=True) for p in section.find_all("p")]
                        if paragraphs:
                            return "\n".join(paragraphs).strip()
                return "\n".join(p.get_text(" ", strip=True) for p in soup.find_all("p")).strip()

            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed for {url}: {e}")
                time.sleep(2 ** attempt)
        return None
    #main Function
    def run_scraper(self):
        all_results = []
        scraped_data = []
        current_year = datetime.now(timezone.utc).year
        min_year = current_year - 10
        for page in range(1, self.max_pages + 1):
            print(f"Fetching search page {page}...")
            html = self.get_page_html(page)
            results = self.parse_search_results(html)
            if not results:
                print(f"No results on page {page}")
                break
            all_results.extend(results)
            time.sleep(1)

        print(f"Found {len(all_results)} articles, getting content")

        for r in all_results:
            # filter by 10 year if date is present
            year = None
            if r["date"]:
                #construct 4 dig date
                match = re.search(r"(20\d{2})", r["date"])
                if match:
                    year = int(match.group(1))
            if year and year < min_year:
                continue#skips old articles

            text = self.extract_article_text(r["url"])
            if not text:
                continue

            scraped_data.append({
                "title": r["title"],
                "url": r["url"],
                "date": r["date"],
                "text": text,
                "scraped_at": datetime.now(timezone.utc).isoformat()
            })
            print(f"Got: {r['title'][:60]}...")

        self.driver.quit()#closes selenium browser
        return scraped_data


if __name__ == "__main__":
    url = ("https://fdic.gov/fdic-search?"
        "query=theft OR wrongful OR unauthorized OR unfair OR deceptive OR fraud OR scam OR fraudulent OR abusive -inactive"
        "&site=&orderby=date&pg=1")
    scraper = FDICScraperSelenium(base_url=url, max_pages=3, headless=True)
    data = scraper.run_scraper()
    if data:
        df = pd.DataFrame(data)
        df.to_csv("fdic_articles.csv", index=False, encoding="utf-8")
        with open("fdic_articles.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved {len(df)} articles to fdic_articles.csv")
    else:
        print("FAILED, no data scraped.")
