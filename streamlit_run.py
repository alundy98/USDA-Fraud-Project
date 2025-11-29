import os
import re
import time
import json
import string
import pandas as pd
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import requests
import pdfplumber
from bs4 import BeautifulSoup
from supabase import create_client, Client
import numpy as np
import openai

# NLP utilities from your embedding script
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources exist (first-run may download)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# -------------------------
# CONFIG - edit as needed
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Tables containing pre-existing embeddings (used by semantic search)
SUPABASE_TABLES = [
    "article_labels",
    "oig_article_labels",
]

# Where newly scraped+embedded+classified articles will be stored
SUPABASE_TARGET_TABLE = os.getenv("SUPABASE_TARGET_TABLE", "non_fraud_dataset")

# OpenAI embedding model and chat model (can be changed via env)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Embedding chunking safe defaults (based on your script)
EMBED_CHUNK_SIZE = 7000  # characters per chunk (approx)
FALLBACK_EMB_DIM = 1536  # used when embedding fails (same dim as text-embedding-3-small)

# ---------------------------------------------------------------------
# Utility functions: Scraping (adapted from FDIC_scraper.py)
# ---------------------------------------------------------------------
def extract_pdf_text(url: str) -> str:
    """Extract text from a PDF URL (no OCR)."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        with pdfplumber.open(BytesIO(resp.content)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        return "\n".join(pages)
    except Exception as e:
        st.warning(f"[PDF ERROR] {url}: {e}")
        return ""


def extract_html_text(url: str) -> str:
    """Extract visible paragraphs from an HTML page."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        return "\n".join(paragraphs)
    except Exception as e:
        st.warning(f"[HTML ERROR] {url}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Clean + lemmatize + remove stopwords and punctuation."""
    if not isinstance(text, str):
        return ""
    # remove urls
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [
        LEMMATIZER.lemmatize(w.lower())
        for w in text.split()
        if w.lower() not in STOPWORDS and len(w) > 2
    ]
    return " ".join(tokens)

def safe_openai_call(func, *args, retries=3, backoff=5, **kwargs):
    """Retry wrapper for OpenAI calls (simple exponential backoff)."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wait = backoff * (attempt + 1)
            st.warning(f"OpenAI call failed (attempt {attempt+1}/{retries}): {e}. Retrying in {wait}s.")
            time.sleep(wait)
    st.error("OpenAI calls failed after retries.")
    return None


def get_ai_embeds(texts: List[str], model_name: str = OPENAI_EMBEDDING_MODEL, chunk_size: int = EMBED_CHUNK_SIZE) -> List[List[float]]:
    embeddings = []
    for text in texts:
        if not isinstance(text, str) or len(text.strip()) == 0:
            embeddings.append([0.0] * FALLBACK_EMB_DIM)
            continue

        chunks = [text[i:i + chunk_size * 4] for i in range(0, len(text), chunk_size * 4)]
        chunk_embs = []
        for chunk in chunks:
            resp = safe_openai_call(openai.Embedding.create, model=model_name, input=chunk)
            if resp is None:
                continue
            try:
                chunk_embs.append(resp["data"][0]["embedding"])
            except Exception:
                continue

        if chunk_embs:
            # average chunk embeddings
            arr = np.array(chunk_embs, dtype=float)
            avg = np.mean(arr, axis=0).tolist()
            embeddings.append(avg)
        else:
            embeddings.append([0.0] * FALLBACK_EMB_DIM)
    return embeddings

def summarize_article(text: str) -> str:
    prompt = f"""
    Summarize the following article in 4–5 concise sentences for a financial crime dataset.
    Focus on capturing:
    1. Who or what organization was involved.
    2. The main fraud or misconduct type (describe briefly).
    3. Whether it was detected after it occurred or prevented beforehand.
    4. The method or mechanism that led to detection or prevention.
    5. Any key outcomes (fines, arrests, policy changes, etc.).

    Be factual and structured so that another model can extract fraud_type and method clearly.

    Article:
    {text}
    """
    resp = safe_openai_call(openai.ChatCompletion.create, model=OPENAI_CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2)
    if resp is None:
        return ""
    try:
        if "choices" in resp and len(resp["choices"]) > 0:
            return resp["choices"][0]["message"]["content"].strip()
        # Fallback: return raw text if shape differs
        return str(resp)
    except Exception:
        return ""


def get_fraud_type_from_summary(summary: str) -> Dict[str, Any]:
    """
    Calls the LLM to extract a JSON object with keys raw_fraud_type, fraud_type, detection_method.
    If parsing fails, return a dict with fraud_type: "Unknown".
    """
    prompt = f"""
You are analyzing a summary of a fraud-related article.

Your tasks:
1. Identify the main raw fraud type mentioned. Be descriptive.
2. Map the raw fraud type to one of the following standard categories:
   - Corporate Fraud
   - Investment Fraud
   - Insurance Fraud
   - Healthcare Fraud
   - Cyber Fraud
   - Identity Fraud
   - Money Laundering Fraud
   - Other Fraud
3. Identify the detection or prevention method mentioned, if any. Use the following list or "Unknown" if not clear:
   - Audit
   - Whistleblower
   - Algorithmic Detection
   - AML Model
   - Routine Inspection
   - Law Enforcement Investigation
   - Internal Controls
   - Public Tip or Complaint
   - Data Analysis
   - Regulatory Reporting
   - Preventive Policy
   - Unknown

Respond ONLY in JSON with the keys:
{{
    "raw_fraud_type": "<raw descriptive fraud type>",
    "fraud_type": "<mapped standard category>",
    "detection_method": "<detection or prevention method>"
}}

Summary:
{summary}
"""
    resp = safe_openai_call(openai.ChatCompletion.create, model=OPENAI_CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.2)
    if resp is None:
        return {"raw_fraud_type": "", "fraud_type": "Unknown", "detection_method": "Unknown"}
    try:
        content = ""
        if "choices" in resp and len(resp["choices"]) > 0:
            content = resp["choices"][0]["message"]["content"].strip()
        else:
            content = str(resp)
        # Attempt to parse JSON
        parsed = json.loads(content)
        # Normalize keys and values
        parsed = {
            "raw_fraud_type": parsed.get("raw_fraud_type", ""),
            "fraud_type": parsed.get("fraud_type", "Unknown"),
            "detection_method": parsed.get("detection_method", "Unknown")
        }
        return parsed
    except Exception:
        # fallback if model didn't return strict JSON
        # Try to heuristically find "fraud_type: XXX" pattern
        m = re.search(r'fraud_type"\s*:\s*"?([^",\n}]+)"?', content, re.IGNORECASE)
        fraud_type = m.group(1).strip() if m else "Unknown"
        return {"raw_fraud_type": "", "fraud_type": fraud_type or "Unknown", "detection_method": "Unknown"}

def init_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase credentials not set in environment.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def insert_articles_to_supabase(supabase: Client, table: str, rows: List[Dict[str, Any]]) -> bool:
    try:
        if not rows:
            return True
        # Supabase may reject very large inserts; insert in small batches
        batch_size = 25
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            res = supabase.table(table).insert(chunk).execute()
            # supabase-py returns .status_code for errors in some versions
            # we simply continue; any exception will be caught
        return True
    except Exception as e:
        st.error(f"Failed inserting articles to Supabase: {e}")
        return False

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)


def fetch_embeddings_from_table(supabase_client: Client, table_name: str, embedding_column: str = "embedding") -> List[Dict]:
    """Fetch candidate rows (limited) from Supabase table."""
    try:
        res = supabase_client.table(table_name).select("*").limit(1000).execute()
    except Exception as e:
        st.warning(f"Error fetching from Supabase table '{table_name}': {e}")
        return []
    rows = res.data or []
    candidates = []
    for r in rows:
        emb = r.get(embedding_column) or r.get("emb") or r.get("vector")
        if isinstance(emb, str):
            try:
                emb = json.loads(emb)
            except Exception:
                emb = None
        if emb is None:
            continue
        candidates.append({
            "id": r.get("id"),
            "title": r.get("title") or r.get("name") or "Untitled",
            "url": r.get("url") or r.get("link") or None,
            "content": r.get("content") or r.get("snippet") or "",
            "embedding": emb
        })
    return candidates


def semantic_search(query: str, supabase_client: Client, table_names: List[str], top_k: int = 5) -> List[Dict]:
    query_emb = get_ai_embeds([query])[0]
    if not query_emb:
        return []
    all_candidates = []
    for t in table_names:
        cand = fetch_embeddings_from_table(supabase_client, t)
        for c in cand:
            c["table_name"] = t
        all_candidates.extend(cand)
    if not all_candidates:
        return []
    for c in all_candidates:
        try:
            c["score"] = cosine_similarity(query_emb, c["embedding"])
        except Exception:
            c["score"] = -1.0
    all_candidates.sort(key=lambda x: x.get("score", -1.0), reverse=True)
    return all_candidates[:top_k]

def summarize_text(text: str) -> str:
    """
    Compatibility wrapper: route to the existing summarization function.
    Uses summarize_article(...) already defined in the file.
    """
    try:
        return summarize_article(text)
    except Exception:
        return ""

def get_embedding(text: str) -> List[float]:
    """
    Compatibility wrapper: embed a single text using get_ai_embeds (batch-style).
    Returns a list of floats (embedding) or an empty list on failure.
    """
    try:
        embs = get_ai_embeds([text])
        if embs and len(embs) > 0:
            return embs[0]
        return []
    except Exception:
        return []

def classify_fraud(text: str) -> Dict[str, Any]:
    """
    Compatibility wrapper: generate a summary and extract fraud labels.
    Returns the parsed label dict from get_fraud_type_from_summary(...).
    If parsing fails, returns a default Unknown label dict.
    """
    try:
        summary = summarize_article(text)
        labels = get_fraud_type_from_summary(summary)
        if not isinstance(labels, dict):
            return {"raw_fraud_type": "", "fraud_type": "Unknown", "detection_method": "Unknown"}
        return labels
    except Exception:
        return {"raw_fraud_type": "", "fraud_type": "Unknown", "detection_method": "Unknown"}


# ---------------------------------------------------------------------
# Scraper-run integrating flow:
#   For each input in the user-provided list:
#     - scrape text
#     - clean text
#     - embed
#     - summarize
#     - classify fraud_type
#     - insert into supabase
# Then fetch the latest 5 non-Unknown fraud_type articles for display
# ---------------------------------------------------------------------
def run_scraper_and_process(user_input: str, supabase):
    """
    Full pipeline:
        1. Convert comma-separated keywords -> FDIC search URL
        2. Scrape multi-page FDIC results
        3. Extract article text (HTML -> PDF fallback)
        4. Clean text
        5. Summarize
        6. Embed
        7. Classify fraud type
        8. Insert into Supabase
        9. Return 5 most recent with known fraud type
    """
    # 1. Convert user input to list
    keywords = [x.strip() for x in user_input.split(",") if x.strip()]
    if not keywords:
        return {"inserted_count": 0, "top5": [], "errors": ["No keywords were provided."]}

    # Build FDIC Search URL
    base_url = "https://www.fdic.gov/fdic-search?query="
    for word in keywords:
        base_url += word
        if word != keywords[-1]:
            base_url += "%20OR%20"
    base_url += "&site=&orderby=date&pg={page}"

    st.info(f"FDIC Search URL generated: {base_url.replace('{page}', '1')}")

    # 2. Scrape Multiple Pages
    MAX_AGE_YEARS = 1
    cutoff_year = datetime.now().year - MAX_AGE_YEARS

    collected = []
    page = 1
    stop = False

    while not stop:
        url = base_url.format(page=page)
        st.write(f"Scraping search results page: {url}")

        try:
            resp = requests.get(url, timeout=20)
        except Exception as e:
            return {"inserted_count": 0, "top5": [], "errors": [f"Error fetching FDIC search page: {e}"]}

        if resp.status_code != 200:
            break

        soup = BeautifulSoup(resp.text, "html.parser")
        articles = soup.select("#block-fdic-theme-content div p a")

        if not articles:
            break

        for link in articles:
            href = link.get("href", "")
            title = link.get_text(strip=True)
            if not href or not title:
                continue

            if not href.startswith("http"):
                href = f"https://www.fdic.gov{href}"

            # Extract date
            parent_p = link.find_parent("p")
            published_year = None
            full_date_str = None

            if parent_p:
                text = parent_p.get_text(" ", strip=True)
                date_match = re.search(r"([A-Za-z]+\.? \d{1,2}, \d{4})", text)
                if date_match:
                    full_date_str = date_match.group(1)
                    for fmt in ("%B %d, %Y", "%b %d, %Y"):
                        try:
                            published_year = datetime.strptime(full_date_str, fmt).year
                            break
                        except:
                            pass

                    if published_year and published_year < cutoff_year:
                        stop = True
                        continue

            # 3. Extract content
            article_text = extract_html_text(href)
            if not article_text.strip():
                article_text = extract_pdf_text(href)

            collected.append({
                "title": title,
                "url": href,
                "published_year": published_year,
                "published_full": full_date_str,
                "text": article_text
            })

            time.sleep(1)

        page += 1
        time.sleep(1)

    if not collected:
        return {"inserted_count": 0, "top5": [], "errors": ["No articles were scraped."]}

    # 4. Clean Text
    for row in collected:
        row["cleaned_text"] = clean_text(row["text"])

    # 5. Summarize
    for row in collected:
        try:
            row["summary"] = summarize_text(row["cleaned_text"])
        except Exception:
            row["summary"] = "Summary unavailable."

    # 6. Embedding
    for row in collected:
        try:
            row["embedding"] = get_embedding(row["cleaned_text"])
        except Exception:
            row["embedding"] = []

    # 7. Fraud Classification
    for row in collected:
        try:
            label = classify_fraud(row["cleaned_text"])
            row["fraud_type"] = label.get("fraud_type")
        except Exception:
            row["fraud_type"] = "Unknown"

    # 8. Insert into Supabase
    inserted_count = 0
    errors = []

    for row in collected:
        try:
            supabase.table("user_search_articles").insert({
                "title": row["title"],
                "url": row["url"],
                "published": row["published_year"],
                "full_date": row["published_full"],
                "summary": row["summary"],
                "text": row["cleaned_text"],
                "fraud_type": row["fraud_type"],
                "embedding": row["embedding"],
            }).execute()
            inserted_count += 1
        except Exception as e:
            errors.append(str(e))

    # 9. Pull 5 Most Recent w/ Known Fraud
    try:
        resp = (
            supabase.table("scraped_fdic_articles")
            .select("*")
            .neq("fraud_type", "Unknown")
            .order("published", desc=True)
            .limit(5)
            .execute()
        )
        top5 = resp.data
    except Exception as e:
        top5 = []
        errors.append(f"Could not fetch results: {e}")

    # Final Return to Streamlit
    return {
        "inserted_count": inserted_count,
        "top5": top5,
        "errors": errors
    }

# Streamlit App
def main():
    st.set_page_config(page_title="USAA Fraud Detection Dashboard", layout="wide")
    st.title("USAA Fraud Detection Dashboard")

    # initialize OpenAI
    if not OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not set in environment.")
        st.stop()
    openai.api_key = OPENAI_API_KEY

    # initialize Supabase
    try:
        supabase = init_supabase_client()
    except Exception as e:
        st.error(f"Supabase init failed: {e}")
        st.stop()

    # Tabs: Visualizations, Interactive Search, Run Scraper
    tab1, tab2, tab3 = st.tabs(["Visualizations and Findings", "Interactive Search", "Run Scraper"])

    # --------------------
    # Tab 1: Visualizations
    # --------------------
    with tab1:
        st.header("Visualizations and Findings")
        col1, col2 = st.columns(2)
        with col1:
            st.image("placeholder.png", caption="Figure 1 - Replace with real visualization", use_column_width=True)
            st.text_area("Figure 1 Description", "Add explanation for visualization #1 here.")
        with col2:
            st.image("placeholder.png", caption="Figure 2 - Replace with real visualization", use_column_width=True)
            st.text_area("Figure 2 Description", "Add explanation for visualization #2 here.")
        st.markdown("---")

    # ---------------------------
    # Tab 2: Interactive Search
    # ---------------------------
    with tab2:
        st.header("Interactive Fraud Knowledge Search")
        user_query = st.text_area("Ask a question about fraud", height=120)
        c1, c2 = st.columns([1, 1])
        with c1:
            top_k = st.number_input("Number of results to return", min_value=1, max_value=20, value=5, step=1)
        with c2:
            run_search = st.button("Search")

        if run_search:
            if not user_query or not user_query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Computing embedding and searching..."):
                    results = semantic_search(user_query, supabase, SUPABASE_TABLES, top_k=int(top_k))
                if not results:
                    st.info("No related articles found. Confirm Supabase embeddings exist.")
                else:
                    st.markdown(f"### Top {len(results)} results")
                    for i, r in enumerate(results, start=1):
                        title = r.get("title", "Untitled")
                        url = r.get("url")
                        score = r.get("score", 0.0)
                        snippet = r.get("content", "")
                        st.markdown(f"**{i}. [{title}]({url})**" if url else f"**{i}. {title}**")
                        st.write(f"Similarity score: {score:.4f}")
                        if snippet:
                            display_snip = snippet if len(snippet) < 600 else snippet[:600] + "..."
                            st.write(display_snip)
                        st.markdown("---")

    # ---------------------------
    # Tab 3: Run Scraper
    # ---------------------------
    with tab3:
        st.header("Run Scraper")
        st.markdown(
            """
            Enter a comma-separated list of **full URLs** (or query-URLs you construct externally).
            Each item will be scraped, embedded, summarized, classified, and inserted into the Supabase table:
            `""" + SUPABASE_TARGET_TABLE + """`.
            After the run, the dashboard will display the 5 most recent articles from this run that do not have an 'Unknown' fraud type.
            """
        )

        scraper_input = st.text_input(
            "Enter comma-separated URLs or query-URLs for scraping",
            placeholder="https://example.com/article1, https://example.com/article2"
        )

        if st.button("Run Scraper and Process"):
            if not scraper_input or not scraper_input.strip():
                st.warning("Please enter one or more URLs / query-URLs for the scraper.")
            else:
                with st.spinner("Running scraper, embeddings, and classification... This can take a while depending on number of items and OpenAI latency."):
                    result = run_scraper_and_process(scraper_input, supabase)

                st.success(f"Inserted {result.get('inserted', 0)} articles (attempted).")
                if result.get("errors"):
                    st.warning("Some items had issues:")
                    for e in result["errors"]:
                        st.write(f"- {e}")

                top5 = result.get("top5", [])
                if not top5:
                    st.info("No recent classified articles (non-Unknown) found for this run.")
                else:
                    st.subheader("Top 5 most recent classified articles (fraud_type != Unknown)")
                    for i, it in enumerate(top5, start=1):
                        st.markdown(f"### {i}. {it.get('title')}")
                        if it.get("url"):
                            st.markdown(f"[View Article]({it.get('url')})")
                        st.write(f"Published: {it.get('published')}")
                        st.write(f"Fraud Type: **{it.get('fraud_type')}**")
                        if it.get("summary"):
                            s = it.get("summary")
                            st.write(s if len(s) < 800 else s[:800] + "...")
                        st.markdown("---")


if __name__ == "__main__":
    main()
