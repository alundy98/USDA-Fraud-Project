
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
import plotly.express as px
import pdfplumber
from bs4 import BeautifulSoup
from supabase import create_client, Client
import numpy as np
import openai
from openai import OpenAI
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict
import dotenv
from dotenv import load_dotenv
load_dotenv()
# Ensure NLTK resources exist
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# CONFIG - edit as needed
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Tables containing pre-existing embeddings (used by semantic search)
SUPABASE_TABLES = [
    "article_labels",
    "oig_article_labels",
]

# Where newly scraped+embedded+classified articles will be stored
SUPABASE_TARGET_TABLE = "user_requested_articles"

# OpenAI embedding model and chat model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"

EMBED_CHUNK_SIZE = 7000  # characters per chunk (approx)
FALLBACK_EMB_DIM = 1536  # used when embedding fails (same dim as text-embedding-3-small)
#initializes the theme from our theme.css file, other customization for color done in .streamlit/config.toml
def load_theme_css():
    css_path = Path(".streamlit/theme.css")
    if css_path.exists():
        with open(css_path, "r") as f:
            st.html(f"<style>{f.read()}</style>")


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
   - Loan Fraud
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


# Scraper-run integrating flow:
#   For each input in the user-provided list:
#     - scrape text
#     - clean text
#     - embed
#     - summarize
#     - classify fraud_type
#     - insert into supabase
# Then fetch the latest 5 non-Unknown fraud_type articles for display
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
    # Clean Text
    for row in collected:
        row["cleaned_text"] = clean_text(row["text"])
    # Summarize
    for row in collected:
        try:
            row["summary"] = summarize_text(row["cleaned_text"])
        except Exception:
            row["summary"] = "Summary unavailable."

    # Embedding
    for row in collected:
        try:
            row["embedding"] = get_embedding(row["cleaned_text"])
        except Exception:
            row["embedding"] = []

    # Fraud Classification
    for row in collected:
        try:
            label = classify_fraud(row["cleaned_text"])
            row["fraud_type"] = label.get("fraud_type")
        except Exception:
            row["fraud_type"] = "Unknown"

    # Insert into Supabase
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
    # last Return to Streamlit
    return {
        "inserted_count": inserted_count,
        "top5": top5,
        "errors": errors
    }
# Streamlit App
def main():
    st.set_page_config(page_title="USAA Fraud Detection Dashboard", layout="wide")
    load_theme_css()
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
    # Visualizations, Interactive Search, Run Scraper
    tab1, tab2, tab3, tab4 = st.tabs(["Visualizations and Findings", "Interactive Search", "Emerging Fraud Trends", "Run Scraper"])
    # visuals
    with tab1:
        st.header("Visualizations and Findings")
        # row 1 two side by side
        col0, col7 = st.columns(2)
        col1, col2 = st.columns(2)
        with col0:
            st.image("New Visuals/Figure_1.png", caption="FDIC Reports", use_container_width=True)
            st.text_area("Prominent fraud types", "Identity Fraud and Cyber Fraud make up most of their reporting.", height=150, key="desc_vis0")
        with col7:
            st.image("New Visuals/Semantic_relation_drift_of_fraud_types2019-2025.png", caption="Semantic Relationship between Cyber and Identity fraud", use_container_width=True)
            st.text_area("Traits of Cyber fraud present in Identity and other fraud types", "Over time the related fraud narratives gradually grow more similar.", height=150, key="desc_vis7")
        st.markdown("---")
        with col1:
            st.image("fraud_group_counts.png", caption="Fraud Type Counts", use_container_width=True)
            st.text_area("Prominent fraud types", "Loan Fraud comprises the majority of FDIC fraud cases.", height=150, key="desc_vis1")
        with col2:
            st.image("loan_fraud_secondary_counts.png", caption="Loan Fraud Secondary Label", use_container_width=True)
            st.text_area("Prominent traits of other fraud types in Loan Fraud cases", "Money Laundering fraud traits and detection methods often show up in loan fraud cases.", height=150, key="desc_vis2")
        st.markdown("---")
        # big visualization
        st.image("visuals/semantic_drift_of_fraud_narratives_over_years.png", caption="Fraud Narrative Semantic Drift", use_container_width=True)
        st.text_area("How much fraud changes year to year", "Shows a big shift in how fraud is carried out in 2018-2020, before declining lower than ever now.", height=150, key="desc_big1")
        st.markdown("---")
        # row 2 two side by side
        col3, col4 = st.columns(2)
        with col3:
            st.image("New Visuals/2019_umap_plot.png", caption="UMAP Plot of Fraud Narrative Semantic Cosine Similarity", use_container_width=True)
            st.text_area("2019", "An even scattering, no clear pattern", height=150, key="desc_vis3")
        with col4:
            st.image("New Visuals/2024_umap_plot.png", caption="UMAP Plot of Fraud Narrative Semantic Cosine Similarity", use_container_width=True)
            st.text_area("2025", "A clear band surrounds cases, much tighter together, cyber fraud in a line through the middle", height=150, key="desc_vis4")
        st.markdown("---")
        col5, col6 = st.columns(2)
        with col5:
            st.image("New Visuals/covid_case_count.png", caption="Number of Covid Related Fraud Cases", use_container_width=True)
            st.text_area("Over 50% of Loan Fraud Covid Related", "Indicates a certain kind of fraud being especially prominent", height=150, key="desc_vis5")
        with col6:
            st.image("New Visuals/loan_fraud_detection_by_cluster.png", caption="Loan Fraud Detection by Clusters ", use_container_width=True)
            st.text_area("Detection Method Proportions", "Shows the detection methods being used for each cluster of Loan Fraud", height=150, key="desc_vis6")
        st.markdown("---")
        # final big visualization
        st.image("New Visuals/loan_fraud_umap_clusters.png", caption="Large Visualization 2", use_container_width=True)
        st.text_area("Description for Large Visualization 2", "Cluster 0: Internal banking misconduct, Asset Diversion Fraud, COVID related loan fraud. " \
        "Cluster 1: Identity Fraud related Loan Fraud cases, lower money amounts. " \
        "Cluster 2:Loan Fraud involving internal account manipulation. " \
        "Cluster 3: Loan Fraud involving false COVID relief claims, no internal misconduct, PPP loans. " \
        "Cluster 4: Employment Misrepresentation, Fake Business PPP loans", height=150, key="desc_big2")


    #semantic search tab
    with tab2:
        st.header("Semantic Search: Fraud Knowledge")
        st.write("Ask a question about fraud and receive a summarized answer along with the most relevant articles from the dataset.")
        question = st.text_area("Enter your question about fraud:")
        num_results = st.number_input("Results to return:", min_value=1, max_value=20, value=5)
        if st.button("Search"):
            if not question.strip():
                st.warning("Please enter a question before searching.")
            else:
                try:
                    article_rows = supabase.table("final_article_label_dataset").select("title,url,full_date,summary,fraud_group_primary,location,clean_text").limit(5000).execute().data
                    emb_rows = supabase.table("final_embeddings_dataset").select("url,embedding").limit(5000).execute().data
                    for row in emb_rows:
                        if isinstance(row["embedding"], str):
                            row["embedding"] = json.loads(row["embedding"])
                    url_to_emb = {row["url"]: row["embedding"] for row in emb_rows}
                    articles_with_emb = [{**article, "embedding": url_to_emb[article["url"]]} for article in article_rows if article["url"] in url_to_emb]
                    filtered_articles = [a for a in articles_with_emb if a.get("location","").strip().lower() != "unknown"]
                    if not filtered_articles:
                        st.warning("No articles with valid locations found.")
                    else:
                        question_embedding = client.embeddings.create(model="text-embedding-3-small", input=question).data[0].embedding
                        for article in filtered_articles:
                            article["score"] = np.dot(np.array(question_embedding), np.array(article["embedding"])) / (np.linalg.norm(question_embedding) * np.linalg.norm(article["embedding"]))
                        top_articles = sorted(filtered_articles, key=lambda x: x["score"], reverse=True)[:num_results]
                        # AI call to format each summary
                        for article in top_articles:
                            prompt = f"Fix the formatting of this text. Ensure proper spaces, punctuation, capitalization, and remove any 'unknown' placeholders:\n\n{article.get('summary','')}"
                            formatted_response = client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role":"user","content":prompt}], temperature=0)
                            article["summary"] = formatted_response.choices[0].message.content.strip()
                        # AIcall answers the question
                        articles_text = "".join([
                        f"Title: {a['title']}\nSummary: {a['summary']}\nLocation: {a['location']}\n\n" for a in top_articles])
                        answer_prompt = (
                            f"Based on the following top {num_results} articles, answer the question concisely and clearly.\n"
                            f"Question: {question}\n"
                            f"Articles:\n{articles_text}\n"
                            "Answer clearly, using proper punctuation, capitalization, and excluding unknown values."
                        )
                        response = client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role":"user","content":answer_prompt}], temperature=0)
                        ai_answer = response.choices[0].message.content.strip()
                        st.subheader("Answer")
                        st.write(ai_answer)
                        st.subheader("Relevant Articles")
                        for article in top_articles:
                            st.markdown(f"**{article['title']}**")
                            st.markdown(f"*Date:* {article.get('full_date', 'Unknown')}")
                            st.markdown(f"*Location:* {article.get('location', 'Unknown')}")
                            st.markdown(f"{article['summary']}")
                            st.markdown("---")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    # tab3 emerging fraud signals dashboard
    with tab3:
        st.header("Emerging Fraud Signals Dashboard")
        st.write("This dashboard identifies rising fraud themes, shows trend charts, and provides cluster-level fraud group & detection insights.")
        if st.button("Analyze Emerging Signals"):
            with st.spinner("Analyzing"):
                try:
                    article_rows = supabase.table("final_article_label_dataset").select(
                        "title,url,full_date,published,clean_text,summary,fraud_group_primary,fraud_group_secondary,detection_method,location,amount_involved,amount_numeric"
                    ).limit(5000).execute().data
                    emb_rows = supabase.table("final_embeddings_dataset").select("url,embedding").limit(5000).execute().data
                    if not article_rows or not emb_rows:
                        st.error("No articles or embeddings found in the dataset.")
                    else:
                        # normalize embeddings
                        for r in emb_rows:
                            if isinstance(r.get("embedding"), str):
                                try:
                                    r["embedding"] = json.loads(r["embedding"])
                                except Exception:
                                    r["embedding"] = None
                        url_to_emb = {r["url"]: r["embedding"] for r in emb_rows if r.get("url") and r.get("embedding")}
                        # join articles with embeddings via url
                        articles = []
                        for a in article_rows:
                            url = a.get("url")
                            emb = url_to_emb.get(url)
                            if not url or emb is None:
                                continue
                            articles.append({
                                "title": a.get("title",""),
                                "url": url,
                                "date": a.get("full_date") or a.get("published") or "",
                                "clean_text": a.get("clean_text") or "",
                                "summary": a.get("summary") or "",
                                "fraud_group_primary": a.get("fraud_group_primary") or "",
                                "fraud_group_secondary": a.get("fraud_group_secondary") or "",
                                "detection_method": a.get("detection_method") or "",
                                "location": a.get("location") or "",
                                "amount_numeric": a.get("amount_numeric") or a.get("amount_involved") or None,
                                "embedding": np.array(emb, dtype=np.float32)
                            })
                        if len(articles) == 0:
                            st.error("No valid embeddings found after matching articles and embeddings.")
                        else:
                            # build data structures
                            df = pd.DataFrame(articles)
                            # cluster on embeddings
                            X = np.vstack(df["embedding"].values)
                            n_components = min(30, X.shape[1])
                            if n_components < 2:
                                X_reduced = X
                            else:
                                pca = PCA(n_components=n_components, random_state=42)
                                X_reduced = pca.fit_transform(X)
                            k = 6
                            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
                            labels = kmeans.fit_predict(X_reduced)
                            df["cluster"] = labels
                            # cluster counter
                            st.subheader("Cluster counts")
                            cluster_counts = df["cluster"].value_counts().sort_index()
                            st.bar_chart(cluster_counts)
                            # monthly trends table
                            st.subheader("Fraud theme trends over time (by cluster)")
                            parsed_months = []
                            for d in df["date"].fillna(""):
                                try:
                                    parsed_months.append(pd.to_datetime(d).strftime("%Y-%m"))
                                except Exception:
                                    parsed_months.append("unknown")
                            df["ym"] = parsed_months
                            monthly = df[df["ym"] != "unknown"].groupby(["ym","cluster"]).size().unstack(fill_value=0)
                            monthly_sorted = monthly.sort_index()
                            if not monthly_sorted.empty:
                                st.line_chart(monthly_sorted)
                            # prepare text for TF-IDF
                            df["aug_text"] = (df["clean_text"].fillna("") + " " + df["fraud_group_primary"].fillna("") + " " + df["detection_method"].fillna("")).str.strip()
                            # extract keywords per cluster
                            cluster_candidates = {}
                            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
                            try:
                                tfidf_all = vectorizer.fit_transform(df["aug_text"].fillna(""))
                                feature_names = np.array(vectorizer.get_feature_names_out())
                                for cl in range(k):
                                    idxs = np.where(df["cluster"].values == cl)[0]
                                    if len(idxs) == 0:
                                        cluster_candidates[cl] = []
                                        continue
                                    submatrix = tfidf_all[idxs]
                                    mean_tfidf = np.asarray(submatrix.mean(axis=0)).ravel()
                                    top_indices = mean_tfidf.argsort()[::-1][:60]
                                    cluster_candidates[cl] = list(feature_names[top_indices])
                            except Exception:
                                for cl in range(k):
                                    texts = df[df["cluster"]==cl]["aug_text"].str.cat(sep=" ")
                                    tokens = re.findall(r"\w[\w\-']+", texts.lower())
                                    most = [w for w,c in Counter(tokens).most_common(30)]
                                    cluster_candidates[cl] = most
                            # enforce exclusivity
                            candidate_counts = Counter()
                            for cl, kws in cluster_candidates.items():
                                for w in kws[:20]:
                                    candidate_counts[w] += 1
                            unique_keywords = {}
                            for cl, kws in cluster_candidates.items():
                                uniq = [w for w in kws if candidate_counts[w] == 1]
                                if len(uniq) < 6:
                                    uniq = [w for w in kws if candidate_counts[w] <= 2][:12]
                                unique_keywords[cl] = uniq[:12]
                            # generate cluster summaries
                            st.subheader("Cluster summaries")
                            cluster_summaries = []
                            for cl in range(k):
                                sub = df[df["cluster"]==cl]
                                if sub.empty:
                                    continue
                                fraud_counts = Counter(sub["fraud_group_primary"].fillna("").astype(str))
                                top_fraud = [f for f,c in fraud_counts.most_common(3) if f and f.lower()!="unknown"]
                                detect_counts = Counter(sub["detection_method"].fillna("").astype(str))
                                top_detect = [d for d,c in detect_counts.most_common(3) if d and d.lower()!="unknown"]
                                # avg amount numeric filtering outliers
                                amounts = pd.to_numeric(sub["amount_numeric"], errors="coerce").dropna()
                                amounts = amounts[(amounts>=1000) & (amounts<=100000000)]
                                avg_amount = float(amounts.mean()) if not amounts.empty else None
                                centroid = kmeans.cluster_centers_[cl]
                                cluster_idxs = sub.index.tolist()
                                cluster_points = X_reduced[cluster_idxs]
                                dists = np.linalg.norm(cluster_points - centroid, axis=1)
                                rep_idxs = np.array(cluster_idxs)[dists.argsort()[:5]].tolist()
                                reps = []
                                for ridx in rep_idxs:
                                    row = df.loc[ridx]
                                    reps.append({"title": row["title"], "url": row["url"], "date": row["date"], "summary": row["summary"] or row["clean_text"][:400]})
                                cluster_summaries.append({
                                    "cluster": cl,
                                    "count": len(sub),
                                    "keywords": unique_keywords.get(cl, [])[:8],
                                    "top_fraud_groups": top_fraud,
                                    "top_detection_methods": top_detect,
                                    "avg_amount": avg_amount,
                                    "representative_articles": reps
                                })
                            # show cluster summaries with cluster numbers aligned to visuals
                            for cs in cluster_summaries:
                                st.markdown(f"### Cluster {cs['cluster']} — {cs['count']} articles")
                                st.markdown(f"**Top unique keywords:** {', '.join(cs['keywords']) if cs['keywords'] else 'None'}")
                                st.markdown(f"**Top fraud groups:** {', '.join(cs['top_fraud_groups']) if cs['top_fraud_groups'] else 'Unknown'}")
                                st.markdown(f"**Top detection methods:** {', '.join(cs['top_detection_methods']) if cs['top_detection_methods'] else 'Unknown'}")
                                if cs['avg_amount'] is not None:
                                    try:
                                        st.markdown(f"**Avg amount involved:** ${cs['avg_amount']:,.2f}")
                                    except Exception:
                                        st.markdown(f"**Avg amount involved:** {cs['avg_amount']}")
                                st.markdown("**Representative articles:**")
                                for r in cs['representative_articles']:
                                    st.markdown(f"- [{r['title']}]({r['url']}) — {r.get('date','')}")
                                st.markdown("---")
                            # compute emerging score per cluster
                            st.subheader("Emerging cluster signals")
                            recent_window = 6
                            monthly = df[df["ym"] != "unknown"].groupby(["ym","cluster"]).size().unstack(fill_value=0)
                            monthly_sorted = monthly.sort_index()
                            growth_scores = {}
                            novelty_scores = {}
                            slope_scores = {}
                            if monthly_sorted.shape[0] >= 2:
                                for col in monthly_sorted.columns:
                                    series = monthly_sorted[col].astype(float)
                                    if len(series) < recent_window + 1:
                                        growth_scores[col] = 0.0
                                        slope_scores[col] = 0.0
                                        novelty_scores[col] = 0.0
                                        continue
                                    recent = series[-recent_window:]
                                    past = series[:-recent_window]
                                    past_mean = past.mean() if len(past)>0 else 0.0
                                    past_std = past.std() if len(past)>0 else 0.0
                                    z = (recent.mean() - past_mean) / past_std if past_std>0 else 0.0
                                    growth_scores[col] = float(z)
                                    try:
                                        idx = np.arange(len(recent))
                                        slope = np.polyfit(idx, recent.values, 1)[0]
                                    except Exception:
                                        slope = 0.0
                                    slope_scores[col] = float(slope)
                                    try:
                                        recent_months = monthly_sorted.index[-recent_window:]
                                        recent_sum = monthly_sorted.loc[recent_months, col].sum()
                                        prev_period = monthly_sorted.index[:-recent_window]
                                        prev_sum = monthly_sorted.loc[prev_period, col].sum() if len(prev_period)>0 else 0.0
                                        novelty_scores[col] = float((recent_sum + 1) / (prev_sum + 1))
                                    except Exception:
                                        novelty_scores[col] = 1.0
                            else:
                                for col in range(k):
                                    growth_scores[col] = 0.0
                                    slope_scores[col] = 0.0
                                    novelty_scores[col] = 1.0
                            def normalize_dict(d):
                                vals = np.array(list(d.values()), dtype=float)
                                if vals.max() - vals.min() == 0:
                                    return {k: 0.0 for k in d.keys()}
                                mn, mx = vals.min(), vals.max()
                                return {k: float((v-mn)/(mx-mn)) for k,v in d.items()}
                            g_norm = normalize_dict(growth_scores)
                            s_norm = normalize_dict(slope_scores)
                            n_norm = normalize_dict(novelty_scores)
                            emerg_scores = {}
                            for col in growth_scores.keys():
                                emerg_scores[col] = 0.5 * g_norm.get(col,0.0) + 0.3 * s_norm.get(col,0.0) + 0.2 * n_norm.get(col,0.0)
                            gs_series = pd.Series(emerg_scores).sort_values(ascending=False)
                            st.write(gs_series)
                            if not gs_series.empty:
                                top_cluster = int(gs_series.index[0])
                                st.subheader(f"Representative articles for emerging cluster {top_cluster}")
                                reps = df[df["cluster"]==top_cluster].sort_values(by="date", ascending=False).head(5)
                                for _, r in reps.iterrows():
                                    st.markdown(f"**[{r['title']}]({r['url']})**")
                                    st.markdown(f"*Date:* {r['date']}  •  *Location:* {r['location'] or 'Unknown'}")
                                    summ = r.get("summary") or r.get("clean_text","")
                                    summ = re.sub(r"[^\x00-\x7F]+"," ", summ)
                                    summ = re.sub(r"\s+"," ", summ).strip()
                                    st.write(summ[:800])
                                    st.markdown("---")
                except Exception as e:
                    st.error(f"An error occurred: {e}")


    # Run Scraper
    with tab4:
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
