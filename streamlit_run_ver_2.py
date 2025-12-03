# streamlit_run.py
import os
import re
import time
import json
import string
import runpy
import importlib
import traceback
import pandas as pd
from io import BytesIO
from datetime import datetime
from pathlib import Path
from FDIC_scraper import extract_pdf_text, extract_html_text, scrape_page as extract_fdic_article
from fdicOIG_scraper import extract_article as extract_oig_article
from typing import List, Dict, Any
import streamlit as st
import requests
import pdfplumber
from bs4 import BeautifulSoup
from supabase import create_client, Client
import numpy as np
import openai
from openai import OpenAI
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
    "final_article_label_dataset",
    "final_embeddings_dataset"
]

# Where newly scraped+embedded+classified articles will be stored
SUPABASE_TARGET_TABLE = "user_requested_articles"

# OpenAI embedding model and chat model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
EMBED_CHUNK_SIZE = 7000
FALLBACK_EMB_DIM = 1536

# External module runner
def _run_external_module(module_name: str, **kwargs) -> Dict[str, Any]:
    """
    Try to call module_name.main(**kwargs) if present, otherwise execute as a script.
    Returns a dict with keys: success (bool), message (str), traceback (str optional)
    """
    try:
        mod = importlib.import_module(module_name)
        importlib.reload(mod)
        if hasattr(mod, "main") and callable(getattr(mod, "main")):
            mod.main(**kwargs)
            return {"success": True, "message": f"{module_name}.main() executed."}
        else:
            runpy.run_module(module_name, run_name="__main__")
            return {"success": True, "message": f"{module_name} executed as script."}
    except Exception as e:
        tb = traceback.format_exc()
        return {"success": False, "message": f"Error running {module_name}: {e}", "traceback": tb}

# -------------------------
# Supabase helpers
# -------------------------
def init_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase credentials not set in environment.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

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
    # Embeddings handled externally in embedding_with_ai.py
    all_candidates = []
    for t in table_names:
        cand = fetch_embeddings_from_table(supabase_client, t)
        for c in cand:
            c["table_name"] = t
        all_candidates.extend(cand)
    if not all_candidates:
        return []
    query_emb = [0.0]*FALLBACK_EMB_DIM  # fallback; actual embeddings generated externally
    for c in all_candidates:
        try:
            c["score"] = cosine_similarity(query_emb, c["embedding"])
        except Exception:
            c["score"] = -1.0
    all_candidates.sort(key=lambda x: x.get("score", -1.0), reverse=True)
    return all_candidates[:top_k]

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

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Visualizations and Findings", "Interactive Search", "Run Scraper"])

    # Tab 1: Visualizations
    with tab1:
        st.header("Visualizations and Findings")
        col1, col2 = st.columns(2)
        with col1:
            st.image("fraud_group_counts.png", caption="Fraud Type Counts", use_container_width=True)
            st.text_area("Prominent fraud types", "Loan Fraud comprises the majority of FDIC fraud cases.", height=150, key="desc_vis1")
        with col2:
            st.image("loan_fraud_secondary_counts.png", caption="Loan Fraud Secondary Label", use_container_width=True)
            st.text_area("Prominent traits of other fraud types in Loan Fraud cases", "Money Laundering fraud traits and detection methods often show up in loan fraud cases.", height=150, key="desc_vis2")
        st.markdown("---")
        st.image("visuals/semantic_drift_of_fraud_narratives_over_years.png", caption="Fraud Narrative Semantic Drift", use_container_width=True)
        st.text_area("How much fraud changes year to year", "Shows a big shift in how fraud is carried out in 2018-2020, before declining lower than ever now.", height=150, key="desc_big1")
        st.markdown("---")
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
        st.image("New Visuals/loan_fraud_umap_clusters.png", caption="Large Visualization 2", use_container_width=True)
        st.text_area("Description for Large Visualization 2", "Cluster 0: Internal banking misconduct, Asset Diversion Fraud, COVID related loan fraud. " \
            "Cluster 1: Identity Fraud related Loan Fraud cases, lower money amounts. " \
            "Cluster 2:Loan Fraud involving internal account manipulation. " \
            "Cluster 3: Loan Fraud involving false COVID relief claims, no internal misconduct, PPP loans. " \
            "Cluster 4: Employment Misrepresentation, Fake Business PPP loans", height=150, key="desc_big2")

    # Tab 2: Interactive Search
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
                with st.spinner("Searching for relevant articles..."):
                    # Step 1: Get top-K relevant articles from Supabase embeddings
                    results = semantic_search(user_query, supabase, SUPABASE_TABLES, top_k=int(top_k))
                
                if not results:
                    st.info("No related articles found. GPT will answer based on its knowledge alone.")
                    context_text = ""
                else:
                    # Prepare context for GPT
                    context_parts = []
                    for r in results:
                        summary = r.get("content", "")
                        title = r.get("title", "Untitled")
                        full_date = r.get("full_date", "Unknown date")
                        url = r.get("url", "")
                        context_parts.append(f"Title: {title}\nDate: {full_date}\nURL: {url}\nSummary: {summary}\n")
                    context_text = "\n---\n".join(context_parts)

                # Step 2: Ask GPT-4o-mini
                prompt = f"""
    You are a knowledgeable fraud detection assistant. Answer the user's question using your knowledge and the following context from FDIC/OIG articles. 

    Context:
    {context_text}

    Instructions:
    1. Answer the question as completely and accurately as possible.
    2. After your answer, list the top {top_k} articles from the context that support your answer, including title, URL, summary, and date. 
    3. If none of the articles are relevant, do not list any.

    User Question: {user_query}
    """

                with st.spinner("Generating GPT answer..."):
                    try:
                        gpt_response = client.chat.completions.create(
                            model=OPENAI_CHAT_MODEL,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                        )
                        answer_text = gpt_response.choices[0].message.content
                        st.markdown("### GPT-4o-mini Answer")
                        st.write(answer_text)
                    except Exception as e:
                        st.error(f"Error generating GPT answer: {e}")
    # Tab 3: Scraper + Embeddings
    with tab3:
        st.header("FDIC / FDIC-OIG Scraper & Embeddings Pipeline")
        st.write(
            """
            Enter a single FDIC or FDIC-OIG article URL.  
            The system will scrape the article, run embeddings, summarize it, label fraud types, cluster articles, and store results.
            """
        )

        article_url = st.text_input("Enter FDIC or FDIC-OIG article URL")

        if st.button("Run Scraper & Embeddings for this URL"):
            if not article_url or not article_url.strip():
                st.warning("Please enter a URL.")
            elif "fdic.gov" not in article_url and "fdicoig.gov" not in article_url:
                st.error("Only FDIC or FDIC-OIG article links are accepted.")
            else:
                st.info("Processing article and running embedding pipeline...")

                try:
                    # Scraper
                    if "fdicoig.gov" in article_url:
                        article_data = extract_oig_article(article_url)
                    else:
                        article_data = extract_fdic_article(article_url)  # custom wrapper for FDIC

                    if not article_data:
                        st.error("Failed to scrape article.")
                    else:
                        st.success("Article scraped successfully.")
                        st.json({
                            "title": article_data.get("title"),
                            "published": article_data.get("full_date"),
                            "url": article_data.get("url")
                        })

                        # -----------------------------
                        # Step 2: Clean Text
                        # -----------------------------
                        def clean_text(text: str) -> str:
                            if not isinstance(text, str):
                                return ""
                            text = re.sub(r"https?://\S+|www\.\S+", " ", text)
                            text = text.translate(str.maketrans("", "", string.punctuation))
                            tokens = [
                                LEMMATIZER.lemmatize(w.lower())
                                for w in text.split()
                                if w.lower() not in STOPWORDS and len(w) > 2
                            ]
                            return " ".join(tokens)

                        clean_article = clean_text(article_data["text"])
                        st.write("Cleaned text preview:")
                        st.write(clean_article[:600] + "..." if len(clean_article) > 600 else clean_article)

                        # -----------------------------
                        # Step 3: Embedding
                        # -----------------------------
                        def get_ai_embeds(texts, model_name=OPENAI_EMBEDDING_MODEL, chunk_size=EMBED_CHUNK_SIZE):
                            embeddings = []
                            for text in texts:
                                if not isinstance(text, str) or len(text.strip()) == 0:
                                    embeddings.append(np.zeros(FALLBACK_EMB_DIM))
                                    continue
                                # split text into chunks
                                chunks = [text[i:i + chunk_size * 4] for i in range(0, len(text), chunk_size * 4)]
                                chunk_embs = []
                                for chunk in chunks:
                                    try:
                                        response = client.embeddings.create(
                                            model=model_name,
                                            input=chunk
                                        )
                                        chunk_embs.append(response.data[0].embedding)
                                    except Exception as e:
                                        print(f"Embedding chunk skipped: {e}")
                                        continue
                                if chunk_embs:
                                    emb = np.mean(chunk_embs, axis=0)
                                else:
                                    emb = np.zeros(FALLBACK_EMB_DIM)
                                embeddings.append(emb)
                            return embeddings

                        embedding = get_ai_embeds([clean_article])[0]

                        # -----------------------------
                        # Step 4: Summarize Article
                        # -----------------------------
                        def summarize_article(text):
                            prompt = f"""
                            Summarize the following article in 4–5 concise sentences for a financial crime dataset.
                            Focus on capturing:
                            1. Who or what organization was involved.
                            2. The main fraud or misconduct type.
                            3. Whether it was detected after it occurred or prevented beforehand.
                            4. Detection method.
                            5. Key outcomes (fines, arrests, policy changes, etc.).
                            6. Amount involved if mentioned.

                            Article:
                            {text}
                            """
                            response = client.chat.completions.create(
                                model=OPENAI_CHAT_MODEL,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3
                            )
                            return response.choices[0].message.content.strip()

                        summary = summarize_article(clean_article)
                        st.write("Summary:")
                        st.write(summary)

                        # -----------------------------
                        # Step 5: Get Fraud Type Labels
                        # -----------------------------
                        def get_fraud_type(summary_text):
                            prompt = f"""
                            You are analyzing a summary of a fraud-related article.
                            Identify:
                            1. Raw descriptive fraud type.
                            2. Primary & secondary fraud groups (from 12 standard groups).
                            3. Detection method.
                            4. Location (City/State or Unknown)
                            5. Amount involved (format $XXX,XXX or Unknown)
                            Respond ONLY in JSON with keys:
                            {{
                                "fraud_type": "",
                                "fraud_group_primary": "",
                                "fraud_group_secondary": "",
                                "detection_method": "",
                                "location": "",
                                "amount_involved": ""
                            }}
                            Summary:
                            {summary_text}
                            """
                            response = client.chat.completions.create(
                                model=OPENAI_CHAT_MODEL,
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.3
                            )
                            try:
                                return json.loads(response.choices[0].message.content.strip())
                            except:
                                return None

                        fraud_labels = get_fraud_type(summary)
                        st.write("Fraud labels:")
                        st.json(fraud_labels)

                        # -----------------------------
                        # Step 6: Clustering & Keywords
                        # -----------------------------
                        cluster = 0
                        cluster_keywords = []
                        st.write("Cluster assigned: 0")
                        st.write("Keywords: None (single article)")

                        # -----------------------------
                        # Step 7: Save Outputs
                        # -----------------------------
                        OUT_DIR = Path("outputs_w_ai")
                        OUT_DIR.mkdir(exist_ok=True)

                        df_article = pd.DataFrame([{
                            "title": article_data["title"],
                            "url": article_data["url"],
                            "published": article_data["full_date"],
                            "text": article_data["text"],
                            "clean_text": clean_article,
                            "embedding": embedding,
                            "summary": summary,
                            "llm_labels": fraud_labels,
                            "cluster": cluster,
                            "cluster_keywords": cluster_keywords
                        }])
                        csv_path = OUT_DIR / "single_article_processed.csv"
                        df_article.to_csv(csv_path, index=False)
                        st.success(f"Saved processed article CSV: {csv_path}")

                        # -----------------------------
                        # Step 8: Upsert to Supabase
                        # -----------------------------
                        records = df_article.to_dict(orient="records")
                        TABLE_NAME = "oig_articles"
                        upsert_resp = supabase.table(TABLE_NAME).upsert(records).execute()
                        st.success("Upserted article to Supabase.")

                        # -----------------------------
                        # Step 9: Display DataFrame in Streamlit
                        # -----------------------------
                        from streamlit_extras.dataframe_explorer import dataframe_explorer
                        st.write("### Processed Article Table")
                        dataframe_explorer(df_article)

                        # -----------------------------
                        # Step 10: Download Button
                        # -----------------------------
                        st.download_button(
                            label="Download Processed CSV",
                            data=df_article.to_csv(index=False).encode("utf-8"),
                            file_name="processed_article.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
