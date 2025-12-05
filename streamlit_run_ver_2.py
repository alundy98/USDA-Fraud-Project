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
from openai import Client as openai_client
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
    query_emb = [0.0]*FALLBACK_EMB_DIM 
    for c in all_candidates:
        try:
            c["score"] = cosine_similarity(query_emb, c["embedding"])
        except Exception:
            c["score"] = -1.0
    all_candidates.sort(key=lambda x: x.get("score", -1.0), reverse=True)
    return all_candidates[:top_k]

def classify_fraud_type(text: str) -> str:
    # 1. Clean text using the same cleaning logic
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", text)
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    tokens = [
        LEMMATIZER.lemmatize(w.lower())
        for w in cleaned.split()
        if w.lower() not in STOPWORDS and len(w) > 2
    ]
    cleaned_text = " ".join(tokens)

    # 2. Summarize
    prompt = f"""
    Summarize the following article in 4–5 concise sentences for a financial crime dataset.
    Focus on capturing:
    - The main fraud or misconduct type
    - Key players and events
    - Amount involved
    - Detection or prevention method
    Article:
    {cleaned_text}
    """
    summary_resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    summary = summary_resp.choices[0].message.content.strip()

    # 3. Extract labels
    label_prompt = f"""
    Identify the following from this summary:
    - fraud_type
    Respond ONLY in JSON with:
    {{
        "fraud_type": ""
    }}

    Summary:
    {summary}
    """
    label_resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": label_prompt}],
        temperature=0
    )

    try:
        labels = json.loads(label_resp.choices[0].message.content.strip())
        return labels.get("fraud_type", "Unknown")
    except:
        return "Unknown"

def classify_detection_type(text: str) -> str:
    # 1. Clean text
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", text)
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    tokens = [
        LEMMATIZER.lemmatize(w.lower())
        for w in cleaned.split()
        if w.lower() not in STOPWORDS and len(w) > 2
    ]
    cleaned_text = " ".join(tokens)

    # 2. Summarize
    prompt = f"""
    Summarize the following article in 4–5 concise sentences for a financial crime dataset.
    Focus on:
    - Fraud type
    - Detection method
    Article:
    {cleaned_text}
    """
    summary_resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    summary = summary_resp.choices[0].message.content.strip()

    # 3. Extract detection method from the summary
    label_prompt = f"""
    Identify the detection method used in the case.
    Respond ONLY in JSON with:
    {{
        "detection_method": ""
    }}

    Summary:
    {summary}
    """
    label_resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": label_prompt}],
        temperature=0
    )

    try:
        labels = json.loads(label_resp.choices[0].message.content.strip())
        return labels.get("detection_method", "Unknown")
    except:
        return "Unknown"


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
    tab1, tab2, tab3, tab4= st.tabs(["Visualizations and Findings", "Interactive Search","Emerging Trends in the Fraud Space", "Run Scraper"])

    # Tab 1: Visualizations
    with tab1:
        st.header("Visualizations and Findings")
        col1, col2 = st.columns(2)
        with col1:
            st.image("/root/assets/fraud_group_counts.png", caption="Fraud Type Counts", use_container_width=True)
            st.text_area("Prominent fraud types", "Loan Fraud comprises the majority of FDIC fraud cases.", height=150, key="desc_vis1")
        with col2:
            st.image("/root/assets/loan_fraud_secondary_counts.png", caption="Loan Fraud Secondary Label", use_container_width=True)
            st.text_area("Prominent traits of other fraud types in Loan Fraud cases", "Money Laundering fraud traits and detection methods often show up in loan fraud cases.", height=150, key="desc_vis2")
        st.markdown("---")
        st.image("/root/assets/Semantic_relation_drift_of_fraud_types2019-2025.png", caption="Fraud Narrative Semantic Drift", use_container_width=True)
        st.text_area("How much fraud changes year to year", "Shows a big shift in how fraud is carried out in 2018-2020, before declining lower than ever now.", height=150, key="desc_big1")
        st.markdown("---")
        col3, col4 = st.columns(2)
        with col3:
            st.image("/root/assets/2019_umap_plot.png", caption="UMAP Plot of Fraud Narrative Semantic Cosine Similarity", use_container_width=True)
            st.text_area("2019", "An even scattering, no clear pattern", height=150, key="desc_vis3")
        with col4:
            st.image("/root/assets/2024_umap_plot.png", caption="UMAP Plot of Fraud Narrative Semantic Cosine Similarity", use_container_width=True)
            st.text_area("2025", "A clear band surrounds cases, much tighter together, cyber fraud in a line through the middle", height=150, key="desc_vis4")
        st.markdown("---")
        col5, col6 = st.columns(2)
        with col5:
            st.image("/root/assets/covid_case_count.png", caption="Number of Covid Related Fraud Cases", use_container_width=True)
            st.text_area("Over 50% of Loan Fraud Covid Related", "Indicates a certain kind of fraud being especially prominent", height=150, key="desc_vis5")
        with col6:
            st.image("/root/assets/loan_fraud_detection_by_cluster.png", caption="Loan Fraud Detection by Clusters ", use_container_width=True)
            st.text_area("Detection Method Proportions", "Shows the detection methods being used for each cluster of Loan Fraud", height=150, key="desc_vis6")
        st.markdown("---")
        st.image("/root/assets/loan_fraud_umap_clusters.png", caption="Large Visualization 2", use_container_width=True)
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
                        import numpy as np
                        for article in filtered_articles:
                            article["score"] = np.dot(np.array(question_embedding), np.array(article["embedding"])) / (np.linalg.norm(question_embedding) * np.linalg.norm(article["embedding"]))
                        top_articles = sorted(filtered_articles, key=lambda x: x["score"], reverse=True)[:num_results]
                        # AI-format each summary
                        for article in top_articles:
                            prompt = f"Fix the formatting of this text. Ensure proper spaces, punctuation, capitalization, and remove any 'unknown' placeholders:\n\n{article.get('summary','')}"
                            formatted_response = client.chat.completions.create(model="gpt-4.1-mini", messages=[{"role":"user","content":prompt}], temperature=0)
                            article["summary"] = formatted_response.choices[0].message.content.strip()
                        # AI answers the question
                        articles_text = "".join([
    f"Title: {a['title']}\nSummary: {a['summary']}\nLocation: {a['location']}\n\n"
    for a in top_articles
])

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

    # tab4 emerging fraud signals dashboard
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
                        import numpy as np
                        import pandas as pd
                        from sklearn.cluster import KMeans
                        from sklearn.decomposition import PCA
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from collections import Counter, defaultdict

                        # normalize embeddings (convert JSON strings to lists if needed)
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
                            # cluster on reduced embeddings
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
                            # cluster counts
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
                            # prepare text for TF-IDF: augment with fraud group + detection method
                            df["aug_text"] = (df["clean_text"].fillna("") + " " + df["fraud_group_primary"].fillna("") + " " + df["detection_method"].fillna("")).str.strip()
                            # Extract candidate keywords per cluster using TF-IDF
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
                                    # mean tfidf per term within cluster
                                    mean_tfidf = np.asarray(submatrix.mean(axis=0)).ravel()
                                    top_indices = mean_tfidf.argsort()[::-1][:60]  # candidate pool
                                    cluster_candidates[cl] = list(feature_names[top_indices])
                            except Exception:
                                # fallback: simple word frequency
                                for cl in range(k):
                                    texts = df[df["cluster"]==cl]["aug_text"].str.cat(sep=" ")
                                    tokens = re.findall(r"\w[\w\-']+", texts.lower())
                                    most = [w for w,c in Counter(tokens).most_common(30)]
                                    cluster_candidates[cl] = most
                            # enforce exclusivity: remove keywords that appear as top candidates in multiple clusters
                            candidate_counts = Counter()
                            for cl, kws in cluster_candidates.items():
                                # consider only top 20 candidates per cluster for overlap counting
                                for w in kws[:20]:
                                    candidate_counts[w] += 1
                            unique_keywords = {}
                            for cl, kws in cluster_candidates.items():
                                uniq = [w for w in kws if candidate_counts[w] == 1]
                                if len(uniq) < 6:
                                    # if too few unique words, allow words that appear at most twice
                                    uniq = [w for w in kws if candidate_counts[w] <= 2][:12]
                                unique_keywords[cl] = uniq[:12]
                            # generate cluster summaries: fraud group counts, detection method counts, avg amount, representative articles
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
                                # avg amount numeric if present
                                amounts = pd.to_numeric(sub["amount_numeric"], errors="coerce").dropna()
                                avg_amount = float(amounts.mean()) if not amounts.empty else None
                                # representative articles: nearest to cluster centroid
                                centroid = kmeans.cluster_centers_[cl]
                                # get embeddings projected to same space (X_reduced)
                                cluster_idxs = sub.index.tolist()
                                cluster_points = X_reduced[cluster_idxs]
                                # compute distances
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
                            # display cluster summaries with detection & fraud group info
                            for cs in cluster_summaries:
                                st.markdown(f"### Cluster {cs['cluster']+1} — {cs['count']} articles")
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
                            # compute emerging score per cluster (z-score growth + trend slope + keyword novelty)
                            st.subheader("Emerging cluster signals")
                            recent_window = 6
                            monthly = df[df["ym"] != "unknown"].groupby(["ym","cluster"]).size().unstack(fill_value=0)
                            monthly_sorted = monthly.sort_index()
                            growth_scores = {}
                            novelty_scores = {}
                            slope_scores = {}
                            if monthly_sorted.shape[0] >= 2:
                                # compute z-score growth (recent mean vs historical mean)
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
                                    # slope (linear fit on recent window)
                                    try:
                                        idx = np.arange(len(recent))
                                        slope = np.polyfit(idx, recent.values, 1)[0]
                                    except Exception:
                                        slope = 0.0
                                    slope_scores[col] = float(slope)
                                    # novelty: for cluster keywords, compare freq in recent_window vs previous 12 months
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
                            # normalize components and combine
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
                            # top emerging cluster details
                            if not gs_series.empty:
                                top_cluster = int(gs_series.index[0])
                                st.subheader(f"Representative articles for emerging cluster {top_cluster+1}")
                                reps = df[df["cluster"]==top_cluster].sort_values(by="date", ascending=False).head(5)
                                for _, r in reps.iterrows():
                                    st.markdown(f"**[{r['title']}]({r['url']})**")
                                    st.markdown(f"*Date:* {r['date']}  •  *Location:* {r['location'] or 'Unknown'}")
                                    # clean summary similarly to Tab2 lightly
                                    summ = r.get("summary") or r.get("clean_text","")
                                    summ = re.sub(r"[^\x00-\x7F]+"," ", summ)
                                    summ = re.sub(r"\s+"," ", summ).strip()
                                    st.write(summ[:800])
                                    st.markdown("---")
                except Exception as e:
                    st.error(f"An error occurred: {e}")


    # Tab 4 Scraper + Embeddings
    with tab4:
        st.header("FDIC Article Keyword Search")

        user_input = st.text_input(
            "Enter keywords (comma-separated):",
            placeholder="fraud, phishing, PPP, loan"
        )

        if st.button("Search FDIC"):
            if not user_input.strip():
                st.error("Please enter at least one keyword.")
                st.stop()

        # -------------------------
        # 1. Parse Keywords
        # -------------------------
            keywords = [x.strip() for x in user_input.split(",") if x.strip()]
            if not keywords:
                st.error("No valid keywords were provided.")
                st.stop()

        # -------------------------
        # 2. Build FDIC Search URL
        # -------------------------
            fdic_url_base = "https://www.fdic.gov/fdic-search?query="

            query_string = ""
            for word in keywords:
                query_string += word
                if word != keywords[-1]:
                    query_string += "%20OR%20"

            fdic_url = (
                fdic_url_base
                + query_string
                + "%20-inactive&site=&orderby=date&pg={page}"
            )

            st.write("**Constructed FDIC Search URL:**")
            st.code(fdic_url.format(page=1))

        # -------------------------
        # 3. Extract first 10 article links
        # -------------------------
            from FDIC_scraper import extract_html_text, extract_pdf_text, scrape_page

            article_links = []
            page = 1

            while len(article_links) < 10:
                search_url = fdic_url.format(page=page)
                try:
                    page_results = scrape_page(search_url)
                    if not page_results:
                        break
                    for link in page_results:
                        if link not in article_links:
                            article_links.append(link)
                        if len(article_links) >= 10:
                            break
                except Exception as e:
                    st.error(f"Error scraping FDIC search page {page}: {e}")
                    break
                page += 1

            if not article_links:
                st.warning("No articles found for the provided keywords.")
                st.stop()

            st.write(f"### Found {len(article_links)} articles")
            for l in article_links:
                st.write("•", l)

        # -------------------------
        # 4. Scrape + Cache Logic
        # -------------------------
            st.subheader("Extracting Article Details...")

            results = []

            for link in article_links:
                st.write(f"Processing: {link}")

            # -------------------------
            # (A) Check cache table
            # -------------------------
                cache_resp = supabase.table("fdic_cache").select("*").eq("url", link).execute()

                if cache_resp.data:
                    # Cached → use existing values
                    row = cache_resp.data[0]
                    results.append({
                        "url": link,
                        "fraud_type": row["fraud_type"],
                        "detection_type": row["detection_type"],
                        "cached": True
                    })
                    st.write(" → Retrieved from cache")
                    continue

            # -------------------------
            # (B) Not cached → scrape + classify
            # -------------------------
                try:
                    if link.endswith(".pdf"):
                        text = extract_pdf_text(link)
                    else:
                        text = extract_html_text(link)
                except Exception as e:
                    results.append({
                        "url": link,
                        "fraud_type": "Error",
                        "detection_type": "Error",
                        "error": str(e),
                        "cached": False
                    })
                    continue

            # classify using your existing ML functions
                try:
                    fraud_type = classify_fraud_type(text)
                    detection_type = classify_detection_type(text)
                except:
                    fraud_type = "Unknown"
                    detection_type = "Unknown"

            # Save computed values
                results.append({
                    "url": link,
                    "fraud_type": fraud_type,
                    "detection_type": detection_type,
                    "cached": False
                })

            # -------------------------
            # (C) Insert into Supabase cache
            # -------------------------
                supabase.table("fdic_cache").insert({
                    "url": link,
                    "fraud_type": fraud_type,
                    "detection_type": detection_type,
                    "scraped_at": datetime.datetime.utcnow().isoformat()
                }).execute()

                st.write(" → Processed & cached")

        # -------------------------
        # 5. Display results
        # -------------------------
            st.subheader("Results for Top 10 Articles")

            for r in results:
                st.markdown(f"### [{r['url']}]({r['url']})")
                st.write("**Fraud Type:**", r["fraud_type"])
                st.write("**Detection Method:**", r["detection_type"])
                st.write("**Source:**", "Cache" if r["cached"] else "Fresh")
                st.write("---")

        # -------------------------
        # 6. Download results as CSV
        # -------------------------
            import pandas as pd
            df_results = pd.DataFrame(results)

            csv_bytes = df_results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Results as CSV",
                data=csv_bytes,
                file_name="fdic_search_results.csv",
                mime="text/csv"
            )


if __name__ == "__main__":
    main()
