import re
import string
from pathlib import Path
from datetime import datetime
from supabase import create_client, Client
import pandas as pd
import numpy as np
import json
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import nltk
import dotenv
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
tqdm.pandas()
nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
OUT_DIR = Path("outputs_new_prompt/last")
OUT_DIR.mkdir(exist_ok=True)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "oig_articles"

# Getting articles from Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
res = supabase.table(TABLE_NAME).select("*").execute()
df = pd.DataFrame(res.data)
print(f"Loaded {len(df)} rows from Supabase table '{TABLE_NAME}'")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello from my API!"}
    ]
)
print(response.choices[0].message.content)
# Cleaning text before embedding
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

print("Cleaning text:")
df["clean_text"] = df["text"].progress_apply(clean_text)
#____________________________________________________________________________________________________________________________________---
#This is where the ai logic flow starts, this gets full article embeddings
# Embedding articles
print("Embedding articles...")
#there was an issue with the 3 small model not liking how big certain articles were
#8500 token limit, biggest we got was around 13977 and broke it.
#new ver. includes splitting into chunks for easier embedding
#combines chunks together at the end if necessary
def get_ai_embeds(texts, model_name="text-embedding-3-small", chunk_size=7000):

    embeddings = []
    print(f"Embedding {len(texts)} documents using {model_name}...")
    for text in tqdm(texts, desc=f"embedding with OpenAI {model_name} model"):
        # Defensive text cleaning
        if not isinstance(text, str) or len(text.strip()) == 0:
            embeddings.append(np.zeros(1536))
            continue
        # split into chunks (4 chars per token roughly)
        chunks = [text[i:i + chunk_size * 4] for i in range(0, len(text), chunk_size * 4)]
        chunk_embs = []
        for chunk in chunks:
            try:
                response = client.embeddings.create(
                    model=model_name,
                    input=chunk
                )
                chunk_embs.append(response.data[0].embedding)
            except openai.BadRequestError as e:
                print(f"Skipped one chunk due to size or invalid input: {e}")
                continue
            except Exception as e:
                print(f"Error while embedding chunk: {e}")
                continue
        if chunk_embs:
            # average chunk embeddings
            emb = np.mean(chunk_embs, axis=0)
        else:
            # fallback vector if chunks failed
            emb = np.zeros(1536)
        embeddings.append(emb)
    return embeddings

df["embedding"] = get_ai_embeds(df["clean_text"].tolist())
embeddings = np.array(df["embedding"].tolist())
#this calls the mdoel to get a summary of each article, necessary to feed the fraud type identifer
#also something we can call in the future for the dashboard
#USE THIS FOR SUMMARIES
def summarize_article(text):
    prompt = f"""
    Summarize the following article in 4–5 concise sentences for a financial crime dataset.
    Focus on capturing:
    1. Who or what organization was involved.
    2. The main fraud or misconduct type (describe briefly, include all relevant details prioritize that).
    3. Whether it was detected after it occurred or prevented beforehand.
    4. The method or mechanism that led to detection or prevention.
    5. Any key outcomes (fines, arrests, policy changes, etc.).
    6. the amount of money incolved if mentioned.

    Be factual and structured so that another model can extract fraud_type and method clearly.

    Article:
    {text}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()
df["summary"] = df["clean_text"].progress_apply(summarize_article)

#this calls the model on the summary to get a fraud_type value and a detection_type value
def get_fraud_type(summary):
    prompt = f"""
    You are analyzing a summary of a fraud-related article.

    Your tasks:
    1. Identify the *raw*, descriptive fraud type mentioned in the article.
    2. Map the fraud to one or two of the following **12 standard fraud groups** 
       (select a secondary one ONLY if it meaningfully applies to fully describe the fraud):

       - Internal Banking Misconduct Fraud
       - Account Manipulation Fraud
       - Loan Fraud
       - Insider Fraud
       - Elder Fraud
       - Payment Fraud
       - Tax Fraud
       - Investment Fraud
       - Insurance Fraud
       - Healthcare Fraud
       - Cyber Fraud
       - Identity Fraud
       - Money Laundering Fraud
       - Government Benefits Fraud

    3. Extract the **detection or prevention method**, using the following list:
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

    4. Identify the **location** (City/State), or respond "Unknown".

    5. Extract the **amount of money involved**, if mentioned.  
       - Return numbers as e.g.,"$850,000".  
       - If not present or fraud not involving finances, return "Unknown".

    Respond **ONLY in JSON** with the following keys:

    {{
        "fraud_type": "<raw descriptive fraud type>",
        "fraud_group_primary": "<primary fraud category>",
        "fraud_group_secondary": "<secondary category or 'None'>",
        "detection_method": "<method or 'Unknown'>",
        "location": "<City/State or 'Unknown'>",
        "amount_involved": "<extracted dollar amount or 'Unknown'>"
    }}

    Summary:
    {summary}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error:", e)
        return None

print("extracting fraud labels from article summaries:")
df["llm_labels"] = df["clean_text"].progress_apply(get_fraud_type)

label_path = OUT_DIR / "final_oig_article_labels.csv"
df.to_csv(label_path, index=False)
print(f"Saved labeled articles: {label_path}")
#import tieimporttieimporttieimporttieimporttieimportteie

import time
#function for saferAI calling, we dont need to use it right now, but call it
#for any time a user in supabase prompts a re-scrape/embed/summarize, so they dont get A BREAKING ERROR
def safe_openai_call(func, *args, retries=3, **kwargs):
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except openai.RateLimitError:
            wait = 5 * (attempt + 1)
            print(f"Rate limited — retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                return None

# Automaticly finding the best number of clusters using HDBSCAN
min_cluster_size = 8
min_samples = 5

print(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

cluster_model = hdbscan.HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=min_samples,
    metric="euclidean"
)
df["cluster"] = cluster_model.fit_predict(embeddings)

# Looking at HDBSCAN results to see how many clusters were made
# If less than 4 clusters were made, fallback to KMeans to manually make 5 clusters to ensure results are useful
# More clusters were tested, any clusters beyond 4 were mostly noise or too small to be significant or useful
valid_clusters = [c for c in df["cluster"].unique() if c != -1]
num_clusters = len(valid_clusters)

print(f"Formed {num_clusters} clusters (excluding outliers)")
print(df["cluster"].value_counts())

if num_clusters < 4:
    print("HDBSCAN formed less than 4 clusters switching to KMeans 4 clusters:")
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(embeddings)
    valid_clusters = sorted(df["cluster"].unique())
    num_clusters = len(valid_clusters)
    print(f"KMeans formed {num_clusters} clusters.")
    print(df["cluster"].value_counts())

# Getting keywords from each cluster
def extract_keywords(texts, top_n=10):
    if len(texts) == 0:
        return []
    vec = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 3),
        min_df=2
    )
    X = vec.fit_transform(texts)
    tfidf_mean = np.asarray(X.mean(axis=0)).ravel()
    top_indices = tfidf_mean.argsort()[::-1][:top_n]
    return [vec.get_feature_names_out()[i] for i in top_indices]

cluster_keywords = {}
for label in valid_clusters:
    texts = df.loc[df["cluster"] == label, "clean_text"].tolist()
    kws = extract_keywords(texts)
    cluster_keywords[label] = kws
    print(f"Cluster {label} keywords:", ", ".join(kws))

# Getting the top 3 articles from each cluster based on similarity score
def get_top_articles_with_scores(label, top_n=3):
    cluster_indices = df.index[df["cluster"] == label]
    if len(cluster_indices) == 0:
        return []
    cluster_embs = embeddings[cluster_indices]
    centroid = cluster_embs.mean(axis=0)
    sims = cosine_similarity([centroid], cluster_embs)[0]
    top_idx = sims.argsort()[-top_n:][::-1]
    top_articles = []
    for idx in top_idx:
        row_idx = cluster_indices[idx]
        title = df.iloc[row_idx].get("title", f"Doc {row_idx}")
        url = df.iloc[row_idx].get("url", "")
        sim = float(sims[idx])
        top_articles.append((title, url, sim))
    return top_articles

cluster_summaries = []
for label in valid_clusters:
    cluster_docs = df[df["cluster"] == label]
    if len(cluster_docs) <= 1:
        continue

    reps = get_top_articles_with_scores(label)
    row = {
        "cluster_id": label,
        "keywords": ", ".join(cluster_keywords[label]),
        "num_articles": len(cluster_docs)
    }
    for i, (title, url, sim) in enumerate(reps, 1):
        row[f"title_{i}"] = title
        row[f"url_{i}"] = url
        row[f"score_{i}"] = round(sim, 3)
    cluster_summaries.append(row)


# Saving summary csve
summary_df = pd.DataFrame(cluster_summaries)
summary_path = OUT_DIR / "fdic_oig_embedding_cluster_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved cluster summary with top articles: {summary_path}")
# Saving full article CSV with cluster info and relevance scores
print("Computing per-article relevance scores")

relevance_scores = []
for label in valid_clusters:
    cluster_indices = df.index[df["cluster"] == label]
    cluster_embs = embeddings[cluster_indices]
    centroid = cluster_embs.mean(axis=0)
    sims = cosine_similarity([centroid], cluster_embs)[0]
    for idx, sim in zip(cluster_indices, sims):
        relevance_scores.append((idx, sim))

# Map the mf relevance score
relevance_map = {idx: sim for idx, sim in relevance_scores}
df["relevance_score"] = df.index.map(relevance_map).fillna(np.nan)

# Replace -1 clusters with None, add keywords for clarity
df["cluster"] = df["cluster"].apply(lambda x: None if x == -1 else x)
df["cluster_keywords"] = df["cluster"].map(
    lambda x: ", ".join(cluster_keywords.get(x, [])) if x is not None else ""
)
embedding_out = OUT_DIR / "oig_article_embedding_full.parquet"
df[["url", "text", "embedding", "cluster", "relevance_score"]].to_parquet(embedding_out, index=False)

