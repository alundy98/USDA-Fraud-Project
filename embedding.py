import re
import string
from pathlib import Path
from datetime import datetime
from supabase import create_client, Client

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

tqdm.pandas()
nltk.download("stopwords")
nltk.download("wordnet")

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

SUPABASE_URL = "https://zciswiifptrhorepsoxb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpjaXN3aWlmcHRyaG9yZXBzb3hiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjE2NTc1NzUsImV4cCI6MjA3NzIzMzU3NX0.kHl_3ucU8dIuctuvevmSTd0BcT1GwURJmHpzt0hYDBc"
TABLE_NAME = "fdic_articles"

# Getting articles from Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
res = supabase.table(TABLE_NAME).select("*").execute()
df = pd.DataFrame(res.data)
print(f"Loaded {len(df)} rows from Supabase table '{TABLE_NAME}'")

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

print("Cleaning text...")
df["clean_text"] = df["text"].progress_apply(clean_text)

# Embedding articles
print("Embedding articles...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["clean_text"].tolist(), show_progress_bar=True, normalize_embeddings=True)

# Automaticly finding the best number of clusters using HDBSCAN
min_cluster_size = 8
min_samples = 5

print(f"\nClustering with HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})...")

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

print(f"\nFormed {num_clusters} clusters (excluding outliers)")
print(df["cluster"].value_counts())

if num_clusters < 4:
    print("\nHDBSCAN formed fewer than 4 clusters. Switching to KMeans with 4 clusters...")
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
    vec = TfidfVectorizer(stop_words="english", max_features=2000)
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

# Getting the top 3 articles for each clustere based on similarity score
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

# Saving summary csv
summary_df = pd.DataFrame(cluster_summaries)
summary_path = OUT_DIR / "embedding_cluster_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved cluster summary with top articles: {summary_path}")
