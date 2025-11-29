import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np
import dotenv
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_FILE = "outputs_w_ai/article_labels.csv"

# === STEP 1: Load your CSV ===
df = pd.read_csv(INPUT_FILE)

# Setting aside unkown detection methods so they don't interfere with embeddings
unknown_mask = df['detection_method'].str.lower() == "unknown"
known_df = df.loc[~unknown_mask].copy()
unknown_df = df.loc[unknown_mask].copy()

methods = known_df['detection_method'].dropna().unique().tolist()
# === STEP 2: Initialize OpenAI client ===
client = OpenAI(api_key=OPENAI_API_KEY)

# === STEP 3: Generate embeddings ===
print("Generating embeddings... this might take a few seconds.")

response = client.embeddings.create(
    input=methods,
    model="text-embedding-3-small"
)

embeddings = np.array([d.embedding for d in response.data])

# Create a dataframe just for methods
methods_df = pd.DataFrame({
    'detection_method': methods
})

# === STEP 4: Cluster the embeddings ===
n_clusters = 10  # adjust 5–10 depending on desired grouping
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
methods_df["cluster"] = kmeans.fit_predict(embeddings)

methods_df["embedding"] = list(embeddings)
cluster_centers = kmeans.cluster_centers_

print("Generating descriptive titles for each cluster...")

cluster_titles = {}
for cluster_id in sorted(methods_df["cluster"].unique()):
    cluster_methods = methods_df.loc[methods_df["cluster"] == cluster_id, 'detection_method'].tolist()
    sample_methods = cluster_methods[:20]  # limit for token efficiency

    prompt = f"""
    You are labeling groups of fraud detection techniques.
    Given the following examples of detection methods:

    {sample_methods}

    Write a concise 2–5 word title summarizing the shared theme of this cluster.
    The title should sound like a fraud detection category name,
    such as "Transaction Velocity Checks", "Device Fingerprinting", or "Behavioral Analytics".
    """

    chat_response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "You are an expert in fraud detection and data science."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
    )

    cluster_title = chat_response.choices[0].message.content.strip()
    cluster_titles[cluster_id] = cluster_title
    print(f"Cluster {cluster_id}: {cluster_title}")

# Map titles to each method
methods_df["cluster_title"] = methods_df["cluster"].map(cluster_titles)

# ============================================================
# STEP 5: MERGE BACK WITH ORIGINAL DATAFRAME
# ============================================================
df = df.merge(methods_df[['detection_method', "cluster", "cluster_title"]], on='detection_method', how="left")
df.loc[df['detection_method'].str.lower() == "unknown", ["cluster", "cluster_title"]] = [None, "Unknown"]



def representative_methods(cluster_id):
    cluster_points = methods_df[methods_df["cluster"] == cluster_id]
    sims = cosine_similarity([cluster_centers[cluster_id]], np.vstack(cluster_points["embedding"]))[0]
    top_idx = sims.argsort()[-3:][::-1]
    return cluster_points.iloc[top_idx]['detection_method'].tolist()

cluster_summaries = {i: representative_methods(i) for i in range(n_clusters)}
# === STEP 6: Save results ===
df.to_csv(INPUT_FILE, index=False)
print(f"\nUpdated {INPUT_FILE} with 'cluster' and 'cluster_title' columns.")

# Optional: print summary
print("\nCluster Titles:")
for cid, title in cluster_titles.items():
    print(f"  {cid}: {title}")