import os
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import umap.umap_ as umap
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
import joblib
from urllib.parse import urlparse

# Load local data
print("Loading article labels...")
labels_df = pd.read_csv("combined_articles.csv")
print("Loading local embeddings parquet...")
emb_df = pd.read_parquet(r"combined_embeddings.parquet")

# Helper functions mostly copy and pasted from embeds and old vis code
def normalize_url(u):
    if pd.isna(u):
        return None
    s = str(u).strip()
    if s == "":
        return None
    p = urlparse(s)
    netloc = p.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = p.path.rstrip("/")
    norm = netloc + path
    return norm

def parse_embedding(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        try:
            v = json.loads(x)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            parts = inner.split()
            try:
                nums = [float(p) for p in parts]
                return nums
            except Exception:
                return None
    return None

def clean_embedding_array(e):
    if isinstance(e, np.ndarray):
        return e if e.size == 1536 else None
    if isinstance(e, list):
        return np.array(e, dtype=float) if len(e) == 1536 else None
    return None

# Prepare embeddings 
print("Parsing embeddings from parquet...")
if "embedding" not in emb_df.columns:
    raise ValueError("Parquet does not contain an 'embedding' column. Check parquet schema.")
emb_df["embedding"] = emb_df["embedding"].apply(parse_embedding)
emb_df["embedding_clean"] = emb_df["embedding"].apply(clean_embedding_array)
emb_df = emb_df[emb_df["embedding_clean"].notnull()].copy()
print(f"Embeddings retained after cleaning: {len(emb_df)}")

# Normalize URLs and merge
labels_df["url_norm"] = labels_df["url"].apply(normalize_url)
emb_df["url_norm"] = emb_df["url"].apply(normalize_url)

# Quick diagnostics if something's strange
labels_only = set(labels_df["url_norm"].dropna().unique()) - set(emb_df["url_norm"].dropna().unique())
emb_only = set(emb_df["url_norm"].dropna().unique()) - set(labels_df["url_norm"].dropna().unique())
print(f"Unique url_norm in labels: {len(labels_df['url_norm'].dropna().unique())}, in emb: {len(emb_df['url_norm'].dropna().unique())}")
if len(labels_only) > 0:
    print(f"Example URLs present only in labels (up to 5): {list(labels_only)[:5]}")
if len(emb_only) > 0:
    print(f"Example URLs present only in embeddings (up to 5): {list(emb_only)[:5]}")

merged = labels_df.merge(emb_df, on="url_norm", how="inner", suffixes=("_labels", "_emb"))
print(f"Merged rows: {len(merged)}")
if len(merged) == 0:
    raise RuntimeError("Merge returned 0 rows after URL normalization. Inspect URL normalization differences between CSV and parquet.")

#use parquet embeddings (embedding_clean) and drop CSV-string embeddings
if "embedding_clean" in merged.columns and merged["embedding_clean"].notna().sum() > 0:
    merged["embedding"] = merged["embedding_clean"]
    print("Using 'embedding_clean' from parquet as 'embedding'.")
elif "embedding_emb" in merged.columns and merged["embedding_emb"].notna().sum() > 0:
    merged["embedding"] = merged["embedding_emb"]
    print("Using 'embedding_emb' from parquet as 'embedding'.")
elif "embedding" in merged.columns:
    merged["embedding"] = merged["embedding"]
    print("Using 'embedding' column available after merge.")
else:
    raise KeyError("No embedding column found after merge. Columns: " + ", ".join(merged.columns.tolist()))
for col in ["embedding_labels", "embedding_x", "embedding_y"]:
    if col in merged.columns and col != "embedding":
        merged.drop(columns=[col], inplace=True)
        print(f"Dropped column {col} to avoid CSV-string embeddings overwriting parquet vectors.")

# Final cleaning check
merged["embedding_clean"] = merged["embedding"].apply(clean_embedding_array)
valid_count = merged["embedding_clean"].notna().sum()
print(f"Valid cleaned embeddings after forcing parquet source: {valid_count}")
merged = merged[merged["embedding_clean"].notna()].copy()
print(f"Loaded {len(merged)} valid articles with embeddings.")

# UMAP Visualization (reverted to original)
def semantic_umap_map(df, save_path="umap_plot.png"):
    emb_matrix = np.vstack(df["embedding_clean"].values)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb_2d = reducer.fit_transform(emb_matrix)
    df["x"], df["y"] = emb_2d[:, 0], emb_2d[:, 1]

    plt.figure(figsize=(10, 7))
    hue_col = "fraud_group_primary" if "fraud_group_primary" in df.columns else None

    if hue_col is None:
        sb.scatterplot(data=df, x="x", y="y", alpha=0.7)
    else:
        sb.scatterplot(
            data=df, x="x", y="y", hue=hue_col, alpha=0.7, palette="tab10"
        )

    plt.title("Semantic Map of Fraud Narratives (UMAP)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

# Semantic Drift Over Time 
def semantic_drift_over_time(df, save_path="semantic_drift_yr_to_yr.png"):
    df = df.copy()

    # Use punlished column directly as the year
    centroids = df.groupby("published")["embedding_clean"].apply(
        lambda x: np.mean(np.vstack(x.values), axis=0)
    )

    years = sorted(centroids.index)
    print("Years found:", years)

    drift = []
    xlabels = []

    for i in range(len(years) - 1):
        y1, y2 = years[i], years[i + 1]
        c1, c2 = centroids[y1], centroids[y2]
        dist = cosine_distances([c1], [c2])[0, 0]
        drift.append(dist)
        xlabels.append(f"{y1}-{y2}")

    if not drift:
        print("Warning: No drift data to plot (need at least two years).")

    plt.figure(figsize=(12, 6))
    plt.plot(xlabels, drift, marker="o")
    plt.title("Semantic Drift of Fraud Narratives (Year-to-Year)")
    plt.xlabel("Year Transitions")
    plt.ylabel("Cosine Distance")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

    # Save drift data for later use
    joblib.dump({"years": years, "drift": drift, "labels": xlabels}, "semantic_drift.joblib")

# Fraud Magnitude by Fraud Group by mean
def fraud_magnitude_by_group(df, save_path="fraud_mean_amount_by_group.png"):
    df = df.copy()

    # Keep only valid numeric amounts > 0
    df = df[df["amount_numeric"].notna() & (df["amount_numeric"] > 0)]

    # Compute mean amount by fraud_group_primary
    agg_df = df.groupby("fraud_group_primary").agg(
        mean_amount=("amount_numeric", "mean"),
        case_count=("amount_numeric", "count")
    ).reset_index()

    # Sort by mean_amount for easier visualization
    agg_df = agg_df.sort_values("mean_amount", ascending=False)

    plt.figure(figsize=(12, 7))
    sb.barplot(
        data=agg_df,
        x="fraud_group_primary",
        y="mean_amount",
        palette="tab10"
    )
    plt.title("Average Amount Involved by Fraud Group")
    plt.xlabel("Fraud Group (Primary)")
    plt.ylabel("Mean Amount Involved")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

# Loan Fraud Secondary Types
def loan_fraud_secondary_hist(df, save_path="loan_fraud_secondary_counts.png"):
    df = df.copy()
    # Filter to only Loan Fraud primary group
    loan_df = df[df["fraud_group_primary"] == "Loan Fraud"]
    # Only keep rows with a valid secondary type
    loan_df = loan_df[loan_df["fraud_group_secondary"].notna() & (loan_df["fraud_group_secondary"] != "")]
    # Count number of cases per secondary type
    counts = loan_df["fraud_group_secondary"].value_counts().reset_index()
    counts.columns = ["fraud_group_secondary", "case_count"]
    # Sort by count for better readability
    counts = counts.sort_values("case_count", ascending=False)
    plt.figure(figsize=(10, 6))
    sb.barplot(
        data=counts,
        x="fraud_group_secondary",
        y="case_count",
        palette="tab10"
    )
    plt.title("Number of Cases by Loan Fraud Secondary Types")
    plt.xlabel("Loan Fraud Secondary Type")
    plt.ylabel("Number of Cases")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

# Fraud Type vs Detection Method Similarity
def fraud_vs_detection_similarity(df, save_path="fraud_detection_similarity_heatmap.png"):
    df = df.copy()
    # Keep only rows with valid fraud_group_primary and detection_method
    df = df[df["fraud_group_primary"].notna() & df["detection_method"].notna()]
    fraud_types = df["fraud_group_primary"].unique()
    detection_methods = df["detection_method"].unique()
    # Compute centroids for each fraud type × detection method
    centroids = {}
    for f in fraud_types:
        centroids[f] = {}
        for d in detection_methods:
            subset = df[(df["fraud_group_primary"] == f) & (df["detection_method"] == d)]
            if len(subset) > 0:
                emb_matrix = np.vstack(subset["embedding_clean"].values)
                centroids[f][d] = np.mean(emb_matrix, axis=0)
            else:
                centroids[f][d] = None
    # Build similarity matrix
    sim_matrix = pd.DataFrame(index=fraud_types, columns=detection_methods, dtype=float)
    for f in fraud_types:
        for d in detection_methods:
            if centroids[f][d] is not None:
                # cosine similarity between fraud type centroid and detection method centroid
                sim = cosine_distances([centroids[f][d]], [centroids[f][d]])  # self distance = 0
                # convert distance to similarity (1 - distance)
                sim_matrix.loc[f, d] = 1 - sim[0][0]
            else:
                sim_matrix.loc[f, d] = np.nan
    # Plot heatmap
    plt.figure(figsize=(12, 8))
    sb.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Similarity'})
    plt.title("Fraud Type vs Detection Method Similarity (Cosine)")
    plt.xlabel("Detection Method")
    plt.ylabel("Fraud Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()

# Hierarchical Clustering Dendrogram of Fraud Groups
def fraud_group_dendrogram(df):
    # find centroids for each fraud_group_primary
    centroids = df.groupby("fraud_group_primary")["embedding_clean"].apply(
        lambda x: np.mean(np.vstack(x.values), axis=0)
    )
    
    # Cosine similarity between centroids
    sim_matrix = cosine_similarity(np.vstack(centroids.values))
    # Convert similarity to distance for clustering
    dist_matrix = 1 - sim_matrix
    
    # Hierarchical clustering
    linked = linkage(dist_matrix, method='average')
    
    plt.figure(figsize=(12, 6))
    dendrogram(
        linked,
        labels=centroids.index.tolist(),
        orientation='top',
        leaf_rotation=45,
        leaf_font_size=12,
        color_threshold=0.5
    )
    plt.title("Hierarchical Clustering of Fraud Groups (Semantic Similarity)")
    plt.ylabel("Distance (1 - Cosine Similarity)")
    plt.tight_layout()
    plt.savefig("fraud_group_dendrogram.png", dpi=300)
    plt.show()
    plt.close()

# MDS Map of Fraud Group Centroids
from sklearn.manifold import MDS
def fraud_group_mds_map(df):
    # Compute centroids
    centroids = df.groupby("fraud_group_primary")["embedding_clean"].apply(
        lambda x: np.mean(np.vstack(x.values), axis=0)
    )
    # Cosine distance
    sim_matrix = cosine_similarity(np.vstack(centroids.values))
    dist_matrix = 1 - sim_matrix
    
    # MDS projection into 2D
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    mds_2d = mds.fit_transform(dist_matrix)
    
    plt.figure(figsize=(10, 7))
    for i, label in enumerate(centroids.index):
        plt.scatter(mds_2d[i, 0], mds_2d[i, 1], s=100)
        plt.text(mds_2d[i, 0]+0.01, mds_2d[i, 1]+0.01, label, fontsize=10)
    
    plt.title("MDS Map of Fraud Group Centroids (Semantic Positions)")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("fraud_group_mds.png", dpi=300)
    plt.show()
    plt.close()

# Semantic + Amount Outlier Analysis
def semantic_amount_outliers(df, top_n=20):
    df = df.copy()
    
    # find group centroids
    centroids = df.groupby("fraud_group_primary")["embedding_clean"].apply(
        lambda x: np.mean(np.vstack(x.values), axis=0)
    )
    # Compfindute semantic distance from group centroid
    def semantic_distance(row):
        group = row["fraud_group_primary"]
        centroid = centroids[group]
        return cosine_distances([row["embedding_clean"]], [centroid])[0, 0]
    
    df["semantic_distance"] = df.apply(semantic_distance, axis=1)
    
    # Filter out rows with zero or unknown amounts
    df = df[df["amount_numeric"].notna() & (df["amount_numeric"] > 0)]
    
    # Log transform amount to remove skew
    df["log_amount"] = np.log1p(df["amount_numeric"])
    
    # Scatterplot: semantic distance v log amount
    plt.figure(figsize=(12, 7))
    sb.scatterplot(
        data=df, 
        x="semantic_distance", 
        y="log_amount", 
        hue="fraud_group_primary",
        alpha=0.7,
        palette="tab10"
    )
    
    # Highlight top N semantic outliers
    top_outliers = df.nlargest(top_n, "semantic_distance")
    plt.scatter(
        top_outliers["semantic_distance"], 
        top_outliers["log_amount"], 
        color="red", 
        edgecolor="black", 
        s=100, 
        label=f"Top {top_n} Semantic Outliers"
    )
    
    # Add small labels for outliers
    for _, row in top_outliers.iterrows():
        label = row.get("fraud_group_secondary", row["fraud_group_primary"])
        plt.text(
            row["log_amount"],
            row["semantic_distance"] + 0.005,
            str(label), 
            fontsize=7,
            alpha=0.8
        )
    
    plt.title("Combined Semantic + Amount Outliers in Fraud Cases")
    plt.xlabel("Semantic Distance from Fraud Group Centroid")
    plt.ylabel("Log(Amount Involved)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("semantic_amount_outliers.png", dpi=300)
    plt.show()
    plt.close()


#the best way to run this is to call funcs directly, otherwise it gets hairy
if __name__ == "__main__":
    semantic_amount_outliers(merged, top_n=20)