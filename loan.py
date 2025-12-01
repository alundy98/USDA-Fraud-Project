import os
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap.umap_ as umap
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull
import joblib
from urllib.parse import urlparse

# ---------------------------------------------------------------------
# Embedding parsing + merge logic (taken from your reference block)
# ---------------------------------------------------------------------
print("Loading article labels...")
labels_df = pd.read_csv(r"outputs_new_prompt/combined_articles.csv")
print("Loading local embeddings parquet...")
emb_df = pd.read_parquet(r"outputs_new_prompt/combined_embeddings.parquet")

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

print("Parsing embeddings from parquet...")
if "embedding" not in emb_df.columns:
    raise ValueError("Parquet does not contain an 'embedding' column. Check parquet schema.")
emb_df["embedding"] = emb_df["embedding"].apply(parse_embedding)
emb_df["embedding_clean"] = emb_df["embedding"].apply(clean_embedding_array)
emb_df = emb_df[emb_df["embedding_clean"].notnull()].copy()
print(f"Embeddings retained after cleaning: {len(emb_df)}")

labels_df["url_norm"] = labels_df["url"].apply(normalize_url)
emb_df["url_norm"] = emb_df["url"].apply(normalize_url)

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

merged["embedding_clean"] = merged["embedding"].apply(clean_embedding_array)
valid_count = merged["embedding_clean"].notna().sum()
print(f"Valid cleaned embeddings after forcing parquet source: {valid_count}")
merged = merged[merged["embedding_clean"].notna()].copy()
print(f"Loaded {len(merged)} valid articles with embeddings.")

# ---------------------------------------------------------------------
# Helper functions and visualization pipeline (from your loan.py)
# ---------------------------------------------------------------------

def load_data(labels_csv_path, embeddings_parquet_path):
    # This loader is retained for backward compatibility if callers prefer paths
    labels_csv_path = labels_csv_path.replace("\\", "/")
    embeddings_parquet_path = embeddings_parquet_path.replace("\\", "/")
    df_labels = pd.read_csv(labels_csv_path)
    df_emb = pd.read_parquet(embeddings_parquet_path)
    # prefer embedding_clean if present else parse
    if "embedding_clean" not in df_emb.columns and "embedding" in df_emb.columns:
        df_emb["embedding"] = df_emb["embedding"].apply(parse_embedding)
        df_emb["embedding_clean"] = df_emb["embedding"].apply(clean_embedding_array)
    # normalize urls and merge as above
    df_labels["url_norm"] = df_labels["url"].apply(normalize_url)
    df_emb["url_norm"] = df_emb["url"].apply(normalize_url)
    merged_local = df_labels.merge(df_emb, on="url_norm", how="inner", suffixes=("_labels", "_emb"))
    if "embedding_clean" in merged_local.columns and merged_local["embedding_clean"].notna().sum() > 0:
        merged_local["embedding"] = merged_local["embedding_clean"]
    else:
        # try fallback columns
        for col in ["embedding_emb", "embedding"]:
            if col in merged_local.columns and merged_local[col].notna().sum() > 0:
                merged_local["embedding"] = merged_local[col]
                break
    merged_local["embedding_clean"] = merged_local["embedding"].apply(clean_embedding_array)
    merged_local = merged_local[merged_local["embedding_clean"].notna()].copy()
    return merged_local

def filter_loan_fraud(df):
    df_lf = df[df["fraud_group_primary"] == "Loan Fraud"].copy()
    if "amount_numeric" in df_lf.columns:
        df_lf = df_lf[df_lf["amount_numeric"].notna()]
    else:
        raise ValueError("amount_numeric column missing for loan-fraud visualization")
    if "detection_method" not in df_lf.columns:
        raise ValueError("detection_method column missing")
    df_lf["fraud_group_secondary"] = df_lf.get("fraud_group_secondary", "Other")
    df_lf["fraud_group_secondary"] = df_lf["fraud_group_secondary"].fillna("Other").astype(str)
    return df_lf

def compute_umap(df_lf):
    embeddings = np.vstack(df_lf["embedding_clean"].values)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, metric="cosine", random_state=42)
    umap_coords = reducer.fit_transform(embeddings)
    df_lf["umap_x"] = umap_coords[:, 0]
    df_lf["umap_y"] = umap_coords[:, 1]
    return df_lf

def prepare_loan_df(df):
    loan = df[df["fraud_group_primary"] == "Loan Fraud"].copy()
    if loan.empty:
        print("no loan fraud cases found")
        return loan
    loan = loan[loan["embedding_clean"].notna()].copy()
    if loan.empty:
        print("no valid embeddings found for loan fraud")
        return loan
    if "amount_numeric" in loan.columns:
        loan["money_amount"] = pd.to_numeric(loan["amount_numeric"], errors="coerce").fillna(0)
    elif "money_amount" in loan.columns:
        loan["money_amount"] = pd.to_numeric(loan["money_amount"], errors="coerce").fillna(0)
    else:
        loan["money_amount"] = 0
    loan["money_log"] = np.log10(loan["money_amount"].replace(0, np.nan)).fillna(0)
    scaler = MinMaxScaler()
    loan["money_scaled"] = scaler.fit_transform(loan["money_log"].values.reshape(-1, 1)).flatten()
    if "fraud_group_secondary" not in loan.columns:
        loan["fraud_group_secondary"] = "Unknown"
    loan["fraud_group_secondary"] = loan["fraud_group_secondary"].fillna("Unknown").astype(str)
    if "detection_method" not in loan.columns:
        loan["detection_method"] = "Unknown"
    loan["detection_method"] = loan["detection_method"].fillna("Unknown").astype(str)
    if "full_date" in loan.columns:
        loan["full_date_dt"] = pd.to_datetime(loan["full_date"], errors="coerce")
    else:
        loan["full_date_dt"] = pd.NaT
    return loan

def add_similarity_to_cyber(df_all, df_loan):
    if "fraud_group_primary" not in df_all.columns:
        print("source dataframe missing fraud_group_primary")
        return df_loan
    cyber_all = df_all[df_all["fraud_group_primary"].str.lower() == "cyber"]
    if cyber_all.empty:
        print("no cyber examples found for centroid")
        return df_loan
    cyber_all = cyber_all[cyber_all["embedding_clean"].notna()].copy()
    if cyber_all.empty:
        print("no valid cyber embeddings found")
        return df_loan
    cyber_matrix = np.vstack(cyber_all["embedding_clean"].values)
    centroid = cyber_matrix.mean(axis=0).reshape(1, -1)
    loan_matrix = np.vstack(df_loan["embedding_clean"].values)
    sims = cosine_similarity(loan_matrix, centroid).flatten()
    df_loan = df_loan.copy()
    df_loan["sim_to_cyber"] = sims
    return df_loan

def cluster_loan_embeddings(df_loan, n_clusters=5, random_state=42):
    if df_loan.empty:
        print("no loan data to cluster")
        return df_loan
    X = np.vstack(df_loan["embedding_clean"].values)
    k = min(n_clusters, len(df_loan))
    if k <= 1:
        df_loan["cluster"] = 0
        return df_loan
    km = KMeans(n_clusters=k, random_state=random_state)
    labels = km.fit_predict(X)
    df_loan = df_loan.copy()
    df_loan["cluster"] = labels
    return df_loan

def plot_umap_clusters(df_loan, cluster_col="cluster", save_path="loan_umap_clusters.png"):
    if df_loan.empty:
        print("no loan data to visualize")
        return
    X = np.vstack(df_loan["embedding_clean"].values)
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(X)
    df = df_loan.copy()
    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 9))
    ax = plt.gca()
    clusters = sorted(df[cluster_col].unique())
    palette = sns.color_palette("tab10", n_colors=max(3, len(clusters)))
    cluster_color_map = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}
    secondaries = df["fraud_group_secondary"].unique()
    edge_palette = sns.color_palette("husl", n_colors=len(secondaries))
    edge_map = {s: edge_palette[i] for i, s in enumerate(secondaries)}
    for c in clusters:
        sub = df[df[cluster_col] == c]
        ax.scatter(sub["umap_x"], sub["umap_y"], s=(sub["money_scaled"] * 120 + 20), c=[cluster_color_map[c]], edgecolors=[edge_map[s] for s in sub["fraud_group_secondary"]], linewidths=0.6, alpha=0.85, label=f"cluster {c}")
    for c in clusters:
        sub = df[df[cluster_col] == c]
        if len(sub) > 3:
            pts = sub[["umap_x", "umap_y"]].values
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                ax.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.06, facecolor=cluster_color_map[c], edgecolor=None)
            except Exception:
                pass
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(title="Clusters", loc="upper right", bbox_to_anchor=(1.25, 1))
    proxies = []
    p_labels = []
    for s in secondaries:
        proxies.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor=edge_map[s], markersize=8, linewidth=0))
        p_labels.append(s)
    sec_leg = ax.legend(proxies, p_labels, title="fraud_group_secondary", loc="lower right", bbox_to_anchor=(1.25, 0))
    ax.add_artist(leg)
    for c in clusters:
        sub = df[df[cluster_col] == c]
        if sub.empty:
            continue
    # annotate only top monetary outliers per cluster
    for c in clusters:
        sub = df[df[cluster_col] == c]
        if sub.empty:
            continue
        thresh = sub["money_amount"].quantile(0.98)
        out = sub[sub["money_amount"] >= thresh]
        for _, r in out.iterrows():
            ax.text(r["umap_x"] + 0.03, r["umap_y"] + 0.03, str(int(r["money_amount"])), fontsize=8)
    ax.set_title("Loan Fraud UMAP Colored by Cluster with Secondary group edges and outliers")
    ax.set_xlabel("umap 1")
    ax.set_ylabel("umap 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_pca_clusters(df_loan, cluster_col="cluster", save_path="loan_pca_clusters.png"):
    if df_loan.empty:
        print("no loan data to visualize")
        return
    X = np.vstack(df_loan["embedding_clean"].values)
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    df = df_loan.copy()
    df["pc1"] = pcs[:, 0]
    df["pc2"] = pcs[:, 1]
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    clusters = sorted(df[cluster_col].unique())
    palette = sns.color_palette("tab10", n_colors=max(3, len(clusters)))
    cluster_color_map = {c: palette[i % len(palette)] for i, c in enumerate(clusters)}
    for c in clusters:
        sub = df[df[cluster_col] == c]
        ax.scatter(sub["pc1"], sub["pc2"], s=(sub["money_scaled"] * 110 + 20), c=[cluster_color_map[c]], alpha=0.9, label=f"cluster {c}", edgecolors="k", linewidths=0.4)
    ax.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("Loan Fraud PCA Projection Colored by Cluster")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_radar_similarity(df_loan, df_all, by="secondary", save_path="loan_radar_similarity.png"):
    if df_loan.empty:
        print("no loan data for radar")
        return
    df = add_similarity_to_cyber(df_all, df_loan)
    if "sim_to_cyber" not in df.columns:
        print("no similarity column available")
        return
    sns.set_style("whitegrid")
    if by == "secondary":
        groups = df["fraud_group_secondary"].unique()
        labels = list(groups)
        values = [df[df["fraud_group_secondary"] == g]["sim_to_cyber"].mean() for g in groups]
        title = "similarity to cyber by secondary fraud group"
    else:
        if "cluster" not in df.columns:
            print("no cluster column for radar")
            return
        groups = sorted(df["cluster"].unique())
        labels = [f"cluster {c}" for c in groups]
        values = [df[df["cluster"] == c]["sim_to_cyber"].mean() for c in groups]
        title = "similarity to cyber by cluster"
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = np.array(values)
    values = np.concatenate([values, values[:1]])
    angles = angles + angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=1.8)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_detection_by_cluster(df_loan, save_path="detection_by_cluster.png"):
    if df_loan.empty:
        print("no loan data for detection plot")
        return
    if "cluster" not in df_loan.columns:
        print("cluster labels missing")
        return
    df = df_loan.copy()
    counts = df.groupby(["cluster", "detection_method"]).size().reset_index(name="count")
    pivot = counts.pivot(index="cluster", columns="detection_method", values="count").fillna(0)
    pivot = pivot.apply(lambda x: x / x.sum(), axis=1)
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    ax = pivot.plot(kind="bar", stacked=True, colormap="tab20", figsize=(12, 8))
    ax.set_ylabel("proportion of detections")
    ax.set_title("detection method distribution by cluster")
    plt.legend(title="detection method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def summarize_clusters(df_loan, save_path="cluster_summary.csv"):
    if df_loan.empty:
        print("no loan data to summarize")
        return pd.DataFrame()
    if "cluster" not in df_loan.columns:
        print("cluster labels missing for summary")
        return pd.DataFrame()
    summary = df_loan.groupby("cluster").agg(
        count=("cluster", "size"),
        mean_money=("money_amount", "mean"),
        median_money=("money_amount", "median")
    ).reset_index().sort_values("cluster")
    summary.to_csv(save_path, index=False)
    print("cluster summary saved")
    return summary

def plot_covid_fraud_stats(df, text_col="clean_text", group_col="fraud_group_primary",
                           bar_path="covid_proportion.png",
                           pie_path="covid_group_proportions.png"):

    # -----------------------------------------------------
    # Check validity
    # -----------------------------------------------------
    if df.empty:
        print("Dataframe is empty — no COVID visualization generated.")
        return

    if text_col not in df.columns:
        print(f"Text column '{text_col}' missing.")
        return

    if group_col not in df.columns:
        print(f"Fraud group column '{group_col}' missing.")
        return

    # -----------------------------------------------------
    # 1. Identify COVID cases
    # -----------------------------------------------------
    df = df.copy()
    df["is_covid"] = df[text_col].str.contains("covid", case=False, na=False)

    total_cases = len(df)
    covid_cases = df["is_covid"].sum()
    non_covid_cases = total_cases - covid_cases

    print(f"Total loan fraud cases: {total_cases}")
    print(f"COVID-related cases: {covid_cases}")
    print(f"Proportion COVID-related: {covid_cases / total_cases:.3f}")

    if covid_cases == 0:
        print("No COVID-related cases detected — no plots produced.")
        return

    covid_df = df[df["is_covid"]]

    # -----------------------------------------------------
    # 2. Compute fraud group proportions (within COVID cases)
    # -----------------------------------------------------
    group_counts = (
        covid_df[group_col]
        .fillna("Unknown")
        .value_counts()
    )

    # -----------------------------------------------------
    # 3. Plot A — Bar Chart (Proportion of COVID vs Non-COVID)
    # -----------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(["COVID-Related", "Not COVID"], [covid_cases, non_covid_cases])
    plt.title("Proportion of Loan Fraud Cases That Are COVID-Related")
    plt.ylabel("Number of Cases")

    # Annotate bars with percentages
    for i, v in enumerate([covid_cases, non_covid_cases]):
        pct = v / total_cases * 100
        plt.text(i, v + max([covid_cases, non_covid_cases]) * 0.02,
                 f"{pct:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(bar_path, dpi=300)
    plt.show()
    print(f"Saved COVID proportion bar chart → {bar_path}")

    # -----------------------------------------------------
    # 4. Plot B — Pie Chart (COVID group proportions)
    # -----------------------------------------------------
    plt.figure(figsize=(8, 8))
    plt.pie(
        group_counts.values,
        labels=group_counts.index,
        autopct="%1.1f%%",
        startangle=140
    )
    plt.title("Fraud Group Proportions Among COVID-Related Cases")
    plt.tight_layout()
    plt.savefig(pie_path, dpi=300)
    plt.show()
    print(f"Saved COVID group proportion pie chart → {pie_path}")

def run_loan_visuals(df_all=None, n_clusters=5, save_prefix="loan_visual"):
    """
    Full pipeline: prepares data, computes UMAP, clusters embeddings,
    generates all visualizations, and saves cluster summary.
    """
    if df_all is None:
        # use merged created at top
        df_all_local = merged
    else:
        df_all_local = df_all

    print("Preparing loan fraud dataframe...")
    loan = prepare_loan_df(df_all_local)
    if loan.empty:
        print("No loan fraud data to visualize.")
        return loan

    print("Computing UMAP coordinates...")
    loan = compute_umap(loan)

    print("Adding similarity to cyber centroid...")
    loan = add_similarity_to_cyber(df_all_local, loan)

    print(f"Clustering embeddings into {n_clusters} clusters...")
    loan = cluster_loan_embeddings(loan, n_clusters=n_clusters)

    print("Plotting UMAP clusters...")
    plot_umap_clusters(loan, cluster_col="cluster", save_path=f"{save_prefix}_umap_clusters.png")

    print("Plotting PCA clusters...")
    plot_pca_clusters(loan, cluster_col="cluster", save_path=f"{save_prefix}_pca_clusters.png")

    print("Plotting radar similarity to cyber...")
    plot_radar_similarity(loan, df_all_local, by="cluster", save_path=f"{save_prefix}_radar_cluster_sim.png")

    print("Plotting detection method by cluster...")
    plot_detection_by_cluster(loan, save_path=f"{save_prefix}_detection_by_cluster.png")

    print("Summarizing clusters...")
    summary = summarize_clusters(loan, save_path=f"{save_prefix}_cluster_summary.csv")
    print("Columns in merged:", loan.columns.tolist())

    plot_covid_fraud_stats(loan, text_col="clean_text", group_col="fraud_group_primary",
                           bar_path="covid_case_count.png",
                           pie_path="covid_group_proportions.png")
    print("All loan fraud visuals complete!")
    return loan

def plot_fraud_group_counts(df, group_col="fraud_group_primary", save_path="fraud_group_counts.png"):
    # Validate dataframe
    if df.empty:
        print("Dataframe is empty — cannot plot fraud groups.")
        return
    if group_col not in df.columns:
        print(f"Column '{group_col}' is missing — cannot plot fraud groups.")
        return
    # Compute counts
    counts = df[group_col].fillna("Unknown").value_counts().sort_values(ascending=False)
    if counts.empty:
        print("No fraud group values found.")
        return
    # Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 7))
    ax = counts.plot(kind="bar", color="royalblue")
    ax.set_title("Number of Instances of Each Fraud Group", fontsize=16)
    ax.set_ylabel("Case Count", fontsize=14)
    ax.set_xlabel("Fraud Group", fontsize=14)
    for i, v in enumerate(counts.values):
        plt.text(i, v + max(counts.values) * 0.01, str(v), ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"Saved fraud group count visualization → {save_path}")
if __name__ == "__main__":
    merged = load_data(r"outputs_new_prompt/combined_articles.csv", r"outputs_new_prompt/combined_embeddings.parquet")
    plot_fraud_group_counts(merged, group_col="fraud_group_primary", save_path="fraud_group_counts.png")
