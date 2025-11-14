#notebook
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
import re
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import joblib
from ast import literal_eval
from model_train import parse_llm_labels, parse_embedding

#for older visuals, you can j use older versions of this .py

#al = article_labels
df = pd.read_csv("outputs_w_ai/article_labels.csv")
embeds = pd.read_parquet("outputs_w_ai/article_embedding_full.parquet")
if isinstance(embeds["embedding"].iloc[0], str):
    embeds["embedding"] = embeds["embedding"].apply(literal_eval)
    embeds["embedding"] = embeds["embedding"].apply(np.array)

common_cols = [c for c in df.columns if c in embeds.columns and c != "url"]

# Drop overlapping columns from df before merging
df = df.drop(columns=common_cols, errors="ignore")#
#fraud grouping:
# Define your fraud category keywords
fraud_groups = {
    "Insider Abuse": ["insider abuse", "internal fraud"],
    "Counterfeit Fraud": ["subscription fraud", "credit card fraud", "counterfeiting", "counterfeit check fraud", "counterfeit money orders","unemployment insurance fraud"],
    "Banking Fraud": ["unfair and deceptive practices", "udap", "non sufficient funds fees", "bank fraud"],
    "Identity Fraud": ["identity fraud", "identity theft", "imposter scams"],
    "Elder Abuse": ["senior life settlements", "elder financial abuse"],
    "Cyber Fraud": ["crypto fraud", "cryptocurrency fraud", "wire", "electronic payment", "unauthorized access"],
    "Real Estate Fraud": ["owner occupancy fraud", "misrepresentation"],
    "Mail Fraud": ["pen pal scam", "mail fraud"],
    "Loan Fraud": ["predatory lending", "indirect auto lending"]
}
detection_groups = {
    "Audit":["audit","rigorous examination proceedures", "analytical methods"],
    "Algorithmic Systems":["cross-system counterparty screening utility","algorithmic detection","AML model","model","mechanisms","robust fraud detection"],
    "Policy": ["collaborating among financial institutions", "compliance and oversight", "collaboration between banks and law enforcement"],
    "Educational": ["educational resources","financial education","examination process"],
    "Regulator Enforcement": ["regulatory scrutiny","rigourous examination procedures"],
    "Internal Review":["interal reviews","supervisory guidance"],
    "Investigation": ["monitoring","monitoring financial accounts", "monitoring billing statements","monitoring and recording", "monitoring account statements" ]
}
#helper: normalize text
def normalize_text(text):
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)

#main grouping logic
def assign_fraud_group(fraud_type):
    text = normalize_text(fraud_type)
    scores = {group: 0 for group in fraud_groups}
    for group, keywords in fraud_groups.items():
        for kw in keywords:
            if kw in text:
                scores[group] += 1
    # Get the group with the most keyword matches
    best_match = max(scores, key=scores.get)
    if scores[best_match] == 0:
        return "Other / Unknown"
    return best_match

def assign_detection_group(detection_method):
    text = normalize_text(detection_method)
    scores = {group: 0 for group in detection_groups}
    for group, keywords in detection_groups.items():
        for kw in keywords:
            if kw in text:
                scores[group] += 1
    # Get the group with the most keyword matches
    best_match = max(scores, key=scores.get)
    if scores[best_match] == 0:
        return "Other / Unknown"
    return best_match

# df = pd.read_csv("article_labels.csv")  # for example
df["fraud_group"] = df["fraud_type"].apply(assign_fraud_group)
df["detection_group"] = df["detection_method"].apply(assign_detection_group)
priority_df_cols = ["fraud_type", "detection_method", "llm_labels","fraud_group","detection_group"]

common_cols = [
    c for c in df.columns 
    if c in embeds.columns and c not in ["url"] + priority_df_cols
]

df = df.drop(columns=common_cols, errors="ignore")

# Merge datasets
al = pd.merge(df, embeds, on="url", how="inner")

#Filter out unknowns
al = al[al["fraud_type"].fillna("").str.lower() != "unknown"].reset_index(drop=True)
al = al[al["fraud_group"].fillna("").str.lower() != "other / unknown"].reset_index(drop=True)

#semantic analysis of fraud, what fraud type's narratives stick close together, like if elderly people are targets commonly in identity theft
#it may be semantically close to cases of elder financial abuse
from sklearn.metrics.pairwise import cosine_similarity
import umap
from umap import UMAP
# Stack embeddings



# Compute centroids by fraud_type and year
centroids = {}

for year, group in al.groupby("published"):
    centroids[year] = {}
    for ftype, fgroup in group.groupby("fraud_group"):
        emb_matrix = np.vstack(fgroup["embedding"].values)
        centroids[year][ftype] = emb_matrix.mean(axis=0)

# Measure semantic drift per fraud type
#(cosine similarity year-to-year)
drift_records = []

fraud_types = al["fraud_group"].unique()

for f in fraud_types:
    # sort years where this fraud type exists
    years_available = sorted([y for y in centroids if f in centroids[y]])
    
    # compute pairwise drift Y_t → Y_(t+1)
    for i in range(len(years_available) - 1):
        y1, y2 = years_available[i], years_available[i+1]
        v1, v2 = centroids[y1][f], centroids[y2][f]

        cos = cosine_similarity([v1], [v2])[0][0]

        drift_records.append({
            "fraud_group": f,
            "year_start": y1,
            "year_end": y2,
            "cosine_similarity": cos,
            "semantic_change": 1 - cos
        })

drift_df = pd.DataFrame(drift_records)

# semantic drift over time
plt.figure(figsize=(12, 7))

# Filter for years 2022–2025 only
filtered = drift_df[(drift_df["year_end"] >= 2022) & (drift_df["year_end"] <= 2025)]

for f in fraud_types:
    subset = filtered[filtered["fraud_group"] == f]
    if len(subset) == 0:
        continue

    plt.plot(
        subset["year_end"],
        subset["cosine_similarity"],
        marker="o",
        label=f
    )

plt.title("Semantic drift of fraud types over time 2019–2025")
plt.xlabel("Year")
plt.xticks([])
plt.ylabel("")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend(title="Fraud Group")
plt.tight_layout()
plt.show()