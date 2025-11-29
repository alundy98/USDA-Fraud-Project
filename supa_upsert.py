import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import numpy as np
import json
import math
import time

# -------------------------
# Load environment variables
# -------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------------
# Load dataset
# -------------------------
path = r"outputs_new_prompt\last\oig_article_embedding_full.parquet"  # raw string for Windows paths
print(f"Loading: {path}")

if path.endswith(".csv"):
    df = pd.read_csv(path)
else:
    df = pd.read_parquet(path)

print("Rows loaded:", len(df))

# -------------------------
# Required columns for Supabase
# -------------------------
columns_in_supabase = {"url", "text", "embedding", "relevance_score"}  # adjust to match your table
missing = {"url", "text", "embedding"} - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------------
# Clean text & drop duplicates
# -------------------------
df = df[df["text"].notna() & (df["text"].str.strip() != "")]
df = df.drop_duplicates(subset=["url"], keep="first")
print("Rows after cleaning text & duplicates:", len(df))

# -------------------------
# Recursive JSON-safe conversion
# -------------------------
def make_json_safe(obj):
    """
    Recursively convert everything to JSON-safe:
    - NaN / inf / -inf -> 0.0
    - numpy types -> Python native types
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif obj is None:
        return None
    return obj

# -------------------------
# Clean embeddings
# -------------------------
def clean_embedding(x):
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except:
            return None
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, list):
        return [0.0 if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else float(v) for v in x]
    return None

df["embedding"] = df["embedding"].apply(clean_embedding)

# -------------------------
# Clean numeric columns
# -------------------------
if "relevance_score" in df.columns:
    df["relevance_score"] = df["relevance_score"].apply(
        lambda x: float(x) if x is not None and not (math.isnan(x) or math.isinf(x)) else None
    )

# -------------------------
# Drop columns not in Supabase
# -------------------------
df = df[[c for c in df.columns if c in columns_in_supabase]]

# -------------------------
# Replace top-level NaN → None
# -------------------------
df = df.where(pd.notnull(df), None)

# -------------------------
# Convert to records & make JSON-safe
# -------------------------
records = df.to_dict(orient="records")
records = [make_json_safe(r) for r in records]

print("Prepared", len(records), "records for upload.")

# -------------------------
# Batch upsert with retries
# -------------------------
BATCH_SIZE = 50  # small batch to avoid timeout
MAX_RETRIES = 3
total_batches = math.ceil(len(records) / BATCH_SIZE)

print("Beginning upsert...")

for i in range(total_batches):
    batch = records[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    for attempt in range(1, MAX_RETRIES+1):
        try:
            print(f"Uploading batch {i+1}/{total_batches} (attempt {attempt})...")
            response = supabase.table("oig_articles_embeddings").upsert(
                batch, on_conflict="url"
            ).execute()
            break  # success
        except Exception as e:
            print(f"Batch {i+1} attempt {attempt} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(2)

print("Upsert complete.")
