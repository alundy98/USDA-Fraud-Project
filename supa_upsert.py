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
# Helper functions
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

def batch_upsert(records, table_name, conflict_key="url", batch_size=50, max_retries=3):
    """
    Upsert records into Supabase in batches with retry logic
    """
    total_batches = math.ceil(len(records) / batch_size)
    print(f"Beginning upsert into {table_name} ({len(records)} records, {total_batches} batches)...")

    for i in range(total_batches):
        batch = records[i*batch_size:(i+1)*batch_size]
        for attempt in range(1, max_retries+1):
            try:
                print(f"Uploading batch {i+1}/{total_batches} (attempt {attempt})...")
                supabase.table(table_name).upsert(batch, on_conflict=conflict_key).execute()
                break  # success
            except Exception as e:
                print(f"Batch {i+1} attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    raise
                time.sleep(2)
    print(f"Upsert into {table_name} complete.\n")


# -------------------------
# Upsert CSV: Article Labels & Text
# -------------------------
csv_path = r"outputs_new_prompt/combined_articles.csv"
print(f"Loading CSV: {csv_path}")
df_csv = pd.read_csv(csv_path)

# Filter out unwanted columns
columns_to_remove = [
    "Unnamed: 18","Unnamed: 19","Unnamed: 20","Unnamed: 21","Unnamed: 22",
    "Unnamed: 23","Unnamed: 24","Unnamed: 25","Unnamed: 26"
]
df_csv = df_csv.drop(columns=[c for c in columns_to_remove if c in df_csv.columns])

# Drop rows without a valid "fraud_group_primary"
df_csv = df_csv[df_csv["fraud_group_primary"].notna() & (df_csv["fraud_group_primary"].str.strip() != "")]
print("Rows after filtering fraud_group_primary:", len(df_csv))

# Drop duplicates based on URL
df_csv = df_csv.drop_duplicates(subset=["url"], keep="first")
def clean_numeric_value(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None

# Apply to amount_numeric if the column exists
if "amount_numeric" in df_csv.columns:
    df_csv["amount_numeric"] = df_csv["amount_numeric"].apply(clean_numeric_value)
# Replace NaN with None for JSON upload
df_csv = df_csv.where(pd.notnull(df_csv), None)

# Convert to records & JSON-safe
records_csv = df_csv.to_dict(orient="records")
records_csv = [make_json_safe(r) for r in records_csv]

# Upsert
batch_upsert(records_csv, "final_article_label_dataset", conflict_key="url")


# -------------------------
# Upsert Parquet: Full Article Embeddings
# -------------------------
parquet_path = r"outputs_new_prompt/combined_embeddings.parquet"
print(f"Loading Parquet: {parquet_path}")
df_parquet = pd.read_parquet(parquet_path)
print("Rows loaded:", len(df_parquet))

# Required columns for Supabase
columns_in_supabase = {"url", "text", "embedding", "relevance_score"}
missing = columns_in_supabase - set(df_parquet.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Drop rows with missing text
df_parquet = df_parquet[df_parquet["text"].notna() & (df_parquet["text"].str.strip() != "")]
df_parquet = df_parquet.drop_duplicates(subset=["url"], keep="first")
print("Rows after cleaning text & duplicates:", len(df_parquet))

# Clean embeddings
df_parquet["embedding"] = df_parquet["embedding"].apply(clean_embedding)

# Clean numeric columns
if "relevance_score" in df_parquet.columns:
    df_parquet["relevance_score"] = df_parquet["relevance_score"].apply(
        lambda x: float(x) if x is not None and not (math.isnan(x) or math.isinf(x)) else None
    )

# Keep only Supabase columns
df_parquet = df_parquet[[c for c in df_parquet.columns if c in columns_in_supabase]]

# Replace NaN with None
df_parquet = df_parquet.where(pd.notnull(df_parquet), None)

# Convert to records & JSON-safe
records_parquet = df_parquet.to_dict(orient="records")
records_parquet = [make_json_safe(r) for r in records_parquet]

# Upsert
batch_upsert(records_parquet, "final_embeddings_dataset", conflict_key="url")
