import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import numpy as np
import json
# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
#I changed this same file a few times to upsert each csv and parquet to supabase, reference the schemas there for more info on the previous ones
# Load Parquet file
df = pd.read_parquet("non_fraud_articles_embeddings.parquet")
# Keep only rows where 'text' is not null and not empty after stripping whitespace
df = df[df["text"].notnull() & (df["text"].str.strip() != "")]
# drop duplicate URLs, keeping the first
df = df.drop_duplicates(subset=["url"], keep="first")
#ensure embedding column JSON serializable
if "embedding" in df.columns:
    df["embedding"] = df["embedding"].apply(
        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
    )
# replace Nan/missing values with None for compatibility
df = df.where(pd.notnull(df), None)
# Convert to list of dicts
records = df.to_dict(orient="records")
#upsert
response = supabase.table("non_fraud_article_embedding").upsert(
    records, on_conflict="url"
).execute()

print(f"Upsert successful! Rows uploaded: {len(records)}")
print(response)
