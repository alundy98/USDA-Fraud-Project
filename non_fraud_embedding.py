import pandas as pd
import numpy as np
from openai import OpenAI
import openai
from supabase import create_client, Client
import os
import dotenv
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import json
from pathlib import Path

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CSV_PATH = "non_fraud_dataset.csv"
OUTPUT_PATH = "non_fraud_articles_embeddings.parquet"
SUPABASE_TABLE = "non_fraud_article_embedding"

# initializing clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
# alias so the chunk function can use `client` as you wrote it
client = openai_client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Loading data
df = pd.read_csv(CSV_PATH)

# checks columns for requirements
required_cols = {"url", "text"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV missing required columns: {required_cols - set(df.columns)}")
print(f"Loaded {len(df)} non-fraud articles for embedding.")

# creating text for embeddings
df["embedding_input"] = (
    df["title"].fillna("") + "\n\n" + df["text"].fillna("")
    if "title" in df.columns
    else df["text"].fillna("")
)

#function slightly changed from embedding with ai but pretty much the same
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
#end of chunking

#generate embeddings using the working chunk logic (so one call - all docs)
print("Generating embeddings (using your chunking logic)...")
texts = df["embedding_input"].tolist()
df["embedding"] = get_ai_embeds(texts, model_name=EMBEDDING_MODEL, chunk_size=7000)

# Save to parquet
out_dir = os.path.dirname(OUTPUT_PATH) or "."
os.makedirs(out_dir, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)
print(f"Embeddings saved to {OUTPUT_PATH}")

# Upsert to Supabase
def chunk_list(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

print("Uploading to Supabase...")

rows = df.to_dict(orient="records")
# convert numpy arrays to lists for JSON storage
for r in rows:
    emb = r.get("embedding")
    if isinstance(emb, np.ndarray):
        r["embedding"] = emb.tolist()
    # if it's a numpy scalar or other, ensure normal Python types
    # also remove any keys Supabase won't accept (optional)
# batch upload to avoid payload size issues
for batch in tqdm(list(chunk_list(rows, 50))):
    response = supabase.table(SUPABASE_TABLE).upsert(batch).execute()
    # Supabase client returns different shapes; check and print errors if any
    try:
        status = getattr(response, "status_code", None)
        if status and status >= 400:
            print(f"Error uploading batch: status_code={status}, response={response}")
    except Exception:
        # fallback: print response for debugging
        print("Upsert response:", response)

print("All non-fraud embeddings uploaded (attempted).")
