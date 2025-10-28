import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
df = pd.read_csv("data/final.csv")
#only rows with text in the text column
df = df[df['text'].notnull() & (df['text'].str.strip() != "")]
#Some duplicate urls, drops anything after the first instance of a url
df = df.drop_duplicates(subset=['url'], keep='first')
#replace Nan / missing values
df = df.where(pd.notnull(df), None)
records = df.to_dict(orient="records")
response = supabase.table("fdic_articles").upsert(records, on_conflict="url").execute()

print("Upsert completed. Response:")
print(response)
