import ast
import os
import numpy as np
import pandas as pd
import json
from supabase import create_client, Client
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)


# Helper func to parse embedding safely
def parse_embedding(x):
    if x is None:
        return None
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
    return None

print("Fetching fraud embeddings from Supabase")
# Get labels, all fraud articles
labels_data = supabase.table("article_labels").select(
    "url, llm_labels"
).execute().data
labels_df = pd.DataFrame(labels_data)

# Parse llm_labels to get fraud_type
def parse_llm_labels(llm_str):
    try:
        # Remove Markdown code fences if present
        cleaned = llm_str.strip().replace("```json", "").replace("```", "")
        # Replace doubled quotes with single quotes for valid JSON
        cleaned = cleaned.replace('""', '"')
        return json.loads(cleaned)
    except Exception as e:
        print(f"Failed to parse LLM string: {e}")
        return {}
labels_df["llm_labels_parsed"] = labels_df["llm_labels"].apply(parse_llm_labels)
labels_df["fraud_type"] = labels_df["llm_labels_parsed"].apply(lambda x: x.get("fraud_type"))

# Get full embeddings from article_embedding_full schema
emb_data = supabase.table("article_embedding_full").select(
    "url, embedding"
).execute().data
emb_df = pd.DataFrame(emb_data)
emb_df["embedding"] = emb_df["embedding"].apply(parse_embedding)

# Merge embeddings with labels on unique key url
fraud_df = labels_df.merge(emb_df, on="url", how="inner")
fraud_df = fraud_df[fraud_df["embedding"].notnull()]
print(f"Fraud rows with embeddings: {len(fraud_df)}")

print("Fetching non-fraud embeddings from Supabase...")
non_fraud_data = supabase.table("non_fraud_article_embedding").select(
    "url, text, embedding"
).execute().data
non_fraud_df = pd.DataFrame(non_fraud_data)
non_fraud_df = non_fraud_df[non_fraud_df["embedding"].notnull()]
non_fraud_df["embedding"] = non_fraud_df["embedding"].apply(parse_embedding)
print(f"Non-fraud rows: {len(non_fraud_df)}")

# MODEL A: Fraud Detection, binary "is fraud present"
print("Training Fraud Detection Model:")
print("Here is where we will actually utilize the data to train mm")
fraud_df["fraud_label"] = 1
non_fraud_df["fraud_label"] = 0
binary_df = pd.concat([fraud_df, non_fraud_df], ignore_index=True)
X_bin = np.array(binary_df["embedding"].tolist())
y_bin = binary_df["fraud_label"].values
X_train, X_val, y_train, y_val = train_test_split(
    X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)
binary_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss"
)

binary_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

os.makedirs("outputs_w_ai", exist_ok=True)
joblib.dump(binary_model, "outputs_w_ai/fraud_detection_model.pkl")
print("Fraud detection model trained and saved.")

# MODEL B: Fraud Type Classification, multi classifier, what type of fraud is it
print("\n=== Training Fraud Type Model ===")
type_df = fraud_df.copy()
# Filter only fraud rows
def parse_llm_labels(llm_str):
    try:
        cleaned = llm_str.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned)
    except:
        return {}
type_df["llm_labels_parsed"] = type_df["llm_labels"].apply(parse_llm_labels)
type_df["fraud_type"] = type_df["llm_labels_parsed"].apply(lambda x: x.get("fraud_type"))

# Remove rows where fraud_type is missing, not useful here
type_df = type_df[type_df["fraud_type"].notnull()]

# remove fraud_types that appear only once, does not work for split train logic
counts = type_df["fraud_type"].value_counts()
valid_types = counts[counts > 1].index
type_df = type_df[type_df["fraud_type"].isin(valid_types)]

print(f"Fraud rows for type classification after filtering rare types: {len(type_df)}")

# Encode fraud_type
le = LabelEncoder()
y_type = le.fit_transform(type_df["fraud_type"].astype(str))
X_type = np.array(type_df["embedding"].tolist())
X_train, X_val, y_train, y_val = train_test_split(
    X_type, y_type, test_size=0.2, random_state=42, stratify=y_type
)

type_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(le.classes_),
    eval_metric="mlogloss"
)

type_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

joblib.dump(type_model, "outputs_w_ai/fraud_type_model.pkl")
joblib.dump(le, "outputs_w_ai/fraud_type_encoder.pkl")

print(f"Fraud type model trained with {len(le.classes_)} classes: {list(le.classes_)}")
print("All models trained and saved successfully.")
