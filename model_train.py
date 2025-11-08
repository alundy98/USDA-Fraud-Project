import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import ast
#this class trains our XGBoost model for: Fraud Detection and Classification, i.e. is fraud present, and if so what kind
class FraudClassifier:
    def __init__(self, model_path=None):
        #initializign model
        self.model = None
        self.model_path = model_path

    def load_data(self, file_path):
        #the embeddings are in a parquet(better for the size of embeddings + ML)
        print(f"Loading data from {file_path} ...")
        if file_path.endswith(".parquet"):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)

        # expect columns: 'embedding'-vector and 'fraud_label' (0/1)
        if isinstance(df.loc[0, "embedding"], str):
            df["embedding"] = df["embedding"].apply(ast.literal_eval)

        X = np.vstack(df["embedding"].values)
        y = df["fraud_label"].astype(int).values

        print(f"loaded {len(df)} samples with embedding dimension {X.shape[1]}")
        return X, y

    def train(self, X, y, test_size=0.2, random_state=42):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False
        )

        print("Training XGBoost model...")
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]

        print("Model evaluation:")
        print(classification_report(y_val, y_pred, digits=3))
        print("Confusion Matrix:", confusion_matrix(y_val, y_pred))
        print("ROC AUC:", roc_auc_score(y_val, y_proba))

    def save(self, path=None):
        """Save trained model to file."""
        if not self.model:
            raise ValueError("Model not trained")
        out_path = path or self.model_path or "outputs_with_ai/fraud_xgb_model.pkl"
        joblib.dump(self.model, out_path)
        print(f"Model saved to {out_path}")
    def load(self, path=None):
        """Load model from file."""
        in_path = path or self.model_path
        if not in_path:
            raise ValueError("No model path")
        self.model = joblib.load(in_path)
        print(f"Model loaded from {in_path}")
    def predict(self, text_embedding):
        #predict fraud probability for a single embedding (list or np.array)
        if self.model is None:
            raise ValueError("Model not loaded or trained ")
        emb = np.array(text_embedding).reshape(1, -1)
        return float(self.model.predict_proba(emb)[0, 1])
