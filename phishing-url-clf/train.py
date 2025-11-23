import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report

# Path to your dataset
CSV = "data/Phishing_Legitimate_full.csv"

# Load data
df = pd.read_csv(CSV)

# Label column
LABEL_COL = "CLASS_LABEL"

# Normalize labels: map -1 -> 0 if present
df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
if df[LABEL_COL].min() < 0:
    df[LABEL_COL] = df[LABEL_COL].replace(-1, 0)
df[LABEL_COL] = df[LABEL_COL].astype(int)

# Drop ID-like columns if present
for col in ["id", "Id", "ID"]:
    if col in df.columns:
        df = df.drop(columns=[col])

# Split features and labels
X = df.drop(columns=[LABEL_COL])
y = df[LABEL_COL].values

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)

# Class weights (for balance)
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
cw = {int(c): float(w) for c, w in zip(classes, weights)}

# Logistic Regression model
model = LogisticRegression(max_iter=2000, solver="liblinear", class_weight=cw)
model.fit(X_train_s, y_train)

# Predictions and evaluation
y_prob = model.predict_proba(X_val_s)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

acc = accuracy_score(y_val, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, average="binary", zero_division=0)
auc = roc_auc_score(y_val, y_prob)

print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=["legit", "phish"], zero_division=0))

# Save trained model and scaler
os.makedirs("artifacts_uploaded", exist_ok=True)
joblib.dump(scaler, "artifacts_uploaded/scaler.joblib")
joblib.dump(model, "artifacts_uploaded/model.joblib")

print("\n Training complete — saved model and scaler to artifacts_uploaded/")
