import os, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, classification_report,
                             confusion_matrix, RocCurveDisplay)
import matplotlib.pyplot as plt

CSV = "data/Phishing_Legitimate_full.csv"
OUT = "artifacts_3split"
os.makedirs(OUT, exist_ok=True)


df = pd.read_csv(CSV)
LABEL = "CLASS_LABEL"
if df[LABEL].min() < 0:
    df[LABEL] = df[LABEL].replace(-1, 0).astype(int)


for c in ["id","Id","ID","index"]:
    if c in df.columns: df = df.drop(columns=[c])

X = df.drop(columns=[LABEL])
y = df[LABEL].astype(int).values


X_trval, X_test, y_trval, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trval, y_trval, test_size=0.1765, stratify=y_trval, random_state=42)

scaler = StandardScaler().fit(X_train)
Xs_tr  = scaler.transform(X_train)
Xs_val = scaler.transform(X_val)
Xs_te  = scaler.transform(X_test)


classes = np.unique(y_train)
cw = compute_class_weight("balanced", classes=classes, y=y_train)
cw = {int(c): float(w) for c, w in zip(classes, cw)}

clf = LogisticRegression(max_iter=2000, solver="liblinear", class_weight=cw)
clf.fit(Xs_tr, y_train)

def eval_split(name, Xs, y):
    prob = clf.predict_proba(Xs)[:,1]
    pred = (prob >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    auc = roc_auc_score(y, prob)
    print(f"{name} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(classification_report(y, pred, target_names=["legit","phish"], zero_division=0))

    
    cm = confusion_matrix(y, pred, labels=[0,1])
    fig, ax = plt.subplots(figsize=(3.5,3.5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xticks([0,1]); ax.set_xticklabels(["legit","phish"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["legit","phish"])
    for (i,j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name.lower()}_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    
    fig2, ax2 = plt.subplots(figsize=(4,3))
    RocCurveDisplay.from_predictions(y, prob, name=name, ax=ax2)
    plt.tight_layout()
    fig2.savefig(os.path.join(OUT, f"{name.lower()}_roc.png"), dpi=150)
    plt.close(fig2)


print("== Validation Metrics ==")
eval_split("Validation", Xs_val, y_val)
print("\n== Test Metrics ==")
eval_split("Test", Xs_te, y_test)

joblib.dump(scaler, os.path.join(OUT, "scaler.joblib"))
joblib.dump(clf,    os.path.join(OUT, "model.joblib"))
print(f"\nSaved artifacts to {OUT}/")
