import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt

DATA_PATH = "data/Phishing_Legitimate_full.csv"
ARTIFACT_DIR = "artifacts_rf"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

print("== Loading dataset ==")
df = pd.read_csv(DATA_PATH)

# Identify label + drop ID-like columns
label_col = "CLASS_LABEL"
id_like = [c for c in df.columns if c.lower() in ["id", "id_", "index"]]
feature_cols = [c for c in df.columns if c not in id_like + [label_col]]

X = df[feature_cols]
y = df[label_col].astype(int)

print(f"Features: {len(feature_cols)} | Rows: {len(df)}")

# 70/15/15 split: train / val / test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)  # 0.1765 of 0.85 ≈ 0.15 overall

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

print("== Training RandomForest model ==")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42,
    class_weight=None  # dataset is already balanced
)

rf.fit(X_train, y_train)


def plot_confusion_matrix(cm, classes, title, filename):
    """Save a confusion matrix heatmap as a PNG."""
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    # Show numbers in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    out_path = os.path.join(ARTIFACT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix: {out_path}")


def plot_roc(y_true, y_prob, title, filename):
    """Save ROC curve as a PNG."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")

    out_path = os.path.join(ARTIFACT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved ROC curve: {out_path}")


def eval_split(name, X_split, y_split, prefix):
    y_prob = rf.predict_proba(X_split)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_split, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_split, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_split, y_prob)

    print(f"\n== {name} Metrics ==")
    print(f"{name} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_split, y_pred, digits=4))

    # Confusion matrix (0 = legit, 1 = phish)
    cm = confusion_matrix(y_split, y_pred, labels=[0, 1])
    plot_confusion_matrix(
        cm,
        classes=["legit (0)", "phish (1)"],
        title=f"{name} Confusion Matrix (RandomForest)",
        filename=f"{prefix}_confusion_matrix.png",
    )

    # ROC curve
    plot_roc(
        y_split,
        y_prob,
        title=f"{name} ROC Curve (RandomForest)",
        filename=f"{prefix}_roc.png",
    )


print("Evaluating on validation and test splits...")
eval_split("Validation", X_val, y_val, prefix="validation_rf")
eval_split("Test", X_test, y_test, prefix="test_rf")

# Save model + feature columns
joblib.dump(rf, os.path.join(ARTIFACT_DIR, "model.joblib"))
joblib.dump(feature_cols, os.path.join(ARTIFACT_DIR, "feature_cols.joblib"))

print(f"\nSaved RandomForest model to {ARTIFACT_DIR}/")
