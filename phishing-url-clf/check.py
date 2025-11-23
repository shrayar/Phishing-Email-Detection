import pandas as pd

# Path to your dataset inside the data folder
CSV_PATH = "data/Phishing_Legitimate_full.csv"

df = pd.read_csv(CSV_PATH)

print("="*60)
print(" Dataset loaded successfully")
print(f"Shape: {df.shape}")
print("="*60)

# Display first few columns and sample rows
print("\nColumns:", df.columns.tolist()[:10], "...")
print("\nPreview:")
print(df.head(5))

# Detect label column (auto if present)
label_col = "CLASS_LABEL" if "CLASS_LABEL" in df.columns else None
if label_col:
    print(f"\n Detected label column: '{label_col}'")
    print("Label distribution:")
    print(df[label_col].value_counts())
else:
    print("\n No 'CLASS_LABEL' column found — check column names above.")

# Optional: detect ID-like columns
for col in ["id", "Id", "ID"]:
    if col in df.columns:
        print(f"\n ID-like column detected: '{col}' (will drop before training)")
