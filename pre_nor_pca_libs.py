import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import os

# -----------------------------
# Paths
# -----------------------------
input_file = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\merged_libs_dataset.csv"
output_file = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_pca.csv"

# -----------------------------
# Load dataset
# -----------------------------
print("📂 Loading merged LIBS dataset...")
df = pd.read_csv(input_file)

# If the first column is unnamed index, drop it
if df.columns[0].lower().startswith("unnamed"):
    df = df.drop(df.columns[0], axis=1)

print(f"✅ Loaded dataset: {df.shape[0]} samples × {df.shape[1]} features")

# -----------------------------
# Preprocessing
# -----------------------------
# Convert all to numeric (just in case), and fill missing values
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

# Drop columns that are entirely zero
df = df.loc[:, (df != 0).any(axis=0)]

print(f"✅ After cleaning: {df.shape[1]} valid spectral features")

# -----------------------------
# Normalization (StandardScaler)
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

print("✅ Normalization complete")

# -----------------------------
# PCA (retain 99% variance)
# -----------------------------
pca = PCA(n_components=0.99, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"✅ PCA complete — reduced to {X_pca.shape[1]} components (99% variance retained)")
print(f"🔹 Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

# -----------------------------
# Save PCA output
# -----------------------------
df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_pca.to_csv(output_file, index=False)

print(f"\n🎉 Saved PCA results to: {output_file}")
print(f"📊 Final shape: {df_pca.shape}")
