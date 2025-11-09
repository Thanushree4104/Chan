import pandas as pd
import numpy as np
import os

# -----------------------------
# File paths  🔧 (update if needed)
# -----------------------------
libs_pca_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_pca.csv"
libs_cluster_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_best_k5_clusters.csv"

apxs_pca_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_pca.csv"
apxs_cluster_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_best_k5_clusters.csv"

output_fused_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\fused_cluster_features.csv"

# -----------------------------
# Load PCA data + cluster labels
# -----------------------------
print("📂 Loading LIBS + APXS cluster and PCA data...")

libs_pca = pd.read_csv(libs_pca_path)
apxs_pca = pd.read_csv(apxs_pca_path)

libs_labels = pd.read_csv(libs_cluster_path)["cluster"]
apxs_labels = pd.read_csv(apxs_cluster_path)["cluster"]

print(f"✅ LIBS PCA: {libs_pca.shape}, clusters: {libs_labels.nunique()}")
print(f"✅ APXS PCA: {apxs_pca.shape}, clusters: {apxs_labels.nunique()}")

# -----------------------------
# Compute per-cluster mean spectra
# -----------------------------
libs_pca["cluster"] = libs_labels
apxs_pca["cluster"] = apxs_labels

libs_means = libs_pca.groupby("cluster").mean().reset_index()
apxs_means = apxs_pca.groupby("cluster").mean().reset_index()

print(f"✅ Computed LIBS cluster means: {libs_means.shape}")
print(f"✅ Computed APXS cluster means: {apxs_means.shape}")

# -----------------------------
# Fuse cluster means (simple concatenation)
# -----------------------------
# ensure both have same number of clusters — here we assume 5
min_clusters = min(len(libs_means), len(apxs_means))
libs_means = libs_means.iloc[:min_clusters, :]
apxs_means = apxs_means.iloc[:min_clusters, :]

# Drop 'cluster' column before fusion to avoid duplication
fused = pd.concat([libs_means.drop(columns=["cluster"]), 
                   apxs_means.drop(columns=["cluster"])], axis=1)

# Re-add a cluster id column
fused["cluster_id"] = range(min_clusters)

# -----------------------------
# Save fused dataset
# -----------------------------
os.makedirs(os.path.dirname(output_fused_path), exist_ok=True)
fused.to_csv(output_fused_path, index=False)

print(f"\n🎉 Fused cluster-level dataset saved to:\n{output_fused_path}")
print(f"📊 Shape: {fused.shape}")
