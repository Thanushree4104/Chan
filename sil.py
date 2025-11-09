import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

# -----------------------------
# File paths
# -----------------------------
libs_file = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_pca.csv"
apxs_file = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_pca.csv"
output_dir = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\results"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Function for clustering
# -----------------------------
def cluster_and_label(data, name, k_range=range(2, 8)):
    print(f"\n🔹 Processing {name} dataset ({data.shape[0]} samples, {data.shape[1]} features)")

    best_k = None
    best_score = -1
    best_labels = None
    all_scores = []

    # Try different cluster counts
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)

        sil = silhouette_score(data, labels)
        ch = calinski_harabasz_score(data, labels)
        db = davies_bouldin_score(data, labels)

        all_scores.append((k, sil, ch, db))

        print(f"k={k}: silhouette={sil:.3f}, CH={ch:.1f}, DB={db:.3f}")

        if sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels

    print(f"\n✅ Best K for {name}: {best_k} (Silhouette={best_score:.3f})")

    # Save pseudo-labels
    pseudo_df = pd.DataFrame(data)
    pseudo_df["cluster"] = best_labels
    pseudo_df.to_csv(os.path.join(output_dir, f"{name.lower()}_pseudo_labels.csv"), index=False)

    # Plot cluster distribution (first two PCs)
    plt.figure(figsize=(7, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=best_labels, cmap='viridis', s=30)
    plt.title(f"{name} Clusters (k={best_k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label="Cluster")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_clusters.png"))
    plt.close()

    # Plot Silhouette score trend
    plt.figure(figsize=(6, 4))
    plt.plot([s[0] for s in all_scores], [s[1] for s in all_scores], marker='o')
    plt.title(f"{name} Silhouette Trend")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name.lower()}_silhouette_trend.png"))
    plt.close()

    return best_k, best_score


# -----------------------------
# Load PCA datasets
# -----------------------------
libs_df = pd.read_csv(libs_file)
apxs_df = pd.read_csv(apxs_file)

# -----------------------------
# Run clustering
# -----------------------------
k_libs, s_libs = cluster_and_label(libs_df, "LIBS", range(2, 8))
k_apxs, s_apxs = cluster_and_label(apxs_df, "APXS", range(2, 8))

# -----------------------------
# Save summary
# -----------------------------
summary = pd.DataFrame({
    "Dataset": ["LIBS", "APXS"],
    "Best_K": [k_libs, k_apxs],
    "Best_Silhouette": [s_libs, s_apxs]
})
summary_file = os.path.join(output_dir, "clustering_summary.csv")
summary.to_csv(summary_file, index=False)

print("\n🎯 Clustering complete!")
print(summary)
print(f"\n📂 Results saved in: {output_dir}")
