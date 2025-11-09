import pandas as pd
from sklearn.cluster import KMeans

# Paths
libs_pca_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_pca.csv"
apxs_pca_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_pca.csv"

# Load datasets
libs = pd.read_csv(libs_pca_path)
apxs = pd.read_csv(apxs_pca_path)

# Apply KMeans with best K=5
k_libs, k_apxs = 5, 5

libs['cluster'] = KMeans(n_clusters=k_libs, random_state=42).fit_predict(libs)
apxs['cluster'] = KMeans(n_clusters=k_apxs, random_state=42).fit_predict(apxs)

# Save results
libs_out = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_best_k5_clusters.csv"
apxs_out = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_best_k5_clusters.csv"

libs.to_csv(libs_out, index=False)
apxs.to_csv(apxs_out, index=False)

print(f"✅ Saved LIBS clusters to: {libs_out}")
print(f"✅ Saved APXS clusters to: {apxs_out}")
