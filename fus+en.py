import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os

# -----------------------------
# Paths (update if needed)
# -----------------------------
libs_pca_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_pca.csv"
apxs_pca_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_pca.csv"
libs_cluster_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_best_k5_clusters.csv"
apxs_cluster_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_best_k5_clusters.csv"

# -----------------------------
# Load datasets
# -----------------------------
libs_pca = pd.read_csv(libs_pca_path)
apxs_pca = pd.read_csv(apxs_pca_path)
libs_labels = pd.read_csv(libs_cluster_path)["cluster"]
apxs_labels = pd.read_csv(apxs_cluster_path)["cluster"]

print(f"LIBS PCA shape: {libs_pca.shape}, APXS PCA shape: {apxs_pca.shape}")

# -----------------------------
# 1️⃣ Align & fuse features
# -----------------------------
min_len = min(len(libs_pca), len(apxs_pca))
X_libs = libs_pca.iloc[:min_len, :]
X_apxs = apxs_pca.iloc[:min_len, :]

# Fusion: concatenate feature spaces
X_fused = pd.concat([X_libs, X_apxs], axis=1)

# Pseudo-labels: average or majority logic (you can choose LIBS or APXS if one is more reliable)
y_fused = ((libs_labels.iloc[:min_len] + apxs_labels.iloc[:min_len]) / 2).round().astype(int)

print(f"Fused features shape: {X_fused.shape}")

# -----------------------------
# 2️⃣ Standardize
# -----------------------------
scaler = StandardScaler()
X_libs_scaled = scaler.fit_transform(X_libs)
X_apxs_scaled = scaler.fit_transform(X_apxs)
X_fused_scaled = scaler.fit_transform(X_fused)

# -----------------------------
# 3️⃣ Split for training/testing
# -----------------------------
def train_test(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Ensemble models
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    xgb = XGBClassifier(eval_metric='mlogloss', random_state=42)
    ensemble = VotingClassifier(estimators=[('rf', rf), ('svm', svm), ('xgb', xgb)], voting='soft')
    
    # Train and evaluate
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {name} Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))
    return acc

acc_libs = train_test(X_libs_scaled, libs_labels.iloc[:min_len], "LIBS-only")
acc_apxs = train_test(X_apxs_scaled, apxs_labels.iloc[:min_len], "APXS-only")
acc_fused = train_test(X_fused_scaled, y_fused, "Fused LIBS+APXS")

# -----------------------------
# 4️⃣ Summary
# -----------------------------
print("\n📊 Final Comparison:")
print(f"LIBS-only Accuracy:  {acc_libs:.3f}")
print(f"APXS-only Accuracy:  {acc_apxs:.3f}")
print(f"Fused Accuracy:      {acc_fused:.3f}")

# Save final results
results = pd.DataFrame({
    "Dataset": ["LIBS", "APXS", "Fusion"],
    "Accuracy": [acc_libs, acc_apxs, acc_fused]
})
os.makedirs(r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\results", exist_ok=True)
results.to_csv(r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\results\fusion_results.csv", index=False)

print("\n✅ Results saved to results/fusion_results.csv")
