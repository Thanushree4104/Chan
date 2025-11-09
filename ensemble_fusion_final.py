import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTGBM_VERBOSITY"] = "0"

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# Load preprocessed datasets
# -----------------------------------------------------------
print("\n📂 Loading processed datasets...")
libs_data = pd.read_csv(r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_pca.csv")
apxs_data = pd.read_csv(r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_pca.csv")

libs_labels = pd.read_csv(r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_best_k5_clusters.csv")["cluster"]
apxs_labels = pd.read_csv(r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_best_k5_clusters.csv")["cluster"]

# -----------------------------------------------------------
# Feature + Label setup
# -----------------------------------------------------------
X_libs, y_libs = libs_data.values, libs_labels.values
X_apxs, y_apxs = apxs_data.values, apxs_labels.values

# For fusion (pseudo-label ensemble fusion)
X_fused = np.hstack([np.mean(X_libs, axis=0, keepdims=True).repeat(len(X_apxs), axis=0), X_apxs])
y_fused = y_apxs

# -----------------------------------------------------------
# Define Base Models
# -----------------------------------------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=150, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=2, eval_metric='logloss', verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=150, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        random_state=42, n_jobs=2, verbose=-1
    ),
    "GradientBoost": GradientBoostingClassifier(n_estimators=200, random_state=42)
}

# -----------------------------------------------------------
# Helper: Evaluate model with clean output
# -----------------------------------------------------------
def evaluate_model(name, model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return acc

# -----------------------------------------------------------
# Train and Evaluate
# -----------------------------------------------------------
results = []

for dataset_name, (X, y) in {
    "LIBS": (X_libs, y_libs),
    "APXS": (X_apxs, y_apxs),
    "Fused": (X_fused, y_fused)
}.items():
    print(f"\n🧩 Evaluating on {dataset_name} dataset...")
    for name, model in models.items():
        acc = evaluate_model(name, model, X, y)
        results.append({"Dataset": dataset_name, "Model": name, "Accuracy": acc})

# -----------------------------------------------------------
# Stacking Ensemble for Fused Dataset
# -----------------------------------------------------------
base_learners = [
    ('xgb', models["XGBoost"]),
    ('lgbm', models["LightGBM"]),
    ('rf', models["RandomForest"])
]
stack = StackingClassifier(estimators=base_learners, final_estimator=GradientBoostingClassifier(), n_jobs=2)
stack_acc = evaluate_model("StackingEnsemble", stack, X_fused, y_fused)
results.append({"Dataset": "Fused", "Model": "StackingEnsemble", "Accuracy": stack_acc})

# -----------------------------------------------------------
# Show Concise Summary
# -----------------------------------------------------------
results_df = pd.DataFrame(results)
summary = results_df.pivot(index="Dataset", columns="Model", values="Accuracy")
print("\n📊 Model Accuracy Summary (concise):")
print(summary.round(3))

# Save results
out_path = r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\ensemble_results_summary.csv"
results_df.to_csv(out_path, index=False)
print(f"\n✅ Results saved to: {out_path}")
