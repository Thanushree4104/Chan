# ======= Pseudo-Label Ensemble Fusion Evaluation ======= #
# Author: Agent TK
# Title: "Pseudo-label Ensemble Fusion of LIBS and APXS for Compositional
#         Discrimination on Chandrayaan-3 Lunar Regolith"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------
# 🔹 Load your preprocessed + PCA reduced datasets
# (replace file paths with your actual ones)
# ---------------------------------------------------------
libs = pd.read_csv(r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\lib-v2\libs_pca.csv")
apxs = pd.read_csv(r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\apx\apxs_pca.csv")
fused = pd.read_csv(r"C:\Users\K THANUSHREE\OneDrive - SSN Trust\Cha3\fused_cluster_features.csv")

# ---------------------------------------------------------
# 🔹 Define feature/label separation for each dataset
# (Assume last column = label; modify if different)
# ---------------------------------------------------------
datasets = {
    "LIBS": (libs.iloc[:, :-1], libs.iloc[:, -1]),
    "APXS": (apxs.iloc[:, :-1], apxs.iloc[:, -1]),
    "Fused": (fused.iloc[:, :-1], fused.iloc[:, -1])
}

# ---------------------------------------------------------
# 🔹 Define models
# ---------------------------------------------------------
models = {
    "GradientBoost": GradientBoostingClassifier(random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ---------------------------------------------------------
# 🔹 Run evaluation for each dataset
# ---------------------------------------------------------
summary = {}

for name, (X, y) in datasets.items():
    print(f"\n📊 Evaluating {name} dataset ...")
    # Stratified split for stability
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    # --- Train base models ---
    for mname, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[mname] = acc
        print(f"  {mname} Accuracy: {acc:.3f}")

    # --- Stacking Ensemble ---
    estimators = [(mname, m) for mname, m in models.items() if mname != "XGBoost"]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    )
    stack.fit(X_train, y_train)
    preds_stack = stack.predict(X_test)
    stack_acc = accuracy_score(y_test, preds_stack)
    results["StackingEnsemble"] = stack_acc

    print(f"  ✅ {name} Ensemble Accuracy: {stack_acc:.3f}")
    print(classification_report(y_test, preds_stack, zero_division=0))

    # Store summary
    summary[name] = results

# ---------------------------------------------------------
# 🔹 Display final summary table
# ---------------------------------------------------------
summary_df = pd.DataFrame(summary).T
print("\n================ Final Accuracy Summary ================\n")
print(summary_df.round(3))
print("\n✅ Evaluation Complete.\n")
