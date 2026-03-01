import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("loan_data.csv")
df.drop("Loan_ID", axis=1, inplace=True, errors="ignore")
df.fillna(df.mode().iloc[0], inplace=True)

# -----------------------
# Feature Engineering
# -----------------------
df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
df["DebtToIncomeRatio"] = df["LoanAmount"] / df["TotalIncome"]

# -----------------------
# Encode Target
# -----------------------
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# -----------------------
# Column Transformer
# -----------------------
categorical_cols = X.select_dtypes(include=["object", "string"]).columns
numeric_cols = X.select_dtypes(exclude=["object", "string"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# -----------------------
# Train-Test Split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit transformer
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# -----------------------
# Handle Imbalance (SMOTE)
# -----------------------
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(
    X_train_processed, y_train
)

# -----------------------
# Models
# -----------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

results = {}
best_model = None
best_auc = 0

# -----------------------
# Train & Evaluate
# -----------------------
for name, model in models.items():
    model.fit(X_train_resampled, y_train_resampled)

    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = acc

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} ROC-AUC: {auc:.4f}")
    print("-" * 40)

    if auc > best_auc:
        best_auc = auc
        best_model = model

print(f"✅ Best Model Selected (Based on AUC): {best_model}")

# -----------------------
# Save Model & Preprocessor
# -----------------------
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(preprocessor, open("pipeline.pkl", "wb"))

# -----------------------
# Accuracy Comparison Plot
# -----------------------
plt.figure()
plt.bar(results.keys(), results.values())
plt.xticks(rotation=45)
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.savefig("accuracy_comparison.png")
plt.close()

# Save best model accuracy
best_accuracy = accuracy_score(
    y_test,
    best_model.predict(X_test_processed)
)

with open("model_metrics.pkl", "wb") as f:
    pickle.dump({"accuracy": best_accuracy, "roc_auc": best_auc}, f)

# -----------------------
# ROC Curve
# -----------------------
y_prob = best_model.predict_proba(X_test_processed)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# -----------------------
# SHAP Explainability (Fixed Properly)
# -----------------------

# Convert to dense if sparse
if hasattr(X_test_processed, "toarray"):
    X_test_dense = X_test_processed.toarray()
else:
    X_test_dense = X_test_processed

# Get feature names after encoding
feature_names = preprocessor.get_feature_names_out()

# Use correct explainer type
if isinstance(best_model, XGBClassifier):
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_dense)
else:
    explainer = shap.Explainer(best_model, X_train_resampled)
    shap_values = explainer(X_test_dense)

# SHAP summary plot
plt.figure()
shap.summary_plot(
    shap_values,
    X_test_dense,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()

print("🔥 Advanced Model Training Complete")