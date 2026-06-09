import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import SMOTE

# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv("loan_data.csv")

df.columns = df.columns.str.strip()

print("Columns in dataset:", df.columns)

# -----------------------
# Clean Target Column
# -----------------------

df["loan_status"] = df["loan_status"].str.strip().str.lower()

df["loan_status"] = df["loan_status"].map({
    "approved": 1,
    "rejected": 0
})

df = df.dropna(subset=["loan_status"])

# -----------------------
# Feature Engineering
# -----------------------

df["emi"] = df["loan_amount"] / df["loan_term"]

df["debt_to_income"] = (
    df["loan_amount"] /
    df["income_annum"]
)

df["total_assets"] = (
    df["residential_assets_value"]
    + df["commercial_assets_value"]
    + df["luxury_assets_value"]
    + df["bank_asset_value"]
)

# -----------------------
# Features & Target
# -----------------------

X = df.drop(
    ["loan_status", "loan_id"],
    axis=1
)

y = df["loan_status"]

# -----------------------
# Column Transformer
# -----------------------

categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns

numeric_cols = X.select_dtypes(
    exclude=["object", "string"]
).columns

preprocessor = ColumnTransformer([
    (
        "num",
        StandardScaler(),
        numeric_cols
    ),
    (
        "cat",
        OneHotEncoder(
            handle_unknown="ignore"
        ),
        categorical_cols
    )
])

# -----------------------
# Train Test Split
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------
# Preprocessing
# -----------------------

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

# -----------------------
# Handle Imbalance
# -----------------------

sm = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = sm.fit_resample(
    X_train_processed,
    y_train
)

# -----------------------
# Logistic Regression
# -----------------------

model = LogisticRegression(
    max_iter=2000
)

model.fit(
    X_train_resampled,
    y_train_resampled
)

# -----------------------
# Predictions
# -----------------------

y_pred = model.predict(
    X_test_processed
)

y_prob = model.predict_proba(
    X_test_processed
)[:, 1]

accuracy = accuracy_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print(
    f"Logistic Regression Accuracy: {accuracy:.4f}"
)

print(
    f"Logistic Regression ROC-AUC: {roc_auc:.4f}"
)

# -----------------------
# Save Model
# -----------------------

pickle.dump(
    model,
    open("best_model.pkl", "wb")
)

pickle.dump(
    preprocessor,
    open("pipeline.pkl", "wb")
)

# -----------------------
# Save Metrics
# -----------------------

with open(
    "model_metrics.pkl",
    "wb"
) as f:

    pickle.dump(
        {
            "accuracy": accuracy,
            "roc_auc": roc_auc
        },
        f
    )

# -----------------------
# ROC Curve
# -----------------------

fpr, tpr, _ = roc_curve(
    y_test,
    y_prob
)

plt.figure()

plt.plot(
    fpr,
    tpr
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Logistic Regression"
)

plt.tight_layout()

plt.savefig(
    "roc_curve.png"
)

plt.close()

# -----------------------
# SHAP Explainability
# -----------------------

if hasattr(
    X_test_processed,
    "toarray"
):
    X_test_dense = X_test_processed.toarray()
else:
    X_test_dense = X_test_processed

explainer = shap.Explainer(
    model,
    X_train_resampled
)

shap_values = explainer(
    X_test_dense
)

feature_names = (
    preprocessor.get_feature_names_out()
)

plt.figure()

shap.summary_plot(
    shap_values,
    X_test_dense,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig(
    "shap_summary.png"
)

plt.close()

print("\nModel saved successfully!")
print("best_model.pkl")
print("pipeline.pkl")
print("model_metrics.pkl")

