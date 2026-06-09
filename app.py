from flask import Flask, render_template, request
import pickle
import pandas as pd
import datetime

app = Flask(__name__)

# -------------------------
# Load Model + Preprocessor
# -------------------------

model = pickle.load(open("best_model.pkl", "rb"))
preprocessor = pickle.load(open("pipeline.pkl", "rb"))

prediction_logs = []

# -------------------------
# Home Page
# -------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -------------------------
# Prediction Route
# -------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.form.to_dict()

    df = pd.DataFrame([data])

    # Convert numeric columns
    numeric_cols = [
        "no_of_dependents",
        "income_annum",
        "loan_amount",
        "loan_term",
        "cibil_score",
        "residential_assets_value",
        "commercial_assets_value",
        "luxury_assets_value",
        "bank_asset_value"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # -------------------------
    # Feature Engineering
    # -------------------------

    df["emi"] = df["loan_amount"] / df["loan_term"]

    df["debt_to_income"] = df["loan_amount"] / df["income_annum"]

    df["total_assets"] = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )

    # -------------------------
    # Preprocess + Predict
    # -------------------------

    processed = preprocessor.transform(df)

    prediction = model.predict(processed)[0]

    result = "Approved" if prediction == 1 else "Rejected"

    # -------------------------
    # Save Prediction Log
    # -------------------------

    prediction_logs.append({

    "timestamp":
    datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),

    "result": result,

    "income":
    data["income_annum"],

    "loan":
    data["loan_amount"],

    "cibil":
    data["cibil_score"]
})

    return render_template("result.html", result=result)


# -------------------------
# Admin Dashboard
# -------------------------

@app.route("/admin")
def admin_dashboard():

    total = len(prediction_logs)

    approved = sum(1 for x in prediction_logs if x["result"] == "Approved")

    rejected = total - approved

    try:
        metrics = pickle.load(open("model_metrics.pkl", "rb"))

        accuracy = round(metrics["accuracy"] * 100, 2)

    except:
        accuracy = 0

    return render_template(
        "admin.html",
        total=total,
        approved=approved,
        rejected=rejected,
        accuracy=accuracy,
        tables=prediction_logs
    )
    
@app.route("/history")
def history():

    return render_template(
        "history.html",
        tables=prediction_logs
    )


@app.route("/analytics")
def analytics():

    total = len(prediction_logs)

    approved = sum(
        1 for x in prediction_logs
        if x["result"] == "Approved"
    )

    rejected = total - approved

    return render_template(
        "analytics.html",
        approved=approved,
        rejected=rejected
    )


@app.route("/settings")
def settings():

    try:
        metrics = pickle.load(
            open("model_metrics.pkl", "rb")
        )

        accuracy = round(
            metrics["accuracy"] * 100,
            2
        )

    except:
        accuracy = 0

    return render_template(
        "settings.html",
        accuracy=accuracy
    )


# -------------------------
# Run App
# -------------------------

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)