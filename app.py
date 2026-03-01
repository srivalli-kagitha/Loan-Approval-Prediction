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

# Store prediction logs
prediction_logs = []

# -------------------------
# Home Route
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

    # Convert numeric columns properly
    numeric_cols = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])

    # Feature Engineering
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["EMI"] = df["LoanAmount"] / df["Loan_Amount_Term"]
    df["DebtToIncomeRatio"] = df["LoanAmount"] / df["TotalIncome"]

    processed = preprocessor.transform(df)

    prediction = model.predict(processed)[0]
    result = "Approved" if prediction == 1 else "Rejected"

    # Save log
    prediction_logs.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "result": result
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


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)