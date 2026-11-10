from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_diabetes_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_comparison.csv"

model = joblib.load(MODEL_PATH)
metrics = pd.read_csv(METRICS_PATH).round(3).to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            patient = pd.DataFrame([{
                "gender": request.form["gender"],
                "age": float(request.form["age"]),
                "hypertension": int(request.form["hypertension"]),
                "heart_disease": int(request.form["heart_disease"]),
                "smoking_history": request.form["smoking_history"],
                "bmi": float(request.form["bmi"]),
                "HbA1c_level": float(request.form["HbA1c_level"]),
                "blood_glucose_level": float(request.form["blood_glucose_level"])
            }])

            if not (0 <= patient.loc[0, "age"] <= 100):
                raise ValueError("Age must be between 0 and 100.")
            if not (10 <= patient.loc[0, "bmi"] <= 80):
                raise ValueError("BMI must be between 10 and 80.")
            if not (3 <= patient.loc[0, "HbA1c_level"] <= 20):
                raise ValueError("HbA1c level must be between 3 and 20.")
            if not (40 <= patient.loc[0, "blood_glucose_level"] <= 500):
                raise ValueError("Blood glucose must be between 40 and 500.")

            result = int(model.predict(patient)[0])
            prediction = "Higher likelihood of diabetes" if result == 1 else "Lower likelihood of diabetes"
            if hasattr(model, "predict_proba"):
                probability = round(float(model.predict_proba(patient)[0][1]) * 100, 1)
        except (ValueError, KeyError) as exc:
            error = str(exc)

    return render_template("index.html", prediction=prediction,
                           probability=probability, error=error, metrics=metrics)

if __name__ == "__main__":
    app.run(debug=True)
