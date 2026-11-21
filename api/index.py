from flask import Flask, request, render_template_string
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load the trained model and encoder
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "iris_model.pkl")
encoder = joblib.load(BASE_DIR / "iris_encoder.pkl")

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Iris Flower Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f1f4f9;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 650px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        h1 {
            color: #172554;
            text-align: center;
        }

        p {
            color: #555;
            text-align: center;
        }

        label {
            display: block;
            margin-top: 15px;
            font-weight: bold;
        }

        input {
            width: 100%;
            box-sizing: border-box;
            padding: 12px;
            margin-top: 7px;
            border: 1px solid #ccc;
            border-radius: 7px;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 14px;
            background: #172554;
            color: white;
            border: none;
            border-radius: 7px;
            cursor: pointer;
            font-size: 16px;
        }

        button:hover {
            background: #263f83;
        }

        .result {
            margin-top: 25px;
            padding: 18px;
            background: #e8f5e9;
            border-radius: 8px;
            text-align: center;
        }

        .error {
            margin-top: 20px;
            color: #b91c1c;
            text-align: center;
        }
    </style>
</head>

<body>
<div class="container">

    <h1>🌸 Iris Flower Classifier</h1>

    <p>
        Enter the flower measurements to predict its species
        using a Random Forest machine learning model.
    </p>

    <form method="POST">

        <label>Sepal Length (cm)</label>
        <input type="number" step="any" name="sepal_length"
               value="{{ values.get('sepal_length', '') }}" required>

        <label>Sepal Width (cm)</label>
        <input type="number" step="any" name="sepal_width"
               value="{{ values.get('sepal_width', '') }}" required>

        <label>Petal Length (cm)</label>
        <input type="number" step="any" name="petal_length"
               value="{{ values.get('petal_length', '') }}" required>

        <label>Petal Width (cm)</label>
        <input type="number" step="any" name="petal_width"
               value="{{ values.get('petal_width', '') }}" required>

        <button type="submit">Predict Iris Species</button>

    </form>

    {% if prediction %}
    <div class="result">
        <h2>Predicted Species: {{ prediction }}</h2>
        <p>Prediction Confidence: {{ confidence }}%</p>
    </div>
    {% endif %}

    {% if error %}
    <div class="error">{{ error }}</div>
    {% endif %}

</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    error = None
    values = {}

    if request.method == "POST":
        try:
            values = {
                feature: request.form[feature]
                for feature in FEATURES
            }

            measurements = [
                float(values[feature])
                for feature in FEATURES
            ]

            if not all(
                0 < value < 20 for value in measurements
            ):
                raise ValueError(
                    "Enter valid positive flower measurements."
                )

            input_data = pd.DataFrame(
                [measurements],
                columns=FEATURES
            )

            predicted_class = model.predict(input_data)
            predicted_species = encoder.inverse_transform(
                predicted_class
            )

            prediction = str(predicted_species[0])

            probabilities = model.predict_proba(input_data)
            confidence = round(
                float(probabilities.max()) * 100, 2
            )

        except (ValueError, KeyError):
            error = "Please enter valid numerical measurements."

        except Exception:
            error = "Prediction failed. Please try again."

    return render_template_string(
        HTML,
        prediction=prediction,
        confidence=confidence,
        error=error,
        values=values
    )