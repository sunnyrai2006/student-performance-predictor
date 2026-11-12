
from flask import Flask, request, jsonify
import os
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------- PATH CONFIGURATION ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = BASE_DIR

# ---------------- LOAD MODELS ----------------

ann_model = load_model(
    os.path.join(MODEL_DIR, "student_performance_ann.keras"),
    compile=False
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

models = {
    "Logistic Regression": joblib.load(
        os.path.join(MODEL_DIR, "Logistic_Regression.pkl")
    ),

    "Random Forest": joblib.load(
        os.path.join(MODEL_DIR, "Random_Forest.pkl")
    ),

    "Support Vector Machine": joblib.load(
        os.path.join(MODEL_DIR, "Support_Vector_Machine.pkl")
    )
}

# ---------------- FEATURE ORDER ----------------

FEATURES = [
    "age",
    "attendance_pct",
    "study_hours_per_week",
    "previous_score",
    "assignment_completion_pct",
    "internal_assessment_score",
    "participation_score",
    "sleep_hours_per_day",
    "extracurricular_hours_per_week",
    "failed_subjects"
]

# ---------------- PREDICTION FUNCTION ----------------

def predict_ann(input_scaled):

    probabilities = ann_model.predict(
        input_scaled,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    prediction = encoder.inverse_transform(
        [predicted_index]
    )[0]

    probability_dict = {
        str(label): round(float(probabilities[i]), 4)
        for i, label in enumerate(encoder.classes_)
    }

    confidence = round(
        float(probabilities[predicted_index]) * 100, 2
    )

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "probabilities": probability_dict
    }


def predict_ml(model, input_scaled):

    predicted_index = int(model.predict(input_scaled)[0])

    prediction = encoder.inverse_transform(
        [predicted_index]
    )[0]

    result = {
        "prediction": str(prediction),
        "confidence": None,
        "probabilities": None
    }

    # Probability is available for models supporting predict_proba
    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_scaled)[0]

        probability_dict = {}

        for i, class_index in enumerate(model.classes_):

            label = encoder.inverse_transform(
                [int(class_index)]
            )[0]

            probability_dict[str(label)] = round(
                float(probabilities[i]), 4
            )

        result["probabilities"] = probability_dict

        result["confidence"] = round(
            float(np.max(probabilities)) * 100, 2
        )

    return result


# ---------------- HOME ROUTE ----------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Student Performance Prediction API is running",
        "available_models": [
            "ANN",
            "Logistic Regression",
            "Random Forest",
            "Support Vector Machine",
            "Compare All Models"
        ]
    })


# ---------------- PREDICTION ROUTE ----------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "error": "Please send input data as JSON."
            }), 400

        # Check missing features
        missing_features = [
            feature for feature in FEATURES
            if feature not in data
        ]

        if missing_features:

            return jsonify({
                "error": "Missing required features",
                "missing_features": missing_features
            }), 400

        # Prepare input
        input_df = pd.DataFrame(
            [[data[feature] for feature in FEATURES]],
            columns=FEATURES
        )

        input_df = input_df.astype(float)

        if not np.isfinite(input_df.to_numpy()).all():

            return jsonify({
                "error": "Input values must be finite numbers."
            }), 400

        # Scale input
        input_scaled = scaler.transform(input_df)

        selected_model = data.get("model", "ANN")

        # ANN prediction
        if selected_model == "ANN":

            result = predict_ann(input_scaled)

            return jsonify({
                "model": "ANN",
                **result
            })

        # Compare all models
        elif selected_model == "Compare All Models":

            results = {}

            results["ANN"] = predict_ann(input_scaled)

            for model_name, model in models.items():

                results[model_name] = predict_ml(
                    model,
                    input_scaled
                )

            return jsonify({
                "model": "Compare All Models",
                "results": results
            })

        # Traditional ML prediction
        elif selected_model in models:

            result = predict_ml(
                models[selected_model],
                input_scaled
            )

            return jsonify({
                "model": selected_model,
                **result
            })

        else:

            return jsonify({
                "error": "Invalid model selected."
            }), 400

    except ValueError as e:

        return jsonify({
            "error": f"Invalid input: {str(e)}"
        }), 400

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ---------------- RUN API ----------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
