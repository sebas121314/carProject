import os
import pandas as pd
from flask import Flask, jsonify, render_template, request
import joblib


MODEL_CANDIDATES = [
    os.path.join("car+evaluation", "best_car_model_sprint3.pkl"),
    os.path.join("car+evaluation", "best_car_model.pkl"),
    "best_car_model_sprint3.pkl",
    "best_car_model.pkl",
]


def load_model():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("No se encontró el modelo. Ejecuta el notebook de sprint 3 para generar el .pkl.")


pipe = load_model()

FEATURE_ORDER = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]

# Opciones para la UI y validación
CHOICES = {
    "buying": ["vhigh", "high", "med", "low"],
    "maint": ["vhigh", "high", "med", "low"],
    "doors": ["2", "3", "4", "5more"],
    "persons": ["2", "4", "more"],
    "lug_boot": ["small", "med", "big"],
    "safety": ["low", "med", "high"],
}

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", choices=CHOICES)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    if not payload:
        return jsonify({"error": "No se recibieron datos"}), 400

    # Validación simple de campos requeridos
    missing = [f for f in FEATURE_ORDER if f not in payload]
    if missing:
        return jsonify({"error": f"Faltan campos: {', '.join(missing)}"}), 400

    # Normalizamos a strings y validamos elección
    row = []
    for feature in FEATURE_ORDER:
        value = str(payload.get(feature, "")).strip()
        if value not in CHOICES[feature]:
            return (
                jsonify(
                    {
                        "error": f"Valor inválido para {feature}: '{value}'. Esperado uno de {CHOICES[feature]}"
                    }
                ),
                400,
            )
        row.append(value)

    # DataFrame con columnas para que el pipeline (ColumnTransformer) encuentre los nombres
    sample_df = pd.DataFrame([row], columns=FEATURE_ORDER)
    pred = pipe.predict(sample_df)[0]
    return jsonify({"prediction": pred})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
