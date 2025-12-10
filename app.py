from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

# ============================
#  CONFIGURACIÓN DEL MODELO
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "student_performance_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

# Cargar modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)

# Mapeo de clases (debe coincidir con el notebook)
CLASS_LABELS = {
    0: "bajo",
    1: "medio",
    2: "alto"
}

# ============================
#  CATEGORÍAS (ONE-HOT)
#  Deben ser EXACTAMENTE las mismas
#  que usaste en el Colab
# ============================

GENDER_CATS = ["female", "male"]
RACE_CATS = ["group A", "group B", "group C", "group D", "group E"]
PARENT_EDU_CATS = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]
LUNCH_CATS = ["free/reduced", "standard"]
PREP_CATS = ["none", "completed"]


def one_hot(value, categories):
    """
    Devuelve un vector one-hot para 'value'
    dado un listado de categories.
    Si no se encuentra, devuelve todo ceros.
    """
    vec = [0] * len(categories)
    if value in categories:
        idx = categories.index(value)
        vec[idx] = 1
    return vec


def encode_input(data: dict) -> np.ndarray:
    """
    data: dict con las claves:
    - gender
    - race_ethnicity
    - parental_level_of_education
    - lunch
    - test_preparation_course
    Retorna: vector numpy de shape (1, 17)
    """
    gender_vec = one_hot(data["gender"], GENDER_CATS)
    race_vec = one_hot(data["race_ethnicity"], RACE_CATS)
    parent_vec = one_hot(data["parental_level_of_education"], PARENT_EDU_CATS)
    lunch_vec = one_hot(data["lunch"], LUNCH_CATS)
    prep_vec = one_hot(data["test_preparation_course"], PREP_CATS)

    full_vec = gender_vec + race_vec + parent_vec + lunch_vec + prep_vec
    return np.array(full_vec, dtype=float).reshape(1, -1)


# ============================
#  RUTAS
# ============================

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "API de predicción de rendimiento académico activa"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "El cuerpo de la petición debe ser JSON"}), 400

        data = request.get_json()

        required_fields = [
            "gender",
            "race_ethnicity",
            "parental_level_of_education",
            "lunch",
            "test_preparation_course",
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Falta el campo requerido: {field}"}), 400

        # Codificar entrada
        x_input = encode_input(data)  # vector 1 x N

        # Predicción
        preds = model.predict(x_input)
        class_idx = int(np.argmax(preds, axis=1)[0])
        prob = float(np.max(preds))

        return jsonify({
            "rendimiento_clase": CLASS_LABELS.get(class_idx, "desconocido"),
            "clase_numerica": class_idx,
            "probabilidad": round(prob, 4)
        }), 200

    except Exception as e:
        # Para debug puedes imprimir el error en consola
        # print("Error en /predict:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Para pruebas locales
    app.run(host="0.0.0.0", port=5000, debug=True)
