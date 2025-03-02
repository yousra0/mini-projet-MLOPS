from flask import Flask, request, jsonify
import joblib
import numpy as np

# 📌 Initialisation de l'application Flask
app = Flask(__name__)

# 📌 Charger le modèle ML
MODEL_PATH = "random_forest_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle Random Forest chargé avec succès (Flask)")
except Exception as e:
    print(f"🚨 Erreur lors du chargement du modèle: {e}")
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Bienvenue sur l'API Flask de prédiction ! 🚀"})


# 📌 Endpoint de prédiction
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "🚨 Modèle non chargé"}), 500

    try:
        # Récupérer les données JSON envoyées par le client
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "❌ Paramètre 'features' manquant"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        # Vérifier que le bon nombre de features est fourni
        if features.shape[1] != 15:
            return jsonify({"error": f"Nombre de features incorrect. Attendu: 15, Reçu: {features.shape[1]}"}), 400

        # Prédiction avec le modèle
        prediction = model.predict(features)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": f"🚨 Erreur lors de la prédiction: {str(e)}"}), 500

# 📌 Lancer le serveur Flask
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
