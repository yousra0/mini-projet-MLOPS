from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import logging
from logger_config import send_log, logger  # Importation du logger et de la fonction d'envoi des logs

# 📌 Étape 1 : Configuration de MLflow et du client de suivi
print("\n" + "—" * 80)
print("🔧 CONFIGURATION DE MLFLOW")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# 🔍 Vérifier la connexion à MLflow
try:
    experiments = client.search_experiments()
    print("✅ Connexion réussie à MLflow ! Expériences disponibles :", [exp.name for exp in experiments])
except Exception as e:
    print(f"🚨 Erreur de connexion à MLflow : {e}")

EXPERIMENT_NAME = "Churn Prediction Experiment"

# 📌 Étape 2 : Vérification de l'existence de l'expérience MLflow
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
    print(f"✅ Nouvelle expérience MLflow créée : ID {experiment_id}")
else:
    experiment_id = experiment.experiment_id
    print(f"🔄 Expérience MLflow existante : ID {experiment_id}")

mlflow.set_experiment(EXPERIMENT_NAME)
print("—" * 80 + "\n")

# 📌 Étape 3 : Chargement du modèle
MODEL_PATH = "random_forest_model.joblib"
try:
    model = joblib.load(MODEL_PATH)
    logger.info("✅ Modèle Random Forest chargé avec succès")
except Exception as e:
    logger.error(f"🚨 Erreur lors du chargement du modèle: {e}")
    raise RuntimeError(f"🚨 Erreur lors du chargement du modèle: {e}")

# 📌 Étape 4 : Initialisation de l'application FastAPI
app = FastAPI()

# 📌 Définition du format d'entrée pour les requêtes de prédiction
class PredictionInput(BaseModel):
    features: list

# 📌 Étape 5 : Endpoints de l'API

## 🏥 Vérifier que le service fonctionne bien
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }

## 👋 Endpoint d'accueil
@app.get("/")
def hello_world():
    return {"message": "👋 Hello, Yousra! 🚀"}

## 🔮 Endpoint de prédiction
@app.post("/predict")
def predict(data: PredictionInput):
    try:
        # Vérifier que les features sont bien une liste non vide
        if not isinstance(data.features, list) or len(data.features) == 0:
            raise ValueError("❌ Les features doivent être une liste non vide.")

        # Transformer l'entrée en tableau numpy et faire la prédiction
        X_input = np.array(data.features).reshape(1, -1)
        prediction = model.predict(X_input)
        prediction_value = int(prediction[0])

        # 🔍 Enregistrer la prédiction avec MLflow
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_metric("prediction", prediction_value)

        # 📡 Envoyer les logs à Elasticsearch
        send_log("mlflow-metrics", {
            "prediction": prediction_value,
            "features": data.features
        }, experiment=EXPERIMENT_NAME)

        return {"prediction": prediction_value}

    except ValueError as ve:
        logger.error(f"⚠️ Erreur de validation des données: {ve}")
        raise HTTPException(status_code=400, detail=f"⚠️ Erreur: {ve}")

    except Exception as e:
        logger.error(f"🚨 Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"🚨 Erreur lors de la prédiction: {e}")

# 📌 Étape 6 : Point d'entrée pour exécuter l'application avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
