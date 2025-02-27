import pytest
import numpy as np
import joblib
import mlflow
from model_pipeline import train_model
from sklearn.ensemble import RandomForestClassifier

# 📌 Configuration de MLflow
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Churn Prediction Experiment"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Vérifier si l'expérience MLflow existe, sinon la créer
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"✅ Expérience créée avec l'ID : {experiment_id}")
else:
    print(f"🔄 L'expérience '{EXPERIMENT_NAME}' existe déjà avec l'ID : {experiment.experiment_id}")

mlflow.set_experiment(EXPERIMENT_NAME)

# 📌 Test 2 : Vérifier que le modèle s'entraîne correctement
def test_train_model():
    """Teste l'entraînement du modèle RandomForestClassifier"""
    
    # 🔹 Générer des données aléatoires
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, 100)
    
    # 🔹 Entraîner le modèle
    model = train_model(X_train, y_train)

    # ✅ Vérifier que le modèle est bien entraîné
    assert model is not None, "Le modèle est None après l'entraînement"
    assert isinstance(model, RandomForestClassifier), "Le modèle n'est pas un RandomForestClassifier"

    # 🔹 Tester la prédiction sur un échantillon
    sample_data = np.random.rand(1, 5)
    prediction = model.predict(sample_data)

    assert prediction is not None, "La prédiction du modèle est None"

    # 🔹 Sauvegarder et recharger le modèle pour tester sa persistance
    model_path = "models/test_model.pkl"
    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)

    assert loaded_model is not None, "Le modèle chargé est None"
    
    print("✅ Test d'entraînement du modèle réussi !")
