import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from yousra_chaieb_ml_project.data_pipeline import prepare_data  # 📌 Import correct
from yousra_chaieb_ml_project.training_pipeline import train_model  # 📌 Import correct

# 📌 1. Test unitaire pour la préparation des données
def test_prepare_data():
    """ Teste si prepare_data() fonctionne correctement. """
    X_train, X_test, y_train, y_test, scaler = prepare_data("data/sample.csv")
    
    # Vérifier que les jeux d'entraînement et de test ne sont pas vides
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

    # Vérifier que le nombre de caractéristiques est le même dans X_train et X_test
    assert X_train.shape[1] == X_test.shape[1]

    print("✅ Test unitaire : prepare_data() réussi !")

# 📌 2. Test unitaire pour l'entraînement du modèle
def test_train_model():
    """ Teste si train_model() entraîne correctement un modèle Random Forest. """
    X_train = np.random.rand(100, 5)  # Données factices
    y_train = np.random.randint(0, 2, 100)  # Classes 0 ou 1

    model = train_model(X_train, y_train)

    # Vérifier que le modèle est bien entraîné
    assert model is not None
    assert isinstance(model, RandomForestClassifier)

    print("✅ Test unitaire : train_model() réussi !")

# 📌 3. Test fonctionnel : Vérifie que l'entraînement fonctionne avec des données réelles
def test_pipeline():
    """ Teste si le pipeline complet fonctionne. """
    X_train, X_test, y_train, y_test, scaler = prepare_data("data/sample.csv")
    model = train_model(X_train, y_train)

    # Vérifier que le modèle peut faire des prédictions
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)

    print("✅ Test fonctionnel : Pipeline complet réussi !")
