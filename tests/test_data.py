import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from model_pipeline import prepare_data, train_model

# 📌 Test 1 : Vérifier que les données sont bien chargées
def test_prepare_data():
    X_train, X_test, y_train, y_test, scaler = prepare_data("churn-bigml-80.csv")

    # Vérifier que les données ne sont pas vides
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

    # Vérifier que les dimensions sont correctes
    assert X_train.shape[1] == X_test.shape[1]
    
    # Vérifier que la normalisation fonctionne
    assert isinstance(scaler, StandardScaler)

    print("✅ Test de préparation des données réussi !")

