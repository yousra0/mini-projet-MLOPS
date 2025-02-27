from fastapi.testclient import TestClient
from app import app  # Assure-toi que `app` est bien importé depuis ton API FastAPI

import sys
sys.path.append(".")

client = TestClient(app)


# 📌 Test 3 : Vérifier que l'API est en bonne santé
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}
    print("✅ Test du endpoint /health réussi !")

# 📌 Test 4 : Vérifier que l'API retourne une prédiction
def test_prediction():
    payload = {"features": [0.5] * 15}  # Assure-toi d'envoyer 15 valeurs
    response = client.post("/predict", json=payload)

    assert response.status_code == 200  # Vérifie si l'API répond correctement
    assert "prediction" in response.json()  # Vérifie que la réponse contient la clé "prediction"

