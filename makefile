.PHONY: install train evaluate-staging evaluate-production test lint data setup notebook run-api test-api deploy clean build run list login push start stop logs notify full-pipeline

# 📌 Définitions des variables
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt

DATA_PATH=/home/yousra/yousra_chaieb_ml_project/churn-bigml-80.csv
TEST_DATA_PATH=/home/yousra/yousra_chaieb_ml_project/churn-bigml-20.csv
MODEL_DIR=models
MODEL_PATH=$(MODEL_DIR)/random_forest_model.pkl

# 🚀 Étape 1 : Installation des dépendances
install:
	@echo "🔧 Création de l'environnement virtuel et installation des dépendances..."
	$(PYTHON) -m venv $(ENV_NAME)
	. $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)

# 🚀 Étape 2 : Entraînement du modèle et enregistrement en Staging
train:
	@echo "📂 Création du dossier pour le modèle..."
	mkdir -p $(MODEL_DIR)
	. $(ENV_NAME)/bin/activate && python3 main.py --data $(DATA_PATH) --train --save $(MODEL_PATH) --stage Staging

# 🚀 Étape 3 : Évaluation du modèle
## 🔹 Évaluation en Staging
evaluate-staging:
	. $(ENV_NAME)/bin/activate && python3 main.py --data $(DATA_PATH) --evaluate --stage Staging

## 🔹 Passer un modèle en Production
evaluate-production:
	. $(ENV_NAME)/bin/activate && python3 main.py --data $(DATA_PATH) --evaluate --stage Production

# 🚀 Étape 4 : Tests et vérifications
## 🔹 Exécution des tests unitaires
test:
	@echo "🧪 Exécution des tests unitaires..."
	. $(ENV_NAME)/bin/activate && pytest tests/

## 🔹 Vérification du code avec Flake8
lint:
	@echo "🔍 Vérification du code avec flake8..."
	. $(ENV_NAME)/bin/activate && flake8 .

# 🚀 Étape 5 : Préparation des données
data:
	@echo "📂 Préparation des données..."
	. $(ENV_NAME)/bin/activate && python /home/yousra/yousra_chaieb_ml_project/model_pipeline.py

# 🚀 Étape 6 : Configuration initiale
setup: install
	@echo "✅ Configuration terminée."

# 🚀 Étape 7 : Exécution des notebooks
notebook:
	@echo "📖 Démarrage de Jupyter Notebook..."
	. $(ENV_NAME)/bin/activate && jupyter notebook

# 🚀 Étape 8 : Exécution et test de l'API FastAPI
## 🔹 Lancer l'API
run-api:
	uvicorn app:app --reload --host 0.0.0.0 --port 8001

## 🔹 Tester l'API via Swagger
test-api:
	@echo "🌍 Ouvrir l'interface Swagger à l'adresse http://127.0.0.1:8001/docs"
	uvicorn app:app --reload --host 127.0.0.1 --port 8001

# 🚀 Étape 9 : Déploiement
deploy:
	@echo "🚀 Déploiement du modèle..."
	uvicorn app:app --host 0.0.0.0 --port 8001 --workers 4

# 🚀 Étape 10 : Nettoyage de l'environnement
clean:
	@echo "🧹 Nettoyage des fichiers temporaires..."
	rm -rf $(ENV_NAME) __pycache__ *.pyc *.pyo .pytest_cache .mypy_cache

# 🚀 Étape 11 : Gestion Docker
## 🔹 Construire l’image Docker
build:
	docker build -t yousra_chaieb_4ds4_mlops .

## 🔹 Exécuter le conteneur
run:
	docker run -p 8000:8000 yousra_chaieb_4ds4_mlops

## 🔹 Lister les images Docker
list:
	docker images

## 🔹 Se connecter à Docker Hub
login:
	docker login

## 🔹 Pousser l’image sur Docker Hub
push:
	docker tag yousra_chaieb_4ds4_mlops yousrachaieb/fastapi-mlflow-app:v1
	docker push yousrachaieb/fastapi-mlflow-app:v1

# 🚀 Étape 12 : Gestion des services avec Docker Compose
## 🔹 Démarrer MLflow et FastAPI
start:
	docker-compose up -d
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
	uvicorn app:app --host 0.0.0.0 --port 8000

## 🔹 Arrêter les services
stop:
	docker-compose down

## 🔹 Afficher les logs des services
logs:
	docker-compose logs -f

# 🚀 Étape 13 : Partie bonus - Notification par e-mail
EMAIL_NOTIFICATION_SCRIPT=send_email.py

# 🔹 Pipeline complète avec notification
full-pipeline: install lint data train test notify

# 🔹 Envoi d'une notification
notify:
	@echo "✅ Pipeline terminé avec succès !" 
	. $(ENV_NAME)/bin/activate && python send_email.py
