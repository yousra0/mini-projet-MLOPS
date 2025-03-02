import mlflow
import mlflow.sklearn
import argparse
import time
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
from mlflow.tracking import MlflowClient
from fastapi import FastAPI
import uvicorn

#configuration de FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

# 📌 Configuration de MLflow
print("\n" + "—"*80)
print("🔧 CONFIGURATION DE MLFLOW")
mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()
model_name = "ChurnPredictionModel"

# 📌 Étape 1 : Fonction principale
def main():
    parser = argparse.ArgumentParser(description="Pipeline de Modélisation du Churn")
    parser.add_argument('--data', type=str, required=True, help='📂 Chemin du fichier CSV')
    parser.add_argument('--train', action='store_true', help='🚀 Entraîner le modèle')
    parser.add_argument('--evaluate', action='store_true', help='📊 Évaluer le modèle')
    parser.add_argument('--save', type=str, default='random_forest_model.joblib', help='💾 Nom du fichier pour sauvegarder le modèle')
    parser.add_argument('--load', type=str, help='🔄 Charger un modèle sauvegardé')
    parser.add_argument('--stage', type=str, default="Production", help="🎯 Définir le stage du modèle (Production uniquement)")

    args = parser.parse_args()

    # 📌 Étape 2 : Chargement des données
    print("\n" + "—"*80)
    print("📂 CHARGEMENT DES DONNÉES")
    print(f"📌 Fichier : {args.data}")
    X_train, X_test, y_train, y_test, scaler = prepare_data(args.data)
    print("✅ Données chargées avec succès !")
    print("—"*80 + "\n")

    model = None  # Initialisation du modèle

    # 🔥 Étape 3 : Démarrer un run MLflow
    with mlflow.start_run(run_name="New_Run") as run:
        print("\n" + "—"*80)
        print("📌 RUN MLFLOW")
        print(f"🏃 ID du Run : {run.info.run_id}")
        mlflow.log_param("data_file", args.data)
        print("—"*80 + "\n")

        # 🔄 Étape 4 : Chargement du modèle existant (si spécifié)
        if args.load:
            print("🔄 CHARGEMENT DU MODÈLE EXISTANT")
            model = load_model(args.load)
            if model is None:
                print("❌ Erreur : Impossible de charger le modèle.")
                return
            print("✅ Modèle chargé avec succès !")
            print("—"*80 + "\n")

        # 🚀 Étape 5 : Entraînement du modèle
        if args.train:
            print("🚀 ENTRAÎNEMENT DU MODÈLE")
            model = train_model(X_train, y_train)
            if model is None:
                print("❌ Erreur : L'entraînement du modèle a échoué.")
                return
            print("✅ Modèle entraîné avec succès !")

            # 🔍 Enregistrement des hyperparamètres
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("max_depth", None)

            # 💾 Sauvegarde du modèle
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_param("algorithm", "RandomForest")

            if args.save:
                save_model(model, args.save)
                print(f"💾 Modèle sauvegardé sous {args.save}")
            print("—"*80 + "\n")

            # 🔥 Étape 6 : Enregistrement dans la Model Registry
            print("🔥 ENREGISTREMENT DU MODÈLE DANS MLFLOW")
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model = mlflow.register_model(model_uri, model_name)
            time.sleep(5)  # Attente pour éviter les erreurs
            model_version = registered_model.version
            print(f"📌 Version du modèle créée : {model_version}")
            print("—"*80 + "\n")

        # 📊 Étape 7 : Évaluation du modèle (si demandée)
        if args.evaluate:
            print("📊 ÉVALUATION DU MODÈLE")
            if model is None:
                print("⚠️ Aucun modèle trouvé ! Veuillez en entraîner un ou en charger un avec --load.")
            else:
                metrics = evaluate_model(model, X_test, y_test)
                for metric, value in metrics.items():
                    print(f"   {metric}: {value:.4f}")
                    mlflow.log_metric(metric, value)
            print("✅ Évaluation terminée !")
            print("—"*80 + "\n")

        # 🏷️ Étape 8 : Ajout de métadonnées et mise à jour du stage
        print("🏷️ AJOUT DES MÉTADONNÉES AU MODÈLE")

        if 'model_version' in locals():
            # Ajouter des tags
            client.set_model_version_tag(model_name, model_version, "stage", args.stage)
            if args.evaluate and 'metrics' in locals():
                client.set_model_version_tag(
                    model_name,
                    model_version,
                    "description",
                    f"Modèle évalué avec une précision de {metrics.get('Accuracy', 0):.4f}"
                )

            # **✅ Transition correcte vers Staging ou Production**
            client.transition_model_version_stage(model_name, model_version, args.stage)
            print(f"🚀 Modèle {model_name} version {model_version} mis en {args.stage} !")
        else:
            print("⚠️ Aucun modèle disponible dans la Model Registry. Vérifiez l'entraînement ou le chargement du modèle.")

        print("—"*80 + "\n")


# 🔥 Lancer le pipeline
if __name__ == "__main__":
    main()
