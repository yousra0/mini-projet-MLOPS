# Import des bibliothèques nécessaires
import argparse  # Pour gérer les arguments passés en ligne de commande
import logging  # Pour afficher des messages d'information ou d'erreur
import os  # Pour interagir avec le système de fichiers

# Import des fonctions du module model_pipeline
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

from sklearn.model_selection import GridSearchCV, cross_val_score
import time
######################################################################################################

#utilisation de la bibliothèque logging pour mieux suivre le déroulement 
#de l'exécution et gérer les erreurs efficacement

# Configuration du système de logging pour afficher les messages avec un format clair
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

######################################################################################################
#PREPARE DATA
# Fonction : Charger les données depuis un fichier CSV
def charger_donnees(data_path: str):
    """Vérifie l'existence du fichier CSV et prépare les données"""
    if not os.path.isfile(data_path):  # Vérifie si le fichier existe
        logging.error(f"❌ Le fichier {data_path} n'existe pas.")
        return None
    logging.info(f"📂 Chargement des données depuis {data_path}...")
    return prepare_data(data_path)  # Prépare les données pour l'entraînement

######################################################################################################
#LOAD MODEL
# Fonction : Charger un modèle pré-entraîné depuis un fichier
def charger_modele(model_path: str):
    """Charge un modèle existant si le fichier est valide"""
    if not os.path.isfile(model_path):  # Vérifie si le modèle existe
        logging.error("❌ Le fichier du modèle spécifié n'existe pas.")
        return None
    logging.info(f"🔄 Chargement du modèle depuis {model_path}...")
    model = load_model(model_path)
    if model:
        logging.info("✅ Modèle chargé avec succès !")
    return model

######################################################################################################
#TRAIN MODEL
# Fonction : Entraîner un nouveau modèle
def entrainer_modele(X_train, y_train):
    """Entraîne un modèle avec recherche d'hyperparamètres et validation croisée"""
    from sklearn.ensemble import RandomForestClassifier

    logging.info("🚀 Entraînement du modèle avec validation croisée et recherche d'hyperparamètres...")

    # 🔍 Définition des hyperparamètres à tester
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 🌲 Initialisation du modèle
    model = RandomForestClassifier(random_state=42)

    # 🔍 Recherche des meilleurs hyperparamètres avec GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1)

    # ⏱️ Mesure du temps d'entraînement
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()

    # 📊 Affichage des résultats
    best_model = grid_search.best_estimator_
    logging.info(f"✅ Meilleurs hyperparamètres : {grid_search.best_params_}")
    logging.info(f"⏱️ Temps d'entraînement : {end_time - start_time:.2f} secondes")

    # 📈 Validation croisée
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    logging.info(f"📊 Score moyen en validation croisée : {cv_scores.mean():.4f}")

    return best_model

######################################################################################################
#SAVE MODEL
# Fonction : Sauvegarder un modèle entraîné
def sauvegarder_modele(model, save_path: str):
    """Sauvegarde le modèle entraîné dans un fichier"""
    save_model(model, save_path)
    logging.info(f"💾 Modèle sauvegardé sous {save_path}")

######################################################################################################
#EVALUATE MODEL
# Fonction : Évaluer les performances du modèle
def evaluer_modele(model, X_test, y_test):
    """Évalue les performances du modèle avec des données de test"""
    logging.info("📊 Évaluation du modèle en cours...")
    metrics = evaluate_model(model, X_test, y_test)  # Retourne les métriques (précision, rappel, etc.)
    logging.info("✅ Résultats de l'évaluation :")
    for metric, value in metrics.items():
        logging.info(f"   {metric}: {value:.4f}")  # Affiche les métriques avec 4 décimales

######################################################################################################

# Fonction principale : Contrôle l'exécution du script
def main():
    # 📌 Utilisation d'argparse pour gérer les arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Pipeline de Modélisation du Churn')
    parser.add_argument('--data', type=str, required=True, help='Chemin du fichier CSV')
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle')
    parser.add_argument('--evaluate', action='store_true', help='Évaluer le modèle')
    parser.add_argument('--save', type=str, default='random_forest_model.pkl', help='Nom du fichier pour sauvegarder le modèle')
    parser.add_argument('--load', type=str, help='Charger un modèle sauvegardé')

    # 📥 Récupération des arguments fournis par l'utilisateur
    args = parser.parse_args()

    # ✅ Étape 1 : Charger les données depuis le fichier CSV
    result = charger_donnees(args.data)
    if result is None:
        return  # Arrête l'exécution si le fichier CSV est invalide
    X_train, X_test, y_train, y_test, scaler = result

    # ✅ Étape 2 : Charger un modèle existant si demandé
    model = None
    if args.load:
        model = charger_modele(args.load)

    # ✅ Étape 3 : Entraîner un nouveau modèle si demandé
    if args.train:
        model = entrainer_modele(X_train, y_train)

    # ✅ Étape 4 : Vérifier si un modèle est disponible
    if model is None:
        logging.warning("⚠️ Aucun modèle trouvé ! Entraînez-en un ou chargez-en un avec --load.")
        return

    # ✅ Étape 5 : Sauvegarder le modèle entraîné
    if args.save:
        sauvegarder_modele(model, args.save)

    # ✅ Étape 6 : Évaluer le modèle si demandé
    if args.evaluate:
        evaluer_modele(model, X_test, y_test)

######################################################################################################

# 🔥 Point d'entrée du programme
if __name__ == '__main__':
    main()
