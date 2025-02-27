# 📌 Étape 1 : Importation des bibliothèques nécessaires

# Manipulation des données
import pandas as pd 

# Sauvegarde et chargement des modèles
import joblib

# Prétraitement des données
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Modèle d'apprentissage supervisé
from sklearn.ensemble import RandomForestClassifier

# Division des données en ensembles d'entraînement et de test
from sklearn.model_selection import train_test_split

# Évaluation des performances du modèle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import mlflow

# 📌 Étape 2 : Chargement d'un modèle MLflow
def load_mlflow_model(model_uri):
    """
    Charge un modèle MLflow à partir d'un URI donné.
    """
    return mlflow.sklearn.load_model(model_uri)

# 📌 Étape 3 : Préparation des données
def prepare_data(filepath):
    """
    Charge et prétraite les données depuis un fichier CSV.
    Effectue l'encodage des variables catégoriques, la normalisation et la séparation en ensembles d'entraînement/test.
    """
    print(f"📂 Chargement des données depuis {filepath}...")
    
    df = pd.read_csv(filepath)

    # Encodage des variables catégoriques
    label_encoders = {}
    for col in ['State', 'International plan', 'Voice mail plan']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Conversion de la colonne cible 'Churn' en binaire
    df['Churn'] = df['Churn'].astype(int)

    # Suppression des colonnes inutiles
    drop_cols = ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']
    df.drop(columns=drop_cols, inplace=True)

    # Séparation des features et de la cible
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    # Normalisation des données
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Séparation des données en train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("✅ Données prétraitées avec succès !")
    return X_train, X_test, y_train, y_test, scaler

# 📌 Étape 4 : Entraînement du modèle
def train_model(X_train, y_train, n_estimators=100, max_depth=None):
    """
    Entraîne un modèle Random Forest sur les données fournies.
    """
    if X_train is None or y_train is None:
        print("❌ Erreur : Les données d'entraînement sont manquantes.")
        return None

    print("🚀 Entraînement du modèle Random Forest...")

    # Enregistrement des hyperparamètres dans MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Initialisation et entraînement du modèle
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    print("✅ Modèle entraîné avec succès !")
    return rf

# 📌 Étape 5 : Évaluation du modèle
def evaluate_model(model, X_test, y_test):
    """
    Évalue un modèle sur les données de test en utilisant plusieurs métriques de performance.
    """
    if model is None:
        print("❌ Erreur : Aucun modèle fourni pour l'évaluation.")
        return {}

    print("📊 Évaluation du modèle en cours...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }

    print("✅ Résultats de l'évaluation :")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

    return metrics

# 📌 Étape 6 : Sauvegarde du modèle
def save_model(model, filename='random_forest_model.joblib'):
    """
    Sauvegarde un modèle entraîné dans un fichier.
    """
    if model is None:
        print("❌ Erreur : Aucun modèle à sauvegarder.")
        return

    joblib.dump(model, filename)
    print(f"💾 Modèle sauvegardé sous {filename}")

# 📌 Étape 7 : Chargement du modèle
def load_model(filename='random_forest_model.joblib'):
    """
    Charge un modèle sauvegardé depuis un fichier.
    """
    try:
        print(f"🔄 Chargement du modèle depuis {filename}...")
        model = joblib.load(filename)
        print("✅ Modèle chargé avec succès !")
        return model
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier {filename} introuvable.")
        return None
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle : {e}")
        return None
