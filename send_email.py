import smtplib  # Bibliothèque pour envoyer des emails via SMTP
import os  # Permet d'accéder aux variables d'environnement
from email.message import EmailMessage  # Pour créer des emails structurés
from dotenv import load_dotenv  # Pour charger les variables depuis un fichier .env

# 📌 Charger les variables d'environnement depuis le fichier .env
load_dotenv()

# 📌 Récupérer les identifiants depuis les variables d'environnement
EMAIL_SENDER = os.getenv("EMAIL_SENDER")  # Adresse email de l'expéditeur
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")  # Mot de passe d'application généré par Google
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")  # Adresse email du destinataire

def envoyer_email():
    """
    Fonction pour envoyer un email de notification après l'exécution du pipeline CI/CD.
    """
    
    # 📌 Vérifier si les informations d'authentification sont disponibles
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        print("❌ Erreur : Informations d'authentification manquantes dans .env")
        return  # Arrêter l'exécution de la fonction si des informations sont manquantes
    
    # 📌 Création du message email
    msg = EmailMessage()
    msg.set_content("✅ Le pipeline CI/CD s'est terminé avec succès ! 🎉")  # Corps du message
    msg['Subject'] = "Notification Pipeline CI/CD"  # Sujet de l'email
    msg['From'] = EMAIL_SENDER  # Adresse email de l'expéditeur
    msg['To'] = EMAIL_RECEIVER  # Adresse email du destinataire

    try:
        # 📌 Connexion sécurisée au serveur SMTP de Gmail
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:  # Utilisation de SSL pour sécuriser la connexion
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)  # Authentification avec l'email et le mot de passe d'application
            smtp.send_message(msg)  # Envoi du message
            print("📧 Email envoyé avec succès !")  # Message de confirmation
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi de l'email : {e}")  # Afficher l'erreur en cas de problème

# 📌 Exécuter la fonction uniquement si le script est lancé directement
if __name__ == "__main__":
    envoyer_email()
