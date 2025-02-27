import logging
from elasticsearch import Elasticsearch
from pythonjsonlogger import jsonlogger

# Connexion à Elasticsearch
es = Elasticsearch(["http://localhost:9200"])

# Configuration du logger
logger = logging.getLogger("mlflow")
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Fonction pour envoyer un log à Elasticsearch
def send_log(metric, value, experiment):
    log_data = {
        "metric": metric,
        "value": value,
        "experiment": experiment
    }
    es.index(index="mlflow-logs", body=log_data)
    logger.info("Log envoyé à Elasticsearch", extra=log_data)
