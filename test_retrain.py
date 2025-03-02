import requests

url = "http://127.0.0.1:8001/retrain"
data = {
    "X_train": [[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]],
    "y_train": [0, 1]
}

response = requests.post(url, json=data)
print("Réponse du serveur :", response.json())
