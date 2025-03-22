import requests
import numpy as np
import json

# URL de l'API Flask
url = "http://127.0.0.1:5000/predict"  # Modifie si l'API est déployée ailleurs

# Générer des données aléatoires (10 lectures, 12 features chacune)
test_data = np.random.rand(10, 12).tolist()

# Création du payload JSON
payload = json.dumps({"data": test_data})

# Envoi de la requête POST
response = requests.post(url, data=payload, headers={"Content-Type": "application/json"})

# Affichage de la réponse
print("Réponse de l'API :", response.json())