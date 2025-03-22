# Utilisation de l'image officielle Python comme base
FROM python:3.9

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Copie des fichiers nécessaires
COPY requirements.txt ./
COPY api/ ./api/

# Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposition du port sur lequel Flask fonctionne
EXPOSE 5000

# Commande pour exécuter l'application Flask
CMD ["python", "api/app.py"]
