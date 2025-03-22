import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

# Charger le modèle LSTM entraîné
MODEL_PATH = "lstm_model_final.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Paramètres du modèle
TIME_STEPS = 10  # Fenêtre temporelle
NUM_FEATURES = 12  # Nombre de features

# Création de l'application Flask
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données envoyées en JSON
        data = request.get_json()
        
        # Vérification du format des données
        if 'readings' not in data:
            return jsonify({'error': 'Missing readings field'}), 400
        
        readings = np.array(data['readings'])
        
        if readings.shape != (TIME_STEPS, NUM_FEATURES):
            return jsonify({'error': f'Invalid input shape, expected ({TIME_STEPS}, {NUM_FEATURES})'}), 400
        
        # Redimensionner pour correspondre au batch input attendu par le modèle
        readings = np.expand_dims(readings, axis=0)
        
        # Prédiction
        predictions = model.predict(readings)
        predicted_class = int(np.argmax(predictions, axis=1)[0])
        confidence = float(np.max(predictions))
        
        return jsonify({'predicted_state': predicted_class, 'confidence': confidence})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)