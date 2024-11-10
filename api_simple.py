import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Charger le modèle
model = tf.keras.models.load_model('model_simple.h5')

def preprocess_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((256, 256))  # Redimensionnez selon les dimensions de votre modèle
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        raise ValueError("Erreur dans le traitement de l'image: " + str(e))

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'API de segmentation d\'image est en ligne'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Vérification de la présence du fichier
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400

    try:
        # Prétraitement de l'image
        image_array = preprocess_image(file.read())
        
        # Prédiction
        prediction = model.predict(image_array)[0]
        
        # Post-traitement : application d'un seuil de binarisation
        prediction = (prediction > 0.5).astype(np.uint8)
        
        # Retour de la prédiction
        return jsonify({'prediction': prediction.tolist()})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Erreur interne: ' + str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
