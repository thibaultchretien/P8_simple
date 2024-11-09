import os
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Charger le modèle (remplacez 'segmentation_model.h5' par le nom de votre fichier de modèle)
model = tf.keras.models.load_model('model_simple.h5')

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((256, 256))  # Redimensionnez selon les dimensions de votre modèle
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nom de fichier vide'}), 400

    try:
        image_array = preprocess_image(file.read())
        prediction = model.predict(image_array)[0]
        
        # Post-traitement : ici, on redimensionne l'image prédite à la taille originale si nécessaire
        prediction = (prediction > 0.5).astype(np.uint8)  # Seuil de binarisation
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
