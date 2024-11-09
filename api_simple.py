from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import io
import base64

# Charger le modèle de segmentation
model = load_model('model_simple.h5')

app = Flask(__name__)

# Fonction pour prétraiter l'image
def preprocess_image(image):
    image = image.resize((256, 256))  # Ajuster la taille à celle attendue par le modèle
    image_array = np.array(image) / 255.0  # Normaliser l'image
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter la dimension batch
    return image_array

# Point de terminaison pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    # Vérifier si l'image est présente dans la requête
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Charger l'image à partir de la requête
    image_file = request.files['image']
    image = Image.open(image_file)

    # Prétraiter l'image
    preprocessed_image = preprocess_image(image)

    # Faire la prédiction
    mask = model.predict(preprocessed_image)
    mask = np.argmax(mask, axis=-1)  # Pour une sortie multi-classe
    mask = np.squeeze(mask, axis=0)

    # Convertir le masque en image et en base64 pour le retour JSON
    mask_image = Image.fromarray((mask * 255).astype(np.uint8))  # Assurez-vous que le masque est bien au format uint8
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Renvoyer le masque encodé en base64
    return jsonify({'mask': mask_base64})

if __name__ == '__main__':
    app.run(debug=True)
