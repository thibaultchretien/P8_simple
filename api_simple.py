from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained segmentation model
model = load_model('model_simple.h5')

# Define image size and categories
img_height, img_width = 256, 256
cats = {
    'void': [0, 1, 2, 3, 4, 5, 6],
    'flat': [7, 8, 9, 10],
    'construction': [11, 12, 13, 14, 15, 16],
    'object': [17, 18, 19, 20],
    'nature': [21, 22],
    'sky': [23],
    'human': [24, 25],
    'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]
}

def preprocess_image(img):
    # Resize and normalize the image
    img = image.img_to_array(image.load_img(img, target_size=(img_height, img_width))) / 255.0
    return np.expand_dims(img, axis=0)

def decode_mask(mask, categories):
    mask_one_hot = np.zeros((img_height, img_width, len(categories)))
    for i in range(-1, 34):
        for idx, cat in enumerate(categories):
            if i in cats[cat]:
                mask_one_hot[:, :, idx] = np.logical_or(mask_one_hot[:, :, idx], (mask == i))
    return mask_one_hot

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # Load and preprocess the image
    img = request.files['image']
    img = preprocess_image(img)

    # Run the model prediction
    mask_pred = model.predict(img)
    mask_pred = np.argmax(mask_pred, axis=-1).squeeze()  # Get predicted class

    # Convert to a list and return the mask
    return jsonify(mask=mask_pred.tolist())

if __name__ == '__main__':
    app.run(debug=True)
