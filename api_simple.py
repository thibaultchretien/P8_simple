import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import io
from PIL import Image
import base64

# Load the model
model = load_model('model_simple.h5')

# Parameters for image preprocessing
img_height, img_width = 256, 256

# Initialize Flask app
app = Flask(__name__)

# Function for preprocessing image and mask
def preprocess_image(img_data):
    # Convert the base64 image data to a PIL image
    img = Image.open(io.BytesIO(base64.b64decode(img_data)))
    img = img.resize((img_width, img_height))  # Resize to match model input size
    img = np.array(img) / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to perform segmentation and return results
def segment_image(img):
    # Predict the segmentation mask
    prediction = model.predict(img)
    
    # Post-process the prediction (convert to one-hot encoding)
    prediction = np.argmax(prediction, axis=-1)  # Get the most probable class per pixel
    return prediction[0]  # Remove the batch dimension

# API route to handle image segmentation
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request (base64 encoded)
    data = request.get_json()
    img_data = data['image']
    
    # Preprocess the image
    img = preprocess_image(img_data)
    
    # Perform the segmentation
    segmented_image = segment_image(img)
    
    # Check if the segmentation has a valid result (e.g., no all-zero values)
    if np.all(segmented_image == 0):
        return jsonify({'error': 'Segmentation failed. The mask is all zeros.'})

    # Convert the segmented mask to a base64-encoded string for returning in the response
    segmented_image_pil = Image.fromarray(segmented_image.astype(np.uint8))  # Convert to image
    buffer = io.BytesIO()
    segmented_image_pil.save(buffer, format="PNG")
    segmented_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Return the base64-encoded segmented image
    return jsonify({
        'segmented_image': segmented_image_base64
    })

if __name__ == '__main__':
    # Ensure Flask app listens on the port provided by Heroku
    port = int(os.environ.get("PORT", 5020))  # Default to 5020 if not provided
    app.run(host="0.0.0.0", port=port)  # Use 0.0.0.0 to allow connections from external sources
