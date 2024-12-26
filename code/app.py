# app.py
from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from pillow_heif import register_heif_opener
import logging
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Register HEIF opener with Pillow
register_heif_opener()

# Load the pre-trained model
model = tf.keras.models.load_model('C:/Users/ASUS/Documents/Project_241/copi_code/trained_model.keras')

# Load label names
label_file = 'C:/Users/ASUS/Documents/Project_241/copi_code/label_names.txt'
with open(label_file, 'r') as file:
    labels = file.read().splitlines()
num_classes = len(labels)

# Function to preprocess the image for the model
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))  # Resize to the input size of the model
    image = image / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)

# Function to postprocess the model output
def postprocess_output(output, original_image):
    output = np.argmax(output, axis=-1)
    output = np.squeeze(output)
    output = cv2.resize(output, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return output

# Function to overlay the segmentation mask on the original image
def overlay_segmentation_mask(image, mask):
    color_map = {
        0: [0, 0, 0],        # Background - Black
        1: [255, 0, 0],      # Potholes - Red
        2: [0, 255, 0],      # Cracks - Green
        3: [0, 0, 255]       # Flooding - Blue
    }
    color_mask = np.zeros_like(image)
    for class_id, color in color_map.items():
        color_mask[mask == class_id] = color
    overlayed_image = cv2.addWeighted(image, 0.7, color_mask, 0.3, 0)
    return overlayed_image

# Function to detect and classify damage using the model
def detect_and_classify_damage(image):
    preprocessed_image = preprocess_image(image)
    logging.debug(f'Preprocessed image shape: {preprocessed_image.shape}')
    output = model.predict(preprocessed_image)
    logging.debug(f'Model output shape: {output.shape}')
    segmented_image = postprocess_output(output, image)
    logging.debug(f'Segmented image shape: {segmented_image.shape}')
    overlayed_image = overlay_segmentation_mask(image, segmented_image)
    return overlayed_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logging.error('No file part in the request')
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error('No selected file')
        return jsonify({'error': 'No selected file'}), 400
    if file and (file.filename.endswith('.png') or file.filename.endswith('.jpg') or file.filename.endswith('.jpeg') or file.filename.endswith('.heic')):
        try:
            logging.debug(f'Processing file: {file.filename}')
            image = Image.open(file.stream)
            logging.debug(f'Image mode: {image.mode}, size: {image.size}')
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_image = detect_and_classify_damage(image)
            _, buffer = cv2.imencode('.png', processed_image)
            logging.debug('Image processed successfully')
            return send_file(BytesIO(buffer), mimetype='image/png')
        except Exception as e:
            logging.error(f'Error processing file: {e}')
            return jsonify({'error': 'Error processing file'}), 500
    else:
        logging.error('Invalid file format')
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)

# Ensure you have the required packages installed:
# pip install flask opencv-python-headless numpy pillow pillow-heif tensorflow