from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from pillow_heif import register_heif_opener
import logging
import tensorflow as tf
import tensorflow_hub as hub

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Register HEIF opener with Pillow
register_heif_opener()

# Load a pre-trained model for road damage detection
model = hub.load("https://tfhub.dev/tensorflow/efficientdet/d0/1")

# Function to detect and highlight damage using basic image processing
def detect_and_highlight_damage(image):
    try:
        logging.debug('Starting damage detection using basic image processing')
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logging.debug('Converted image to grayscale')
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        logging.debug('Applied Gaussian blur')
        
        # Use color thresholding to create a mask for the road surface
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 255])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        logging.debug('Created mask for road surface')
        
        # Apply the mask to the blurred grayscale image
        masked_gray = cv2.bitwise_and(blurred, blurred, mask=mask)
        logging.debug('Applied mask to grayscale image')
        
        # Perform edge detection
        edges = cv2.Canny(masked_gray, 50, 150)
        logging.debug('Performed edge detection')
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logging.debug(f'Found {len(contours)} contours')
        
        # Draw contours on the original image
        highlighted_image = image.copy()
        cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), 2)  # Draw contours in red
        logging.debug('Highlighted damaged areas')
        
        return highlighted_image
    except Exception as e:
        logging.error(f'Error in detect_and_highlight_damage: {e}')
        raise

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
            image = np.array(image)
            logging.debug(f'Image converted to numpy array with shape: {image.shape} and dtype: {image.dtype}')
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            logging.debug(f'Converted image shape: {image.shape}')
            processed_image = detect_and_highlight_damage(image)
            
            # Resize the processed image to make it larger
            scale_percent = 225  # percent of original size (150% * 1.5)
            width = int(processed_image.shape[1] * scale_percent / 100)
            height = int(processed_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized_image = cv2.resize(processed_image, dim, interpolation=cv2.INTER_LINEAR)
            logging.debug(f'Resized image to dimensions: {dim}')
            
            _, buffer = cv2.imencode('.png', resized_image)
            logging.debug('Image processed successfully')
            return send_file(BytesIO(buffer), mimetype='image/png')
        except Exception as e:
            logging.error(f'Error processing file: {e}')
            return jsonify({'error': f'Error processing file: {e}'}), 500
    else:
        logging.error('Invalid file format')
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(debug=True)

# Ensure you have the required packages installed:
# pip install flask opencv-python-headless numpy pillow pillow-heif tensorflow tensorflow-hub