import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# Load the pre-trained model
model = tf.keras.models.load_model('C:/Users/ASUS/Documents/Project_241/copi_code/trained_model.keras')

def load_label_names(label_file):
    with open(label_file, 'r') as file:
        labels = file.read().splitlines()
    return labels

def load_dataset(image_dir, mask_dir, image_size=(128, 128), num_classes=4):
    images = []
    masks = []
    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            image_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, filename)  # Assuming masks have the same filename as images
            image = Image.open(image_path).resize(image_size)
            mask = Image.open(mask_path).resize(image_size)
            images.append(np.array(image))
            mask_array = np.array(mask)
            mask_one_hot = np.zeros((*mask_array.shape, num_classes))
            for c in range(num_classes):
                mask_one_hot[:, :, c] = (mask_array == c).astype(np.float32)
            masks.append(mask_one_hot)
    return np.array(images), np.array(masks)

# Load label names
label_file = 'C:/Users/ASUS/Documents/Project_241/copi_code/dataset/label_names.txt'
labels = load_label_names(label_file)
num_classes = len(labels)

# Load your dataset
image_dir = 'C:/Users/ASUS/Documents/Project_241/copi_code/dataset/images'
mask_dir = 'C:/Users/ASUS/Documents/Project_241/copi_code/dataset/masks'
X_val, Y_val = load_dataset(image_dir, mask_dir, image_size=(128, 128), num_classes=num_classes)

# Normalize images
X_val = X_val / 255.0

# Evaluate the model
loss, accuracy = model.evaluate(X_val, Y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')
