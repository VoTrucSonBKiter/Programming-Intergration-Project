import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('C:/Users/ASUS/Documents/Project_241/copi_code/trained_model.keras')

# Function to preprocess the image for the model
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))  # Resize to the input size of the model
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

# Load a test image
test_image_path = 'C:/Users/ASUS/Documents/Project_241/copi_code/dataset/images/021.png'
test_image = Image.open(test_image_path)
test_image = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR)

# Preprocess the test image
preprocessed_image = preprocess_image(test_image)

# Run the model on the test image
output = model.predict(preprocessed_image)

# Postprocess the output
segmented_image = postprocess_output(output, test_image)

# Overlay the segmentation mask on the original image
overlayed_image = overlay_segmentation_mask(test_image, segmented_image)

# Visualize the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 2)
plt.title('Segmented Mask')
plt.imshow(segmented_image, cmap='jet')

plt.subplot(1, 3, 3)
plt.title('Overlayed Image')
plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))

plt.show()
