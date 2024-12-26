import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
import os
from PIL import Image

def unet_model(input_size=(128, 128, 3), num_classes=4):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, 3, activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.3)(pool1)

    conv2 = Conv2D(64, 3, activation='relu', padding='same')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.3)(pool2)

    conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.3)(pool3)

    conv4 = Conv2D(256, 3, activation='relu', padding='same')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, 3, activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.3)(pool4)

    conv5 = Conv2D(512, 3, activation='relu', padding='same')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, 3, activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(256, 3, activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same')(conv8)
    conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same')(conv9)
    conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

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
X_train, Y_train = load_dataset(image_dir, mask_dir, num_classes=num_classes)

# Normalize images
X_train = X_train / 255.0

# Define the callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Define class weights
# class_weights = {0: 1.0, 1: 2.0, 2: 1.5, 3: 3.0}

# Train the model
model = unet_model(num_classes=num_classes)
model.fit(X_train, Y_train, epochs=80, batch_size=32, validation_split=0.1, callbacks=[reduce_lr, early_stopping])

# Save the model in the native Keras format
model.save('C:/Users/ASUS/Documents/Project_241/copi_code/trained_model.keras')