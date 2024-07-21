import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Function to preprocess images and create labels
def prepare_data(data_folder):
    data = []
    labels = []
    for person_folder in tqdm(os.listdir(data_folder)):
        person_path = os.path.join(data_folder, person_folder)
        if not os.path.isdir(person_path):
            continue
        
        label = person_folder  # Assuming folder name is the person's name
        
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            img = cv2.resize(img, (50, 50))  # Resize image to 50x50 pixels
            
            # Normalize pixel values to be between 0 and 1
            img = img / 255.0
            
            # Append image and its label to the dataset
            data.append(img)
            labels.append(label)
    
    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Convert labels to one-hot encoded vectors
    label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
    y = np.array([label_map[label] for label in labels])
    global num_classes
    num_classes = len(label_map)
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    
    return data, y, label_map

# Replace with your collected samples directory
data_folder = r'C:\Users\lenovo\Desktop\face recognition model\dataset'

# Prepare data
X, y, label_map = prepare_data(data_folder)

# Reshape X to (samples, height, width, channels) for Conv2D input
X = np.expand_dims(X, axis=-1)  # Add single channel dimension (grayscale)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shape information
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Save label mapping for future reference
np.save("label_map.npy", label_map)

# Define the number of classes
global num_classes

# Define the CNN architecture using Keras API
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    

    
    Flatten(),
    Dense(2048, activation='relu'),
    Dropout(0.8),
    Dense(1024, activation='relu'),
    Dropout(0.8),
    Dense(512, activation='relu'),
    Dropout(0.7),
    Dense(256, activation='relu'),
    Dropout(0.6),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.4),
    
    Dense(num_classes, activation='softmax')  # Output layer with num_classes neurons for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=52, validation_data=(X_test, y_test), verbose=1)

# Save the trained model
model.save("face_recognition_model_2.h5")

