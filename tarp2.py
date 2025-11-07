import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Dataset directory
DATASET_DIR = "C:/Users/motak/OneDrive/Desktop/TARP/dataset"

# Function to load images
def load_images_from_folder(folder, label):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # ensure consistent size
            img = img / 255.0
            images.append((img, label))
    return images

# Load open and closed eye images
open_images = load_images_from_folder(os.path.join(DATASET_DIR, "open"), 0)
closed_images = load_images_from_folder(os.path.join(DATASET_DIR, "closed"), 1)

# Combine and prepare dataset
data = open_images + closed_images
X = np.array([i[0] for i in data]).reshape(-1, 128, 128, 1)
y = np.array([i[1] for i in data])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# Evaluate model and print accuracy
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

# Save the trained model
model.save("C:/Users/motak/OneDrive/Desktop/TARP/eye_state_cnn.h5")
