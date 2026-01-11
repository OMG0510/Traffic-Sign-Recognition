import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------- CONFIG --------
DATA_DIR = "../dataset/train"
IMG_SIZE = 32
NUM_CLASSES = 43
EPOCHS = 10
BATCH_SIZE = 64

# -------- LOAD DATA --------
images = []
labels = []

print("Loading images...")

for class_id in range(NUM_CLASSES):
    class_path = os.path.join(DATA_DIR, str(class_id))
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        images.append(img)
        labels.append(class_id)

images = np.array(images) / 255.0
labels = to_categorical(labels, NUM_CLASSES)

print("Images loaded:", images.shape)

# -------- BUILD CNN MODEL --------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------- TRAIN --------
print("Training model...")
model.fit(
    images,
    labels,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2
)

# -------- SAVE MODEL --------
model.save("model.h5")
print("Model saved as model.h5")