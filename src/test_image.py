import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("model.h5")

# Image to test (put any traffic sign image path here)
IMAGE_PATH = "../dataset/test/00000.png"   # change if needed
IMG_SIZE = 32

# Read and preprocess image
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
class_id = np.argmax(prediction)

print("Predicted class ID:", class_id)