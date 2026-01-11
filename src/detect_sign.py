import cv2
import numpy as np
from tensorflow.keras.models import load_model
from labels import labels
from speak import speak
import time


# ---------------- CONFIG ----------------
IMG_SIZE = 32
CONFIDENCE_THRESHOLD = 0.7

# Load trained model
model = load_model("model.h5")

# Open camera
cap = cv2.VideoCapture(0)

#Speech related variables
stable_sign = None
stable_count = 0
STABLE_FRAMES = 15   # adjust: higher = less repetition


while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --------- Crop center area ---------
    h, w, _ = frame.shape
    crop = frame[h//4:3*h//4, w//4:3*w//4]

    # --------- Preprocess ---------
    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # --------- Predict ---------
    prediction = model.predict(img, verbose=0)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    current_time = time.time()

    if confidence > CONFIDENCE_THRESHOLD:
        sign_name = labels.get(class_id, "Unknown")

        if sign_name != "Unknown":
            if sign_name == stable_sign:
                stable_count += 1
            else:
                stable_sign = sign_name
                stable_count = 1

            # Speak only when stable for enough frames
            if stable_count == STABLE_FRAMES:
                speak(sign_name)
    else:
        sign_name = "Detecting..."
        stable_sign = None
        stable_count = 0

    # --------- Display ---------
    cv2.rectangle(
        frame,
        (w//4, h//4),
        (3*w//4, 3*h//4),
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"{sign_name} ({confidence:.2f})",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Traffic Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()