# Traffic Sign Recognition System

A real-time Traffic Sign Recognition system built using **Deep Learning
(CNN)** and **Computer Vision**, inspired by ADAS traffic sign modules
used in modern vehicles.

------------------------------------------------------------------------

## 🔹 Features

-   Real-time traffic sign detection using webcam
-   CNN-based image classification (43 traffic sign classes)
-   Confidence-based prediction filtering
-   Temporal stability to reduce false alerts
-   Voice output for detected traffic signs
-   Designed as an ADAS-style prototype

------------------------------------------------------------------------

## 🔹 Tech Stack

-   Python 3.10
-   TensorFlow / Keras
-   OpenCV
-   NumPy
-   Windows Speech API (SAPI)

------------------------------------------------------------------------

## 🔹 Dataset

-   **GTSRB (German Traffic Sign Recognition Benchmark)**
-   Dataset is not included due to size constraints

------------------------------------------------------------------------

## 🔹 Project Structure

    src/
    ├── train_model.py
    ├── detect_sign.py
    ├── test_image.py
    ├── speak.py
    └── labels.py

------------------------------------------------------------------------

## 🔹 How to Run

### 1. Install dependencies

``` bash
pip install -r requirements.txt
```

### 2. Train the model

``` bash
python src/train_model.py
```

### 3. Run real-time detection

``` bash
python src/detect_sign.py
```

Press **Q** to exit.

------------------------------------------------------------------------

## 🔹 Notes

-   Model trained on clean images; real-time predictions use confidence
    filtering and stability logic
-   This project is a prototype and not a production-grade ADAS system

------------------------------------------------------------------------

## 🔹 Future Improvements

-   YOLO-based bounding box detection
-   Multiple sign detection
-   Raspberry Pi deployment
-   Accuracy benchmarking

------------------------------------------------------------------------

## 🔹 Author

Om
