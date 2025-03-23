# Face Detection Model

## 📌 Project Overview
This project implements a **Face Detection Model** using **TensorFlow and Keras**. The model detects faces in images and predicts bounding box coordinates along with class labels.

## 🚀 Features
- Trains a deep learning model for face detection
- Uses **TensorFlow, Keras, OpenCV, and NumPy**
- Compatible with **Google Colab** (with a workaround for webcam input)
- Supports both **real-time and pre-captured image** face detection

## 📂 Folder Structure
```
├── dataset/                 # Folder containing training images
├── models/                  # Saved trained model
├── notebooks/               # Jupyter Notebooks
│   ├── Face_detection_model.ipynb   # Main training and testing notebook
├── utils/                   # Utility scripts (data processing, visualization)
├── README.md                # Project documentation
```

## 🛠 Installation
### **1️⃣ Install Dependencies**
Ensure you have Python installed, then install required libraries:
```bash
pip install tensorflow keras opencv-python numpy matplotlib
```

### **2️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/face-detection-model.git
cd face-detection-model
```

## 🎯 How to Use
### **1️⃣ Train the Model**
Run the Jupyter Notebook **Face_detection_model.ipynb** to train the model:
```bash
jupyter notebook
```

### **2️⃣ Capture Image for Prediction (Colab Only)**
Since Google Colab does not support direct webcam access, use the following workaround:
```python
from google.colab.output import eval_js
from IPython.display import Javascript

def capture_photo():
    display(Javascript('''
        async function takePhoto() {
            const video = document.createElement('video');
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            await video.play();
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            stream.getTracks().forEach(track => track.stop());
            return canvas.toDataURL('image/jpeg', 0.8);
        }
        takePhoto().then(data => google.colab.kernel.invokeFunction('notebook.capture_photo', [data], {}));
    '''))

capture_photo()
```

### **3️⃣ Run the Face Detection Model**
```python
import cv2
import numpy as np

# Load the image and resize to match model input
img = cv2.imread("photo.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized_img = cv2.resize(img, (224, 224))  # Adjust based on your model
resized_img = np.expand_dims(resized_img, axis=0)

# Run prediction
yhat = facetracker.predict(resized_img)
print("Model Output:", yhat)
```

## 🛠 Troubleshooting
- **ValueError: Input Shape Mismatch**
  - Ensure your image input size matches the model's expected input (e.g., `(224, 224, 3)`).
  - Resize images before passing them into the model.

- **Colab Webcam Not Working**
  - Use JavaScript workaround to capture images.

## 🤖 Technologies Used
- **TensorFlow & Keras**: Model training and prediction
- **OpenCV**: Image processing
- **NumPy & Matplotlib**: Data handling and visualization

## 📜 License
This project is licensed under the MIT License.

---
💡 **Contributions & Feedback**: Feel free to fork, contribute, or report issues!

