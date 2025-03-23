# Face Detection Model

This repository contains a deep learning-based face detection model implemented using TensorFlow and Keras. The model is trained on a dataset to detect faces and predict their bounding boxes.

## Features

- Uses a convolutional neural network (CNN) for face detection.
- Detects faces in images and returns bounding box coordinates.
- Implements real-time face detection (if running on a local system).
- Supports training and inference on **Google Colab** and **Jupyter Notebook**.

## Installation

### 1. Clone this repository:

```bash
git clone https://github.com/your-username/face-detection-model.git
cd face-detection-model
```

### 2. Install Dependencies

After navigating into the project directory, install the required dependencies using:

```bash
pip install -r requirements.txt
```

This ensures all necessary Python libraries (TensorFlow, Keras, OpenCV, etc.) are installed.

## Running the Model

### A. Training the Model

To train the model, run the following script:

```bash
python train.py
```

- Ensure your dataset is correctly formatted and available in the appropriate directory.
- Modify hyperparameters in `config.py` if needed.

### B. Running Inference

For testing on an image, use:

```bash
python detect_faces.py --image sample.jpg
```

For real-time webcam detection (if running locally):

```bash
python detect_faces.py --webcam
```

*(Note: Webcam access is not supported on Google Colab.)*

## Running on Google Colab

If you are running this project on **Google Colab**, upload your dataset and execute the notebook:

```bash
Face_detection_model.ipynb
```

Make sure to **mount Google Drive** if accessing datasets from there:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Troubleshooting

- **Shape Mismatch Error:** Ensure the dataset images are resized to `(224, 224, 3)`.
- **Missing Dataset:** Upload your dataset to the correct location before training.
- **Webcam Issues:** Use **local execution** instead of Colab for real-time webcam detection.

## Contributors

- **Your Name** – [GitHub Profile](https://github.com/your-username)

## License

This project is licensed under the **MIT License**.


