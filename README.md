Breast Cancer Detection Web App
This is a Breast Cancer Detection web application developed using Flask and a pre-trained EfficientNet model for binary classification (Benign or Malignant). The app allows users to upload an image of a breast mass, and it predicts whether it is Benign or Malignant.

Features
Real-time image upload and prediction.

EfficientNet model for classification.

Flask backend to handle image preprocessing, prediction, and result display.

Prerequisites
Before you begin, make sure you have the following installed:
Python 3.6+
pip (Python package installer)

Required Libraries
This project uses the following libraries:

Flask: A lightweight WSGI web application framework.

TensorFlow: Open-source library for machine learning and deep learning.

TensorFlow Hub: Pre-trained model serving platform for TensorFlow.

NumPy: Essential package for scientific computing with Python.

Pillow: Image processing library used for loading and manipulating images.

Install the required libraries by running the following command:
pip install -r requirements.txt

How to Run the Project Locally
1. Clone this repository
First, clone this repository to your local machine using Git:

git clone https://github.com/Peddikavya/Breast-Cancer-Detection/


2. Install the required dependencies
If you haven’t installed the libraries yet, run:

pip install -r requirements.txt

3. Run the Flask app
Now, you can run the Flask application with the following command:

python app.py

4. Access the app
Once the Flask server starts, open your browser and go to:

http://127.0.0.1:5000/
You should see the homepage of your web application, where you can upload an image for classification.

How the Web App Works
Upload Image: The user uploads an image of a breast mass.

Model Prediction: The image is processed, and the EfficientNet model predicts whether the mass is Benign or Malignant.

Display Result: The result (Benign or Malignant) is shown on the web page.
