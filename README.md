# Face Age Estimation Project

A deep learning project to estimate the age of a person from a facial image using Convolutional Neural Networks (CNN). This repository also includes a baseline Random Forest model for performance comparison.

## 📌 Overview

This project utilizes **TensorFlow/Keras** to build and train a CNN regression model. It preprocesses facial images, detects faces using **OpenCV**, and predicts the age as a continuous value.

### Key Features
* **CNN Model:** Custom Convolutional Neural Network for regression.
* **Face Detection:** Automatic face detection using OpenCV Haar Cascades.
* **Baseline Comparison:** A Random Forest Regressor (`train_baseline.py`) to benchmark the deep learning model.
* **Data Processing:** Automated label extraction from the IMDB-WIKI dataset filenames.

⚙️ Installation
Clone the repository:

git clone [https://github.com/YOUR_USERNAME/face-age-estimation.git](https://github.com/AlperenKuzucu/face-age-estimation.git)
cd face-age-estimation


Install dependencies:

pip install -r requirements.txt


🚀 Usage
1. Training the Model
To train the CNN model from scratch (requires the IMDB-WIKI dataset in data/raw/imdb_crop):

python src/train_cnn.py

This will save the trained model as face_age_cnn.h5 in the root directory.

To train the baseline Random Forest model:

python src/train_baseline.py



2. Prediction
To predict the age of a person in an image:


python src/predict_cnn.py "path/to/your/image.jpg"


📊 Dataset



https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/?utm_source=chatgpt.com
