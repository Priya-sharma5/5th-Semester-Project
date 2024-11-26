# Autism Spectrum Detection System (ASDS)

## Overview
The Autism Spectrum Detection System (ASDS) is a Python-based machine learning project that uses **Convolutional Neural Networks (CNNs)** for detecting autism spectrum disorder (ASD) in individuals based on facial expressions. The system leverages real-time facial recognition techniques to analyze facial features and predict the likelihood of autism.

The model has been trained using a large image dataset and achieves a high accuracy rate of **96.25%** in detecting ASD. The project focuses on using image processing and deep learning techniques to provide a reliable tool for autism detection.

## Project Objective
The primary goal of this project is to:
- Detect Autism Spectrum Disorder (ASD) from facial images using machine learning.
- Utilize **CNNs** to extract key facial features associated with ASD.
- Implement an efficient **data preprocessing pipeline** to clean and prepare the dataset for training and validation.
- Achieve high accuracy to ensure reliable predictions.

## Features
- **Facial Image Analysis**: The model detects features like facial expressions, eye movements, and gestures.
- **High Accuracy**: The system has achieved an accuracy of 96.25% on the test dataset.
- **Data Preprocessing**: Includes image normalization, resizing, and augmentation techniques to enhance dataset quality.
- **Real-Time Testing**: Uses webcam input for real-time facial recognition and ASD prediction.

## Methodology
1. **Dataset**: The model is trained on a large dataset of facial images labeled with ASD status.
2. **CNN Architecture**: A Convolutional Neural Network is used to extract deep features from the facial images. The CNN is trained with multiple layers, including convolutional, pooling, and fully connected layers.
3. **Preprocessing**: The images are resized and normalized to ensure consistent input quality. Data augmentation techniques such as rotation and flipping are applied to increase model robustness.
4. **Evaluation**: The model's performance is evaluated using accuracy, precision, recall, and F1-score metrics.
5. **Prediction**: After training, the model can predict ASD status in real-time based on facial expressions captured through a webcam.

## Requirements
- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Pandas

## Installation
1. Clone this repository to your local machine:

```bash
git clone https://github.com/Priya-sharma5/Autism-Spectrum-Detection-System.git
