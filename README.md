# Captcha-OCR-CNN-RNN

This project implements an Optical Character Recognition (OCR) model from scratch using Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and Connectionist Temporal Classification (CTC) loss. The model is designed to read captchas and has achieved impressive accuracy metrics.

## Features

- **High Accuracy**: 
  - Train Accuracy: 99.68%
  - Validation Accuracy: 99.04%
- **Image Processing**: Handles image loading, preprocessing, and augmentation.
- **Deep Learning**: Combines CNNs for feature extraction and RNNs for sequence prediction.
- **Custom Loss Function**: Uses CTC loss to handle the alignment between predicted and true sequences.
- **Prediction Analysis**: Displays incorrect predictions for both training and validation datasets.

## Project Structure

- `CAPTHA_OCR.py`: Main script containing the implementation of the OCR model.
- `captcha_images/`: Folder containing captcha images.


### Code Explanation

1. **Data Loading and Preprocessing**:
    - Loads images from the `captcha_images` folder.
    - Converts images to grayscale and resizes them.
    - Maps characters to integers for model training.

2. **Model Architecture**:
    - Combines CNNs for feature extraction with RNNs (Bidirectional LSTM) for sequence prediction.
    - Uses a custom CTC loss layer to compute the training loss.

3. **Training**:
    - Trains the model with early stopping based on validation loss.
    - Prints the accuracy for both training and validation datasets.
    - Displays incorrect predictions for analysis.

4. **Inference**:
    - Uses the trained model to make predictions on validation data.
    - Decodes the predictions to human-readable text.

### Results

- **Train Accuracy**: 99.68%
- **Validation Accuracy**: 99.04%

