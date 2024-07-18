# Emotion Recognition Using Audio

This project aims to recognize emotions from audio recordings using machine learning and deep learning techniques. The dataset used for this project is the TESS Toronto emotional speech set.

## Introduction

Emotion recognition from audio is a significant area of research in the field of human-computer interaction. This project leverages the TESS Toronto emotional speech set to train a model capable of classifying emotions based on audio features.

## Dataset

The dataset used in this project is the [TESS Toronto emotional speech set](https://tspace.library.utoronto.ca/handle/1807/24487), which includes audio recordings of actors expressing various emotions.

## Model Architecture

We use a simple Artificial Neural Network (ANN) with the following architecture:

Input Layer: Dense layer with 100 neurons
Hidden Layer 1: Dense layer with 200 neurons, followed by a Dropout layer
Hidden Layer 2: Dense layer with 100 neurons, followed by a Dropout layer
Output Layer: Dense layer with the number of emotion classes and softmax activation

## Feature Extraction

The features are extracted using the `librosa` library. Specifically, we use Mel-frequency cepstral coefficients (MFCCs) to represent the audio signals.

"""python
def features_extract(file): 
    audio, sam = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sam, n_mfcc=40)
    mfcc_scaled = np.mean(mfccs_features.T, axis=0)
    
    return mfcc_scaled"""

## Model Architecture

We use a simple Artificial Neural Network (ANN) with the following architecture:

Input Layer: Dense layer with 100 neurons
Hidden Layer 1: Dense layer with 200 neurons, followed by a Dropout layer
Hidden Layer 2: Dense layer with 100 neurons, followed by a Dropout layer
Output Layer: Dense layer with the number of emotion classes and softmax activation



## Training

The model is trained using categorical cross-entropy loss and the Adam optimizer. We save the best model during training using a checkpoint.

## Results

The model achieved an accuracy of approximately 95% on the test set. 

