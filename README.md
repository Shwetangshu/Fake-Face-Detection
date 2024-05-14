# Fake-Face-Detection
Real vs Fake Face Detection using Deep Learning

This repository contains a deep learning project for detecting real and fake faces using PyTorch and torchvision. The project includes data loading, augmentation, model definition (ResNet9 and a pretrained VGG16), and training/validation/testing routines.
Requirements

    Python 3.x
    PyTorch
    torchvision
    numpy
    pandas
    scikit-learn
    plotly

Setup

    Device Configuration: The code automatically detects and uses a GPU if available.

    Directory Paths: Define the paths for training, validation, and testing datasets.

    Data Augmentation: Apply various transformations like random horizontal flips, rotations, and perspective changes to the training dataset.

    Data Loading: Load the datasets using torchvision.datasets.ImageFolder and create DataLoaders for training, validation, and testing.

Model Architecture
ResNet9

The custom ResNet9 architecture is defined with multiple convolutional and residual blocks, followed by a fully connected layer for classification.
Pretrained VGG16

The pretrained VGG16 model from torchvision is used, with the final classifier layer modified to fit the binary classification task (real vs fake faces).
Training and Evaluation
Helper Class

A helper class manages the training and validation steps, including computing loss and updating model parameters.
Training Loop

The training loop handles:

    Training over multiple epochs.
    Validation at the end of each epoch.
    Saving the model state.

Evaluation Metrics

The model's performance is evaluated using:

    Accuracy
    F1 Score
    Precision
    Recall
    ROC AUC

Usage

    Train the Model: Call the model_trainer function with appropriate parameters to train the ResNet9 or VGG16 model.
    Evaluate the Model: Use the model_tester function to evaluate the trained model on the test dataset.
    Save/Load Model: Save the trained model's state dictionary for future use.
