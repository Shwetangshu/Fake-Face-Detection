# Fake Face Detection using Deep Learning

This repository contains a deep learning project for detecting real and fake faces using PyTorch and torchvision. The project includes data loading, augmentation, model definition (ResNet9 and a pretrained VGG16), and training/validation/testing routines.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- plotly

## Setup

- **Device Configuration**: The code automatically detects and uses a GPU if available.
- **Directory Paths**: Define the paths for training, validation, and testing datasets.
- **Data Augmentation**: Apply various transformations like random horizontal flips, rotations, and perspective changes to the training dataset.
- **Data Loading**: Load the datasets using `torchvision.datasets.ImageFolder` and create DataLoaders for training, validation, and testing.

## Model Architecture

- **ResNet9**: 
  - Custom ResNet9 architecture with multiple convolutional and residual blocks.
  - Includes a fully connected layer for classification.

- **Pretrained VGG16**: 
  - Uses the pretrained VGG16 model from torchvision.
  - Modifies the final classifier layer for binary classification (real vs fake faces).

## Training and Evaluation

- **Helper Class**: Manages training and validation steps, including computing loss and updating model parameters.
- **Training Loop**:
  - Trains over multiple epochs.
  - Validates at the end of each epoch.
  - Saves the model state.
- **Evaluation Metrics**: 
  - Accuracy
  - F1 Score
  - Precision
  - Recall
  - ROC AUC

## Usage

1. **Train the Model**: 
   - Call the `model_trainer` function with appropriate parameters to train the ResNet9 or VGG16 model.
2. **Evaluate the Model**: 
   - Use the `model_tester` function to evaluate the trained model on the test dataset.
3. **Save/Load Model**: 
   - Save the trained model's state dictionary for future use.

## License

This project is licensed under the MIT License.
