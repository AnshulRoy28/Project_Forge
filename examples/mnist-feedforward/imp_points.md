# Important Points for MNIST Digit Classification Model

This document summarizes the key architectural and implementation decisions for building a basic feedforward neural network for MNIST digit classification using PyTorch.

## 1. Architecture

*   **Model Type**: Feedforward Neural Network (FNN).
*   **Reasoning**: User explicitly requested a "very basic neural network" and "feed forward neural network" for simple digit classification.
*   **Input Layer**: Flattens the 28x28 pixel input images into a 784-feature vector.
*   **Hidden Layers**:
    *   **Number**: 1 hidden layer.
    *   **Neurons**: 48 neurons.
    *   **Activation**: ReLU (Rectified Linear Unit).
*   **Output Layer**:
    *   **Neurons**: 10 neurons (corresponding to digits 0-9).
    *   **Activation**: Softmax (implicitly handled by `CrossEntropyLoss` in PyTorch, which expects logits).

## 2. Dataset

*   **Source**: `torchvision.datasets.MNIST`.
*   **Name**: MNIST.
*   **Format**: Grayscale images (28x28 pixels) and integer labels (0-9).
*   **Preprocessing**:
    *   Convert images to PyTorch Tensors.
    *   Normalize pixel values: Standard normalization for MNIST (`mean=(0.1307,)`, `std=(0.3081,)`).
    *   Flatten images from `(1, 28, 28)` to `(784,)` for input to the feedforward network.

## 3. Training Strategy

*   **Optimizer**: Adam.
*   **Learning Rate**: 0.001 (default for Adam, good starting point).
*   **Learning Rate Schedule**: Fixed learning rate (no schedule specified).
*   **Batch Size**: 64.
*   **Epochs**: 10.
*   **Loss Function**: `torch.nn.CrossEntropyLoss`.
    *   **Reasoning**: Standard loss function for multi-class classification in PyTorch, combines `LogSoftmax` and Negative Log Likelihood Loss.
*   **Primary Metric**: Accuracy.
    *   **Reasoning**: User explicitly stated "accuracy is the thing that is important for me".

## 4. Input/Output

*   **Model Input**:
    *   **Shape**: `(batch_size, 784)` after flattening.
    *   **Data Type**: `torch.float32`.
    *   **Normalization**: Yes, standard MNIST normalization applied to pixel values.
*   **Model Output (Logits)**:
    *   **Shape**: `(batch_size, 10)`.
    *   **Data Type**: `torch.float32`.
*   **Target Labels**:
    *   **Shape**: `(batch_size,)`.
    *   **Data Type**: `torch.long`.

## 5. Compute

*   **Device Preference**: GPU if available, otherwise CPU.
*   **Memory Constraints**: Assumed standard, no specific constraints mentioned.
*   **Mixed Precision**: Not used (full precision `float32`).

## 6. Export

*   **Target Format**: None specified. The project goal is "just want to train a model," implying no immediate export requirement.

## 7. Key Constraints

*   **Model Simplicity**: Must be a "very basic neural network" with only one hidden layer.
*   **Hidden Layer Configuration**: Exactly 48 neurons with ReLU activation in the single hidden layer.
*   **Primary Evaluation**: Focus solely on classification accuracy.

## 8. Checkpointing

*   **Frequency**: Save the model's state dictionary at the end of each epoch, or specifically when validation accuracy improves.
*   **What to Save**:
    *   Model's `state_dict()`.
    *   Optimizer's `state_dict()`.
    *   Current epoch number.
    *   Current training and validation loss/accuracy.