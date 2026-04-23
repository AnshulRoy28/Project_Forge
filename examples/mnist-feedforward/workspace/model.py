import torch
import torch.nn as nn
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

class FeedForwardNN(nn.Module):
    """
    A basic Feedforward Neural Network for MNIST digit classification.

    Architecture:
    - Input Layer: Flattens 28x28 images to a 784-feature vector.
    - Hidden Layer: One hidden layer with a specified number of neurons and ReLU activation.
    - Output Layer: 10 neurons for digits 0-9, outputting logits.
    """

    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        """
        Initializes the FeedForwardNN model.

        Args:
            input_size (int): The size of the input features (e.g., 784 for 28x28 flattened images).
            hidden_size (int): The number of neurons in the single hidden layer.
            num_classes (int): The number of output classes (e.g., 10 for MNIST digits 0-9).
        """
        super(FeedForwardNN, self).__init__()
        
        if not all(isinstance(arg, int) and arg > 0 for arg in [input_size, hidden_size, num_classes]):
            logger.error(f"Invalid model initialization parameters: input_size={input_size}, hidden_size={hidden_size}, num_classes={num_classes}. All must be positive integers.")
            raise ValueError("Model initialization parameters must be positive integers.")

        try:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size, num_classes)
            logger.info(f"Model initialized: Input={input_size}, Hidden={hidden_size}, Output={num_classes}")
        except Exception as e:
            logger.error(f"Failed to initialize model layers: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create neural network layers. Fix: Ensure input_size, hidden_size, and num_classes are valid for nn.Linear. Error: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor, expected to be of shape (batch_size, input_size).

        Returns:
            torch.Tensor: The output logits tensor of shape (batch_size, num_classes).
        """
        if x.dim() > 2:
            logger.debug(f"Flattening input tensor of shape {x.shape} for forward pass.")
            # Flatten the input if it's not already (batch_size, input_size)
            # e.g., from (batch_size, 1, 28, 28) to (batch_size, 784)
            x = x.view(x.size(0), -1)

        try:
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            return out
        except Exception as e:
            logger.error(f"Error during model forward pass: {e}", exc_info=True)
            raise RuntimeError(f"Model forward pass failed. Fix: Check input tensor shape and model layer compatibility. Error: {e}")