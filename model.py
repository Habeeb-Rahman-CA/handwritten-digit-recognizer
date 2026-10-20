"""
Neural Network Model Architecture for Handwritten Digit Recognition.

This module defines two models:
1. DigitMLP: A classic Multi-Layer Perceptron (fully connected neural network).
   Ideal for learning the core concepts of neurons, weights, biases, and layers.
2. DigitCNN: A simple Convolutional Neural Network.
   Shows how 2D spatial features (edges, curves) are learned in computer vision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitMLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for 28x28 grayscale handwritten digits.

    Architecture:
      - Input: 28 x 28 = 784 flattened pixel intensities (values between 0.0 and 1.0)
      - Hidden Layer 1: Linear(784, 128) -> ReLU -> Dropout(0.2)
      - Hidden Layer 2: Linear(128, 64)  -> ReLU -> Dropout(0.2)
      - Output Layer:   Linear(64, 10)   -> Raw logits for digits 0 through 9
    """

    def __init__(self, input_size: int = 28 * 28, hidden1: int = 128, hidden2: int = 64, num_classes: int = 10):
        super(DigitMLP, self).__init__()

        # Linear Layer 1: Connects all 784 inputs to 128 hidden neurons
        # Each neuron computes: z = sum(w_i * x_i) + b
        self.fc1 = nn.Linear(input_size, hidden1)

        # Dropout: randomly zeroes 20% of activations during training
        # to prevent neurons from co-adapting (reduces overfitting)
        self.dropout1 = nn.Dropout(0.2)

        # Linear Layer 2: Connects 128 hidden neurons to 64 second-layer neurons
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.dropout2 = nn.Dropout(0.2)

        # Output Layer: Connects 64 features to 10 class scores (digits 0-9)
        self.fc3 = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass: how an input tensor travels through neurons and layers.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784)
            
        Returns:
            Logits of shape (batch_size, 10)
        """
        # Step 1: Flatten the 2D image (28x28) into a 1D vector (784)
        x = x.view(x.size(0), -1)

        # Step 2: Pass through Hidden Layer 1 + ReLU activation
        # ReLU(z) = max(0, z) introduces non-linearity
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # Step 3: Pass through Hidden Layer 2 + ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # Step 4: Output Layer (returns raw logits; CrossEntropyLoss applies Softmax internally)
        logits = self.fc3(x)
        return logits


class DigitCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) for handwritten digits.
    Uses 2D convolution filters to detect spatial patterns (edges, loops, curves).
    """

    def __init__(self, num_classes: int = 10):
        super(DigitCNN, self).__init__()
        # Conv layer 1: 1 input channel (grayscale) -> 16 feature maps, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Conv layer 2: 16 channels -> 32 feature maps, 3x3 kernel
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)  # Halves spatial dimensions (28x28 -> 14x14 -> 7x7)
        self.dropout = nn.Dropout(0.25)

        # Fully connected classifier: 32 channels * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 16, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 32, 7, 7)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # Flatten -> (batch, 1568)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


def get_model(model_type: str = "mlp") -> nn.Module:
    """Factory helper to retrieve the requested model type."""
    if model_type.lower() == "cnn":
        return DigitCNN()
    return DigitMLP()

