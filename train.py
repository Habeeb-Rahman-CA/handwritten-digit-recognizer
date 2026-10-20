"""
Training script for Handwritten Digit Recognizer.

This script demonstrates the full supervised deep learning pipeline:
1. Load training and validation data (MNIST).
2. Instantiate the neural network (MLP or CNN).
3. Set up Loss Function (CrossEntropyLoss) and Optimizer (Adam).
4. Run Epochs:
   - Forward pass (predict logits from pixel inputs)
   - Loss calculation (measure error against true labels)
   - Backpropagation (calculate gradients using the chain rule)
   - Weight update (nudge weights to reduce error)
5. Evaluate test accuracy and save the best model weights.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import get_model


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Executes one epoch of training over the training dataset."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\n--- Epoch {epoch} Training ---")
    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        # 1. Zero out accumulated gradients from previous step
        optimizer.zero_grad()

        # 2. Forward pass: compute predictions (logits)
        outputs = model(images)

        # 3. Compute loss (CrossEntropyLoss includes softmax calculation)
        loss = criterion(outputs, labels)

        # 4. Backward pass: compute gradients (backpropagation)
        loss.backward()

        # 5. Optimizer step: update weights (w = w - lr * grad)
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(dataloader):
            batch_acc = 100.0 * correct / total
            avg_loss = running_loss / total
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] | "
                  f"Loss: {avg_loss:.4f} | Acc: {batch_acc:.2f}%")

    elapsed = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    print(f"Epoch {epoch} finished in {elapsed:.1f}s | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluates model performance on the unseen test dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disables gradient tracking to save memory and speed up inference
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / total
    test_acc = 100.0 * correct / total
    return test_loss, test_acc


def main():
    parser = argparse.ArgumentParser(description="Train a Handwritten Digit Recognizer")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"],
                        help="Model architecture: 'mlp' (Multi-Layer Perceptron) or 'cnn' (Convolutional)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--save-path", type=str, default="digit_model.pth", help="Path to save model weights")
    args = parser.parse_args()

    # Device selection: GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # Load MNIST Data
    print(f"[Data] Loading MNIST dataset (batch size: {args.batch_size})...")
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size, download=True)
    print(f"  Training samples: {len(train_loader.dataset):,}")
    print(f"  Test samples:     {len(test_loader.dataset):,}")

    # Instantiate Model
    print(f"[Model] Initializing '{args.model.upper()}'...")
    model = get_model(args.model).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_test_acc = 0.0

    print("\n" + "=" * 50)
    print(f"Starting Training for {args.epochs} Epochs...")
    print("=" * 50)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f"-> Test Evaluation: Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_type": args.model,
                "model_state_dict": model.state_dict(),
                "test_acc": test_acc
            }, args.save_path)
            print(f"   ★ Best model saved to {args.save_path} ({best_test_acc:.2f}%)")

    print("\n" + "=" * 50)
    print(f"Training Complete! Best Test Accuracy: {best_test_acc:.2f}%")
    print(f"Weights saved at: {args.save_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()

