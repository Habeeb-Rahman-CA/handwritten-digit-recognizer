"""
Inference script for Handwritten Digit Recognizer.

Features:
1. Predict random or specific samples from the MNIST test dataset.
2. Render the 28x28 digit directly in your terminal using ASCII art!
3. Predict on custom external image files (PNG/JPG).
4. Display a probability bar chart for all 10 digits (0-9).
"""

import argparse
import os
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from dataset import get_data_loaders, ascii_render, MNIST_MEAN, MNIST_STD
from model import get_model


def load_trained_model(weights_path: str = "digit_model.pth", device: torch.device = None):
    """Loads saved model checkpoint and weights."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file '{weights_path}' not found! Please run `python train.py` first."
        )

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model_type = checkpoint.get("model_type", "mlp")
    model = get_model(model_type).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"[Loaded] Model architecture: {model_type.upper()} (Checkpoint Test Acc: {checkpoint.get('test_acc', 0.0):.2f}%)")
    return model


def print_confidence_bar(probabilities: torch.Tensor):
    """Prints a terminal bar chart of predicted probabilities for digits 0 to 9."""
    print("\nDigit Probabilities:")
    for digit, prob in enumerate(probabilities):
        pct = prob.item() * 100
        bar = "█" * int(pct / 4)
        print(f"  Digit {digit}: {pct:5.1f}% | {bar}")


def predict_sample(model, test_dataset, index: int, device):
    """Predicts a specific index from the MNIST test dataset."""
    image, true_label = test_dataset[index]
    image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension -> (1, 1, 28, 28)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        confidence, predicted_digit = torch.max(probs, dim=0)

    print("\n" + "=" * 40)
    print(f"Sample #{index} from MNIST Test Set")
    print("=" * 40)
    print(ascii_render(image))
    print_confidence_bar(probs)
    print("=" * 40)
    print(f"True Label:      {true_label}")
    print(f"Model Predicted: {predicted_digit.item()} ({confidence.item() * 100:.2f}% confidence)")
    is_correct = "✔ CORRECT" if predicted_digit.item() == true_label else "✘ WRONG"
    print(f"Result:          {is_correct}")
    print("=" * 40)


def predict_custom_image(model, image_path: str, device):
    """
    Preprocesses a custom user image (e.g. drawn in Paint or on paper):
    1. Converts to grayscale ('L')
    2. Inverts colors if background is white and stroke is dark (MNIST uses white stroke on black background)
    3. Resizes to 28x28
    4. Normalizes with MNIST mean & standard deviation
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Load and convert to grayscale
    img = Image.open(image_path).convert("L")

    # If corner pixels are bright, background is white -> invert
    corners = [img.getpixel((0, 0)), img.getpixel((img.width - 1, 0)),
               img.getpixel((0, img.height - 1)), img.getpixel((img.width - 1, img.height - 1))]
    if sum(corners) / len(corners) > 128:
        img = ImageOps.invert(img)

    # Resize to 28x28
    img = img.resize((28, 28), Image.Resampling.BILINEAR)

    # Convert to Tensor and normalize
    import torchvision.transforms.functional as TF
    tensor = TF.to_tensor(img)
    normalized = TF.normalize(tensor, [MNIST_MEAN], [MNIST_STD]).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(normalized)
        probs = F.softmax(logits, dim=1).squeeze(0)
        confidence, predicted_digit = torch.max(probs, dim=0)

    print("\n" + "=" * 40)
    print(f"Predicting on Custom Image: {image_path}")
    print("=" * 40)
    print(ascii_render(tensor))
    print_confidence_bar(probs)
    print("=" * 40)
    print(f"Model Predicted: {predicted_digit.item()} ({confidence.item() * 100:.2f}% confidence)")
    print("=" * 40)


def main():
    parser = argparse.ArgumentParser(description="Predict handwritten digits using trained PyTorch model")
    parser.add_argument("--weights", type=str, default="digit_model.pth", help="Path to model weights file")
    parser.add_argument("--sample", type=int, default=None, help="Index of MNIST test sample (0 to 9999). Default is random.")
    parser.add_argument("--image", type=str, default=None, help="Path to custom image file to classify")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(args.weights, device)

    if args.image:
        predict_custom_image(model, args.image, device)
    else:
        _, test_loader = get_data_loaders(batch_size=1, download=True)
        test_dataset = test_loader.dataset
        import random
        idx = args.sample if args.sample is not None else random.randint(0, len(test_dataset) - 1)
        predict_sample(model, test_dataset, idx, device)


if __name__ == "__main__":
    main()

