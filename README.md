# 🖋️ Handwritten Digit Recognizer

A beginner-friendly, hands-on deep learning project built with **PyTorch** to classify handwritten digits (0–9) using the classic **MNIST** dataset.

Includes a terminal visualizer (ASCII art) and an interactive browser-based drawing canvas.

---

## 📚 Core Concepts to Learn

### 1. What is a Neural Network? (The Basic Idea)
A neural network is a mathematical function inspired by the human brain. It takes input numbers (in our case, the pixel intensities of a digit), passes them through a series of interconnected computational nodes (**neurons**), and produces output numbers (the probability that the image represents a `0`, `1`, `2`, ..., or `9`).

```
[28x28 Image (784 Pixels)] 
           │
           ▼
   [ Input Layer ]       784 values (brightness from 0.0 to 1.0)
           │
           ▼
   [ Hidden Layer 1 ]    128 Neurons (Detects strokes, edges, loops)
           │
           ▼
   [ Hidden Layer 2 ]    64 Neurons (Combines strokes into digit shapes)
           │
           ▼
   [ Output Layer ]      10 Neurons (Scores for digits 0 through 9)
           │
           ▼
  [ Softmax Function ]   Converts scores into % probabilities (sum = 100%)
```

---

### 2. What is a Neuron?
An artificial neuron is the fundamental building block. Every neuron performs two simple math operations:

1. **Weighted Sum ($z$):**
   $$z = \sum_{i=1}^{n} (w_i \cdot x_i) + b = (w_1 x_1 + w_2 x_2 + \dots + w_n x_n) + b$$
   - **$x_i$ (Inputs):** Incoming signals (e.g., pixel brightness values).
   - **$w_i$ (Weights):** Importance or strength of each input connection. Can be positive (stimulatory) or negative (inhibitory).
   - **$b$ (Bias):** An adjustable threshold offset that shifts activation up or down.

2. **Activation Function ($\sigma$ or $\text{ReLU}$):**
   $$a = f(z)$$
   Without an activation function, stacking multiple linear layers would collapse into a single giant linear equation ($W_2(W_1 x) = W_{combined} x$). Non-linear activation functions allow neural networks to learn curves, corners, and complex non-linear patterns.

#### The Activation Functions Used Here:
- **ReLU (Rectified Linear Unit):**
  $$\text{ReLU}(z) = \max(0, z)$$
  If $z > 0$, pass it through; if $z \le 0$, output 0. It is fast, efficient, and prevents vanishing gradients.
- **Softmax (Output Layer):**
  $$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=0}^{9} e^{z_j}}$$
  Turns raw model outputs (logits) into a probability distribution where all 10 probabilities add up to 1.0 (100%).

---

### 3. Layers Explained
- **Input Layer (784 nodes):** Unrolls a 2D image of $28 \times 28$ pixels into a 1D vector of 784 numbers.
- **Hidden Layer 1 (128 neurons):** Each neuron looks at all 784 pixels with its own set of 784 weights. Early neurons learn basic stroke features (horizontal bar, vertical line, curve).
- **Hidden Layer 2 (64 neurons):** Looks at the 128 activations from Layer 1. It combines strokes into compound shapes (e.g. top circle + bottom circle = 8; top curve + vertical stem = 7).
- **Output Layer (10 neurons):** Each neuron corresponds to one digit class `0` through `9`. The neuron with the highest value is the model's prediction!

---

### 4. How the Network Learns (Training Loop)
1. **Forward Pass:** Input data flows forward through layers to compute predictions.
2. **Loss Calculation (Error Measurement):**
   - **CrossEntropyLoss** measures how far off the predicted probabilities are from the real target label.
3. **Backpropagation (Backward Pass):**
   - Using the chain rule from calculus, PyTorch calculates the gradient $\frac{\partial \text{Loss}}{\partial w}$ for every single weight in the network. This tells us whether increasing or decreasing each weight will reduce error.
4. **Optimization (Gradient Descent / Adam):**
   - Weights are updated by taking a small step in the opposite direction of the gradient:
     $$w \leftarrow w - \eta \cdot \nabla_w \text{Loss}$$
   - With each training epoch, error decreases and accuracy climbs!

---

## 🚀 Quickstart Guide

### 1. Environment Setup
The project already includes a virtual environment `.venv` with `PyTorch` and `torchvision` ready to go. To activate it in your terminal:

```bash
source .venv/bin/activate
```

---

### 2. Train the Model
Train the neural network on the 60,000 MNIST training images:

```bash
python train.py --model mlp --epochs 3
```

You will see:
- Epoch loss decreasing
- Test accuracy reaching **97.0%+** in just 3 epochs (~15 seconds on CPU!)
- Saved checkpoint: `digit_model.pth`

*(Optional)* You can also train a Convolutional Neural Network (CNN):
```bash
python train.py --model cnn --epochs 3
```

---

### 3. Test Predictions in Terminal (With ASCII Art)
Test the trained model on test samples from MNIST. The digit will be rendered directly in your terminal using ASCII characters!

```bash
# Test random sample
python predict.py

# Test a specific sample index (e.g. sample #42 or #0)
python predict.py --sample 42
```

Sample output:
```
+----------------------------+
|        ░█▒      ░▓▒        |
|        ▓▓░      ▒▓▓        |
|        ▓▓     ▒▓▓▓░        |
|       ▒▓▓    ░▓▓▓▓         |
|        ▒▓▒░▓▓▓▓▓▓░         |
|        ░▒▓▓▓▓▒▓▓▓          |
|              ▓▓▓▒          |
|              ▒▓▓           |
+----------------------------+

Digit Probabilities:
  Digit 4:  99.8% | ████████████████████████
  Digit 9:   0.2% | 

True Label:      4
Model Predicted: 4 (99.78% confidence)
Result:          ✔ CORRECT
```

---

### 4. Interactive Drawing Canvas (Browser Demo)
Draw your own digits and watch the neural network recognize them in real-time!

```bash
python web_app.py 5000
```
Open **`http://localhost:5000`** in your browser:
- ✏️ Draw any digit with your mouse or touchscreen.
- 📊 Real-time confidence bar chart for numbers 0–9.
- 🔬 Live neuron diagnostics showing how many neurons activate in Hidden Layer 1 and Hidden Layer 2!

---

### 5. Predict on Any Image File
Have a picture of a handwritten digit? You can run:

```bash
python predict.py --image path/to/my_digit.png
```

---

## 📁 Project Structure

```
handwritten-digit-recognizer/
├── model.py            # Neural network architectures (DigitMLP & DigitCNN)
├── dataset.py          # MNIST data loader, preprocessing & terminal ASCII renderer
├── train.py            # Training loop with forward, backward, and evaluation steps
├── predict.py          # Inference script (evaluates samples or custom images)
├── web_app.py          # Interactive drawing canvas & live neuron visualizer
├── digit_model.pth     # Trained model checkpoint (97% accuracy)
├── data/               # Downloaded MNIST dataset
└── README.md           # This comprehensive guide
```

