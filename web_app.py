"""
Interactive Web Application for Handwritten Digit Recognition.
Zero external web frameworks required (uses Python standard library http.server).

Features:
- HTML5 Canvas where you can draw any digit with mouse or touchscreen.
- Real-time recognition via POST /predict.
- Interactive visualization of active neurons across layers.
- Preset MNIST test samples button to test known digits.
"""

import base64
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import io
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import numpy as np

from dataset import MNIST_MEAN, MNIST_STD
from model import get_model


# Load model weights globally
DEVICE = torch.device("cpu")
CHECKPOINT_PATH = "digit_model.pth"

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError("Please run `python train.py` first to generate digit_model.pth!")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
model_type = checkpoint.get("model_type", "mlp")
model = get_model(model_type).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def preprocess_canvas_image(image_data_uri: str) -> torch.Tensor:
    """Decodes data URL, extracts grayscale, centers digit, scales to 28x28."""
    # Strip base64 header
    header, encoded = image_data_uri.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # The canvas is drawn with white stroke on black background
    # Find bounding box of drawing to center digit (MNIST standard preprocessing)
    np_img = np.array(pil_img)
    non_empty_coords = np.argwhere(np_img > 20)

    if non_empty_coords.size > 0:
        ymin, xmin = non_empty_coords.min(axis=0)
        ymax, xmax = non_empty_coords.max(axis=0)
        # Crop to digit bounding box
        cropped = pil_img.crop((xmin, ymin, xmax + 1, ymax + 1))
        # Resize maintaining aspect ratio to fit inside 20x20 box
        w, h = cropped.size
        if w > h:
            new_w = 20
            new_h = max(1, int(h * (20.0 / w)))
        else:
            new_h = 20
            new_w = max(1, int(w * (20.0 / h)))
        resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Paste onto centered 28x28 black canvas
        final_img = Image.new("L", (28, 28), 0)
        paste_x = (28 - new_w) // 2
        paste_y = (28 - new_h) // 2
        final_img.paste(resized, (paste_x, paste_y))
    else:
        final_img = pil_img.resize((28, 28), Image.Resampling.BILINEAR)

    # Convert to Tensor & Normalize
    arr = np.array(final_img, dtype=np.float32) / 255.0
    normalized = (arr - MNIST_MEAN) / MNIST_STD
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 28, 28)
    return tensor, arr.tolist()


def predict_and_explain(tensor: torch.Tensor):
    """Computes predictions and returns layer activations for educational inspection."""
    with torch.no_grad():
        if model_type == "mlp":
            flat = tensor.view(1, -1)
            h1 = F.relu(model.fc1(flat))
            h2 = F.relu(model.fc2(h1))
            logits = model.fc3(h2)

            probs = F.softmax(logits, dim=1).squeeze(0).tolist()
            predicted = int(torch.argmax(logits, dim=1).item())

            return {
                "prediction": predicted,
                "confidence": max(probs),
                "probabilities": probs,
                "layer_stats": {
                    "input_active_pixels": int((flat > 0).sum().item()),
                    "h1_active_neurons": int((h1 > 0).sum().item()),
                    "h1_total_neurons": 128,
                    "h2_active_neurons": int((h2 > 0).sum().item()),
                    "h2_total_neurons": 64,
                }
            }
        else:
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).tolist()
            return {
                "prediction": int(torch.argmax(logits, dim=1).item()),
                "confidence": max(probs),
                "probabilities": probs,
                "layer_stats": {}
            }


HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Handwritten Digit Recognizer - Neural Network Demo</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --border: #334155;
    --primary: #38bdf8;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #22c55e;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    padding: 24px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }
  .container {
    max-width: 900px;
    width: 100%;
  }
  header {
    text-align: center;
    margin-bottom: 24px;
  }
  h1 { font-size: 2rem; color: var(--primary); margin-bottom: 8px; }
  p.subtitle { color: var(--muted); font-size: 1rem; }
  .grid {
    display: grid;
    grid-template-columns: 320px 1fr;
    gap: 24px;
  }
  @media (max-width: 720px) {
    .grid { grid-template-columns: 1fr; }
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .card-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 14px;
    color: #e2e8f0;
  }
  #canvas {
    background: #000;
    border-radius: 8px;
    cursor: crosshair;
    display: block;
    touch-action: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
  }
  .button-row {
    display: flex;
    gap: 10px;
    margin-top: 14px;
  }
  button {
    flex: 1;
    padding: 10px 16px;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
  }
  .btn-primary { background: #0284c7; color: white; }
  .btn-primary:hover { background: #0369a1; }
  .btn-secondary { background: #334155; color: #cbd5e1; }
  .btn-secondary:hover { background: #475569; }
  .prediction-hero {
    text-align: center;
    padding: 16px;
    background: #090d16;
    border-radius: 10px;
    margin-bottom: 18px;
    border: 1px solid #1e293b;
  }
  .prediction-number {
    font-size: 4rem;
    font-weight: 800;
    color: var(--accent);
    line-height: 1;
  }
  .prediction-label {
    color: var(--muted);
    font-size: 0.9rem;
    margin-top: 4px;
  }
  .bars { display: flex; flex-direction: column; gap: 8px; }
  .bar-row { display: flex; align-items: center; gap: 10px; font-size: 0.9rem; }
  .bar-label { width: 24px; font-weight: 700; color: #cbd5e1; }
  .bar-track {
    flex: 1;
    height: 14px;
    background: #0f172a;
    border-radius: 7px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    width: 0%;
    background: var(--primary);
    transition: width 0.2s ease-out;
  }
  .bar-fill.top { background: var(--accent); }
  .bar-pct { width: 45px; text-align: right; font-size: 0.8rem; color: var(--muted); }
  .explanation-box {
    margin-top: 24px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .stats-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 10px;
  }
  .badge {
    background: #0f172a;
    border: 1px solid var(--border);
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.85rem;
  }
  .badge strong { color: var(--primary); }
</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Handwritten Digit Recognizer</h1>
    <p class="subtitle">Draw any digit from 0 to 9 in the box below</p>
  </header>

  <div class="grid">
    <div class="card">
      <div class="card-title">Draw Digit Here</div>
      <canvas id="canvas" width="280" height="280"></canvas>
      <div class="button-row">
        <button class="btn-secondary" onclick="clearCanvas()">Clear</button>
        <button class="btn-primary" onclick="predict()">Recognize</button>
      </div>
      <p style="font-size: 0.8rem; color: var(--muted); margin-top: 12px; text-align: center;">
        Tip: Draw with thick, centered strokes.
      </p>
    </div>

    <div class="card">
      <div class="card-title">Model Predictions</div>
      <div class="prediction-hero">
        <div class="prediction-number" id="predNumber">-</div>
        <div class="prediction-label" id="predConfidence">Draw a digit and click Recognize</div>
      </div>

      <div class="bars" id="barsContainer">
        <!-- Bars generated dynamically -->
      </div>
    </div>
  </div>

  <div class="explanation-box">
    <div class="card-title">Neural Network Live Diagnostics</div>
    <p style="color: var(--muted); font-size: 0.9rem;">
      When you draw a digit, it is converted into a 28x28 matrix (784 numbers). Watch how many neurons activate in each layer:
    </p>
    <div class="stats-badges">
      <div class="badge">Input Pixels: <strong id="statInputs">0</strong> / 784</div>
      <div class="badge">Hidden Layer 1 (ReLU): <strong id="statH1">0</strong> / 128 active</div>
      <div class="badge">Hidden Layer 2 (ReLU): <strong id="statH2">0</strong> / 64 active</div>
      <div class="badge">Output Layer: <strong>10</strong> classes (0-9)</div>
    </div>
  </div>
</div>

<script>
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  let isDrawing = false;
  let hasDrawn = false;

  // Initialize black background
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#ffffff';

  // Build probability bar rows
  const barsContainer = document.getElementById('barsContainer');
  for (let i = 0; i <= 9; i++) {
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-label">${i}</div>
      <div class="bar-track"><div class="bar-fill" id="bar-${i}"></div></div>
      <div class="bar-pct" id="pct-${i}">0%</div>
    `;
    barsContainer.appendChild(row);
  }

  function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height)
    };
  }

  function startDraw(e) {
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();
    const pos = getPos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  }

  function stopDraw(e) {
    if (!isDrawing) return;
    isDrawing = false;
    predict(); // Auto-predict on stroke completion!
  }

  canvas.addEventListener('mousedown', startDraw);
  canvas.addEventListener('mousemove', draw);
  window.addEventListener('mouseup', stopDraw);

  canvas.addEventListener('touchstart', startDraw, { passive: false });
  canvas.addEventListener('touchmove', draw, { passive: false });
  canvas.addEventListener('touchend', stopDraw);

  function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;
    document.getElementById('predNumber').innerText = '-';
    document.getElementById('predConfidence').innerText = 'Draw a digit and click Recognize';
    for (let i = 0; i <= 9; i++) {
      const bar = document.getElementById(`bar-${i}`);
      bar.style.width = '0%';
      bar.classList.remove('top');
      document.getElementById(`pct-${i}`).innerText = '0%';
    }
    document.getElementById('statInputs').innerText = '0';
    document.getElementById('statH1').innerText = '0';
    document.getElementById('statH2').innerText = '0';
  }

  async function predict() {
    if (!hasDrawn) return;
    const dataUri = canvas.toDataURL('image/png');
    try {
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUri })
      });
      const data = await response.json();
      if (data.error) return alert(data.error);

      document.getElementById('predNumber').innerText = data.prediction;
      const conf = (data.confidence * 100).toFixed(1);
      document.getElementById('predConfidence').innerText = `Confidence: ${conf}%`;

      data.probabilities.forEach((p, idx) => {
        const pct = (p * 100).toFixed(1);
        const bar = document.getElementById(`bar-${idx}`);
        bar.style.width = `${pct}%`;
        if (idx === data.prediction) {
          bar.classList.add('top');
        } else {
          bar.classList.remove('top');
        }
        document.getElementById(`pct-${idx}`).innerText = `${pct}%`;
      });

      if (data.layer_stats) {
        document.getElementById('statInputs').innerText = data.layer_stats.input_active_pixels || 0;
        document.getElementById('statH1').innerText = data.layer_stats.h1_active_neurons || 0;
        document.getElementById('statH2').innerText = data.layer_stats.h2_active_neurons || 0;
      }
    } catch (err) {
      console.error(err);
    }
  }
</script>
</body>
</html>
"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                payload = json.loads(body)
                img_data_uri = payload.get("image", "")
                tensor, raw_grid = preprocess_canvas_image(img_data_uri)
                result = predict_and_explain(tensor)
                result["grid"] = raw_grid

                response_bytes = json.dumps(result).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_bytes)
            except Exception as e:
                err_resp = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(err_resp)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Silence verbose request logs
        return


def run_server(port: int = 5000):
    server = HTTPServer(("127.0.0.1", port), RequestHandler)
    print(f"\n=======================================================")
    print(f" Handwritten Digit Recognizer Web App is Live!")
    print(f" URL: http://localhost:{port}")
    print(f" Draw any digit in your browser and see predictions live!")
    print(f" Press Ctrl+C to stop.")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    run_server(port)

