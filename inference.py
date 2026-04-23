#!/usr/bin/env python3
"""
MNIST Digit Inference Server

A web-based Paint canvas where you draw a digit and the model predicts it.
Runs inside Docker — zero local dependencies needed.

Usage:
    python inference.py [--port 8080] [--model PATH]
"""

import io
import json
import base64
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# =============================================================================
# MODEL (matches training architecture: 784 → 48 → 10)
# =============================================================================

class FeedForwardNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=48, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))


# =============================================================================
# HTML PAGE (embedded — no external files needed)
# =============================================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MNIST Digit Recognizer</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a1a;
    color: #e0e0e0;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }

  /* Animated background */
  body::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 30% 20%, rgba(233,69,96,0.08) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, rgba(83,52,131,0.08) 0%, transparent 50%);
    animation: drift 20s ease-in-out infinite alternate;
    z-index: 0;
    pointer-events: none;
  }
  @keyframes drift {
    0%   { transform: translate(0, 0) rotate(0deg); }
    100% { transform: translate(-5%, 3%) rotate(3deg); }
  }

  .container {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 20px;
  }

  h1 {
    font-size: 1.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e94560, #533483);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
  }
  .subtitle {
    color: #666;
    font-size: 0.85rem;
    margin-top: -12px;
  }

  .main {
    display: flex;
    gap: 24px;
    align-items: stretch;
  }

  /* Canvas */
  .canvas-wrap {
    border-radius: 16px;
    padding: 3px;
    background: linear-gradient(135deg, #e94560, #533483);
    box-shadow: 0 0 40px rgba(233,69,96,0.15);
    transition: box-shadow 0.3s;
  }
  .canvas-wrap:hover {
    box-shadow: 0 0 60px rgba(233,69,96,0.25);
  }

  canvas {
    display: block;
    border-radius: 14px;
    cursor: crosshair;
    background: #000;
    touch-action: none;
  }

  /* Prediction panel */
  .panel {
    background: rgba(22,33,62,0.6);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 24px 20px;
    width: 220px;
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  .panel-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #666;
    margin-bottom: 4px;
  }

  .prediction {
    font-size: 5rem;
    font-weight: 700;
    color: #e94560;
    line-height: 1;
    transition: transform 0.15s, color 0.3s;
  }
  .prediction.pop {
    transform: scale(1.15);
  }

  .confidence {
    font-size: 0.9rem;
    color: #888;
    margin-bottom: 20px;
  }

  /* Probability bars */
  .bars {
    width: 100%;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .bar-row {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .bar-digit {
    width: 16px;
    font-size: 0.75rem;
    color: #888;
    text-align: right;
    font-family: 'Consolas', monospace;
  }
  .bar-track {
    flex: 1;
    height: 10px;
    background: rgba(15,52,96,0.6);
    border-radius: 5px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 5px;
    width: 0%;
    transition: width 0.3s ease, background 0.3s;
    background: #533483;
  }
  .bar-fill.active {
    background: linear-gradient(90deg, #e94560, #ff6b81);
  }
  .bar-pct {
    width: 36px;
    font-size: 0.7rem;
    color: #555;
    font-family: 'Consolas', monospace;
  }
  .bar-pct.active { color: #e94560; }

  /* Clear button */
  .clear-btn {
    margin-top: 8px;
    padding: 10px 32px;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: white;
    background: linear-gradient(135deg, #e94560, #c0392b);
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.3s;
  }
  .clear-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(233,69,96,0.3);
  }
  .clear-btn:active { transform: translateY(0); }

  .hint {
    font-size: 0.75rem;
    color: #444;
    margin-top: -8px;
  }

  @media (max-width: 640px) {
    .main { flex-direction: column; align-items: center; }
    .panel { width: 280px; }
  }
</style>
</head>
<body>
<div class="container">
  <h1>✏️ MNIST Digit Recognizer</h1>
  <p class="subtitle">Draw a digit on the canvas — the model predicts in real-time</p>

  <div class="main">
    <div class="canvas-wrap">
      <canvas id="canvas" width="280" height="280"></canvas>
    </div>

    <div class="panel">
      <div class="panel-title">Prediction</div>
      <div class="prediction" id="pred">—</div>
      <div class="confidence" id="conf">Draw something!</div>

      <div class="bars" id="bars"></div>
    </div>
  </div>

  <button class="clear-btn" onclick="clearCanvas()">🗑️ Clear Canvas</button>
  <p class="hint">Model: FeedForwardNN (784→48→10) · 97.2% accuracy</p>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const predEl = document.getElementById('pred');
const confEl = document.getElementById('conf');
const barsEl = document.getElementById('bars');

let drawing = false;
let debounceTimer = null;

// Build bar rows
for (let i = 0; i < 10; i++) {
  barsEl.innerHTML += `
    <div class="bar-row">
      <span class="bar-digit">${i}</span>
      <div class="bar-track"><div class="bar-fill" id="bf${i}"></div></div>
      <span class="bar-pct" id="bp${i}"></span>
    </div>`;
}

// Drawing
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.lineWidth = 18;
ctx.strokeStyle = 'white';

function startDraw(e) {
  drawing = true;
  const p = pos(e);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  // Draw a dot on click
  ctx.lineTo(p.x + 0.1, p.y + 0.1);
  ctx.stroke();
}

function draw(e) {
  if (!drawing) return;
  e.preventDefault();
  const p = pos(e);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
}

function endDraw() {
  if (!drawing) return;
  drawing = false;
  ctx.beginPath();
  // Debounce prediction
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(predict, 150);
}

function pos(e) {
  const r = canvas.getBoundingClientRect();
  const t = e.touches ? e.touches[0] : e;
  return { x: t.clientX - r.left, y: t.clientY - r.top };
}

// Mouse
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', endDraw);
canvas.addEventListener('mouseleave', endDraw);

// Touch
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDraw(e); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });
canvas.addEventListener('touchend', endDraw);

function clearCanvas() {
  ctx.clearRect(0, 0, 280, 280);
  predEl.textContent = '—';
  predEl.style.color = '#e94560';
  confEl.textContent = 'Draw something!';
  for (let i = 0; i < 10; i++) {
    document.getElementById('bf' + i).style.width = '0%';
    document.getElementById('bf' + i).className = 'bar-fill';
    document.getElementById('bp' + i).textContent = '';
    document.getElementById('bp' + i).className = 'bar-pct';
  }
}

async function predict() {
  // Check if canvas is blank
  const pixels = ctx.getImageData(0, 0, 280, 280).data;
  let hasContent = false;
  for (let i = 3; i < pixels.length; i += 4) {
    if (pixels[i] > 10) { hasContent = true; break; }
  }
  if (!hasContent) return;

  // Send canvas as PNG
  const blob = await new Promise(r => canvas.toBlob(r, 'image/png'));
  const form = new FormData();
  form.append('image', blob);

  try {
    const res = await fetch('/predict', { method: 'POST', body: form });
    const data = await res.json();

    // Animate prediction
    predEl.textContent = data.prediction;
    predEl.classList.add('pop');
    setTimeout(() => predEl.classList.remove('pop'), 150);

    confEl.textContent = (data.confidence * 100).toFixed(1) + '% confidence';

    // Update bars
    const maxP = Math.max(...data.probabilities);
    for (let i = 0; i < 10; i++) {
      const p = data.probabilities[i];
      const fill = document.getElementById('bf' + i);
      const pct = document.getElementById('bp' + i);
      fill.style.width = (p / maxP * 100).toFixed(1) + '%';
      fill.className = i === data.prediction ? 'bar-fill active' : 'bar-fill';
      pct.textContent = (p * 100).toFixed(0) + '%';
      pct.className = i === data.prediction ? 'bar-pct active' : 'bar-pct';
    }
  } catch (err) {
    confEl.textContent = 'Server error';
  }
}
</script>
</body>
</html>
"""


# =============================================================================
# HTTP SERVER
# =============================================================================

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

model = None  # loaded at startup


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Convert drawn canvas PNG to a normalized 784-dim tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    img = img.resize((28, 28), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MNIST_MEAN) / MNIST_STD
    return torch.tensor(arr).view(1, -1)


class Handler(BaseHTTPRequestHandler):
    """Serves the HTML page and handles /predict POST requests."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return

        # Parse multipart form data (image field)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Extract image bytes from multipart form
        image_bytes = self._extract_image(body)
        if image_bytes is None:
            self.send_error(400, "No image data")
            return

        # Predict
        tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        predicted = int(probs.argmax().item())
        confidence = float(probs[predicted].item())

        result = {
            "prediction": predicted,
            "confidence": confidence,
            "probabilities": [round(float(p), 4) for p in probs],
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def _extract_image(self, body: bytes) -> bytes | None:
        """Extract image bytes from multipart/form-data body."""
        # Find the boundary from Content-Type header
        content_type = self.headers.get("Content-Type", "")
        if "boundary=" not in content_type:
            return None

        boundary = content_type.split("boundary=")[1].strip()
        # Handle quoted boundary
        if boundary.startswith('"') and boundary.endswith('"'):
            boundary = boundary[1:-1]

        parts = body.split(f"--{boundary}".encode())
        for part in parts:
            if b"image" in part and b"\r\n\r\n" in part:
                return part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n--")
        return None

    def log_message(self, format, *args):
        """Suppress default request logging (too noisy)."""
        pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    global model

    parser = argparse.ArgumentParser(description="MNIST Digit Inference Server")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    parser.add_argument(
        "--model",
        type=str,
        default="/workspace/checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model} ...")
    m = FeedForwardNN()
    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    model = m
    print(f"Model loaded (best accuracy: {ckpt.get('best_accuracy', 'N/A')})")

    # Start server
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\n{'='*50}")
    print(f"  MNIST Inference Server running!")
    print(f"  Open:  http://localhost:{args.port}")
    print(f"{'='*50}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
