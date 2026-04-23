#!/usr/bin/env python3
"""
MNIST Digit Inference — Draw & Predict

A Paint-like canvas where you draw a digit and the trained model predicts it.
Uses the model from .nnb/nnb-20260423-091628-af157ba1/workspace/checkpoints/best_model.pth

Usage:
    python inference.py
"""

import tkinter as tk
from tkinter import font as tkfont
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn


# =============================================================================
# MODEL (must match training architecture exactly)
# =============================================================================

class FeedForwardNN(nn.Module):
    """Feedforward NN matching the trained model: 784 → 48 → 10."""

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
# APP
# =============================================================================

CHECKPOINT = ".nnb/nnb-20260423-091628-af157ba1/workspace/checkpoints/best_model.pth"
CANVAS_SIZE = 280          # Drawing canvas (pixels)
MODEL_INPUT = 28           # MNIST resolution
BRUSH_RADIUS = 10          # Brush thickness
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class DigitRecognizer:
    """Tkinter app with a drawing canvas and live prediction."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MNIST Digit Recognizer")
        self.root.resizable(False, False)
        self.root.configure(bg="#1a1a2e")

        # Load model
        self.model = self._load_model()

        # Off-screen image for pixel-accurate capture
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)

        self._build_ui()
        self._bind_events()

    # ── Model ────────────────────────────────────────────────────────────

    def _load_model(self) -> FeedForwardNN:
        model = FeedForwardNN()
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    # ── UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Fonts
        title_font = tkfont.Font(family="Segoe UI", size=16, weight="bold")
        pred_font = tkfont.Font(family="Segoe UI", size=72, weight="bold")
        conf_font = tkfont.Font(family="Segoe UI", size=14)
        hint_font = tkfont.Font(family="Segoe UI", size=10)
        bar_label_font = tkfont.Font(family="Consolas", size=10)

        bg = "#1a1a2e"
        fg = "#e0e0e0"
        accent = "#e94560"

        # ── Title ──
        tk.Label(
            self.root, text="✏️  Draw a Digit", font=title_font,
            bg=bg, fg=fg
        ).pack(pady=(16, 4))

        tk.Label(
            self.root, text="Draw on the canvas below, then see the prediction →",
            font=hint_font, bg=bg, fg="#888"
        ).pack()

        # ── Main frame (canvas + prediction) ──
        main = tk.Frame(self.root, bg=bg)
        main.pack(padx=20, pady=12)

        # Canvas
        canvas_frame = tk.Frame(main, bg=accent, bd=0, highlightthickness=2,
                                highlightbackground=accent)
        canvas_frame.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(
            canvas_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="black", cursor="crosshair", highlightthickness=0
        )
        self.canvas.pack()

        # Prediction panel
        pred_panel = tk.Frame(main, bg="#16213e", width=220)
        pred_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(16, 0))
        pred_panel.pack_propagate(False)

        tk.Label(
            pred_panel, text="Prediction", font=conf_font,
            bg="#16213e", fg="#888"
        ).pack(pady=(20, 0))

        self.pred_label = tk.Label(
            pred_panel, text="—", font=pred_font,
            bg="#16213e", fg=accent
        )
        self.pred_label.pack(pady=(0, 4))

        self.conf_label = tk.Label(
            pred_panel, text="Draw something!", font=conf_font,
            bg="#16213e", fg="#888"
        )
        self.conf_label.pack()

        # ── Probability bars ──
        bars_frame = tk.Frame(pred_panel, bg="#16213e")
        bars_frame.pack(fill=tk.X, padx=14, pady=(16, 10))

        self.bar_canvases = []
        self.bar_labels = []

        for i in range(10):
            row = tk.Frame(bars_frame, bg="#16213e")
            row.pack(fill=tk.X, pady=1)

            lbl = tk.Label(row, text=str(i), font=bar_label_font,
                           bg="#16213e", fg="#aaa", width=2, anchor="e")
            lbl.pack(side=tk.LEFT)

            bar_cv = tk.Canvas(row, height=12, bg="#0f3460",
                               highlightthickness=0)
            bar_cv.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

            pct_lbl = tk.Label(row, text="", font=bar_label_font,
                               bg="#16213e", fg="#666", width=5, anchor="w")
            pct_lbl.pack(side=tk.LEFT, padx=(4, 0))

            self.bar_canvases.append(bar_cv)
            self.bar_labels.append(pct_lbl)

        # ── Clear button ──
        btn_frame = tk.Frame(self.root, bg=bg)
        btn_frame.pack(pady=(0, 16))

        self.clear_btn = tk.Button(
            btn_frame, text="🗑️  Clear Canvas", font=conf_font,
            bg=accent, fg="white", activebackground="#c0392b",
            activeforeground="white", bd=0, padx=24, pady=8,
            cursor="hand2", command=self._clear
        )
        self.clear_btn.pack()

    # ── Events ───────────────────────────────────────────────────────────

    def _bind_events(self):
        self.canvas.bind("<B1-Motion>", self._paint)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        # Also draw on click (not just drag)
        self.canvas.bind("<Button-1>", self._paint)

    def _paint(self, event):
        r = BRUSH_RADIUS
        x, y = event.x, event.y
        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill="white", outline="white"
        )
        self.pil_draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=255
        )

    def _on_release(self, _event):
        self._predict()

    def _clear(self):
        self.canvas.delete("all")
        self.pil_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        self.pred_label.config(text="—")
        self.conf_label.config(text="Draw something!")
        for i in range(10):
            self.bar_canvases[i].delete("all")
            self.bar_labels[i].config(text="")

    # ── Prediction ───────────────────────────────────────────────────────

    def _predict(self):
        # Preprocess: resize to 28×28, center, normalize like MNIST training
        img = self.pil_image.copy()

        # Add slight Gaussian blur (mimics antialiasing of real MNIST)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))

        # Resize to 28×28 with antialiasing
        img = img.resize((MODEL_INPUT, MODEL_INPUT), Image.LANCZOS)

        # Convert to tensor and normalize (same as training)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - MNIST_MEAN) / MNIST_STD
        tensor = torch.tensor(arr).view(1, -1)  # (1, 784)

        # Infer
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        predicted = probs.argmax().item()
        confidence = probs[predicted].item()

        # Update UI
        self.pred_label.config(text=str(predicted))
        self.conf_label.config(text=f"{confidence:.1%} confidence")

        # Update probability bars
        max_prob = probs.max().item()
        accent = "#e94560"
        dim = "#0f3460"

        for i in range(10):
            p = probs[i].item()
            cv = self.bar_canvases[i]
            cv.delete("all")
            cv.update_idletasks()
            w = cv.winfo_width()

            bar_w = int(w * p / max(max_prob, 0.01))
            color = accent if i == predicted else "#533483"
            cv.create_rectangle(0, 0, bar_w, 12, fill=color, outline="")

            self.bar_labels[i].config(
                text=f"{p:.0%}",
                fg=accent if i == predicted else "#666"
            )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()
