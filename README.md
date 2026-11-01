# NeuroLab · Interactive Neural Network Visualizer

A single-file (`index.html`) web app that trains a **Multi-Layer Perceptron written from scratch in vanilla JavaScript** and visualizes it live on HTML5 Canvas.

## Features
- **Topology diagram**: glowing neurons, weighted edges (cyan = positive, purple = negative, thickness = |w|), animated weight flow, forward-signal pulses (left → right) and backprop gradient pulses (right → left, rose/amber).
- **Datasets**: Logical XOR, Logical AND, Binary Classification (circle), Simple Regression (sine).
- **Controls**: hidden layers (1–3), neurons per layer (2–8), activation (Sigmoid / Tanh / Leaky ReLU), learning rate (log slider), target epochs, simulation speed, **Train Step**, **Auto-Run**, **Reset Weights**, custom-input inference.
- **Dashboard**: epoch, MSE loss (with trend), accuracy / R², parameter count, log-scale loss curve, decision boundary / regression fit, neuron inspector (z = Σw·a + b), equations panel, computation log.
- Keyboard: `Space` run/pause · `S` step · `R` reset.

## Pages
| Page | What it shows |
|---|---|
| `index.html` | **Basic example**: 1–3 hidden layers of equal width, one output. XOR, AND, circle classification, sine regression. |
| `complex.html` | **Deep Network Lab**: 2–4 hidden layers, each with its own width (2–12), several inputs and outputs, SGD with momentum. |

### Complex examples (`complex.html`)
- **7-Segment Display → 4-bit Binary**: `7 → 10 → 8 → 4`. Segments a–g in, the digit's binary code out.
- **Binary Full Adder**: `3 → 6 → 6 → 2`. A, B, carry-in in; Sum and Carry-out out.
- **3-Class Pinwheel**: `2 → 8 → 8 → 3`. One-hot outputs, argmax decides the class.
- **Two Intertwined Spirals**: `2 → 12 → 12 → 8 → 1`. A classic benchmark that shallow networks fail.

It adds a per-layer width editor, a momentum (β) slider, clickable bit toggles and a training-sample picker for inference, and a prediction matrix (inputs, targets and ŷ for every sample). The diagram labels which layer-to-layer hand-off each pulse is crossing.

## Run locally
Open `index.html` (or `complex.html`) in a browser, or serve the folder:

```bash
npx serve .
```

## Deploy to Vercel
No build step is needed. Import the repo in Vercel (framework preset: **Other**), or run:

```bash
npx vercel --prod
```

> Styling uses the Tailwind Play CDN (`cdn.tailwindcss.com`), which compiles CSS in the browser. That suits a demo; for heavy production traffic, precompile Tailwind into a static stylesheet.
