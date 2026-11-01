# NeuroLab · Interactive Neural Network Visualizer

A single-file (`index.html`) web app that trains a **Multi-Layer Perceptron written from scratch in vanilla JavaScript** and visualizes it live on HTML5 Canvas.

## Features
- **Topology diagram**: glowing neurons, weighted edges (cyan = positive, purple = negative, thickness = |w|), animated weight flow, forward-signal pulses (left → right) and backprop gradient pulses (right → left, rose/amber).
- **Datasets**: Logical XOR, Logical AND, Binary Classification (circle), Simple Regression (sine).
- **Controls**: hidden layers (1–3), neurons per layer (2–8), activation (Sigmoid / Tanh / Leaky ReLU), learning rate (log slider), target epochs, simulation speed, **Train Step**, **Auto-Run**, **Reset Weights**, custom-input inference.
- **Dashboard**: epoch, MSE loss (with trend), accuracy / R², parameter count, log-scale loss curve, decision boundary / regression fit, neuron inspector (z = Σw·a + b), equations panel, computation log.
- Keyboard: `Space` run/pause · `S` step · `R` reset.

## Run locally
Open `index.html` in a browser, or serve the folder:

```bash
npx serve .
```

## Deploy to Vercel
No build step is needed. Import the repo in Vercel (framework preset: **Other**), or run:

```bash
npx vercel --prod
```

> Styling uses the Tailwind Play CDN (`cdn.tailwindcss.com`), which compiles CSS in the browser. That suits a demo; for heavy production traffic, precompile Tailwind into a static stylesheet.
