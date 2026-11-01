# NeuroLab · Learn How Neural Networks and Transformers Work

An educational, browser-only set of pages that **compute everything live**: a neural network written from scratch in plain JavaScript, trained in real time, plus an animated walkthrough of how AI language models tokenize and process a sentence. Explanations use simple language, and every technical term has an ⓘ info icon with a plain-English definition.

## Pages
| Page | What it teaches |
|---|---|
| `index.html` | **Neural network basics.** A small network learns XOR, AND, a circle, or a sine wave. Watch the forward pass (guess), the loss (mistake), backpropagation (blame) and gradient descent (adjust). |
| `complex.html` | **Deep Network Lab.** 2–4 hidden layers, each with its own width, passing information from several inputs to several outputs: a 7-segment display → binary, a full adder, a 3-class pinwheel and two spirals. Uses momentum. |
| `transformer.html` | **Transformer & Tokens.** Type a sentence, press **Transform** and watch 10 animated stages: characters → normalization → pre-tokenization → WordPiece subwords → token IDs → embeddings (with a meaning map) → positional encoding → 2-head self-attention (arcs + table) → feed-forward network → next-token probabilities, with "add the word and run again" to show autoregressive generation. |

## Code layout
```
nn-core.js      shared engine (global `NN`, a classic script so file:// works)
  ├─ NN.MLP       multi-layer perceptron: forward, MSE, backprop, SGD + momentum
  ├─ NN.ACT       sigmoid / tanh / leaky ReLU / linear
  ├─ NN.render    canvas: network diagram + pulses, loss curve, decision regions
  ├─ NN.anim      forward/backward animation clock
  ├─ NN.ui        status chip, logger, neuron inspector, canvas hover, shortcuts
  ├─ NN.GLOSSARY  plain-English definitions for every term
  └─ NN.Info      info-icon popovers: <button class="info-i" data-info="key">
nn-theme.css    shared dark/neon look (panels, buttons, sliders, info popover)
index.html      datasets + wiring for the basic page
complex.html    datasets, per-layer width controls, prediction matrix
transformer.html  toy tokenizer + transformer; its feed-forward layer is a real NN.MLP
```

The transformer page is a **toy model** on purpose: 8-number vectors, a vocabulary of about 670 tokens, one transformer block. The pipeline and maths (WordPiece, sinusoidal positions, scaled dot-product attention with softmax, causal mask, residual + layer norm, ReLU feed-forward, softmax with temperature) are real. The weights are small and partly hand-made, so the attention patterns and predictions are readable instead of random.

## Run locally
Open any page directly in a browser, or serve the folder:

```bash
npx serve .
```

## Deploy
No build step is needed: it is a static site. Import the repo in Vercel (framework preset: **Other**), or run:

```bash
npx vercel --prod
```

The repo also includes a GitHub Pages workflow (`.github/workflows/static.yml`).

> Styling uses the Tailwind Play CDN (`cdn.tailwindcss.com`), which compiles CSS in the browser. That suits a demo; for heavy production traffic, precompile Tailwind into a static stylesheet.
