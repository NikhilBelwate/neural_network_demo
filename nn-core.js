/* =====================================================================
   nn-core.js: shared engine for the NeuroLab pages
   ---------------------------------------------------------------------
   Exposes one global, `NN`, with:
     NN.util      small math / formatting helpers
     NN.rand      seeded random numbers (mulberry32, gaussian, hashString)
     NN.ACT       activation functions (sigmoid, tanh, leaky ReLU, linear)
     NN.MLP       a from-scratch multi-layer perceptron (forward pass,
                  MSE loss, backpropagation, SGD with optional momentum)
     NN.render    canvas drawing: network diagram, pulses, loss curve,
                  2D decision regions
     NN.anim      forward/backward animation clock
     NN.ui        shared DOM helpers (status chip, logger, inspector...)
     NN.GLOSSARY  plain-English definitions used by the info icons
     NN.Info      info-icon popovers ( <button class="info-i" data-info="key"> )

   Loaded as a classic script (not an ES module) so the pages also work
   when opened straight from disk (file://).
   ===================================================================== */
(function (global) {
  'use strict';

  /* ------------------------------------------------------------------
     Utilities
     ------------------------------------------------------------------ */
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const fmt = (v, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : 'NaN');
  const fmtLoss = (v) => (!Number.isFinite(v) ? 'NaN' : v < 1e-3 ? v.toExponential(2) : v.toFixed(4));
  const easeInOut = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  const argmax = (arr) => arr.reduce((best, v, i) => (v > arr[best] ? i : best), 0);
  const subscript = (n) => String(n).split('').map((d) => String.fromCharCode(8320 + +d)).join('');
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const prefersReducedMotion = !!(global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches);
  const softmax = (arr, T = 1) => {
    const m = Math.max(...arr);
    const e = arr.map((v) => Math.exp((v - m) / T));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  };

  /* ------------------------------------------------------------------
     Seeded randomness
     ------------------------------------------------------------------ */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussian(rng) {
    let u = 0; while (u === 0) u = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
  }
  /** FNV-1a 32-bit hash: turns a string into a stable seed. */
  function hashString(str) {
    let h = 0x811c9dc5;
    for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
    return h >>> 0;
  }

  /* ------------------------------------------------------------------
     Activation functions
     ------------------------------------------------------------------ */
  const ACT = {
    sigmoid: {
      name: 'Sigmoid', short: 'σ', f: (z) => 1 / (1 + Math.exp(-z)), d: (z, a) => a * (1 - a),
      eq: 'σ(z) = 1 / (1 + e<sup>−z</sup>)', deq: "σ'(z) = σ(z)·(1 − σ(z))",
      words: 'squashes any number into the range 0 to 1',
    },
    tanh: {
      name: 'Tanh', short: 'tanh', f: (z) => Math.tanh(z), d: (z, a) => 1 - a * a,
      eq: 'tanh(z) = (e<sup>z</sup> − e<sup>−z</sup>) / (e<sup>z</sup> + e<sup>−z</sup>)', deq: "tanh'(z) = 1 − tanh²(z)",
      words: 'squashes any number into the range −1 to 1',
    },
    relu: {
      name: 'Leaky ReLU', short: 'ReLU', f: (z) => (z > 0 ? z : 0.01 * z), d: (z) => (z > 0 ? 1 : 0.01),
      eq: 'ReLU(z) = max(0.01·z, z)', deq: "ReLU'(z) = 1 if z &gt; 0 else 0.01",
      words: 'keeps positive numbers, shrinks negative ones almost to 0',
    },
    linear: {
      name: 'Linear', short: 'id', f: (z) => z, d: () => 1,
      eq: 'g(z) = z', deq: "g'(z) = 1",
      words: 'passes the number through unchanged',
    },
  };

  /* ------------------------------------------------------------------
     Multi-Layer Perceptron
     ------------------------------------------------------------------ */
  class MLP {
    /**
     * @param {number[]} sizes  neurons per layer, e.g. [2, 4, 4, 1]
     * @param {string} hiddenAct key of ACT for hidden layers
     * @param {string} outAct    key of ACT for the output layer
     * @param {number} seed      seed for the random starting weights
     */
    constructor(sizes, hiddenAct, outAct, seed) {
      this.sizes = sizes.slice();
      this.L = sizes.length;
      this.hiddenAct = hiddenAct;
      this.outAct = outAct;
      const rng = mulberry32(seed);
      this.W = []; // W[l][j][i]: weight from neuron i (layer l) to neuron j (layer l+1)
      this.b = []; // b[l][j]: bias of neuron j in layer l+1
      this.vW = []; this.vb = []; // momentum "velocity"
      for (let l = 0; l < this.L - 1; l++) {
        const nIn = sizes[l], nOut = sizes[l + 1];
        const act = l === this.L - 2 ? outAct : hiddenAct;
        // He init for ReLU, scaled Xavier/Glorot otherwise
        const std = act === 'relu' ? Math.sqrt(2 / nIn) : Math.sqrt(2 / (nIn + nOut)) * (act === 'sigmoid' ? 2 : 1.2);
        this.W.push(Array.from({ length: nOut }, () => Array.from({ length: nIn }, () => gaussian(rng) * std)));
        this.b.push(Array.from({ length: nOut }, () => (rng() - 0.5) * 0.2));
        this.vW.push(Array.from({ length: nOut }, () => new Array(nIn).fill(0)));
        this.vb.push(new Array(nOut).fill(0));
      }
      this.z = sizes.map((n) => new Array(n).fill(0));
      this.a = sizes.map((n) => new Array(n).fill(0));
      this.resetGradStats();
    }

    get paramCount() {
      let w = 0, b = 0;
      for (let l = 0; l < this.L - 1; l++) { w += this.sizes[l] * this.sizes[l + 1]; b += this.sizes[l + 1]; }
      return { w, b, total: w + b };
    }

    /** Activation used by neuron layer l (1..L-1). */
    actOf(l) { return ACT[l === this.L - 1 ? this.outAct : this.hiddenAct]; }

    resetGradStats() {
      this.gradW = this.W.map((Wl) => Wl.map((row) => new Array(row.length).fill(0))); // mean |∂L/∂w|
      this.nodeGrad = this.sizes.map((n) => new Array(n).fill(0));                       // mean |δ|
    }

    /** Forward pass. With store=true the z / a values are kept for drawing. */
    forward(x, store = true) {
      const zs = [x.slice()], as = [x.slice()];
      let a = as[0];
      for (let l = 0; l < this.L - 1; l++) {
        const act = this.actOf(l + 1), Wl = this.W[l], bl = this.b[l];
        const z = new Array(Wl.length), na = new Array(Wl.length);
        for (let j = 0; j < Wl.length; j++) {
          let s = bl[j];
          const row = Wl[j];
          for (let i = 0; i < row.length; i++) s += row[i] * a[i];
          z[j] = s; na[j] = act.f(s);
        }
        zs.push(z); as.push(na); a = na;
      }
      if (store) { this.z = zs; this.a = as; }
      return { zs, as, out: a };
    }

    predict(x) { return this.forward(x, false).out; }

    /**
     * One gradient-descent update on one sample. Returns its MSE (before the update).
     * Update rule: v ← β·v − η·∇,  θ ← θ + v   (β = 0 gives plain SGD)
     */
    trainSample(x, y, lr, beta = 0) {
      const { zs, as } = this.forward(x, false);
      const L = this.L, out = as[L - 1], oAct = this.actOf(L - 1);
      let loss = 0;
      let delta = new Array(out.length);
      for (let j = 0; j < out.length; j++) {
        const e = out[j] - y[j];
        loss += e * e;
        // dL/dz = dL/da · da/dz  with  L = (1/K)Σ(ŷ−y)²
        delta[j] = clamp((2 * e / out.length) * oAct.d(zs[L - 1][j], out[j]), -5, 5);
      }
      loss /= out.length;

      for (let l = L - 2; l >= 0; l--) {
        const Wl = this.W[l], bl = this.b[l], aPrev = as[l], vWl = this.vW[l], vbl = this.vb[l];
        const ng = this.nodeGrad[l + 1];
        for (let j = 0; j < delta.length; j++) ng[j] += Math.abs(delta[j]);

        // pass the error back one layer before changing the weights
        let prevDelta = null;
        if (l > 0) {
          const act = this.actOf(l);
          prevDelta = new Array(aPrev.length);
          for (let i = 0; i < aPrev.length; i++) {
            let s = 0;
            for (let j = 0; j < delta.length; j++) s += Wl[j][i] * delta[j];
            prevDelta[i] = clamp(s * act.d(zs[l][i], aPrev[i]), -5, 5);
          }
        }
        const gW = this.gradW[l];
        for (let j = 0; j < delta.length; j++) {
          const dj = delta[j], row = Wl[j], vRow = vWl[j], gRow = gW[j];
          for (let i = 0; i < row.length; i++) {
            const g = dj * aPrev[i];
            vRow[i] = beta * vRow[i] - lr * g;
            row[i] += vRow[i];
            gRow[i] += Math.abs(g);
          }
          vbl[j] = beta * vbl[j] - lr * dj;
          bl[j] += vbl[j];
        }
        if (prevDelta) delta = prevDelta;
      }
      return loss;
    }

    /** One epoch: every sample once, in a shuffled order. Returns mean loss. */
    trainEpoch(data, lr, rng, beta = 0) {
      this.resetGradStats();
      const idx = data.map((_, i) => i);
      for (let i = idx.length - 1; i > 0; i--) { const k = Math.floor(rng() * (i + 1)); [idx[i], idx[k]] = [idx[k], idx[i]]; }
      let total = 0;
      for (const i of idx) total += this.trainSample(data[i].x, data[i].y, lr, beta);
      const n = data.length;
      for (const Wl of this.gradW) for (const row of Wl) for (let i = 0; i < row.length; i++) row[i] /= n;
      for (const ng of this.nodeGrad) for (let i = 0; i < ng.length; i++) ng[i] /= n;
      return total / n;
    }

    /**
     * Score the whole dataset.
     * metric 'bits'   every output rounded at 0.5 must match the target
     *        'argmax' the biggest output must be the right class (one-hot)
     *        'r2'     regression: R² = 1 − SS_res / SS_tot (first output)
     */
    evaluate(data, metric) {
      let loss = 0, correct = 0, bitsOk = 0, bitsTotal = 0, ssRes = 0, ssTot = 0, mean = 0;
      if (metric === 'r2') { for (const d of data) mean += d.y[0]; mean /= data.length; }
      for (const d of data) {
        const out = this.predict(d.x);
        let l = 0;
        for (let j = 0; j < out.length; j++) { const e = out[j] - d.y[j]; l += e * e; }
        loss += l / out.length;
        if (metric === 'r2') {
          ssRes += (out[0] - d.y[0]) ** 2; ssTot += (d.y[0] - mean) ** 2;
        } else if (metric === 'argmax') {
          if (argmax(out) === argmax(d.y)) correct++;
        } else {
          let all = true;
          for (let j = 0; j < out.length; j++) {
            const ok = (out[j] >= 0.5 ? 1 : 0) === d.y[j];
            bitsOk += ok ? 1 : 0; bitsTotal++;
            if (!ok) all = false;
          }
          if (all) correct++;
        }
      }
      const score = metric === 'r2' ? (ssTot > 0 ? 1 - ssRes / ssTot : 0) : correct / data.length;
      return { loss: loss / data.length, score, bitScore: bitsTotal ? bitsOk / bitsTotal : NaN };
    }
  }

  /* ------------------------------------------------------------------
     Canvas rendering
     ------------------------------------------------------------------ */
  const COLORS = {
    cyan: [34, 211, 238], purple: [168, 85, 247], violet: [192, 132, 252],
    emerald: [52, 211, 153], rose: [251, 113, 133], amber: [251, 191, 36],
  };
  const CLASS_COLORS = [COLORS.cyan, COLORS.purple, COLORS.amber];
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
  const layerColor = (l, L) => (l === 0 ? COLORS.cyan : l === L - 1 ? COLORS.emerald : (l % 2 ? COLORS.purple : COLORS.violet));
  const layerName = (l, L) => (l === 0 ? 'Input' : l === L - 1 ? 'Output' : `Hidden ${l}`);

  /** Size a canvas for the device pixel ratio; returns ctx + CSS size. */
  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(global.devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(rect.width * dpr)), h = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  function computeLayout(sizes, w, h, opts = {}) {
    const L = sizes.length;
    const maxN = Math.max(...sizes);
    const top = opts.top ?? 46, bottom = opts.bottom ?? 34;
    const usableH = h - top - bottom;
    const r = clamp(Math.min(usableH / (maxN * 2.6), (w / L) / 5.5), opts.minR ?? 6, opts.maxR ?? 22);
    const padX = clamp(w * 0.08, r * 2.8, 100);
    const pos = [];
    for (let l = 0; l < L; l++) {
      const n = sizes[l];
      const x = L === 1 ? w / 2 : padX + (l * (w - 2 * padX)) / (L - 1);
      const spacing = Math.min(usableH / n, r * 3.4);
      const startY = top + usableH / 2 - (spacing * (n - 1)) / 2;
      pos.push(Array.from({ length: n }, (_, j) => ({ x, y: startY + j * spacing, r })));
    }
    return pos;
  }

  function drawPulse(ctx, x, y, r, col, alpha, dx, dy) {
    const len = Math.hypot(dx, dy) || 1;
    const ux = dx / len, uy = dy / len;
    for (let k = 3; k >= 1; k--) { // fading trail
      ctx.fillStyle = rgba(col, alpha * 0.12 * (4 - k));
      ctx.beginPath(); ctx.arc(x - ux * k * r * 1.4, y - uy * k * r * 1.4, r * (1 - k * 0.15), 0, Math.PI * 2); ctx.fill();
    }
    const g = ctx.createRadialGradient(x, y, 0, x, y, r * 2.8);
    g.addColorStop(0, rgba(col, alpha * 0.55)); g.addColorStop(1, rgba(col, 0));
    ctx.fillStyle = g;
    ctx.beginPath(); ctx.arc(x, y, r * 2.8, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgba(255,255,255,${alpha})`;
    ctx.beginPath(); ctx.arc(x, y, r * 0.55, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = rgba(col, alpha);
    ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill();
  }

  /**
   * Draw the network: weighted edges, glowing neurons, forward / backward pulses.
   * @param {HTMLCanvasElement} canvas
   * @param {object} s  { net, anim, opts:{values,pulses,flow}, selected, hover, time,
   *                      labelFor(l,j), nameFor(l,L), layout? }
   * @returns node positions (for hit-testing)
   */
  function drawNetwork(canvas, s) {
    const { ctx, w, h } = fitCanvas(canvas);
    const net = s.net, L = net.L, opts = s.opts || {};
    const anim = s.anim || { phase: null, t: 0 };
    const nameFor = s.nameFor || layerName;
    const labelFor = s.labelFor || ((l, j) => `${l}.${j}`);
    ctx.clearRect(0, 0, w, h);
    const pos = computeLayout(net.sizes, w, h, s.layout || {});
    const segs = L - 1;

    let fwdSeg = -1, bwdSeg = -1, frac = 0;
    if (anim.phase === 'fwd') { fwdSeg = Math.min(Math.floor(anim.t), segs - 1); frac = easeInOut(clamp(anim.t - fwdSeg, 0, 1)); }
    if (anim.phase === 'bwd') { const k = Math.min(Math.floor(anim.t), segs - 1); bwdSeg = segs - 1 - k; frac = easeInOut(clamp(anim.t - k, 0, 1)); }

    let maxW = 1e-9, maxG = 1e-12;
    for (let l = 0; l < segs; l++) for (const row of net.W[l]) for (const v of row) maxW = Math.max(maxW, Math.abs(v));
    for (let l = 0; l < segs; l++) for (const row of net.gradW[l]) for (const v of row) maxG = Math.max(maxG, v);
    const wScale = Math.max(1.5, maxW);
    const totalEdges = net.paramCount.w;
    const dense = totalEdges > 120;

    // ---- edges ----
    ctx.lineCap = 'round';
    for (let l = 0; l < segs; l++) {
      const Wl = net.W[l], gl = net.gradW[l];
      for (let j = 0; j < Wl.length; j++) {
        for (let i = 0; i < Wl[j].length; i++) {
          const wv = Wl[j][i], aw = Math.abs(wv) / wScale;
          const p0 = pos[l][i], p1 = pos[l + 1][j];
          const col = wv >= 0 ? COLORS.cyan : COLORS.purple;
          ctx.strokeStyle = rgba(col, (dense ? 0.07 : 0.12) + 0.5 * aw);
          ctx.lineWidth = 0.5 + (dense ? 2.2 : 3) * aw;
          ctx.setLineDash([]);
          ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();

          // animated "flow" along strong weights; direction follows the phase
          if (opts.flow && !prefersReducedMotion && aw > (dense ? 0.25 : 0)) {
            const dir = anim.phase === 'bwd' ? 1 : -1;
            ctx.setLineDash([3, 11]);
            ctx.lineDashOffset = (dir * (s.time || 0) * 0.025) % 14;
            ctx.strokeStyle = rgba(col, 0.2 + 0.5 * aw);
            ctx.lineWidth = 0.8 + 1.4 * aw;
            ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
            ctx.setLineDash([]);
          }
          if (l === bwdSeg) {
            const g = gl[j][i] / maxG;
            if (g > 0.04) {
              ctx.strokeStyle = rgba(COLORS.rose, 0.12 + 0.55 * g * (1 - Math.abs(frac - 0.5)));
              ctx.lineWidth = 1 + 2.5 * g;
              ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
            }
          }
        }
      }
    }

    // ---- glowing band over the layer hand-off being processed ----
    const activeSeg = anim.phase === 'fwd' ? fwdSeg : anim.phase === 'bwd' ? bwdSeg : -1;
    if (activeSeg >= 0 && L > 2) {
      const x0 = pos[activeSeg][0].x, x1 = pos[activeSeg + 1][0].x;
      const g = ctx.createLinearGradient(x0, 0, x1, 0);
      const c = anim.phase === 'fwd' ? COLORS.cyan : COLORS.rose;
      g.addColorStop(0, rgba(c, 0)); g.addColorStop(0.5, rgba(c, 0.05)); g.addColorStop(1, rgba(c, 0));
      ctx.fillStyle = g; ctx.fillRect(x0, 40, x1 - x0, h - 72);
    }

    // ---- weight labels (small networks only) ----
    if (opts.values && totalEdges <= 30) {
      ctx.font = '500 9.5px "JetBrains Mono", monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      for (let l = 0; l < segs; l++) {
        const Wl = net.W[l];
        for (let j = 0; j < Wl.length; j++) for (let i = 0; i < Wl[j].length; i++) {
          const p0 = pos[l][i], p1 = pos[l + 1][j];
          const x = lerp(p0.x, p1.x, 0.32), y = lerp(p0.y, p1.y, 0.32);
          const txt = fmt(Wl[j][i], 2);
          const tw = ctx.measureText(txt).width + 6;
          ctx.fillStyle = 'rgba(2,6,23,0.85)'; ctx.fillRect(x - tw / 2, y - 6.5, tw, 13);
          ctx.fillStyle = Wl[j][i] >= 0 ? '#67e8f9' : '#d8b4fe'; ctx.fillText(txt, x, y);
        }
      }
    }

    // ---- pulses ----
    if (opts.pulses && anim.phase) {
      if (anim.phase === 'fwd' && fwdSeg >= 0) {
        const l = fwdSeg, Wl = net.W[l], aPrev = net.a[l];
        let maxC = 1e-9;
        for (let j = 0; j < Wl.length; j++) for (let i = 0; i < Wl[j].length; i++) maxC = Math.max(maxC, Math.abs(Wl[j][i] * aPrev[i]));
        for (let j = 0; j < Wl.length; j++) for (let i = 0; i < Wl[j].length; i++) {
          const c = Math.abs(Wl[j][i] * aPrev[i]) / maxC;
          if (dense && c < 0.08) continue;
          const p0 = pos[l][i], p1 = pos[l + 1][j];
          drawPulse(ctx, lerp(p0.x, p1.x, frac), lerp(p0.y, p1.y, frac), 1.4 + 3.2 * c, Wl[j][i] >= 0 ? COLORS.cyan : COLORS.purple, 0.3 + 0.7 * c, p1.x - p0.x, p1.y - p0.y);
        }
      } else if (anim.phase === 'bwd' && bwdSeg >= 0) {
        const l = bwdSeg, gl = net.gradW[l];
        for (let j = 0; j < gl.length; j++) for (let i = 0; i < gl[j].length; i++) {
          const g = clamp(gl[j][i] / maxG, 0, 1);
          if (dense && g < 0.05) continue;
          const p0 = pos[l + 1][j], p1 = pos[l][i];
          drawPulse(ctx, lerp(p0.x, p1.x, frac), lerp(p0.y, p1.y, frac), 1.3 + 3.4 * Math.sqrt(g), g > 0.5 ? COLORS.amber : COLORS.rose, 0.3 + 0.7 * Math.sqrt(g), p1.x - p0.x, p1.y - p0.y);
        }
      }
    }

    // ---- neurons ----
    let maxNG = 1e-12;
    for (let l = 1; l < L; l++) for (const v of net.nodeGrad[l]) maxNG = Math.max(maxNG, v);
    for (let l = 0; l < L; l++) {
      const col = layerColor(l, L);
      const acts = net.a[l];
      const maxA = Math.max(1, ...acts.map(Math.abs));
      let lit = 1;
      if (anim.phase === 'fwd') lit = l <= fwdSeg ? 1 : l === fwdSeg + 1 ? 0.35 + 0.65 * frac : 0.35;
      for (let j = 0; j < pos[l].length; j++) {
        const p = pos[l][j];
        const inten = clamp(Math.abs(acts[j]) / maxA, 0, 1);
        const glow = (0.25 + 0.75 * inten) * lit;

        ctx.save();
        ctx.shadowColor = rgba(col, 0.9); ctx.shadowBlur = 6 + 20 * glow;
        ctx.fillStyle = '#060b1a';
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
        ctx.restore();

        const grd = ctx.createRadialGradient(p.x - p.r * 0.3, p.y - p.r * 0.3, p.r * 0.1, p.x, p.y, p.r);
        grd.addColorStop(0, rgba(col, 0.25 + 0.75 * glow));
        grd.addColorStop(1, rgba(col, 0.06 + 0.25 * glow));
        ctx.fillStyle = grd;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
        ctx.lineWidth = 1.5; ctx.strokeStyle = rgba(col, 0.6 + 0.4 * glow); ctx.stroke();

        // error ring: how much blame this neuron received (|δ|)
        if (l > 0) {
          const g = net.nodeGrad[l][j] / maxNG;
          const strong = anim.phase === 'bwd' && (bwdSeg + 1 === l || bwdSeg === l);
          if (g > 0.01) {
            ctx.strokeStyle = rgba(COLORS.rose, strong ? 0.35 + 0.65 * g : 0.08 + 0.25 * g);
            ctx.lineWidth = strong ? 2.5 : 1.5;
            ctx.beginPath(); ctx.arc(p.x, p.y, p.r + 4, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * clamp(g, 0.05, 1)); ctx.stroke();
          }
        }

        const isSel = s.selected && s.selected.l === l && s.selected.j === j;
        const isHov = s.hover && s.hover.l === l && s.hover.j === j;
        if (isSel || isHov) {
          ctx.setLineDash(isSel ? [4, 3] : []);
          ctx.strokeStyle = isSel ? 'rgba(255,255,255,0.85)' : 'rgba(255,255,255,0.45)';
          ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.arc(p.x, p.y, p.r + 8, 0, Math.PI * 2); ctx.stroke();
          ctx.setLineDash([]);
        }

        if (opts.values && p.r >= 12) {
          ctx.font = `600 ${p.r >= 17 ? 10 : 8.5}px "JetBrains Mono", monospace`;
          ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
          ctx.fillStyle = glow > 0.6 ? '#020617' : '#e2e8f0';
          ctx.fillText(fmt(acts[j], 2), p.x, p.y + 0.5);
        }
        if (l === 0 || l === L - 1) {
          ctx.font = '600 11px "JetBrains Mono", monospace';
          ctx.textBaseline = 'middle';
          ctx.fillStyle = rgba(col, 0.95);
          ctx.textAlign = l === 0 ? 'right' : 'left';
          let lab = labelFor(l, j);
          if (l === L - 1 && opts.values && p.r < 12) lab += ` ${fmt(acts[j], 2)}`;
          ctx.fillText(lab, l === 0 ? p.x - p.r - 10 : p.x + p.r + 10, p.y);
        }
      }
      const x = pos[l][0].x;
      ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
      ctx.font = '600 10px Inter, sans-serif';
      ctx.fillStyle = rgba(col, 0.9);
      ctx.fillText(nameFor(l, L).toUpperCase(), x, 20);
      ctx.font = '500 9.5px "JetBrains Mono", monospace';
      ctx.fillStyle = 'rgba(148,163,184,0.7)';
      ctx.fillText(l === 0 ? `${net.sizes[l]} inputs` : `${net.sizes[l]} · ${net.actOf(l).name}`, x, 34);
    }

    // ---- phase banner ----
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    if (anim.phase && s.banner !== false) {
      const isF = anim.phase === 'fwd';
      const seg = isF ? fwdSeg : bwdSeg;
      ctx.font = '600 10px "JetBrains Mono", monospace';
      ctx.fillStyle = isF ? 'rgba(103,232,249,0.9)' : 'rgba(253,164,175,0.9)';
      ctx.fillText(isF
        ? `FORWARD PASS ▶▶ ${nameFor(seg, L)} → ${nameFor(seg + 1, L)}: making a guess`
        : `◀◀ BACKPROPAGATION ${nameFor(seg + 1, L)} → ${nameFor(seg, L)}: sharing the blame`, 12, h - 12);
    }
    return pos;
  }

  /** Loss curve on a log scale. */
  function drawLoss(canvas, history, emptyText = 'Train the network to see the error go down') {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const hist = history.filter(Number.isFinite);
    const m = { l: 46, r: 10, t: 10, b: 24 };
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    if (pw <= 0 || ph <= 0) return;
    const minV = Math.max(1e-6, Math.min(...hist, 1));
    const maxV = Math.max(...hist, 1e-5);
    let lo = Math.floor(Math.log10(minV)), hi = Math.ceil(Math.log10(maxV));
    if (hi <= lo) hi = lo + 1;
    const yOf = (v) => m.t + ph - ((Math.log10(Math.max(v, 1e-6)) - lo) / (hi - lo)) * ph;
    const n = Math.max(hist.length - 1, 1);
    const xOf = (i) => m.l + (i / n) * pw;

    ctx.font = '9.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let e = lo; e <= hi; e++) {
      const y = yOf(Math.pow(10, e));
      ctx.strokeStyle = 'rgba(148,163,184,0.10)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(m.l + pw, y); ctx.stroke();
      ctx.fillStyle = 'rgba(148,163,184,0.6)'; ctx.fillText(`1e${e}`, m.l - 6, y);
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let k = 0; k <= 4; k++) {
      const i = Math.round((k / 4) * n);
      ctx.fillStyle = 'rgba(148,163,184,0.6)'; ctx.fillText(String(i), xOf(i), m.t + ph + 6);
    }
    if (hist.length < 2) {
      ctx.fillStyle = 'rgba(148,163,184,0.5)'; ctx.font = '12px Inter, sans-serif'; ctx.textBaseline = 'middle';
      ctx.fillText(emptyText, m.l + pw / 2, m.t + ph / 2);
      return;
    }
    const step = Math.max(1, Math.floor(hist.length / (pw * 1.5)));
    const pts = [];
    for (let i = 0; i < hist.length; i += step) pts.push([xOf(i), yOf(hist[i])]);
    pts.push([xOf(hist.length - 1), yOf(hist[hist.length - 1])]);

    const fill = ctx.createLinearGradient(0, m.t, 0, m.t + ph);
    fill.addColorStop(0, 'rgba(168,85,247,0.28)'); fill.addColorStop(1, 'rgba(34,211,238,0.0)');
    ctx.beginPath(); ctx.moveTo(pts[0][0], m.t + ph);
    pts.forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.lineTo(pts[pts.length - 1][0], m.t + ph); ctx.closePath();
    ctx.fillStyle = fill; ctx.fill();
    const stroke = ctx.createLinearGradient(m.l, 0, m.l + pw, 0);
    stroke.addColorStop(0, '#a855f7'); stroke.addColorStop(1, '#22d3ee');
    ctx.save();
    ctx.shadowColor = 'rgba(168,85,247,0.8)'; ctx.shadowBlur = 10;
    ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.beginPath(); pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
    ctx.restore();
    const [hx, hy] = pts[pts.length - 1];
    ctx.fillStyle = '#22d3ee'; ctx.beginPath(); ctx.arc(hx, hy, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = 'rgba(34,211,238,0.4)'; ctx.lineWidth = 6; ctx.beginPath(); ctx.arc(hx, hy, 5, 0, Math.PI * 2); ctx.stroke();
  }

  function drawProbe(ctx, x, y) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1.2;
    ctx.shadowColor = '#fff'; ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.arc(x, y, 7, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x - 12, y); ctx.lineTo(x - 4, y); ctx.moveTo(x + 4, y); ctx.lineTo(x + 12, y);
    ctx.moveTo(x, y - 12); ctx.lineTo(x, y - 4); ctx.moveTo(x, y + 4); ctx.lineTo(x, y + 12); ctx.stroke();
    ctx.restore();
  }

  /**
   * 2D input space coloured by the network's answer, with the data points on top.
   * @param {object} s { net, data, domain:[d0,d1], metric:'bits'|'argmax', probe:[x,y] }
   */
  const heatCanvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
  function drawRegions(canvas, s) {
    const { ctx, w, h } = fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    const { net, data, domain, metric, probe } = s;
    const m = { l: 34, r: 10, t: 8, b: 22 };
    const pw = w - m.l - m.r, ph = h - m.t - m.b;
    if (pw <= 0 || ph <= 0) return;
    const [d0, d1] = domain;
    const side = Math.min(pw, ph);
    const ox = m.l + (pw - side) / 2, oy = m.t + (ph - side) / 2;
    const toX = (v) => ox + ((v - d0) / (d1 - d0)) * side;
    const toY = (v) => oy + side - ((v - d0) / (d1 - d0)) * side;
    const multi = metric === 'argmax';

    const RES = 64;
    heatCanvas.width = RES; heatCanvas.height = RES;
    const hctx = heatCanvas.getContext('2d');
    const img = hctx.createImageData(RES, RES);
    for (let py = 0; py < RES; py++) {
      for (let px = 0; px < RES; px++) {
        const x = d0 + ((px + 0.5) / RES) * (d1 - d0);
        const y = d1 - ((py + 0.5) / RES) * (d1 - d0);
        const out = net.predict([x, y]);
        const k = (py * RES + px) * 4;
        let r, g, b, a;
        if (multi) {
          const sum = out.reduce((t, v) => t + v, 0) || 1;
          r = g = b = 0;
          out.forEach((v, c) => { const wgt = v / sum; r += CLASS_COLORS[c][0] * wgt; g += CLASS_COLORS[c][1] * wgt; b += CLASS_COLORS[c][2] * wgt; });
          const conf = Math.max(...out) / sum;
          r *= 0.6; g *= 0.6; b *= 0.6; a = 90 + 140 * conf;
        } else {
          const p = clamp(out[0], 0, 1);
          const edge = 1 - Math.min(1, Math.abs(p - 0.5) * 6); // bright line on the boundary
          r = lerp(COLORS.purple[0], COLORS.emerald[0], p) * 0.55 + 255 * edge * 0.25;
          g = lerp(COLORS.purple[1], COLORS.emerald[1], p) * 0.55 + 255 * edge * 0.25;
          b = lerp(COLORS.purple[2], COLORS.emerald[2], p) * 0.55 + 255 * edge * 0.25;
          a = 120 + 180 * Math.abs(p - 0.5);
        }
        img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = a;
      }
    }
    hctx.putImageData(img, 0, 0);
    ctx.save(); ctx.imageSmoothingEnabled = true; ctx.globalAlpha = 0.9;
    ctx.drawImage(heatCanvas, ox, oy, side, side);
    ctx.restore();
    ctx.strokeStyle = 'rgba(148,163,184,0.25)'; ctx.lineWidth = 1; ctx.strokeRect(ox, oy, side, side);

    const ticks = d1 - d0 > 2 ? [-1, 0, 1] : [0, 0.5, 1];
    ctx.font = '9.5px "JetBrains Mono", monospace'; ctx.fillStyle = 'rgba(148,163,184,0.65)';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    ticks.forEach((t) => ctx.fillText(String(t), toX(t), oy + side + 5));
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ticks.forEach((t) => ctx.fillText(String(t), ox - 5, toY(t)));

    const big = data.length <= 8;
    for (const d of data) {
      const x = toX(d.x[0]), y = toY(d.x[1]);
      const out = net.predict(d.x);
      let col, ok;
      if (multi) { const c = argmax(d.y); col = CLASS_COLORS[c]; ok = argmax(out) === c; }
      else { col = d.y[0] ? COLORS.emerald : COLORS.purple; ok = (out[0] >= 0.5 ? 1 : 0) === d.y[0]; }
      ctx.save();
      ctx.shadowColor = rgba(col, 1); ctx.shadowBlur = big ? 14 : 4;
      ctx.fillStyle = rgba(col, 1);
      ctx.beginPath(); ctx.arc(x, y, big ? 8 : 3.2, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      ctx.strokeStyle = ok ? 'rgba(255,255,255,0.8)' : 'rgba(251,113,133,1)';
      ctx.lineWidth = ok ? (big ? 1 : 0.8) : 1.8;
      ctx.stroke();
    }
    if (probe) drawProbe(ctx, toX(probe[0]), toY(probe[1]));
  }

  /* ------------------------------------------------------------------
     Forward / backward animation clock
     ------------------------------------------------------------------ */
  const anim = {
    create: (phase = null, queue = []) => ({ phase, t: 0, queue: queue.slice() }),
    /** Advance by dt ms. One "segment" = one layer-to-layer hop. */
    step(A, dt, segs, running) {
      if (!A.phase) return;
      A.t += dt / (running ? 250 : 450);
      if (A.t >= segs) {
        if (A.queue.length) { A.phase = A.queue.shift(); A.t = 0; }
        else if (running) { A.phase = A.phase === 'fwd' ? 'bwd' : 'fwd'; A.t = 0; }
        else { A.phase = null; A.t = 0; }
      }
    },
  };

  /* ------------------------------------------------------------------
     Shared DOM helpers
     ------------------------------------------------------------------ */
  const $ = (id) => document.getElementById(id);
  const ui = {
    $,
    setFill(range) {
      const pct = ((range.value - range.min) / (range.max - range.min)) * 100;
      range.style.setProperty('--fill', pct + '%');
    },
    LR_MIN: 0.001, LR_MAX: 2,
    lrToSlider(lr) { return Math.round((Math.log(lr / ui.LR_MIN) / Math.log(ui.LR_MAX / ui.LR_MIN)) * 1000); },
    sliderToLr(v) { return +(ui.LR_MIN * Math.pow(ui.LR_MAX / ui.LR_MIN, v / 1000)).toPrecision(3); },

    /** Header status chip (#statusChip / #statusDot / #statusText). */
    setStatus(state) {
      const map = {
        idle: ['Idle', 'bg-slate-400', 'bg-slate-800/80 text-slate-300 border-slate-600/40', false],
        training: ['Learning…', 'bg-emerald-400 shadow-[0_0_8px_#34d399]', 'bg-emerald-400/10 text-emerald-300 border-emerald-400/40', true],
        done: ['Finished', 'bg-cyan-400 shadow-[0_0_8px_#22d3ee]', 'bg-cyan-400/10 text-cyan-300 border-cyan-400/40', false],
        error: ['Diverged', 'bg-rose-400', 'bg-rose-400/10 text-rose-300 border-rose-400/40', false],
        running: ['Animating…', 'bg-purple-400 shadow-[0_0_8px_#a855f7]', 'bg-purple-400/10 text-purple-200 border-purple-400/40', true],
      };
      const [text, dot, chip, pulse] = map[state] || map.idle;
      if ($('statusText')) $('statusText').textContent = text;
      if ($('statusDot')) $('statusDot').className = `dot ${dot} ${pulse ? 'pulse-dot' : ''}`;
      if ($('statusChip')) $('statusChip').className = `chip hidden sm:inline-flex border ${chip}`;
    },

    /** Small "Forward pass / Backpropagation / Idle" chip above the diagram. */
    updatePhaseChip(chip, phase) {
      const key = phase || 'idle';
      if (chip.dataset.phase === key) return;
      chip.dataset.phase = key;
      if (phase === 'fwd') { chip.textContent = '▶ Forward pass'; chip.className = 'chip bg-cyan-400/10 text-cyan-300 border border-cyan-400/40'; }
      else if (phase === 'bwd') { chip.textContent = '◀ Backpropagation'; chip.className = 'chip bg-rose-400/10 text-rose-300 border border-rose-400/40'; }
      else { chip.textContent = 'Idle'; chip.className = 'chip bg-slate-800/70 text-slate-400 border border-slate-600/30'; }
    },

    /** Returns log(msg, kind) bound to a scrolling log element. */
    createLogger(el, max = 160) {
      const COLORS_ = { info: 'text-slate-400', fwd: 'text-cyan-300', bwd: 'text-rose-300', ok: 'text-emerald-300', warn: 'text-amber-300', sys: 'text-purple-300' };
      return function log(msg, kind = 'info') {
        const line = document.createElement('div');
        line.className = `log-line ${COLORS_[kind] || COLORS_.info}`;
        const ts = new Date().toLocaleTimeString([], { hour12: false });
        line.innerHTML = `<span class="text-slate-600">${ts}</span> ${msg}`;
        el.appendChild(line);
        while (el.childElementCount > max) el.removeChild(el.firstChild);
        el.scrollTop = el.scrollHeight;
      };
    },

    /** Neuron Inspector: shows z = Σ w·a + b and a = g(z) for one neuron. */
    renderInspector(el, tagEl, s) {
      const { net, sel, probe, labelFor } = s;
      if (!net || !sel) return;
      const { l, j } = sel, L = net.L;
      tagEl.textContent = `${layerName(l, L)} · ${labelFor(l, j)}`;
      if (l === 0) {
        el.innerHTML = `<div><span class="text-cyan-300">${labelFor(0, j)}</span> = ${fmt(net.a[0][j], 4)}</div>
          <div class="mt-1 font-sans text-xs text-slate-400">An <b>input neuron</b> does no maths: it just holds one input number and hands it to every neuron in the next layer.</div>
          <div class="mt-2 text-slate-500">Outgoing weights: ${net.W[0].map((row, k) => `<span class="${row[j] >= 0 ? 'text-cyan-300' : 'text-purple-300'}">${fmt(row[j])}</span>→${labelFor(1, k)}`).join(', ')}</div>`;
        return;
      }
      const act = net.actOf(l), w = net.W[l - 1][j], b = net.b[l - 1][j], aPrev = net.a[l - 1];
      const terms = w.map((wi, i) => `<span class="${wi >= 0 ? 'text-cyan-300' : 'text-purple-300'}">(${fmt(wi)})</span>·<span class="text-slate-400">${fmt(aPrev[i])}</span>`);
      const shown = terms.length > 8 ? terms.slice(0, 8).join(' + ') + ` + … <span class="text-slate-500">(${terms.length} terms)</span>` : terms.join(' + ');
      const z = net.z[l][j], a = net.a[l][j];
      el.innerHTML = `
        <div class="font-sans text-xs text-slate-400 mb-2">This neuron <b class="text-slate-200">multiplies</b> each of its ${aPrev.length} incoming values by a weight, <b class="text-slate-200">adds</b> them up with its bias, then <b class="text-slate-200">squashes</b> the total with ${act.name} (${act.words}).</div>
        <div class="text-slate-500">Input x = [${probe.map((v) => fmt(v, 2)).join(', ')}]</div>
        <div class="mt-1"><span class="text-slate-500">1. weighted sum</span> z = ${shown} + <span class="text-amber-300">bias(${fmt(b)})</span></div>
        <div>&nbsp;&nbsp;&nbsp;&nbsp;= <span class="text-slate-100">${fmt(z, 4)}</span></div>
        <div class="mt-1"><span class="text-slate-500">2. activation</span> a = ${act.short}(${fmt(z, 4)}) = <span class="neon-text-emerald">${fmt(a, 4)}</span>${l < L - 1 ? ` <span class="text-slate-500">→ sent to ${net.sizes[l + 1]} neurons in ${layerName(l + 1, L)}</span>` : ''}</div>
        <div class="mt-1 text-rose-300"><span class="text-slate-500">blame from last round</span> |δ| = ${net.nodeGrad[l][j].toExponential(3)}</div>`;
    },

    /** Hover tooltip + click-to-select on the network canvas. */
    bindNetworkCanvas({ canvas, tip, wrap, getPositions, getNet, labelFor, onHover, onSelect }) {
      const hit = (ev) => {
        const rect = canvas.getBoundingClientRect();
        const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
        const pos = getPositions() || [];
        for (let l = 0; l < pos.length; l++) for (let j = 0; j < pos[l].length; j++) {
          const p = pos[l][j];
          if (Math.hypot(mx - p.x, my - p.y) <= p.r + 5) return { l, j, p };
        }
        return null;
      };
      canvas.addEventListener('mousemove', (ev) => {
        const h = hit(ev);
        onHover(h ? { l: h.l, j: h.j } : null);
        canvas.style.cursor = h ? 'pointer' : 'default';
        if (!h) { tip.style.opacity = 0; return; }
        const net = getNet(), { l, j } = h, L = net.L;
        let html = `<div class="text-cyan-300 font-semibold mb-1">${layerName(l, L)} · ${labelFor(l, j)}</div>`;
        if (l === 0) html += `<div>value = ${fmt(net.a[0][j], 4)}</div>`;
        else {
          html += `<div>z (weighted sum) = ${fmt(net.z[l][j], 4)}</div><div>a (output) = ${net.actOf(l).short}(z) = <span class="text-emerald-300">${fmt(net.a[l][j], 4)}</span></div>`;
          html += `<div>bias = <span class="text-amber-300">${fmt(net.b[l - 1][j], 4)}</span></div>`;
          html += `<div class="text-rose-300">blame |δ| = ${net.nodeGrad[l][j].toExponential(2)}</div>`;
          const ws = net.W[l - 1][j];
          html += `<div class="mt-1 text-slate-500">incoming weights (${ws.length}): ${ws.slice(0, 8).map((v) => `<span class="${v >= 0 ? 'text-cyan-300' : 'text-purple-300'}">${fmt(v, 2)}</span>`).join(' ')}${ws.length > 8 ? ' …' : ''}</div>`;
        }
        tip.innerHTML = html;
        const wr = wrap.getBoundingClientRect();
        const tw = tip.offsetWidth, th = tip.offsetHeight;
        let x = h.p.x + h.p.r + 12, y = h.p.y - th / 2;
        if (x + tw > wr.width - 8) x = h.p.x - h.p.r - 12 - tw;
        y = clamp(y, 6, wr.height - th - 6);
        tip.style.left = `${x}px`; tip.style.top = `${y}px`; tip.style.opacity = 1;
      });
      canvas.addEventListener('mouseleave', () => { onHover(null); tip.style.opacity = 0; });
      canvas.addEventListener('click', (ev) => { const h = hit(ev); if (h) onSelect({ l: h.l, j: h.j }); });
    },

    /** Space = run/pause, S = step, R = reset (ignored while typing). */
    bindShortcuts({ toggle, step, reset }) {
      global.addEventListener('keydown', (e) => {
        const tag = (e.target.tagName || '').toLowerCase();
        if (['input', 'select', 'textarea', 'button'].includes(tag) || e.metaKey || e.ctrlKey || e.altKey) return;
        if (e.code === 'Space' && toggle) { e.preventDefault(); toggle(); }
        else if ((e.key === 's' || e.key === 'S') && step) step();
        else if ((e.key === 'r' || e.key === 'R') && reset) reset();
      });
    },
  };

  /* ------------------------------------------------------------------
     Glossary: plain-English explanations for the info icons
     t = title, s = simple explanation, x = example / extra detail
     ------------------------------------------------------------------ */
  const GLOSSARY = {
    // ----- neural network basics -----
    neuralNetwork: { t: 'Neural network', s: 'A computer program made of many tiny calculators (neurons) wired together in layers. It learns from examples instead of following hand-written rules.', x: 'Loosely inspired by the brain, but really just a lot of multiply-and-add.' },
    neuron: { t: 'Neuron', s: 'A tiny calculator. It takes numbers in, multiplies each by a weight, adds them up with a bias, then squashes the result with an activation function.', x: '<b>Formula:</b> output = activation(w₁·x₁ + w₂·x₂ + … + bias)' },
    weight: { t: 'Weight', s: 'A number on each connection that says how much one neuron should listen to another. Big = important, near 0 = ignored, negative = "do the opposite".', x: 'Learning = slowly adjusting all the weights. <b>Cyan</b> lines are positive weights, <b>purple</b> are negative; thicker means stronger.' },
    bias: { t: 'Bias', s: 'An extra number each neuron adds on its own. It lets the neuron switch on more or less easily, even when all inputs are 0.', x: 'Like the y-intercept "b" in the line y = m·x + b.' },
    activation: { t: 'Activation function', s: 'A "squashing" step after the weighted sum. Without it, stacking layers would be no smarter than a single straight line.', x: 'It is what lets a network learn curves and complex shapes.' },
    sigmoid: { t: 'Sigmoid', s: 'Turns any number into a value between 0 and 1. Handy when the answer is a yes/no probability.', x: 'σ(z) = 1 / (1 + e⁻ᶻ). Big positive → near 1, big negative → near 0.' },
    tanh: { t: 'Tanh', s: 'Like sigmoid but outputs between −1 and 1, centred on 0. Often learns faster in hidden layers.', x: 'tanh(0) = 0, tanh(2) ≈ 0.96, tanh(−2) ≈ −0.96' },
    relu: { t: 'ReLU (Leaky)', s: 'Keeps positive numbers as they are and shrinks negative numbers to almost zero. Simple and very popular in big models.', x: 'Leaky ReLU keeps a tiny slope (0.01) for negatives so neurons never fully "die".' },
    linear: { t: 'Linear output', s: 'No squashing at all. Used when the answer can be any number (for example, a price or a temperature).', x: '' },
    layers: { t: 'Layers', s: 'Neurons are grouped into layers. Information always flows from one layer to the next: input → hidden → … → output.', x: '' },
    inputLayer: { t: 'Input layer', s: 'Where the data enters. Each input neuron simply holds one number, such as an x-coordinate or a 0/1 switch.', x: 'It does no maths itself.' },
    hiddenLayer: { t: 'Hidden layer', s: 'The layers in the middle. They build their own internal "features" that help solve the problem. We never tell them what to look for.', x: 'More hidden layers = a "deeper" network that can learn more complex patterns.' },
    outputLayer: { t: 'Output layer', s: 'The final layer that gives the answer, such as a probability, a class or a number.', x: '' },
    hiddenLayers: { t: 'Number of hidden layers', s: 'How many middle layers the network has. Each extra layer lets the network combine the previous layer\'s ideas into more complex ones.', x: 'Deep learning simply means "many hidden layers".' },
    neurons: { t: 'Neurons per layer', s: 'How wide each hidden layer is. More neurons = more capacity to learn, but also more weights to train.', x: '' },
    topology: { t: 'Network diagram', s: 'A picture of the network. Circles are neurons, lines are weights. Glowing circles are strongly "on" for the current input.', x: '<b>Cyan pulses</b> = the forward pass (making a guess). <b>Rose/amber pulses</b> = backpropagation (sending the error back). The rose ring around a neuron shows how much blame it got.' },
    forward: { t: 'Forward pass (forward propagation)', s: 'Making a guess: numbers flow left-to-right through the layers, each neuron does its multiply-add-squash, and the output layer gives an answer.', x: '' },
    backward: { t: 'Backpropagation', s: 'Learning from the mistake: the error at the output is sent backwards through the network so every weight finds out how much it contributed to the mistake.', x: 'Uses the chain rule from calculus. It is how almost every modern AI is trained.' },
    gradient: { t: 'Gradient', s: 'For each weight: "if I nudge this weight up a little, does the error go up or down, and by how much?" It tells us which way to change each weight.', x: 'Gradient descent = always step "downhill" on the error.' },
    loss: { t: 'Loss (error)', s: 'A single number that measures how wrong the network is. 0 would be perfect. Training tries to make it as small as possible.', x: '' },
    mse: { t: 'MSE: Mean Squared Error', s: 'Take each answer, subtract the correct answer, square it (so negatives don\'t cancel out), then average over all examples.', x: 'Guess 0.8, correct 1 → error² = 0.04' },
    lossCurve: { t: 'Loss curve', s: 'The error over time. A curve that goes down means the network is learning. The scale is logarithmic: each grid line is 10× smaller.', x: '' },
    epoch: { t: 'Epoch', s: 'One full pass over every training example. Networks usually need hundreds or thousands of epochs to learn.', x: '' },
    targetEpochs: { t: 'Target epochs', s: 'When to stop training automatically. Raise it if the network has not finished learning yet.', x: '' },
    learningRate: { t: 'Learning rate (η)', s: 'How big a step each weight takes when it learns. Too small = very slow learning. Too big = it overshoots and the error jumps around or explodes.', x: 'Try 0.5 on XOR, then 2.0 to see what "too big" looks like.' },
    momentum: { t: 'Momentum (β)', s: 'Lets learning build up speed in a consistent direction, like a ball rolling downhill. It helps deep networks train faster and smoother.', x: 'β = 0.9 keeps 90% of the previous step and adds the new one.' },
    speed: { t: 'Simulation speed', s: 'How many epochs to train per animation frame. This only changes how fast you watch it, not how the network learns.', x: '' },
    trainStep: { t: 'Train Step', s: 'Runs exactly one epoch and plays the animation slowly: the forward pass first, then backpropagation. Good for following each step.', x: '' },
    autoRun: { t: 'Auto-Run', s: 'Keeps training epoch after epoch so you can watch the network learn in real time. Press again to pause.', x: 'Shortcut: Space' },
    reset: { t: 'Reset weights', s: 'Throws away everything learned and starts again from new random weights. Different starting points can learn at different speeds.', x: '' },
    parameters: { t: 'Parameters', s: 'All the numbers the network can learn: every weight plus every bias. These are the "knobs" training turns.', x: 'Small demo: tens or hundreds. ChatGPT-class models: hundreds of billions.' },
    accuracy: { t: 'Accuracy', s: 'The share of training examples the network currently gets right. 100% = every example correct.', x: '' },
    exactMatch: { t: 'Exact-match accuracy', s: 'With several outputs, an example only counts as correct if <b>all</b> its outputs are right. "Per-bit" counts each output separately.', x: '' },
    r2: { t: 'R² score', s: 'For number-predicting (regression) tasks: how much of the pattern the model explains. 1.0 = perfect fit, 0 = no better than always guessing the average.', x: '' },
    decisionBoundary: { t: 'Decision boundary', s: 'Every point in the square is coloured by what the network would answer there. The bright edge between colours is where it changes its mind.', x: 'Dots are the training examples. A red outline means the network gets that dot wrong.' },
    regressionFit: { t: 'Regression fit', s: 'The green curve is the network\'s prediction for every x. The dashed line is the true function. The cyan dots are the noisy training data.', x: '' },
    predictionMatrix: { t: 'Prediction matrix', s: 'One row per training example: its inputs, the correct answer (target) and the network\'s answer (ŷ). Brighter green = closer to 1.', x: 'A ✓ means every output is right; a red outline marks a wrong bit.' },
    inspector: { t: 'Neuron inspector', s: 'Click any neuron in the diagram to see its exact calculation for the current input: the weighted sum, the bias and the activation.', x: '' },
    equations: { t: 'Equations', s: 'The maths behind every step, with a plain-English version next to each formula. You don\'t need to understand it to use the demo.', x: '' },
    computationLog: { t: 'Computation log', s: 'A running record of what the network just did: guesses, error values and learning updates.', x: '' },
    customInput: { t: 'Try your own input', s: 'After (or during) training, type your own input and run a forward pass to see what the network answers. This is called inference.', x: '' },
    inference: { t: 'Inference', s: 'Using a trained network to answer a new question, without learning anything. Only the forward pass runs.', x: '' },
    dataset: { t: 'Dataset', s: 'The examples the network learns from. Each example is an input together with its correct answer.', x: '' },
    xor: { t: 'XOR (exclusive OR)', s: 'Output 1 when exactly one of the two inputs is 1. No single straight line can separate the answers, so the network needs a hidden layer.', x: '0,0→0 · 0,1→1 · 1,0→1 · 1,1→0' },
    classification: { t: 'Classification', s: 'Sorting each input into a category, such as "inside the circle" or "outside the circle".', x: '' },
    regression: { t: 'Regression', s: 'Predicting a number on a smooth scale instead of a category.', x: '' },
    oneHot: { t: 'One-hot & argmax', s: 'With 3 classes we use 3 output neurons. The correct class is marked [1,0,0], [0,1,0] or [0,0,1]. The network\'s answer is whichever output is biggest (argmax).', x: '' },
    deep: { t: 'Deep network', s: 'A network with several hidden layers. Early layers learn simple patterns, later layers combine them into more complex ones, like letters → words → sentences.', x: '' },

    // ----- NLP & transformers -----
    nlp: { t: 'NLP: Natural Language Processing', s: 'The part of AI that works with human language: reading, translating, summarising and writing text.', x: 'Chatbots, translation apps, spam filters and voice assistants all use NLP.' },
    transformer: { t: 'Transformer', s: 'The neural-network design behind ChatGPT, Claude, Gemini and most modern AI. Its key trick, attention, lets every word look at every other word at once.', x: 'Introduced in the 2017 paper "Attention Is All You Need".' },
    llm: { t: 'LLM: Large Language Model', s: 'A very big transformer trained on huge amounts of text to predict the next token. Chat assistants are LLMs.', x: '' },
    tokenization: { t: 'Tokenization', s: 'Chopping text into small pieces called tokens. Computers can\'t read letters, so each token is later swapped for a number.', x: 'It is always the very first step of any language AI.' },
    token: { t: 'Token', s: 'One piece of text the model reads: a whole word, part of a word, a punctuation mark or a special marker.', x: 'Rule of thumb in English: 1 token ≈ ¾ of a word ≈ 4 characters.' },
    characters: { t: 'Characters & Unicode', s: 'Every letter, digit, space and emoji is stored in the computer as a number (its Unicode code point).', x: '"A" = 65, "a" = 97, space = 32' },
    normalization: { t: 'Normalization (clean-up)', s: 'Tidying the text before splitting it: removing extra spaces, simplifying accents and (in this demo) making everything lower-case.', x: 'Many real models keep capital letters, because "Apple" and "apple" can mean different things.' },
    pretokenization: { t: 'Pre-tokenization', s: 'A first rough split at spaces and punctuation, so we know where words begin and end.', x: '"Hello, world!" → Hello | , | world | !' },
    subword: { t: 'Subword tokenization', s: 'Rare or long words are broken into smaller, common pieces. The model can then handle any word, even ones it has never seen.', x: '"unhappiness" → un + ##happi + ##ness. The "##" means "glued to the previous piece". Real methods: BPE (GPT), WordPiece (BERT), SentencePiece.' },
    vocabulary: { t: 'Vocabulary', s: 'The fixed list of every token the model knows. Each one has its own ID number.', x: 'This demo knows about 670 tokens. GPT-4-class models know about 100,000–200,000.' },
    tokenId: { t: 'Token ID', s: 'The position of a token in the vocabulary list. From here on, the model only sees these numbers, not the text.', x: '' },
    specialTokens: { t: 'Special tokens', s: 'Marker tokens that aren\'t real text, such as [START] (beginning of text) or [UNK] (an unknown symbol).', x: 'Real models use names such as <|endoftext|>, [CLS], [SEP] or <s>.' },
    unknownToken: { t: '[UNK]: unknown token', s: 'Used when a symbol isn\'t in the vocabulary at all, such as an emoji in this tiny demo. Modern tokenizers fall back to raw bytes, so they almost never need it.', x: '' },
    embedding: { t: 'Embedding', s: 'Each token ID is swapped for a list of numbers (a vector) that captures its meaning. Words with similar meanings get similar vectors.', x: 'The vectors are learned during training. The famous example: king − man + woman ≈ queen.' },
    dimensions: { t: 'Vector dimensions', s: 'How many numbers are in each token\'s vector. Each number captures some aspect of meaning.', x: 'This demo uses 8. Real models use 768 to more than 12,000.' },
    meaningMap: { t: 'Meaning map', s: 'The 8 numbers of each token squashed down to 2, so we can draw them. Tokens close together have similar meanings.', x: '' },
    positionalEncoding: { t: 'Positional encoding', s: 'Attention by itself doesn\'t know word order ("dog bites man" vs "man bites dog"). So we add a unique wave pattern for each position to every token\'s vector.', x: 'The original transformer uses sine and cosine waves of different speeds.' },
    attention: { t: 'Self-attention', s: 'Every token looks at the other tokens and decides which ones matter for understanding it. It then mixes in information from them.', x: 'In "The animal didn\'t cross the street because <b>it</b> was tired", attention helps "it" link to "animal".' },
    qkv: { t: 'Query, Key, Value', s: '<b>Query</b> = "what am I looking for?" · <b>Key</b> = "what do I contain?" · <b>Value</b> = "what I share if you pick me". A match between a query and a key gives a high score.', x: 'score = (Query · Key) / √d, then softmax turns the scores into percentages.' },
    softmax: { t: 'Softmax', s: 'Turns a list of scores into percentages that add up to 100%. Bigger scores get a much bigger share.', x: 'scores [2, 1, 0] → [67%, 24%, 9%]' },
    multiHead: { t: 'Multi-head attention', s: 'Several attention "heads" run side by side, each looking for a different kind of relationship (meaning, grammar, nearby words…). Their results are combined.', x: 'Large models use dozens of heads in each layer.' },
    causalMask: { t: 'Causal mask', s: 'GPT-style models generate text left to right, so a token may only look at itself and earlier tokens, never future ones. Turn it off to see BERT-style "look both ways" attention.', x: '' },
    residual: { t: 'Residual connection ("add")', s: 'The original vector is added back after each step, so information is never lost and deep stacks stay easy to train.', x: 'output = input + change' },
    layerNorm: { t: 'Layer normalization', s: 'Rescales each token\'s vector so its numbers stay in a healthy range (average 0, spread 1). It keeps training stable.', x: '' },
    feedForward: { t: 'Feed-forward network', s: 'After attention, each token\'s vector goes through a small neural network on its own, exactly like the networks on the other two pages. This is where much of the model\'s "knowledge" lives.', x: 'Usually it widens the vector to 4× its size (here 8 → 16), applies ReLU, then shrinks it back.' },
    transformerBlock: { t: 'Transformer block', s: 'One round of: attention → add & normalize → feed-forward → add & normalize. Real models stack many blocks on top of each other.', x: 'GPT-2 small: 12 blocks. Big modern models: around 100.' },
    logits: { t: 'Logits', s: 'Raw scores the model gives every word in the vocabulary for "what comes next". Softmax turns them into probabilities.', x: '' },
    nextToken: { t: 'Next-token prediction', s: 'A language model\'s only real job: given the text so far, guess the probability of every possible next token.', x: 'Chatbots write whole answers by repeating this one token at a time.' },
    temperature: { t: 'Temperature', s: 'Controls how adventurous the choice is. Low (0.3) = almost always the top word (safe, repetitive). High (1.5) = more random and creative.', x: '' },
    autoregressive: { t: 'Autoregressive generation', s: 'Add the chosen word to the text, then run the whole model again to predict the next one. Repeat. This is how AI writes sentences.', x: '' },
    contextWindow: { t: 'Context window', s: 'The maximum number of tokens a model can look at in one go.', x: 'Modern models handle anywhere from thousands to over a million tokens.' },
    toyModel: { t: 'Why a "toy" model?', s: 'A real transformer learns its numbers by reading billions of sentences. Here the numbers are small and partly hand-made so the patterns are easy to see. Every step does the same kind of maths a real model does, just smaller.', x: '' },
  };

  /* ------------------------------------------------------------------
     Info icons: <button class="info-i" data-info="key"></button>
     ------------------------------------------------------------------ */
  const Info = {
    dict: GLOSSARY,
    pop: null, current: null, pinned: false, hideTimer: null,
    /** HTML for an inline info icon. */
    icon(key, extraClass = '') {
      const g = GLOSSARY[key];
      return `<button type="button" class="info-i ${extraClass}" data-info="${key}" aria-label="What is ${g ? escapeHtml(g.t) : key}?">i</button>`;
    },
    init(extra = {}) {
      Object.assign(GLOSSARY, extra);
      if (!this.pop) {
        this.pop = document.createElement('div');
        this.pop.id = 'infoPop';
        this.pop.setAttribute('role', 'tooltip');
        document.body.appendChild(this.pop);
        this.pop.addEventListener('mouseenter', () => clearTimeout(this.hideTimer));
        this.pop.addEventListener('mouseleave', () => { if (!this.pinned) this.hide(); });
      }
      this.decorate(document);
      const find = (e) => e.target.closest && e.target.closest('.info-i');
      document.addEventListener('mouseover', (e) => { const b = find(e); if (b && !this.pinned) this.show(b); });
      document.addEventListener('mouseout', (e) => { const b = find(e); if (b && !this.pinned) this.hideSoon(); });
      document.addEventListener('focusin', (e) => { const b = find(e); if (b) this.show(b); });
      document.addEventListener('focusout', (e) => { const b = find(e); if (b && !this.pinned) this.hideSoon(); });
      document.addEventListener('click', (e) => {
        const b = find(e);
        if (b) {
          e.preventDefault(); e.stopPropagation();
          if (this.pinned && this.current === b) { this.pinned = false; this.hide(); }
          else { this.pinned = true; this.show(b); }
        } else if (this.pinned && !this.pop.contains(e.target)) { this.pinned = false; this.hide(); }
      }, true);
      document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { this.pinned = false; this.hide(); } });
      global.addEventListener('scroll', () => { if (this.current && this.pop.classList.contains('show')) this.place(this.current); }, { passive: true });
    },
    /** Give every icon an accessible label and the "i" glyph. */
    decorate(root) {
      root.querySelectorAll('.info-i').forEach((b) => {
        if (!b.textContent.trim()) b.textContent = 'i';
        if (!b.getAttribute('type')) b.setAttribute('type', 'button');
        const g = GLOSSARY[b.dataset.info];
        if (g && !b.getAttribute('aria-label')) b.setAttribute('aria-label', `What is ${g.t}?`);
      });
    },
    show(btn) {
      const g = GLOSSARY[btn.dataset.info];
      if (!g) return;
      clearTimeout(this.hideTimer);
      if (this.current && this.current !== btn) this.current.setAttribute('aria-expanded', 'false');
      this.current = btn;
      btn.setAttribute('aria-expanded', 'true');
      this.pop.innerHTML = `<div class="ip-title">${g.t}</div><div>${g.s}</div>${g.x ? `<div class="ip-ex">${g.x}</div>` : ''}`;
      this.pop.classList.add('show');
      this.place(btn);
    },
    place(btn) {
      const r = btn.getBoundingClientRect();
      const pw = this.pop.offsetWidth, ph = this.pop.offsetHeight;
      let x = r.left + r.width / 2 - pw / 2;
      x = clamp(x, 12, global.innerWidth - pw - 12);
      let y = r.bottom + 8;
      if (y + ph > global.innerHeight - 8) y = r.top - ph - 8;
      this.pop.style.left = `${x}px`; this.pop.style.top = `${Math.max(8, y)}px`;
    },
    hideSoon() { clearTimeout(this.hideTimer); this.hideTimer = setTimeout(() => this.hide(), 160); },
    hide() {
      this.pop.classList.remove('show');
      if (this.current) this.current.setAttribute('aria-expanded', 'false');
      this.current = null;
    },
  };

  global.NN = {
    util: { clamp, lerp, fmt, fmtLoss, easeInOut, easeOut, argmax, subscript, escapeHtml, sleep, softmax, prefersReducedMotion },
    rand: { mulberry32, gaussian, hashString },
    ACT, MLP,
    render: { COLORS, CLASS_COLORS, rgba, layerColor, layerName, fitCanvas, computeLayout, drawPulse, drawNetwork, drawLoss, drawProbe, drawRegions },
    anim, ui, GLOSSARY, Info,
  };
})(window);
