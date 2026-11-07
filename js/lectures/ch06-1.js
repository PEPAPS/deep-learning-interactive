/* =========================================================
   Lecture 4 · Ch 6.1–6.4 · Deep Feedforward Networks
   ========================================================= */
'use strict';

const ACTS = {
  relu: { name: 'ReLU', f: z => Math.max(0, z), d: z => (z > 0 ? 1 : 0), tex: 'g(z) = max{0, z}', note: 'Default choice. Slope exactly 1 when active, 0 when inactive (a “dead” unit gets no gradient).' },
  leaky: { name: 'Leaky ReLU', f: z => (z > 0 ? z : 0.01 * z), d: z => (z > 0 ? 1 : 0.01), tex: 'g(z) = max(0, z) + 0.01·min(0, z)', note: 'Small slope α = 0.01 for z < 0, so the unit never completely stops learning.' },
  abs: { name: 'Absolute value', f: z => Math.abs(z), d: z => (z > 0 ? 1 : -1), tex: 'g(z) = |z|  (α = −1)', note: 'Used for images where a feature should not care about reversing light/dark polarity.' },
  prelu: { name: 'PReLU (α = 0.25)', f: z => (z > 0 ? z : 0.25 * z), d: z => (z > 0 ? 1 : 0.25), tex: 'g(z) = max(0, z) + α·min(0, z), α learned', note: 'Parametric ReLU: the negative slope α is a parameter learned by gradient descent (0.25 shown).' },
  maxout: { name: 'Maxout (k = 3)', f: z => Math.max(-0.5 * z - 0.5, 0.3 * z, 1.4 * z - 1.2), d: z => { const v = [-0.5 * z - 0.5, 0.3 * z, 1.4 * z - 1.2]; return [-0.5, 0.3, 1.4][v.indexOf(Math.max(...v))]; }, tex: 'g(z)ᵢ = max_{j∈𝔾⁽ⁱ⁾} zⱼ  (here 3 linear pieces)', note: 'Takes the max over a group of k linear functions → learns a convex, piecewise-linear activation itself.' },
  sigmoid: { name: 'Logistic sigmoid', f: z => sigmoid(z), d: z => sigmoid(z) * (1 - sigmoid(z)), tex: 'σ(z) = 1 / (1 + e⁻ᶻ)', note: 'Saturates on both sides (slope → 0); max slope only 0.25. Discouraged as a hidden unit.' },
  tanh: { name: 'tanh', f: z => Math.tanh(z), d: z => 1 - Math.tanh(z) ** 2, tex: 'tanh(z) = 2σ(2z) − 1', note: 'Also saturates, but tanh(0) = 0 and slope 1 at 0 — behaves like the identity for small inputs, so it trains better than σ.' },
  softplus: { name: 'Softplus', f: z => Math.log1p(Math.exp(-Math.abs(z))) + Math.max(z, 0), d: z => sigmoid(z), tex: 'ζ(z) = log(1 + eᶻ)', note: 'A smooth ReLU. Surprisingly works worse than ReLU in practice — hidden-unit performance is hard to predict.' },
  hardtanh: { name: 'Hard tanh', f: z => Math.max(-1, Math.min(1, z)), d: z => (Math.abs(z) < 1 ? 1 : 0), tex: 'g(z) = max(−1, min(1, z))', note: 'Like tanh but piecewise linear; bounded, unlike the rectifier.' },
  rbf: { name: 'RBF (template 0, σ = 1)', f: z => Math.exp(-0.5 * z * z), d: z => -z * Math.exp(-0.5 * z * z), tex: 'hᵢ = exp(−‖W:,ᵢ − x‖² / σᵢ²)', note: 'Most active when the input is near a template; saturates to 0 almost everywhere → hard to optimize.' },
  cos: { name: 'cos', f: z => Math.cos(z), d: z => -Math.sin(z), tex: 'h = cos(Wx + b)', note: 'An “unusual” unit that still reached < 1% error on MNIST: many differentiable functions work fine.' }
};

/* ---------- 1) A live MLP forward pass ---------- */
Widgets.mlpDiagram = (el) => {
  UI.head(el, 'Information flows forward through layers', 'A small network with random weights. Move the inputs and watch every unit’s value change. Click a unit to see its calculation.');
  let L = 2, width = 4, act = 'relu', x = [0.8, -0.5], seed = 3, sel = [1, 0];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Input x₁', min: -2, max: 2, step: 0.05, value: x[0], fmt: v => v.toFixed(2), onInput: v => { x[0] = v; draw(); } });
  UI.slider(ctr, { label: 'Input x₂', min: -2, max: 2, step: 0.05, value: x[1], fmt: v => v.toFixed(2), onInput: v => { x[1] = v; draw(); } });
  UI.slider(ctr, { label: 'Hidden layers (depth)', min: 1, max: 3, step: 1, value: L, fmt: v => String(v), onInput: v => { L = v; build(); draw(); } });
  UI.slider(ctr, { label: 'Units per layer (width)', min: 2, max: 6, step: 1, value: width, fmt: v => String(v), onInput: v => { width = v; build(); draw(); } });
  UI.select(ctr, 'Activation g', [['relu', 'ReLU'], ['tanh', 'tanh'], ['sigmoid', 'sigmoid']], act, v => { act = v; draw(); });
  const btns = h('div', { class: 'btns' }); UI.button(btns, 'New random weights', () => { seed++; build(); draw(); }); ctr.append(btns);
  const svgBox = h('div');
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, svgBox, calc);
  let Ws = [], bs = [];
  function build() {
    const rng = mulberry32(seed * 17);
    const sizes = [2, ...Array(L).fill(width), 1];
    Ws = []; bs = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      Ws.push(Array.from({ length: sizes[l + 1] }, () => Array.from({ length: sizes[l] }, () => gaussian(rng) * Math.sqrt(2 / sizes[l]))));
      bs.push(Array.from({ length: sizes[l + 1] }, () => 0.1 * gaussian(rng)));
    }
    sel = [1, 0];
  }
  function forward() {
    const g = ACTS[act].f;
    const hs = [x.slice()], zs = [null];
    Ws.forEach((W, l) => {
      const z = W.map((row, j) => dotv(row, hs[l]) + bs[l][j]);
      zs.push(z);
      hs.push(l === Ws.length - 1 ? z : z.map(g));
    });
    return { hs, zs };
  }
  function draw() {
    const c = Colors.get();
    const { hs, zs } = forward();
    const ns = 'http://www.w3.org/2000/svg', Wd = 600, Hd = 260;
    const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${Wd} ${Hd}`); svg.setAttribute('class', 'diagram');
    const mk = (t, a, txt) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.append(e); return e; };
    const cols = hs.length, xs = hs.map((_, k) => 50 + k * (Wd - 100) / (cols - 1));
    const ys = hs.map(v => v.map((_, i) => Hd / 2 + (i - (v.length - 1) / 2) * 40));
    Ws.forEach((W, l) => W.forEach((row, j) => row.forEach((w, i) => {
      mk('line', { x1: xs[l], y1: ys[l][i], x2: xs[l + 1], y2: ys[l + 1][j], stroke: w > 0 ? c.s1 : c.s8, 'stroke-width': Math.min(4, 0.5 + Math.abs(w) * 1.5), opacity: 0.55 });
    })));
    hs.forEach((v, l) => v.forEach((a, i) => {
      const isSel = sel[0] === l && sel[1] === i;
      const fillC = a >= 0 ? c.s1 : c.s8, op = Math.min(1, Math.abs(a) / 2);
      const circ = mk('circle', { cx: xs[l], cy: ys[l][i], r: 16, fill: c.surface, stroke: isSel ? c.text : c.axis, 'stroke-width': isSel ? 3 : 1.5, style: 'cursor:pointer' });
      mk('circle', { cx: xs[l], cy: ys[l][i], r: 14, fill: fillC, 'fill-opacity': op, style: 'pointer-events:none' });
      mk('text', { x: xs[l], y: ys[l][i] + 4, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, fill: c.text, style: 'pointer-events:none' }, a.toFixed(2));
      if (l > 0) circ.addEventListener('click', () => { sel = [l, i]; draw(); });
    }));
    hs.forEach((_, l) => mk('text', { x: xs[l], y: 18, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: c.text2 }, l === 0 ? 'input x' : l === hs.length - 1 ? 'output ŷ' : `h⁽${l}⁾ = f⁽${l}⁾`));
    svgBox.innerHTML = ''; svgBox.append(svg);
    const [l, j] = sel, row = Ws[l - 1][j], prev = hs[l - 1], z = zs[l][j];
    const params = Ws.reduce((s, W) => s + W.length * W[0].length + W.length, 0);
    calc.innerHTML = `<span class="c">selected unit: layer ${l}, unit ${j + 1}  — every unit computes  g(Σᵢ wᵢ·hᵢ + b)</span>
z = ${row.map((w, i) => `(${w.toFixed(2)})(${prev[i].toFixed(2)})`).join(' + ')} + (${bs[l - 1][j].toFixed(2)}) = <b>${z.toFixed(3)}</b>
${l === hs.length - 1 ? 'output layer is linear: ŷ = z = <b>' + z.toFixed(3) + '</b>' : `h = ${act}(z) = <b>${hs[l][j].toFixed(3)}</b>`}
<span class="c">depth = ${L + 1} layers of weights; parameters = Σ (inputs × outputs + outputs) = <b>${params}</b>;  f(x) = f⁽${L + 1}⁾(…f⁽²⁾(f⁽¹⁾(x)))</span>`;
  }
  build(); draw();
  return { redraw: draw };
};

/* tiny 1-hidden-layer MLP trained by full-batch gradient descent */
function trainMLP(X, Y, opt) {
  const { hidden, steps, lr, seed, act = 'relu', out = 'sigmoid', wScale = 1, b0 = 0.1, record = 0 } = opt;
  const rng = mulberry32(seed);
  const nIn = X[0].length;
  const W = Array.from({ length: hidden }, () => Array.from({ length: nIn }, () => gaussian(rng) * wScale * Math.sqrt(2 / nIn)));
  const c = Array(hidden).fill(b0);
  const w = Array.from({ length: hidden }, () => gaussian(rng) * wScale * Math.sqrt(1 / hidden));
  let b = 0;
  const g = ACTS[act];
  const curve = [];
  const fwd = x => { const z = W.map((r, j) => dotv(r, x) + c[j]); const hh = z.map(g.f); const o = dotv(w, hh) + b; return { z, hh, o, p: out === 'sigmoid' ? sigmoid(o) : o }; };
  const loss = () => mean(X.map((x, i) => { const p = fwd(x).p; return out === 'sigmoid' ? -(Y[i] * Math.log(p + 1e-12) + (1 - Y[i]) * Math.log(1 - p + 1e-12)) : (p - Y[i]) ** 2; }));
  for (let t = 0; t < steps; t++) {
    const gW = W.map(r => r.map(() => 0)), gc = Array(hidden).fill(0), gw = Array(hidden).fill(0); let gb = 0;
    X.forEach((x, i) => {
      const { z, hh, p } = fwd(x);
      const dO = out === 'sigmoid' ? (p - Y[i]) : 2 * (p - Y[i]);
      gb += dO;
      for (let j = 0; j < hidden; j++) {
        gw[j] += dO * hh[j];
        const dz = dO * w[j] * g.d(z[j]);
        gc[j] += dz;
        for (let k = 0; k < nIn; k++) gW[j][k] += dz * x[k];
      }
    });
    const m = X.length;
    for (let j = 0; j < hidden; j++) { w[j] -= lr * gw[j] / m; c[j] -= lr * gc[j] / m; for (let k = 0; k < nIn; k++) W[j][k] -= lr * gW[j][k] / m; }
    b -= lr * gb / m;
    if (record && t % record === 0) curve.push([t, loss()]);
  }
  return { W, c, w, b, fwd, loss: loss(), curve };
}

/* ---------- 2) Choosing the feature map φ ---------- */
Widgets.phiFeatures = (el) => {
  UI.head(el, 'A linear model on top of features φ(x)', 'Two classes: an inner disk and a ring. A linear classifier (logistic regression) is trained on top of three different feature maps.');
  const rng = mulberry32(31);
  const pts = [];
  for (let i = 0; i < 45; i++) { const r = 0.55 * Math.sqrt(rng()), t = 2 * Math.PI * rng(); pts.push([r * Math.cos(t), r * Math.sin(t), 0]); }
  for (let i = 0; i < 45; i++) { const r = 0.9 + 0.45 * rng(), t = 2 * Math.PI * rng(); pts.push([r * Math.cos(t), r * Math.sin(t), 1]); }
  let mode = 'raw';
  const segBox = h('div');
  UI.seg(segBox, [['raw', '1 · φ(x) = x (no features)'], ['hand', '2 · hand-made φ = (x₁², x₂²)'], ['learn', '3 · learned φ (8 ReLU units)']], mode, v => { mode = v; draw(); });
  const two = h('div', { class: 'row2' });
  const w1 = h('div', { class: 'canvas-wrap' }), w2 = h('div', { class: 'canvas-wrap' });
  const c1 = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Input space with decision regions' }), c2 = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Feature space' });
  w1.append(h('div', { class: 'note' }, 'input space x'), c1); w2.append(h('div', { class: 'note' }, 'feature space φ(x)'), c2); two.append(w1, w2);
  const out = h('div');
  el.append(segBox, two, out);
  const cache = {};
  function model(m) {
    if (cache[m]) return cache[m];
    let r;
    if (m === 'learn') {
      const net = trainMLP(pts.map(p => [p[0], p[1]]), pts.map(p => p[2]), { hidden: 8, steps: 1500, lr: 0.5, seed: 4 });
      r = { prob: (x, y) => net.fwd([x, y]).p, feat: (x, y) => { const hh = net.fwd([x, y]).hh; return [hh[0], hh[1]]; } };
    } else {
      const phi = m === 'raw' ? ((x, y) => [x, y]) : ((x, y) => [x * x, y * y]);
      let w = [0, 0], b = 0;
      for (let t = 0; t < 2000; t++) {
        let g0 = 0, g1 = 0, gb = 0;
        pts.forEach(([x, y, l]) => { const f = phi(x, y), e = sigmoid(w[0] * f[0] + w[1] * f[1] + b) - l; g0 += e * f[0]; g1 += e * f[1]; gb += e; });
        w[0] -= 1.0 * g0 / pts.length; w[1] -= 1.0 * g1 / pts.length; b -= 1.0 * gb / pts.length;
      }
      r = { prob: (x, y) => { const f = phi(x, y); return sigmoid(w[0] * f[0] + w[1] * f[1] + b); }, feat: phi, w, b };
    }
    cache[m] = r; return r;
  }
  function draw() {
    const c = Colors.get();
    const M = model(mode);
    const W = boxWidth(w1), Hh = W;
    let ctx = fitCanvas(c1, W, Hh);
    let P = makePlot(W, Hh, [-1.5, 1.5], [-1.5, 1.5], { l: 4, r: 4, t: 4, b: 4 });
    const G = 50;
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      const x = -1.5 + (i + .5) * 3 / G, y = -1.5 + (j + .5) * 3 / G;
      ctx.fillStyle = alpha(M.prob(x, y) > 0.5 ? c.s2 : c.s1, .18);
      ctx.fillRect(P.X(x - 1.5 / G), P.Y(y + 1.5 / G), W / G + 1, Hh / G + 1);
    }
    pts.forEach(([x, y, l]) => dot(ctx, P.X(x), P.Y(y), 3.5, l ? c.s2 : c.s1, c.surface));
    const F = pts.map(([x, y, l]) => [...M.feat(x, y), l]);
    const fx = F.map(f => f[0]), fy = F.map(f => f[1]);
    const xr = [Math.min(...fx), Math.max(...fx)], yr = [Math.min(...fy), Math.max(...fy)];
    const pad = (r) => { const d = (r[1] - r[0]) * 0.1 + 1e-3; return [r[0] - d, r[1] + d]; };
    ctx = fitCanvas(c2, W, Hh);
    P = makePlot(W, Hh, pad(xr), pad(yr), { l: 30, b: 22 });
    drawAxes(ctx, P, { nx: 4, ny: 4 });
    F.forEach(([a, b2, l]) => dot(ctx, P.X(a), P.Y(b2), 3.5, l ? c.s2 : c.s1, c.surface));
    if (M.w) {
      const [w0, w1v] = M.w, bb = M.b;
      ctx.save(); P.clip(ctx); ctx.strokeStyle = c.text; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
      const X0 = P.xr[0], X1 = P.xr[1];
      if (Math.abs(w1v) > 1e-6) { ctx.beginPath(); ctx.moveTo(P.X(X0), P.Y(-(w0 * X0 + bb) / w1v)); ctx.lineTo(P.X(X1), P.Y(-(w0 * X1 + bb) / w1v)); ctx.stroke(); }
      ctx.restore();
    }
    const acc = pts.filter(([x, y, l]) => (M.prob(x, y) > 0.5 ? 1 : 0) === l).length / pts.length;
    out.innerHTML = '';
    out.append(h('div', { class: 'callout ' + (acc > 0.95 ? 'good' : 'bad') }, h('span', { class: 'ico' }, acc > 0.95 ? '✓' : '✗'), h('div', {}, `Training accuracy ${(acc * 100).toFixed(0)}%. `,
      { raw: 'A straight line can’t separate a disk from a ring — linear models can’t capture interactions between inputs.', hand: 'With hand-designed features (x₁², x₂²) the classes become linearly separable (dashed line). But someone had to know to square the inputs.', learn: 'The hidden layer learned its own features from data (two of the 8 learned features are plotted). No human design needed — this is the deep-learning strategy.' }[mode])));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) XOR, step by step ---------- */
Widgets.xorNet = (el) => {
  UI.head(el, 'Solving XOR with one hidden layer', 'Inputs (0,0),(0,1),(1,0),(1,1) → targets 0,1,1,0. Step through the network and watch the hidden layer move the points so that a line can separate them.');
  const X = [[0, 0], [0, 1], [1, 0], [1, 1]], Y = [0, 1, 1, 0];
  let mode = 'book', stage = 4, seed = 2;
  const segBox = h('div');
  UI.seg(segBox, [['book', 'The textbook solution'], ['linear', 'Linear model only'], ['relu-off', 'Hidden layer without ReLU'], ['trained', 'Trained by gradient descent']], mode, v => { mode = v; draw(); });
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Show computation up to step', min: 1, max: 4, step: 1, value: stage, fmt: v => ['', '1 · XW', '2 · XW + c', '3 · ReLU', '4 · output'][v], onInput: v => { stage = v; draw(); } });
  const btns = h('div', { class: 'btns' }); UI.button(btns, 'Train again (new random start)', () => { seed++; trained = null; mode = 'trained'; draw(); }); ctr.append(btns);
  const two = h('div', { class: 'row2' });
  const w1 = h('div', { class: 'canvas-wrap' }), w2 = h('div', { class: 'canvas-wrap' });
  const c1 = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Input space' }), c2 = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Hidden space' });
  w1.append(h('div', { class: 'note' }, 'original x space'), c1); w2.append(h('div', { class: 'note' }, 'learned h space (after the hidden layer)'), c2); two.append(w1, w2);
  const calc = h('div', { class: 'calcbox' });
  const out = h('div');
  el.append(segBox, ctr, two, calc, out);
  let trained = null;
  function params() {
    if (mode === 'book') return { W: [[1, 1], [1, 1]], c: [0, -1], w: [1, -2], b: 0, relu: true };
    if (mode === 'relu-off') return { W: [[1, 1], [1, 1]], c: [0, -1], w: [1, -2], b: 0, relu: false };
    if (mode === 'linear') return null;
    if (!trained) {
      let best = null;
      for (let s = 0; s < 6 && !best; s++) { const n = trainMLP(X, Y, { hidden: 2, steps: 4000, lr: 0.2, seed: seed * 10 + s, out: 'linear' }); if (n.loss < 1e-3) best = n; if (!best && s === 5) best = n; }
      trained = best;
    }
    return { W: [[trained.W[0][0], trained.W[1][0]], [trained.W[0][1], trained.W[1][1]]], c: trained.c, w: trained.w, b: trained.b, relu: true };
  }
  function draw() {
    const c = Colors.get();
    const p = params();
    const f2 = v => v.map(n => fmt(n, 2)).join(', ');
    let H = null, preds, lines;
    if (!p) {
      preds = X.map(() => 0.5);
      lines = `<span class="c">linear model f(x) = xᵀw + b, solved with the normal equations:</span>
w = (0, 0), b = 0.5  →  output 0.5 for every input.  MSE = (0.5²·4)/4 = <b>0.25</b>  (best any line can do)`;
    } else {
      const XW = X.map(x => [x[0] * p.W[0][0] + x[1] * p.W[1][0], x[0] * p.W[0][1] + x[1] * p.W[1][1]]);
      const XWc = XW.map(r => [r[0] + p.c[0], r[1] + p.c[1]]);
      const Hh = XWc.map(r => p.relu ? r.map(v => Math.max(0, v)) : r);
      preds = Hh.map(r => r[0] * p.w[0] + r[1] * p.w[1] + p.b);
      H = [XW, XWc, Hh][Math.min(stage, 3) - 1];
      lines = `<span class="c">W = [[${f2(p.W[0])}], [${f2(p.W[1])}]],  c = (${f2(p.c)}),  w = (${f2(p.w)}),  b = ${fmt(p.b, 2)}</span>
1 · XW      = ${XW.map(r => '[' + f2(r) + ']').join(' ')}
2 · XW + c  = ${XWc.map(r => '[' + f2(r) + ']').join(' ')}
3 · ${p.relu ? 'max{0, ·}' : '(no ReLU)'} = ${Hh.map(r => '[' + f2(r) + ']').join(' ')}
4 · h·w + b = <b>${preds.map(v => fmt(v, 2)).join(', ')}</b>   targets 0, 1, 1, 0   MSE = ${fmt(mean(preds.map((v, i) => (v - Y[i]) ** 2)), 4)}`;
    }
    calc.innerHTML = lines;
    const W = boxWidth(w1);
    const drawSpace = (cv, pts, ttl, showLine) => {
      const ctx = fitCanvas(cv, W, W);
      const all = pts.flat(); const lo = Math.min(-0.3, ...all) - 0.2, hi = Math.max(1.3, ...all) + 0.2;
      const P = makePlot(W, W, [lo, hi], [lo, hi], { l: 26, b: 22 });
      drawAxes(ctx, P, { nx: 4, ny: 4 });
      if (showLine && p && stage >= 3) {
        ctx.save(); P.clip(ctx); ctx.strokeStyle = c.text; ctx.setLineDash([5, 4]); ctx.lineWidth = 2;
        const [w0, w1v] = p.w, k = 0.5 - p.b;
        if (Math.abs(w1v) > 1e-6) { ctx.beginPath(); ctx.moveTo(P.X(lo), P.Y((k - w0 * lo) / w1v)); ctx.lineTo(P.X(hi), P.Y((k - w0 * hi) / w1v)); ctx.stroke(); }
        ctx.restore();
      }
      pts.forEach((q, i) => { dot(ctx, P.X(q[0]), P.Y(q[1]), 8, Y[i] ? c.s2 : c.s1, c.surface); label(ctx, `${X[i].join('')}→${Y[i]}`, P.X(q[0]) + 10, P.Y(q[1]) - 10, { color: c.text, size: 10, bold: true }); });
    };
    drawSpace(c1, X, 'x', false);
    drawSpace(c2, H || X, 'h', true);
    out.innerHTML = '';
    const ok = preds.every((v, i) => Math.abs(v - Y[i]) < 0.05);
    out.append(h('div', { class: 'callout ' + (ok ? 'good' : 'bad') }, h('span', { class: 'ico' }, ok ? '✓' : '✗'), h('div', {},
      ok ? 'All four outputs are correct. In h-space the two “1” points land on the same spot and the dashed line wᵀh + b = 0.5 separates the classes — the hidden layer created a representation where a linear output layer works.'
        : mode === 'relu-off' ? 'Without the nonlinearity, the two layers collapse into one linear function (xᵀW w), which cannot represent XOR.' : 'A linear model outputs 0.5 everywhere: XOR is not linearly separable.')));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) Sensitivity to the initial weights ---------- */
Widgets.xorInits = (el) => {
  UI.head(el, 'Same network, different starting points', 'Ten runs of gradient descent on XOR (2 hidden ReLU units, MSE), each from a different random initialization. Non-convex loss → the result depends on the start.');
  let wScale = 1, b0 = 0.1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Initial weight scale', min: 0.1, max: 3, step: 0.1, value: wScale, fmt: v => v.toFixed(1) + '×', onInput: v => { wScale = v; draw(); } });
  UI.select(ctr, 'Initial hidden biases', [['0.1', 'small positive (0.1)'], ['0', 'zero'], ['-0.5', 'negative (−0.5)']], '0.1', v => { b0 = +v; draw(); });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 3000], yr: [1e-5, 1], logY: true, xlabel: 'gradient-descent step', ylabel: 'MSE', xname: 'step', aspect: .5, maxH: 280 });
  const out = h('div');
  el.append(out);
  function draw() {
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]], Y = [0, 1, 1, 0];
    const runs = Array.from({ length: 10 }, (_, s) => trainMLP(X, Y, { hidden: 2, steps: 3000, lr: 0.2, seed: 100 + s, out: 'linear', wScale, b0, record: 50 }));
    chart.set(runs.map((r, i) => ({ name: 'run ' + (i + 1), color: r.loss < 1e-3 ? 's3' : 's8', data: r.curve.map(([t, l]) => [t, Math.max(1.1e-5, l)]), width: 1.6 })));
    const ok = runs.filter(r => r.loss < 1e-3).length;
    out.innerHTML = '';
    out.append(UI.legend(h('div'), [{ label: `solved XOR (${ok} runs)`, color: Colors.get().s3 }, { label: `stuck at a worse point (${10 - ok} runs) — e.g. a “dead” ReLU unit`, color: Colors.get().s8 }]).el,
      h('div', { class: 'callout info', style: 'margin-top:6px' }, h('span', { class: 'ico' }, 'i'), h('div', {}, 'This is why the course recommends small random weights and small positive biases: they keep ReLU units active at the start so gradients can flow. Try zero or negative biases.')));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Output unit ↔ distribution ↔ loss ---------- */
Widgets.outputPicker = (el, ds) => {
  UI.head(el, 'Pick the output type → get the unit and the loss', 'The task decides p(y|x); maximum likelihood then decides the loss. Notice the gradient w.r.t. the pre-activation is always <b>prediction − target</b>.');
  let kind = ds.kind || 'binary', z = 1.2, yT = 0, zs = [2, 0.5, -1], k = 0, yR = 1.5;
  const segBox = h('div');
  UI.seg(segBox, [['real', 'Real number'], ['binary', 'Yes / no'], ['multi', 'One of k classes']], kind, v => { kind = v; build(); });
  const ctr = h('div', { class: 'controls' });
  const out = h('div', { class: 'stack', style: 'gap:8px' });
  el.append(segBox, ctr, out);
  function build() {
    ctr.innerHTML = '';
    if (kind === 'real') {
      UI.slider(ctr, { label: 'Network output ŷ = Wᵀh + b', min: -3, max: 4, step: 0.05, value: z, fmt: v => v.toFixed(2), onInput: v => { z = v; draw(); } });
      UI.slider(ctr, { label: 'True target y', min: -3, max: 4, step: 0.05, value: yR, fmt: v => v.toFixed(2), onInput: v => { yR = v; draw(); } });
    } else if (kind === 'binary') {
      UI.slider(ctr, { label: 'Logit z = wᵀh + b', min: -8, max: 8, step: 0.1, value: z, fmt: v => v.toFixed(1), onInput: v => { z = v; draw(); } });
      UI.select(ctr, 'True label y', [['0', '0'], ['1', '1']], String(yT), v => { yT = +v; draw(); });
    } else {
      zs.forEach((v, i) => UI.slider(ctr, { label: `Logit z${i + 1}`, min: -5, max: 5, step: 0.1, value: v, fmt: x => x.toFixed(1), onInput: x => { zs[i] = x; draw(); } }));
      UI.select(ctr, 'True class', [['0', 'class 1'], ['1', 'class 2'], ['2', 'class 3']], String(k), v => { k = +v; draw(); });
    }
    draw();
  }
  function draw() {
    const c = Colors.get();
    out.innerHTML = '';
    let rows, calcTxt;
    if (kind === 'real') {
      const d = z - yR;
      rows = [['Distribution', 'Gaussian p(y|x) = 𝒩(y; ŷ, I)'], ['Output unit', 'linear: ŷ = Wᵀh + b (never saturates)'], ['Loss (NLL)', '½(y − ŷ)² + const  = MSE'], ['Gradient ∂L/∂ŷ', 'ŷ − y']];
      calcTxt = `L = ½(${yR.toFixed(2)} − ${z.toFixed(2)})² = <b>${(0.5 * d * d).toFixed(4)}</b>     ∂L/∂ŷ = ŷ − y = <b>${d.toFixed(3)}</b>`;
    } else if (kind === 'binary') {
      const p = sigmoid(z), L = Math.log1p(Math.exp(-Math.abs((1 - 2 * yT) * z))) + Math.max((1 - 2 * yT) * z, 0);
      rows = [['Distribution', 'Bernoulli: P(y = 1|x) = ŷ'], ['Output unit', 'sigmoid: ŷ = σ(wᵀh + b)'], ['Loss (NLL)', '−log σ((2y − 1)z) = ζ((1 − 2y)z)  (softplus)'], ['Gradient ∂L/∂z', 'σ(z) − y']];
      calcTxt = `ŷ = σ(${z.toFixed(1)}) = 1/(1+e^(${(-z).toFixed(1)})) = ${p.toFixed(4)}
L = ζ((1 − 2·${yT})·${z.toFixed(1)}) = log(1 + e^(${((1 - 2 * yT) * z).toFixed(1)})) = <b>${L.toFixed(4)}</b>     ∂L/∂z = σ(z) − y = <b>${(p - yT).toFixed(4)}</b>`;
    } else {
      const mx = Math.max(...zs), ex = zs.map(v => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0), p = ex.map(v => v / s);
      rows = [['Distribution', 'Multinoulli: P(y = i|x) = ŷᵢ'], ['Output unit', 'softmax: ŷᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)'], ['Loss (NLL)', '−zᵧ + log Σⱼ exp(zⱼ)'], ['Gradient ∂L/∂z', 'ŷ − onehot(y)']];
      calcTxt = `softmax = (${p.map(v => v.toFixed(3)).join(', ')})   true class ${k + 1}
L = −log ŷ_${k + 1} = −log ${p[k].toFixed(4)} = <b>${(-Math.log(p[k])).toFixed(4)}</b>     ∂L/∂z = (${p.map((v, i) => (v - (i === k ? 1 : 0)).toFixed(3)).join(', ')})`;
    }
    out.append(h('table', { class: 'readout' }, h('tbody', {}, rows.map(([a, b]) => h('tr', {}, h('th', { style: 'width:150px' }, a), h('td', {}, b))))));
    out.append(h('div', { class: 'calcbox', html: calcTxt }));
    void c;
  }
  build();
  return { redraw: draw };
};

/* ---------- 6) Why the log matters: NLL vs MSE for sigmoid outputs ---------- */
Widgets.nllShapes = (el) => {
  UI.head(el, 'Sigmoid output: log-likelihood vs. squared error', 'Loss as a function of the logit z for the true label y. Dashed lines are the gradients ∂loss/∂z. Watch the far side where the model is confidently WRONG.');
  let y = 1, z = -6;
  const ctr = h('div', { class: 'controls' });
  UI.select(ctr, 'True label y', [['1', 'y = 1'], ['0', 'y = 0']], '1', v => { y = +v; draw(); });
  UI.slider(ctr, { label: 'Current logit z', min: -8, max: 8, step: 0.1, value: z, fmt: v => v.toFixed(1), onInput: v => { z = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [-8, 8], yr: [-1.1, 4], xlabel: 'z', xname: 'z', aspect: .5, maxH: 290, refY: 0, vfmt: v => v.toFixed(4) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const sp = t => Math.log1p(Math.exp(-Math.abs(t))) + Math.max(t, 0);
  function draw() {
    const c = Colors.get();
    const zs = Array.from({ length: 161 }, (_, i) => -8 + i * 0.1);
    const nll = t => sp((1 - 2 * y) * t), dn = t => sigmoid(t) - y;
    const ms = t => (sigmoid(t) - y) ** 2, dm = t => 2 * (sigmoid(t) - y) * sigmoid(t) * (1 - sigmoid(t));
    chart.set([
      { name: 'NLL = softplus((1−2y)z)', color: 's1', data: zs.map(t => [t, Math.min(4, nll(t))]) },
      { name: '∂NLL/∂z', color: 's1', data: zs.map(t => [t, dn(t)]), dash: [5, 4] },
      { name: 'MSE = (σ(z) − y)²', color: 's2', data: zs.map(t => [t, ms(t)]) },
      { name: '∂MSE/∂z', color: 's2', data: zs.map(t => [t, dm(t)]), dash: [5, 4] },
      { name: 'now', color: 'text', style: 'dots', r: 5, noHover: true, data: [[z, Math.min(4, nll(z))], [z, ms(z)]] }
    ]);
    lg.set([{ label: 'negative log-likelihood', color: c.s1 }, { label: 'its gradient', color: c.s1, kind: 'dash' }, { label: 'squared error', color: c.s2 }, { label: 'its gradient', color: c.s2, kind: 'dash' }]);
    const wrong = (y === 1 && z < -2) || (y === 0 && z > 2);
    calc.innerHTML = `<span class="c">at z = ${z.toFixed(1)}, y = ${y}:  σ(z) = ${sigmoid(z).toFixed(4)}</span>
NLL = ${nll(z).toFixed(4)},   ∂NLL/∂z = σ(z) − y = <b>${dn(z).toFixed(4)}</b>
MSE = ${ms(z).toFixed(4)},   ∂MSE/∂z = 2(σ − y)σ(1 − σ) = <b>${dm(z).toFixed(4)}</b>
<span class="c">${wrong ? `confidently wrong: the NLL gradient is ${Math.abs(dn(z) / (dm(z) || 1e-12)).toFixed(0)}× larger → NLL corrects the mistake fast; MSE barely moves (saturated).` : 'move z to the wrong side (y = 1, z ≪ 0) to see the difference.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) Softmax playground ---------- */
Widgets.softmaxPlay = (el) => {
  UI.head(el, 'Softmax: logits → probabilities → loss → gradient', 'Four classes. Move the logits zᵢ. Bars show the probabilities, the one-hot target, and the gradient ∂L/∂z = softmax(z) − onehot(y).');
  let zs = [2.0, 1.0, 0.1, -1.0], k = 1, shift = 0;
  const ctr = h('div', { class: 'controls' });
  zs.forEach((v, i) => UI.slider(ctr, { label: `logit z${i + 1}`, min: -6, max: 6, step: 0.1, value: v, fmt: x => x.toFixed(1), onInput: x => { zs[i] = x; draw(); } }));
  UI.select(ctr, 'Correct class y', [['0', 'class 1'], ['1', 'class 2'], ['2', 'class 3'], ['3', 'class 4']], String(k), v => { k = +v; draw(); });
  UI.slider(ctr, { label: 'Add the same constant c to every logit', min: -5, max: 5, step: 0.1, value: shift, fmt: v => v.toFixed(1), onInput: v => { shift = v; draw(); } });
  const bars = h('div');
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, bars, calc);
  function draw() {
    const c = Colors.get();
    const z = zs.map(v => v + shift), mx = Math.max(...z), ex = z.map(v => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0), p = ex.map(v => v / s);
    const grad = p.map((v, i) => v - (i === k ? 1 : 0));
    bars.innerHTML = '';
    const grid = h('div', { style: 'display:grid;grid-template-columns:70px repeat(4,1fr);gap:6px;align-items:end;font-size:.75rem' });
    const cell = (v, col, neg) => h('div', { style: 'height:90px;position:relative;border-bottom:1px solid var(--axis)' },
      h('div', { style: `position:absolute;left:15%;right:15%;${neg && v < 0 ? 'top:100%' : 'bottom:0'};height:${Math.abs(v) * 88}px;background:${col};border-radius:3px` }),
      h('div', { style: 'position:absolute;top:0;width:100%;text-align:center;font-family:var(--mono)' }, v.toFixed(2)));
    grid.append(h('div', {}, 'softmax(z)'), ...p.map(v => cell(v, c.s1)));
    grid.append(h('div', {}, 'target'), ...p.map((_, i) => cell(i === k ? 1 : 0, c.s3)));
    bars.append(grid);
    bars.append(h('div', { style: 'display:grid;grid-template-columns:70px repeat(4,1fr);gap:6px;font-size:.75rem;margin-top:6px' }, h('div', {}, 'gradient'), ...grad.map(g => h('div', { style: `text-align:center;font-family:var(--mono);color:${g < 0 ? c.s3 : c.s8};font-weight:700` }, (g > 0 ? '+' : '') + g.toFixed(3)))));
    const lse = mx + Math.log(s);
    calc.innerHTML = `<span class="c">softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ);  stable version subtracts max z = ${mx.toFixed(2)} first (same answer, no overflow)</span>
exp(z − max) = (${ex.map(v => v.toFixed(3)).join(', ')}),  sum = ${s.toFixed(3)}  →  p = (${p.map(v => v.toFixed(3)).join(', ')})
log-likelihood: log softmax(z)ᵧ = zᵧ − log Σ exp(zⱼ) = ${z[k].toFixed(2)} − ${lse.toFixed(3)} = ${(z[k] - lse).toFixed(4)}   (log Σ exp ≈ max zⱼ = ${mx.toFixed(2)})
loss L = −log pᵧ = <b>${(lse - z[k]).toFixed(4)}</b>    <span class="c">adding c = ${shift.toFixed(1)} to all logits changes nothing: softmax(z) = softmax(z + c)</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 8) Activation explorer ---------- */
Widgets.activationExplorer = (el, ds) => {
  const keys = (ds.acts || 'relu,leaky,abs,prelu,maxout').split(',');
  UI.head(el, ds.title || 'Hidden-unit activation functions', 'Solid = g(z), dashed = slope g′(z) (what the gradient is multiplied by). Bottom: share of units that barely learn when pre-activations are spread with std s.');
  let key = keys[0], s = 2, z0 = 1;
  const segBox = h('div', { class: 'gridbtns' });
  const btns = keys.map(k => { const b = h('button', { type: 'button' }, ACTS[k].name); b.addEventListener('click', () => { key = k; draw(); }); segBox.append(b); return [k, b]; });
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Evaluate at z', min: -5, max: 5, step: 0.05, value: z0, fmt: v => v.toFixed(2), onInput: v => { z0 = v; draw(); } });
  UI.slider(ctr, { label: 'Spread of pre-activations s', min: 0.2, max: 6, step: 0.1, value: s, fmt: v => v.toFixed(1), onInput: v => { s = v; draw(); } });
  el.append(segBox, ctr);
  const chart = new LineChart(el, { xr: [-5, 5], yr: [-2, 3], xlabel: 'z', xname: 'z', aspect: .5, maxH: 270, refY: 0, vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    btns.forEach(([k, b]) => b.classList.toggle('on', k === key));
    const A = ACTS[key];
    const zs = Array.from({ length: 201 }, (_, i) => -5 + i * 0.05);
    chart.set([
      { name: A.name, color: 's1', data: zs.map(t => [t, clamp(A.f(t), -2, 3)]), width: 2.6 },
      { name: 'slope g′(z)', color: 's2', data: zs.map(t => [t, clamp(A.d(t), -2, 3)]), dash: [5, 4] },
      { name: 'z', color: 'text', style: 'dots', r: 5, noHover: true, data: [[z0, clamp(A.f(z0), -2, 3)]] }
    ]);
    lg.set([{ label: A.name + ':  ' + A.tex, color: c.s1 }, { label: 'derivative', color: c.s2, kind: 'dash' }]);
    const rng = mulberry32(5); let low = 0; const N = 4000;
    for (let i = 0; i < N; i++) if (Math.abs(A.d(s * gaussian(rng))) < 0.05) low++;
    calc.innerHTML = `<span class="c">${A.note}</span>
g(${z0.toFixed(2)}) = <b>${A.f(z0).toFixed(4)}</b>      g′(${z0.toFixed(2)}) = <b>${A.d(z0).toFixed(4)}</b>
with z ~ 𝒩(0, ${s.toFixed(1)}²): <b>${(low / N * 100).toFixed(1)}%</b> of units have |g′(z)| < 0.05 (saturated or dead → almost no gradient)`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 9) Depth folds space: exponentially many pieces ---------- */
Widgets.foldDemo = (el) => {
  UI.head(el, 'Why depth helps: each layer folds the input', 'Each layer applies the same “tent” t(x) = 1 − |2x − 1|, built from just 2 ReLU units: t(x) = 2·ReLU(x) − 4·ReLU(x − ½). Compose it k times.');
  let k = 3;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Number of layers k', min: 1, max: 7, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 1], yr: [-0.05, 1.05], xlabel: 'x', ylabel: 'f(x)', xname: 'x', aspect: .45, maxH: 250, vfmt: v => v.toFixed(3) });
  const stats = h('div', { class: 'stat-row' });
  const calc = h('div', { class: 'calcbox' });
  el.append(stats, calc);
  const t = x => 2 * Math.max(0, x) - 4 * Math.max(0, x - 0.5);
  function draw() {
    const xs = Array.from({ length: 1025 }, (_, i) => i / 1024);
    const fk = x => { let v = x; for (let i = 0; i < k; i++) v = t(v); return v; };
    chart.set([{ name: `f = t∘…∘t (${k}×)`, color: 's1', data: xs.map(x => [x, fk(x)]) }]);
    stats.innerHTML = '';
    stats.append(
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'ReLU units used (deep)'), h('div', { class: 'v' }, String(2 * k))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Linear pieces produced'), h('div', { class: 'v' }, String(2 ** k))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Units a 1-hidden-layer net needs'), h('div', { class: 'v' }, '≈ ' + 2 ** k)));
    calc.innerHTML = `<span class="c">one layer folds [0, 1] onto itself; k layers fold k times</span>
pieces = 2^k = 2^${k} = <b>${2 ** k}</b> using only 2k = ${2 * k} ReLU units.  A shallow network needs about one unit per kink (≈ ${2 ** k - 1}).
<span class="c">depth = an exponential saving: “compose simple functions” is exactly the prior deep networks encode.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 10) Universal approximation with one wide hidden layer ---------- */
Widgets.uatWidth = (el) => {
  UI.head(el, 'One hidden layer can approximate any continuous function…', '…given enough units. Hidden layer: k ReLU units with random kinks; output layer: best linear combination (fitted exactly). Increase the width.');
  const targets = {
    wave: ['sin wave', x => Math.sin(6 * x) + 0.3 * Math.cos(17 * x)],
    bump: ['bumps', x => Math.exp(-(((x - 0.3) / 0.06) ** 2)) + 0.6 * Math.exp(-(((x - 0.7) / 0.12) ** 2))],
    step: ['step (discontinuous)', x => (x > 0.5 ? 1 : 0)]
  };
  let tk = 'wave', lk = 0.6;
  const ctr = h('div', { class: 'controls' });
  UI.select(ctr, 'Target function', Object.entries(targets).map(([k, v]) => [k, v[0]]), tk, v => { tk = v; draw(); });
  UI.slider(ctr, { label: 'Hidden units k (width)', min: 0, max: 2.3, step: 0.05, value: lk, fmt: v => String(Math.round(10 ** v)), onInput: v => { lk = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 1], yr: [-1.6, 1.6], xlabel: 'x', xname: 'x', aspect: .45, maxH: 260, vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const k = Math.round(10 ** lk), f = targets[tk][1];
    const rng = mulberry32(9 + k);
    const units = Array.from({ length: k }, () => [rng(), rng() < 0.5 ? 1 : -1]);
    const feat = x => [1, x, ...units.map(([u, s]) => Math.max(0, s * (x - u)))];
    const xs = Array.from({ length: 300 }, (_, i) => i / 299);
    const wv = ridgeFit(xs.map(feat), xs.map(f), 1e-7);
    const approx = xs.map(x => dotv(wv, feat(x)));
    const err = mean(xs.map((x, i) => (approx[i] - f(x)) ** 2));
    chart.set([{ name: 'target', color: 'muted', data: xs.map(x => [x, f(x)]), width: 3 }, { name: `network, k = ${k}`, color: 's2', data: xs.map((x, i) => [x, approx[i]]) }], { yr: tk === 'wave' ? [-1.6, 1.6] : [-0.3, 1.3] });
    lg.set([{ label: 'target function', color: c.muted }, { label: `1 hidden layer with ${k} ReLU units`, color: c.s2 }]);
    calc.innerHTML = `ŷ(x) = Σⱼ wⱼ·max{0, ±(x − uⱼ)} + w₀ + w₁x     (k = ${k} kinks at random positions uⱼ)
mean squared approximation error = <b>${err.toExponential(2)}</b>
<span class="c">the theorem guarantees such weights exist for large k — it does NOT promise that training finds them or that k is small.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 11) Ways to connect two layers ---------- */
Widgets.connectivity = (el) => {
  UI.head(el, 'How should two layers be connected?', 'Compare full connectivity, sparse (local) connectivity, and skip connections.');
  let mode = 'full';
  const segBox = h('div');
  UI.seg(segBox, [['full', 'Fully connected'], ['local', 'Sparse / local'], ['skip', 'Skip connection']], mode, v => { mode = v; draw(); });
  const svgBox = h('div'), out = h('div');
  el.append(segBox, svgBox, out);
  function draw() {
    const c = Colors.get();
    const ns = 'http://www.w3.org/2000/svg', W = 560, H = 220;
    const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('class', 'diagram');
    const mk = (t, a, txt) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.append(e); return e; };
    const n = 6, rows = mode === 'skip' ? [40, 110, 180] : [50, 170];
    const X = i => 90 + i * 76;
    let edges = 0;
    const conn = (r1, r2, local, col) => { for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { if (local && Math.abs(i - j) > 1) continue; edges++; mk('line', { x1: X(i), y1: rows[r1], x2: X(j), y2: rows[r2], stroke: col, 'stroke-width': 1.4, opacity: .7 }); } };
    if (mode === 'full') conn(0, 1, false, c.s1);
    if (mode === 'local') conn(0, 1, true, c.s1);
    if (mode === 'skip') {
      conn(0, 1, false, c.s1); conn(1, 2, false, c.s1);
      for (let i = 0; i < n; i++) mk('path', { d: `M ${X(i) - 12} ${rows[0]} C ${X(i) - 50} ${rows[0] + 40}, ${X(i) - 50} ${rows[2] - 40}, ${X(i) - 12} ${rows[2]}`, fill: 'none', stroke: c.s3, 'stroke-width': 2.2 });
    }
    rows.forEach((y, r) => { for (let i = 0; i < n; i++) mk('circle', { cx: X(i), cy: y, r: 11, fill: c.surface, stroke: c.text, 'stroke-width': 2 }); mk('text', { x: 20, y: y + 4, 'font-size': 12, fill: c.text2, 'font-weight': 700 }, ['layer i', 'layer i+1', 'layer i+2'][r]); });
    svgBox.innerHTML = ''; svgBox.append(svg);
    out.innerHTML = '';
    out.append(h('div', { class: 'callout info' }, h('span', { class: 'ico' }, 'i'), h('div', {}, {
      full: `Every input unit connects to every output unit: ${edges} weights for 6 → 6 units (n_in × n_out). The default layer, a matrix W.`,
      local: `Each output unit only sees a small neighbourhood: ${edges} weights instead of 36. Fewer parameters and less computation — but the right pattern depends on the problem (convolutional nets use exactly this for images).`,
      skip: 'Skip connections (green) jump from layer i to layer i + 2 or higher. They give the gradient a short path from the output back to early layers, making deep nets easier to train.'
    }[mode])));
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   Quiz
   ========================================================= */
window.QUIZ = [
  { q: 'Why are hidden layers called “hidden”?', opts: ['They are not used at test time', 'The training data does not say what their output should be', 'They have no weights', 'They are always linear'], a: 1, why: 'Training examples specify only the output layer’s target y. The learning algorithm must decide how to use the intermediate layers.' },
  { q: 'A network with two layers and NO nonlinearity, f(x) = xᵀW w, is…', opts: ['able to learn XOR', 'still just a linear function of x', 'a universal approximator', 'convex in W and w jointly'], a: 1, why: 'xᵀW w = xᵀw′ with w′ = W w. Stacking linear layers gives a linear function, which can’t represent XOR.' },
  { q: 'In the XOR solution, what does the ReLU hidden layer accomplish?', opts: ['It memorizes the labels', 'It maps the 4 points into a space h where a linear model can separate them', 'It removes the bias', 'It makes the loss convex'], a: 1, why: 'After h = max{0, Wᵀx + c} the points (0,1) and (1,0) coincide at (1, 0), and wᵀh + b gives 0, 1, 1, 0.' },
  { q: 'Most neural networks are trained with which cost?', opts: ['Accuracy', 'Negative log-likelihood (cross-entropy between data and model)', 'Number of parameters', 'Hinge loss only'], a: 1, why: 'J(θ) = −E log p_model(y|x). Its exact form follows from the chosen output distribution (Gaussian → MSE, Bernoulli → binary cross-entropy, multinoulli → softmax cross-entropy).' },
  { q: 'With a sigmoid output, why is MSE a poor loss?', opts: ['It is too slow to compute', 'When the model is confidently wrong the sigmoid saturates and the MSE gradient vanishes', 'It cannot be negative', 'It only works for regression'], a: 1, why: 'The NLL −log σ((2y−1)z) = ζ((1−2y)z) undoes the exp, so its gradient → ±1 when very wrong; MSE’s gradient is multiplied by σ′(z) ≈ 0.' },
  { q: 'The gradient of the softmax cross-entropy w.r.t. the logits z is…', opts: ['z − y', 'softmax(z) − onehot(y)', 'log softmax(z)', '1 − softmax(z)'], a: 1, why: 'L = −z_y + log Σ exp(z_j). Differentiating gives softmax(z)ᵢ − 1[i = y]: push the correct logit up, all logits down in proportion to their probability.' },
  { q: 'Adding the same constant to every input of a softmax…', opts: ['changes all probabilities', 'does not change the output', 'makes it sum to 2', 'causes saturation'], a: 1, why: 'softmax(z) = softmax(z + c). That is why the stable implementation subtracts max z first.' },
  { q: 'Why are ReLUs easy to optimize?', opts: ['They saturate on both sides', 'Their derivative is 1 whenever the unit is active, so gradients stay large and consistent', 'They are bounded', 'They are differentiable everywhere'], a: 1, why: 'Being nearly linear, they pass gradients through unchanged when active. Their drawback: no gradient when inactive (fixed by leaky ReLU / PReLU / maxout).' },
  { q: 'Which statement about tanh vs. logistic sigmoid hidden units is correct?', opts: ['sigmoid is preferred', 'tanh usually trains better because tanh(0) = 0 and it resembles the identity near 0', 'they never saturate', 'tanh is piecewise linear'], a: 1, why: 'tanh(z) = 2σ(2z) − 1. Near zero it behaves like a linear model, which is easier to train — but both saturate for large |z|.' },
  { q: 'The universal approximation theorem guarantees that…', opts: ['gradient descent will find the right weights', 'a wide enough single-hidden-layer network can represent any continuous function on a bounded domain to any accuracy', 'one hidden unit is enough', 'deeper is always better'], a: 1, why: 'It is about representation only: the network may need exponentially many units, and training may still fail or overfit. Deeper networks can need far fewer units.' }
];
