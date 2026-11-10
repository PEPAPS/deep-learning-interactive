/* =========================================================
   Lecture 12 · Ch 14 · Autoencoders
   ========================================================= */
'use strict';

/* ---------- helpers ---------- */
const fx = (v, d = 2) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d).replace('-', '−');
const vec = (a, d = 2) => '[' + a.map(v => fx(v, d)).join(', ') + ']';
const sq = v => v * v;
const sum = a => a.reduce((s, v) => s + v, 0);
const calcBox = () => h('div', { class: 'calcbox' });
const stat = (k, v) => h('div', { class: 'stat' }, h('div', { class: 'k' }, k), h('div', { class: 'v', html: v }));
const gw = (cap, node) => h('div', { class: 'gwrap' }, h('div', { class: 'gcap' }, cap), node);
const gop = s => h('div', { class: 'gop' }, s);
const rowBox = (...kids) => h('div', { class: 'grids', style: 'align-items:center;gap:8px 10px;padding:6px 10px;border-radius:10px;background:var(--surface-2)' }, kids);

/* an 8×8 grayscale image drawn as a grid of shaded cells */
function imgGrid(px, o = {}) {
  const n = o.n || 8, cs = o.cs || 18, c = Colors.get();
  const g = h('div', { class: 'cgrid' + (o.onClick ? ' click' : ''), style: `grid-template-columns:repeat(${n}, ${cs}px);--cs:${cs}px`, title: o.title || null });
  for (let i = 0; i < n * n; i++) {
    const v = clamp(px[i], 0, 1);
    const cell = h('div', { class: 'cell' });
    cell.style.background = alpha(o.color || c.text, 0.04 + 0.86 * v);
    g.append(cell);
  }
  if (o.onClick) g.addEventListener('click', o.onClick);
  return g;
}
/* a code vector drawn as a row of signed cells (blue +, orange −) */
function codeRow(hv, o = {}) {
  const c = Colors.get(), cs = o.cs || 18;
  const scale = o.scale || Math.max(0.3, ...hv.map(Math.abs));
  const g = h('div', { class: 'cgrid', style: `grid-template-columns:repeat(${hv.length}, ${cs}px);--cs:${cs}px` });
  hv.forEach(v => {
    const cell = h('div', { class: 'cell', title: fx(v) });
    cell.style.background = alpha(v >= 0 ? c.s1 : c.s2, 0.1 + 0.8 * clamp(Math.abs(v) / scale, 0, 1));
    g.append(cell);
  });
  return g;
}

/* symmetric eigen-decomposition (cyclic Jacobi). Returns eigenvalues (desc) and eigenvectors. */
function symEig(A) {
  const n = A.length, a = A.map(r => r.slice());
  const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j];
    if (off < 1e-14) break;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-15) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = (th >= 0 ? 1 : -1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k][p], akq = a[k][q]; a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p][k], aqk = a[q][k]; a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = V[k][p], vkq = V[k][q]; V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq; }
    }
  }
  const order = a.map((_, i) => i).sort((i, j) => a[j][j] - a[i][i]);
  return { vals: order.map(i => a[i][i]), vecs: order.map(i => V.map(row => row[i])) };
}

/* The toy dataset: 72 images (8×8) made of vertical and/or horizontal bars, plus a linear
   autoencoder trained on it (= PCA: encode with the top-k principal directions). */
const BARS = (() => {
  const rng = mulberry32(1414);
  const N = 72, D = 64, X = [], fam = [];
  const add = (img, kind, idx, amp) => { for (let t = 0; t < 8; t++) { const i = kind === 'v' ? t * 8 + idx : idx * 8 + t; img[i] = Math.min(1, img[i] + amp); } };
  for (let n = 0; n < N; n++) {
    const f = n % 3; // 0 vertical, 1 horizontal, 2 both
    const img = new Array(D).fill(0);
    const nb = 1 + Math.floor(rng() * 2);
    if (f !== 1) for (let b = 0; b < nb; b++) add(img, 'v', Math.floor(rng() * 8), 0.55 + 0.45 * rng());
    if (f !== 0) for (let b = 0; b < nb; b++) add(img, 'h', Math.floor(rng() * 8), 0.55 + 0.45 * rng());
    for (let i = 0; i < D; i++) img[i] = clamp(img[i] + 0.05 * (rng() - 0.5), 0, 1);
    X.push(img); fam.push(f);
  }
  const mu = new Array(D).fill(0);
  X.forEach(x => x.forEach((v, i) => { mu[i] += v / N; }));
  const C = Array.from({ length: D }, () => new Array(D).fill(0));
  X.forEach(x => { const d = x.map((v, i) => v - mu[i]); for (let i = 0; i < D; i++) for (let j = i; j < D; j++) C[i][j] += d[i] * d[j] / N; });
  for (let i = 0; i < D; i++) for (let j = 0; j < i; j++) C[i][j] = C[j][i];
  const { vals, vecs } = symEig(C);
  const total = sum(vals);
  const encode = (x, k) => { const d = x.map((v, i) => v - mu[i]); return vecs.slice(0, k).map(v => dotv(v, d)); };
  const decode = hv => { const r = mu.slice(); hv.forEach((hi, j) => { const v = vecs[j]; for (let i = 0; i < D; i++) r[i] += hi * v[i]; }); return r; };
  const recon = (x, k) => decode(encode(x, k));
  const mseOf = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += sq(a[i] - b[i]); return s / a.length; };
  const explained = k => sum(vals.slice(0, k)) / total;
  const diag = new Array(D).fill(0);
  for (let t = 0; t < 8; t++) { diag[t * 8 + t] = 1; if (t < 7) diag[t * 8 + t + 1] = 0.55; }
  const noise = seed => { const r = mulberry32(seed); return Array.from({ length: D }, () => r()); };
  return { N, D, X, fam, mu, vals, vecs, encode, decode, recon, mse: mseOf, explained, diag, noise, FAM: ['vertical bars', 'horizontal bars', 'both kinds'] };
})();

/* ---------- 1) a tiny autoencoder, number by number ---------- */
Widgets.aeAnatomy = (el) => {
  UI.head(el, 'A tiny autoencoder, number by number', 'Input x = 4 numbers, code h = 2 numbers. Encoder h = W<sub>e</sub>x averages each pair of inputs; decoder r = W<sub>d</sub>h writes each code number twice. Move the inputs and watch what survives the bottleneck.');
  const We = [[.5, .5, 0, 0], [0, 0, .5, .5]], Wd = [[1, 0], [1, 0], [0, 1], [0, 1]];
  let x = [1, 1, 0.4, 0.4];
  const ctr = h('div', { class: 'controls' });
  const sl = x.map((v, i) => UI.slider(ctr, { label: `x${i + 1}`, min: -1.5, max: 1.5, step: 0.1, value: v, fmt: v => fx(v, 1), onInput: v => { x[i] = v; draw(); } }));
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Typical input (pairs agree)', () => set([1, 1, 0.4, 0.4]));
  UI.button(btns, 'Unusual input (pairs disagree)', () => set([1, -1, 0.5, -0.2]));
  UI.button(btns, 'Random input', () => set(x.map(() => Math.round((Math.random() * 3 - 1.5) * 10) / 10)));
  function set(v) { x = v.slice(); sl.forEach((s, i) => s.set(x[i])); draw(); }
  el.append(ctr, btns);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'autoencoder with input, code and reconstruction values' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 240, ctx = fitCanvas(cv, W, H);
    const hv = We.map(row => dotv(row, x)), r = Wd.map(row => dotv(row, hv));
    const L = sum(x.map((v, i) => sq(v - r[i])));
    const X0 = W * 0.16, X1 = W * 0.5, X2 = W * 0.84, ys4 = i => 40 + i * 50, ys2 = i => 65 + i * 100;
    for (let i = 0; i < 4; i++) for (let j = 0; j < 2; j++) if (We[j][i]) { ctx.strokeStyle = alpha(c.s1, .6); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X0 + 20, ys4(i)); ctx.lineTo(X1 - 22, ys2(j)); ctx.stroke(); }
    for (let i = 0; i < 4; i++) for (let j = 0; j < 2; j++) if (Wd[i][j]) { ctx.strokeStyle = alpha(c.s2, .6); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X1 + 22, ys2(j)); ctx.lineTo(X2 - 20, ys4(i)); ctx.stroke(); }
    const node = (xx, yy, rr, txt, fill, ring) => { dot(ctx, xx, yy, rr, fill, ring); label(ctx, txt, xx, yy, { align: 'center', size: 11, bold: true, color: c.text }); };
    x.forEach((v, i) => node(X0, ys4(i), 20, fx(v, 1), c.surface2, c.border));
    hv.forEach((v, j) => node(X1, ys2(j), 22, fx(v, 2), alpha(c.s1, .25), c.s1));
    r.forEach((v, i) => { const ok = Math.abs(v - x[i]) < 0.05; node(X2, ys4(i), 20, fx(v, 2), alpha(ok ? c.good : c.critical, .22), ok ? c.good : c.critical); });
    label(ctx, 'input x', X0, 12, { align: 'center', size: 11, bold: true, color: c.text });
    label(ctx, 'code h = f(x)', X1, 12, { align: 'center', size: 11, bold: true, color: c.s1 });
    label(ctx, 'reconstruction r = g(h)', X2, 12, { align: 'center', size: 11, bold: true, color: c.s2 });
    label(ctx, 'encoder f (weights 0.5)', (X0 + X1) / 2, H - 12, { align: 'center', size: 10, color: c.s1 });
    label(ctx, 'decoder g (weights 1)', (X1 + X2) / 2, H - 12, { align: 'center', size: 10, color: c.s2 });
    stats.innerHTML = '';
    stats.append(stat('code size', '2 <small>of 4 numbers</small>'), stat('reconstruction loss L', fx(L, 3)), stat('verdict', L < 0.01 ? 'copied ✓' : 'lossy copy'));
    calc.innerHTML = `h₁ = 0.5·x₁ + 0.5·x₂ = 0.5·${fx(x[0], 1)} + 0.5·${fx(x[1], 1)} = <b>${fx(hv[0])}</b>\nh₂ = 0.5·x₃ + 0.5·x₄ = 0.5·${fx(x[2], 1)} + 0.5·${fx(x[3], 1)} = <b>${fx(hv[1])}</b>\nr = g(h) = (h₁, h₁, h₂, h₂) = ${vec(r)}\nL(x, r) = Σ(xᵢ − rᵢ)² = ${x.map((v, i) => `(${fx(v, 1)} − ${fx(r[i])})²`).join(' + ')} = <b>${fx(L, 3)}</b>\n<span class="c">${L < 0.01 ? 'The pairs agree, so nothing is lost in the bottleneck: this input “resembles the training data”.' : 'The code keeps only the average of each pair; the difference inside a pair is thrown away. An input like this is not copied well — the autoencoder only copies what it was built to prioritize.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) deterministic vs stochastic encoder/decoder ---------- */
Widgets.stochasticAE = (el) => {
  UI.head(el, 'Deterministic vs. stochastic encoder and decoder', 'The data live on the green curve. A deterministic encoder gives one code h = f(x) (its position along the curve); a stochastic one gives a distribution p<sub>encoder</sub>(h | x). Decoding samples of h gives a cloud of reconstructions.');
  let s0 = 0.35, off = 0.2, sig = 0.05, tau = 0.03, seed = 1;
  const curve = s => [0.08 + 0.84 * s, 0.55 + 0.28 * Math.sin(2 * Math.PI * s)];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Where along the curve the input sits', min: 0, max: 1, step: 0.01, value: s0, fmt: v => fx(v), onInput: v => { s0 = v; draw(); } });
  UI.slider(ctr, { label: 'How far off the curve it is', min: -0.3, max: 0.3, step: 0.01, value: off, fmt: v => fx(v), onInput: v => { off = v; draw(); } });
  UI.slider(ctr, { label: 'Encoder spread σ (0 = deterministic)', min: 0, max: 0.15, step: 0.005, value: sig, fmt: v => fx(v, 3), onInput: v => { sig = v; draw(); } });
  UI.slider(ctr, { label: 'Decoder spread τ (0 = deterministic)', min: 0, max: 0.1, step: 0.005, value: tau, fmt: v => fx(v, 3), onInput: v => { tau = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Draw new samples', () => { seed++; draw(); });
  el.append(ctr, btns);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'data curve, input point, code axis and sampled reconstructions' });
  wrap.append(cv);
  const calc = calcBox();
  el.append(wrap, calc);
  function nearest(x) { let best = 0, bd = 1e9; for (let i = 0; i <= 300; i++) { const s = i / 300, p = curve(s), d = sq(p[0] - x[0]) + sq(p[1] - x[1]); if (d < bd) { bd = d; best = s; } } return best; }
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 310, ctx = fitCanvas(cv, W, H);
    const p = makePlot(W, H, [0, 1], [0, 1], { l: 30, r: 12, t: 12, b: 26 });
    drawAxes(ctx, p, { xlabel: 'x₁', ylabel: 'x₂' });
    ctx.strokeStyle = c.s3; ctx.lineWidth = 3; ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const q = curve(i / 200); const X = p.X(q[0]), Y = p.Y(q[1]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
    ctx.stroke();
    label(ctx, 'data manifold (where training data live)', p.X(0.5), p.Y(0.93), { align: 'center', size: 10, color: c.s3 });
    const base = curve(s0), b2 = curve(Math.min(1, s0 + 1e-3)), tx = b2[0] - base[0], ty = b2[1] - base[1], tl = Math.hypot(tx, ty) || 1, nx = -ty / tl, ny = tx / tl;
    const x = [base[0] + off * nx, base[1] + off * ny];
    const hd = nearest(x), rd = curve(hd);
    const rng = mulberry32(seed * 7919 + 1);
    const hs = [], rs = [];
    for (let i = 0; i < 40; i++) { const hh = clamp(hd + sig * gaussian(rng), 0, 1); hs.push(hh); const q = curve(hh); rs.push([q[0] + tau * gaussian(rng), q[1] + tau * gaussian(rng)]); }
    rs.forEach(q => dot(ctx, p.X(q[0]), p.Y(q[1]), 3, alpha(c.s2, .55)));
    const yb = p.Y(0.06);
    ctx.strokeStyle = c.s1; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(p.X(0.08), yb); ctx.lineTo(p.X(0.92), yb); ctx.stroke();
    label(ctx, 'code h = position along the curve (0 … 1)', p.X(0.5), yb + 12, { align: 'center', size: 10, color: c.s1 });
    hs.forEach(hh => dot(ctx, p.X(0.08 + 0.84 * hh), yb, 3, alpha(c.s1, .5)));
    if (sig > 0) {
      ctx.strokeStyle = c.s1; ctx.lineWidth = 1.5; ctx.beginPath();
      for (let i = 0; i <= 100; i++) { const hh = i / 100, pdf = Math.exp(-0.5 * sq((hh - hd) / sig)); const X = p.X(0.08 + 0.84 * hh), Y = yb - 38 * pdf; i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
      ctx.stroke();
      label(ctx, 'p(h | x)', p.X(0.08 + 0.84 * hd) + 8, yb - 32, { size: 10, color: c.s1 });
    }
    arrow(ctx, p.X(x[0]), p.Y(x[1]), p.X(0.08 + 0.84 * hd), yb - 6, alpha(c.s1, .8), 1.5, 7);
    dot(ctx, p.X(x[0]), p.Y(x[1]), 6, c.text, c.surface);
    label(ctx, 'x (input)', p.X(x[0]) + 10, p.Y(x[1]) - 8, { size: 12, bold: true, color: c.text });
    dot(ctx, p.X(rd[0]), p.Y(rd[1]), 6, c.s2, c.surface);
    label(ctx, 'g(f(x))', p.X(rd[0]) + 10, p.Y(rd[1]) + 11, { size: 11, bold: true, color: c.s2 });
    dot(ctx, p.X(0.08 + 0.84 * hd), yb, 5, c.s1, c.surface);
    const err = sq(x[0] - rd[0]) + sq(x[1] - rd[1]);
    calc.innerHTML = `x = ${vec(x)}   (off the curve by ${fx(off)})\nencoder: h = f(x) = <b>${fx(hd, 3)}</b>${sig > 0 ? `;  p_encoder(h | x) = N(${fx(hd, 3)}, σ² = ${fx(sig * sig, 4)})` : '  (deterministic: σ = 0)'}\ndecoder: g(h) = ${vec(rd)}${tau > 0 ? `;  p_decoder(x | h) = N(g(h), τ² = ${fx(tau * tau, 4)})` : '  (deterministic: τ = 0)'}\nreconstruction error ‖x − g(f(x))‖² = <b>${fx(err, 4)}</b>  → a point off the curve is pulled back onto it\n<span class="c">${sig > 0 || tau > 0 ? 'Orange cloud: 40 reconstructions drawn from the stochastic encoder/decoder — a small generative model of inputs that look like x.' : 'With σ = τ = 0 the cloud collapses to the single point g(f(x)): the classic autoencoder is the special case.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) undercomplete autoencoder on 8×8 images ---------- */
Widgets.bottleneckImages = (el) => {
  UI.head(el, 'Undercomplete autoencoder on 8×8 images', 'Training set: 72 images made of vertical and/or horizontal bars. The code keeps only k of the 64 numbers. Linear encoder + linear decoder + MSE = the principal subspace (PCA). A diagonal stroke was never in the training set.');
  let k = 16, page = 0;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Code size k (numbers kept)', min: 1, max: 64, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Other training images', () => { page = (page + 1) % 6; draw(); });
  el.append(ctr, btns);
  const rows = h('div', { class: 'stack' }), stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(rows, stats, calc);
  function draw() {
    rows.innerHTML = '';
    const idx = [0, 1, 2, 3].map(i => (page * 4 + i * 7) % BARS.N);
    const show = (x, cap, novel) => {
      const hv = BARS.encode(x, k), r = BARS.decode(hv), e = BARS.mse(x, r);
      rows.append(rowBox(gw(cap, imgGrid(x)), gop('→ f →'), gw(`code (${k} numbers)`, codeRow(hv.slice(0, Math.min(k, 16)), { cs: 12 })), k > 16 ? h('span', { class: 'note' }, `… +${k - 16} more`) : null, gop('→ g →'), gw(`r · MSE ${fx(e, 4)}`, imgGrid(r)), novel ? h('span', { class: 'pill', style: 'background:color-mix(in srgb,var(--critical) 18%,var(--surface))' }, 'never seen in training') : null));
      return e;
    };
    idx.forEach(i => show(BARS.X[i], `train #${i + 1} (${BARS.FAM[BARS.fam[i]]})`));
    const en = show(BARS.diag, 'diagonal stroke', true);
    const all = mean(BARS.X.map(x => BARS.mse(x, BARS.recon(x, k))));
    stats.innerHTML = '';
    stats.append(stat('compression', `64 → ${k} <small>(${(64 / k).toFixed(1)}×)</small>`), stat('variance kept', (100 * BARS.explained(k)).toFixed(1) + '%'), stat('train MSE (all 72)', fx(all, 4)), stat('MSE on the diagonal', fx(en, 4)));
    calc.innerHTML = `h = f(x) = V_kᵀ(x − μ)   → the k code numbers are the projections of x onto the k main directions of variation\nr = g(h) = μ + V_k h       → rebuilt from those k numbers only\nL(x, g(f(x))) = mean squared error = ${fx(all, 4)} on the training set, but <b>${fx(en, 4)}</b> on the diagonal stroke\n<span class="c">${k >= 24 ? 'With a big code even the diagonal gets copied: the autoencoder prioritizes less and less.' : 'A small code forces the autoencoder to prioritize the most salient features of the training data — bars, not diagonals. The diagonal is copied badly because the code has no room for it.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) linear autoencoder with a 1-number code = PCA ---------- */
Widgets.pcaLine = (el) => {
  UI.head(el, 'A linear autoencoder with a 1-number code is PCA', 'Each 2-D point is encoded as its position along a line (one number) and decoded as the point on the line. Turn the line and watch the total squared reconstruction error; “Train” runs gradient descent on the angle.');
  let shape = 'cloud', th = 20, timer = null;
  const data = { cloud: [], banana: [] };
  {
    const r = mulberry32(21);
    for (let i = 0; i < 60; i++) {
      const u = gaussian(r), v = gaussian(r), a = 0.6;
      data.cloud.push([u * 1.6 * Math.cos(a) - v * .45 * Math.sin(a), u * 1.6 * Math.sin(a) + v * .45 * Math.cos(a)]);
      const t = -1.6 + 3.2 * r();
      data.banana.push([t + 0.1 * gaussian(r), 0.8 * t * t - 1 + 0.1 * gaussian(r)]);
    }
  }
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, [['cloud', 'Elongated cloud'], ['banana', 'Curved data']], shape, v => { shape = v; stop(); draw(); });
  const ctr = h('div', { class: 'controls' });
  const sTh = UI.slider(ctr, { label: 'Angle of the line θ', min: 0, max: 180, step: 1, value: th, fmt: v => v + '°', onInput: v => { th = v; stop(); draw(); } });
  const btns = h('div', { class: 'btns' });
  const bTrain = UI.button(btns, 'Train θ by gradient descent', train, 'btn primary');
  el.append(seg, ctr, btns);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'points projected onto a line' });
  wrap.append(cv);
  const calc = calcBox();
  el.append(wrap, calc);
  const J = (pts, t) => { const a = t * Math.PI / 180, ux = Math.cos(a), uy = Math.sin(a); return sum(pts.map(([x, y]) => { const s = x * ux + y * uy; return sq(x - s * ux) + sq(y - s * uy); })); };
  const pcaAngle = pts => { let sxx = 0, syy = 0, sxy = 0; pts.forEach(([x, y]) => { sxx += x * x; syy += y * y; sxy += x * y; }); return ((Math.atan2(2 * sxy, sxx - syy) / 2) * 180 / Math.PI + 180) % 180; };
  const curveErr = pts => sum(pts.map(([x, y]) => { let bd = 1e9; for (let i = 0; i <= 240; i++) { const t = -1.8 + 3.6 * i / 240; bd = Math.min(bd, sq(x - t) + sq(y - (0.8 * t * t - 1))); } return bd; }));
  function stop() { if (timer) { clearInterval(timer); timer = null; bTrain.textContent = 'Train θ by gradient descent'; } }
  function train() {
    if (timer) { stop(); return; }
    bTrain.textContent = 'Stop';
    timer = setInterval(() => {
      const pts = data[shape], g = (J(pts, th + 0.5) - J(pts, th - 0.5)); // dJ/dθ per degree
      th = ((th - 0.5 * g) % 180 + 180) % 180;
      sTh.set(Math.round(th)); draw();
      if (Math.abs(g) < 0.02) stop();
    }, 60);
  }
  function draw() {
    const pts = data[shape], c = Colors.get(), W = boxWidth(wrap), H = 300, ctx = fitCanvas(cv, W, H);
    const p = makePlot(W, H, [-3, 3], [-2.4, 2.4], { l: 30, r: 10, t: 10, b: 24 });
    drawAxes(ctx, p, {});
    const a = th * Math.PI / 180, ux = Math.cos(a), uy = Math.sin(a), tp = pcaAngle(pts), ap = tp * Math.PI / 180;
    ctx.save(); p.clip(ctx);
    ctx.setLineDash([5, 4]); ctx.strokeStyle = c.s3; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(p.X(-4 * Math.cos(ap)), p.Y(-4 * Math.sin(ap))); ctx.lineTo(p.X(4 * Math.cos(ap)), p.Y(4 * Math.sin(ap))); ctx.stroke(); ctx.setLineDash([]);
    if (shape === 'banana') { ctx.strokeStyle = alpha(c.s2, .75); ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i <= 100; i++) { const t = -1.8 + 3.6 * i / 100, X = p.X(t), Y = p.Y(0.8 * t * t - 1); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); } ctx.stroke(); }
    ctx.strokeStyle = c.s1; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(p.X(-4 * ux), p.Y(-4 * uy)); ctx.lineTo(p.X(4 * ux), p.Y(4 * uy)); ctx.stroke();
    pts.forEach(([x, y]) => {
      const s = x * ux + y * uy;
      ctx.strokeStyle = alpha(c.critical, .5); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.X(x), p.Y(y)); ctx.lineTo(p.X(s * ux), p.Y(s * uy)); ctx.stroke();
      dot(ctx, p.X(s * ux), p.Y(s * uy), 2.5, c.s1);
      dot(ctx, p.X(x), p.Y(y), 3.5, c.text, c.surface);
    });
    ctx.restore();
    label(ctx, `your line, θ = ${Math.round(th)}° (code = position along it)`, p.pad.l + 6, p.pad.t + 10, { size: 11, bold: true, color: c.s1 });
    label(ctx, `PCA direction θ* = ${tp.toFixed(0)}° (dashed)`, p.pad.l + 6, p.pad.t + 26, { size: 11, color: c.s3 });
    label(ctx, 'red = reconstruction error of each point', p.pad.l + 6, p.pad.t + 42, { size: 10, color: c.critical });
    const j = J(pts, th), jp = J(pts, tp);
    calc.innerHTML = `encode: h = uᵀx with u = (cos θ, sin θ) = (${fx(ux)}, ${fx(uy)})      decode: r = h·u\nJ(θ) = Σ‖x − r‖² = <b>${fx(j, 2)}</b>   (the red segments, squared and added up)\nbest line = first principal component: θ* = ${tp.toFixed(1)}°, J(θ*) = ${fx(jp, 2)}${shape === 'banana' ? `\n<span class="c">A straight line cannot follow a curve: even the best line leaves J = ${fx(jp, 2)}. A nonlinear encoder/decoder using the orange curve as its 1-number code would leave only J ≈ ${fx(curveErr(pts), 2)} — a nonlinear generalization of PCA.</span>` : '\n<span class="c">Minimizing the copying loss finds the PCA direction as a side effect — the autoencoder learned the principal subspace without being told to.</span>'}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) the identity trap ---------- */
Widgets.capacityTrap = (el) => {
  UI.head(el, 'The identity trap: a code as big as the input', 'Same bar images. With k = 64 code numbers the linear autoencoder can copy anything — even pure noise — perfectly. Perfect copying of everything means it learned nothing about what makes the data special.');
  let k = 16, seed = 3;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Code size k', min: 1, max: 64, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'New noise image', () => { seed++; draw(); });
  el.append(ctr, btns);
  const rows = h('div', { class: 'stack' });
  el.append(rows);
  const curves = (() => { const tr = [], nz = [], noise = BARS.noise(99); for (let kk = 1; kk <= 64; kk++) { tr.push([kk, mean(BARS.X.map(x => BARS.mse(x, BARS.recon(x, kk))))]); nz.push([kk, BARS.mse(noise, BARS.recon(noise, kk))]); } return { tr, nz }; })();
  const ymax = Math.max(...curves.tr.map(d => d[1]), ...curves.nz.map(d => d[1])) * 1.1;
  const chart = new LineChart(el, { xr: [1, 64], yr: [0, ymax], aspect: .4, minH: 170, maxH: 220, xlabel: 'code size k', ylabel: 'MSE', xname: 'k', vfmt: v => fx(v, 4), snapX: v => Math.round(v), aria: 'reconstruction error against code size' });
  const calc = calcBox();
  el.append(calc);
  function draw() {
    rows.innerHTML = '';
    const x = BARS.X[5], nz = BARS.noise(seed);
    const show = (img, cap) => { const r = BARS.recon(img, k), e = BARS.mse(img, r); rows.append(rowBox(gw(cap, imgGrid(img)), gop('→ f → g →'), gw(`r · MSE ${fx(e, 4)}`, imgGrid(r)))); return e; };
    const e1 = show(x, 'a training image'), e2 = show(nz, 'pure noise (no structure)');
    chart.set([{ name: 'training images', color: 's1', data: curves.tr }, { name: 'pure noise', color: 's2', data: curves.nz }, { name: 'current k', color: 'muted', data: [[k, 0], [k, ymax]], dash: [4, 4], noHover: true }]);
    calc.innerHTML = `k = ${k}:   MSE(training image) = ${fx(e1, 4)}     MSE(noise) = <b>${fx(e2, 4)}</b>\n<span class="c">${k >= 64 ? 'k = 64 = input size: both errors are 0. The autoencoder is the identity g(f(x)) = x — it copies noise as happily as data, so its code says nothing about the data distribution.' : k >= 40 ? 'Large code: noise is already copied quite well — the autoencoder is sliding toward the identity function.' : 'Small code: data are copied well and noise badly — the code is forced to capture the structure of the data (bars). That gap is the sign of a useful representation.'}</span>`;
  }
  draw();
  return { redraw: () => { chart.draw(); draw(); } };
};

/* ---------- 6) build a regularized autoencoder ---------- */
Widgets.regMenu = (el) => {
  UI.head(el, 'Build a regularized autoencoder', 'Switch the ingredients on and off. Each one gives the autoencoder a reason to learn structure instead of copying — so the code may even be bigger than the input (overcomplete) and still be useful.');
  let sp = false, dn = false, ct = false;
  const tog = h('div', { class: 'toggles' });
  UI.chip(tog, 'Sparsity penalty Ω(h) = λΣ|hᵢ|', '--s4', sp, v => { sp = v; draw(); });
  UI.chip(tog, 'Corrupt the input: x̃ = x + noise', '--s2', dn, v => { dn = v; draw(); });
  UI.chip(tog, 'Penalize derivatives: λ‖∂f/∂x‖²_F', '--s7', ct, v => { ct = v; draw(); });
  el.append(tog);
  const out = h('div', { class: 'math' }), name = h('div', { class: 'stat-row' }), notes = h('div', { class: 'compare' });
  el.append(out, name, notes);
  function draw() {
    const inp = dn ? '\\tilde{\\mathbf x}' : '\\mathbf x';
    let f = `L\\big(\\mathbf x,\\ g(f(${inp}))\\big)`;
    if (sp) f += ' + \\lambda\\sum_i |h_i|';
    if (ct) f += ' + \\lambda\\Big\\|\\tfrac{\\partial f(\\mathbf x)}{\\partial \\mathbf x}\\Big\\|_F^2';
    out.innerHTML = `\\[ J = ${f} \\]`;
    renderMath(out);
    const nm = [sp && 'sparse', dn && 'denoising', ct && 'contractive'].filter(Boolean);
    name.innerHTML = '';
    name.append(stat('this is a…', nm.length ? nm.join(' + ') + ' autoencoder' : 'plain autoencoder'), stat('code bigger than x allowed?', nm.length ? 'yes — it still learns structure' : 'no — it would just copy'), stat('what stops it from copying', nm.length ? 'the extra ingredient(s)' : 'only a small code (undercomplete)'));
    notes.innerHTML = '';
    notes.append(
      h('div', { style: sp ? 'outline:2px solid var(--s4)' : '' }, h('b', {}, 'Sparsity Ω(h) = λΣ|hᵢ|'), sp ? 'ON: for each input only a few code units may be non-zero. The code must respond to specific statistical features of the data.' : 'off — every unit may fire for every input.'),
      h('div', { style: dn ? 'outline:2px solid var(--s2)' : '' }, h('b', {}, 'Corrupted input x̃'), dn ? 'ON: the target is the clean x but the input is damaged. To undo the damage the network must know what clean data look like.' : 'off — input and target are the same x.'),
      h('div', { style: ct ? 'outline:2px solid var(--s7)' : '' }, h('b', {}, 'Derivative penalty ‖∂f/∂x‖²_F'), ct ? 'ON: features must change little when x changes slightly — the encoder contracts the neighbourhood of each training point.' : 'off — features may change as fast as they like.'));
  }
  draw();
  return { redraw: () => {} };
};

/* ---------- 7) how the L1 penalty makes a code sparse ---------- */
Widgets.sparseCode = (el) => {
  UI.head(el, 'How the L¹ penalty makes a code sparse', 'Suppose the reconstruction term alone would like code unit i to be aᵢ (grey). Adding λ|hᵢ| pulls every unit toward 0, and any unit with |aᵢ| ≤ λ becomes exactly 0 (yellow). Slide λ.');
  const a = [1.6, -0.2, 0.9, 0.15, -1.1, 0.4, 0.05, 2.0, -0.6, 0.3, -0.1, 0.7];
  let lam = 0.5;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Penalty strength λ', min: 0, max: 1.5, step: 0.05, value: lam, fmt: v => fx(v), onInput: v => { lam = v; draw(); } });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'code units before and after the sparsity penalty' });
  wrap.append(cv);
  const lg = UI.legend(el, []);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 240, ctx = fitCanvas(cv, W, H);
    const hv = a.map(v => Math.sign(v) * Math.max(Math.abs(v) - lam, 0));
    const p = makePlot(W, H, [-0.5, a.length - 0.5], [-1.5, 2.3], { l: 34, r: 10, t: 14, b: 30 });
    drawAxes(ctx, p, { xticks: a.map((_, i) => i), xfmt: t => 'h' + (t + 1), ylabel: 'value of the code unit' });
    ctx.fillStyle = alpha(c.s4, .12); ctx.fillRect(p.pad.l, p.Y(lam), p.pw, p.Y(-lam) - p.Y(lam));
    const bw = p.pw / a.length * 0.36;
    a.forEach((v, i) => {
      const X = p.X(i);
      ctx.fillStyle = alpha(c.muted, .45); ctx.fillRect(X - bw - 1, Math.min(p.Y(0), p.Y(v)), bw, Math.abs(p.Y(v) - p.Y(0)));
      ctx.fillStyle = hv[i] === 0 ? alpha(c.critical, .35) : c.s4; ctx.fillRect(X + 1, Math.min(p.Y(0), p.Y(hv[i])), bw, Math.max(2, Math.abs(p.Y(hv[i]) - p.Y(0))));
    });
    ctx.strokeStyle = c.axis; ctx.beginPath(); ctx.moveTo(p.pad.l, p.Y(0) + .5); ctx.lineTo(W - p.pad.r, p.Y(0) + .5); ctx.stroke();
    label(ctx, `dead zone: |a| ≤ λ = ${fx(lam)} → h = 0`, W - p.pad.r - 4, p.Y(lam) - 9, { align: 'right', size: 10, color: c.s4 });
    lg.set([{ color: alpha(c.muted, .6), label: 'aᵢ — what reconstruction alone wants' }, { color: c.s4, label: 'hᵢ — with the L¹ penalty' }, { color: alpha(c.critical, .5), label: 'switched off (exactly 0)' }]);
    const nz = hv.filter(v => v !== 0).length, om = lam * sum(hv.map(Math.abs)), rec = 0.5 * sum(a.map((v, i) => sq(v - hv[i])));
    stats.innerHTML = '';
    stats.append(stat('active units', `${nz} <small>/ 12</small>`), stat('Ω(h) = λΣ|hᵢ|', fx(om)), stat('reconstruction part ½Σ(aᵢ−hᵢ)²', fx(rec)), stat('total', fx(rec + om)));
    calc.innerHTML = `per unit: minimize ½(aᵢ − hᵢ)² + λ|hᵢ|   ⇒   hᵢ = sign(aᵢ)·max(|aᵢ| − λ, 0)   (the “soft threshold”)\nh₁: a = 1.6 → ${fx(hv[0])}      h₄: a = 0.15 → ${fx(hv[3])}      h₈: a = 2.0 → ${fx(hv[7])}      h₂: a = −0.2 → ${fx(hv[1])}\nΩ(h) = λ Σ|hᵢ| = ${fx(lam)} × ${fx(sum(hv.map(Math.abs)))} = <b>${fx(om)}</b>;   active units: <b>${nz} / 12</b>\n<span class="c">${lam === 0 ? 'λ = 0: every unit fires — no sparsity, nothing stops the identity function.' : nz <= 4 ? 'Only the strongest features survive: the code must respond to specific statistical features of the data instead of copying everything.' : 'Weak units are switched off and strong ones shrink by λ: the code becomes sparse.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 8) the sparsity penalty is a Laplace prior ---------- */
Widgets.laplacePrior = (el) => {
  UI.head(el, 'The sparsity penalty is a prior in disguise', 'Put a Laplace prior p(hᵢ) = (λ/2)·e^(−λ|hᵢ|) on each code unit. Its negative log is λ|hᵢ| + constant — exactly the L¹ penalty. A Gaussian prior with the same variance gives a parabola instead (that would be weight decay).');
  let lam = 1.5, view = 'nlog', h0 = 0.8;
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, [['pdf', 'the prior p(h)'], ['nlog', '−log p(h) = the penalty']], view, v => { view = v; draw(); });
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'λ', min: 0.3, max: 4, step: 0.1, value: lam, fmt: v => fx(v, 1), onInput: v => { lam = v; draw(); } });
  UI.slider(ctr, { label: 'inspect at h =', min: -3, max: 3, step: 0.1, value: h0, fmt: v => fx(v, 1), onInput: v => { h0 = v; draw(); } });
  el.append(seg, ctr);
  const chart = new LineChart(el, { xr: [-3, 3], yr: [0, 2], aspect: .45, minH: 180, maxH: 240, xlabel: 'h', xname: 'h', vfmt: v => fx(v, 3), aria: 'Laplace and Gaussian prior' });
  const calc = calcBox();
  el.append(calc);
  function draw() {
    const sd = Math.sqrt(2) / lam;
    const lap = x => lam / 2 * Math.exp(-lam * Math.abs(x)), gau = x => normPdf(x, 0, sd);
    const xs = []; for (let i = 0; i <= 240; i++) xs.push(-3 + 6 * i / 240);
    if (view === 'pdf') {
      const top = Math.max(lam / 2, gau(0)) * 1.1;
      chart.set([{ name: 'Laplace prior', color: 's4', data: xs.map(x => [x, lap(x)]) }, { name: 'Gaussian, same variance', color: 's1', data: xs.map(x => [x, gau(x)]) }, { name: 'h', color: 'muted', data: [[h0, 0], [h0, top]], dash: [4, 4], noHover: true }], { yr: [0, top], ylabel: 'p(h)' });
    } else {
      const nl = x => lam * Math.abs(x) - Math.log(lam / 2), ng = x => x * x / (2 * sd * sd) + Math.log(sd * Math.sqrt(2 * Math.PI));
      const top = Math.max(nl(3), ng(3)) * 1.05;
      chart.set([{ name: '−log p (Laplace) = λ|h| + c', color: 's4', data: xs.map(x => [x, nl(x)]) }, { name: '−log p (Gaussian) = h²/2σ² + c', color: 's1', data: xs.map(x => [x, ng(x)]) }, { name: 'h', color: 'muted', data: [[h0, 0], [h0, top]], dash: [4, 4], noHover: true }], { yr: [0, top], ylabel: '−log p(h)  (= penalty)' });
    }
    calc.innerHTML = `Laplace: p(h) = (λ/2)e^(−λ|h|)   at h = ${fx(h0, 1)}:  −log p = λ|h| − log(λ/2) = ${fx(lam, 1)}×${fx(Math.abs(h0), 1)} + ${fx(-Math.log(lam / 2))} = <b>${fx(lam * Math.abs(h0) - Math.log(lam / 2))}</b>\n   → the penalty λ|h| plus a constant that does not depend on h: Ω(h) = λΣ|hᵢ|\nGaussian with the same variance 2/λ² = ${fx(2 / (lam * lam))}:  −log p = h²/(2σ²) + const = <b>${fx(h0 * h0 / (2 * 2 / (lam * lam)))}</b> + ${fx(Math.log(sd * Math.sqrt(2 * Math.PI)))}\n   → a parabola: flat near 0, so it shrinks units but rarely makes them exactly 0; the V-shape has a kink at 0 that does\nMAP view: maximize log p(x | h) + log p(h)  ⇔  minimize  L(x, g(h)) + λΣ|hᵢ|  — the code describes latent variables that explain x`;
  }
  draw();
  return { redraw: () => chart.draw() };
};

/* ---------- 9) denoising ---------- */
Widgets.denoiseDemo = (el) => {
  UI.head(el, 'Denoising: input x̃ (corrupted), target x (clean)', 'The autoencoder sees a damaged copy and must output the original. Here encoder and decoder are the k-number linear autoencoder trained on clean bar images.');
  let img = 4, mode = 'gauss', lvl = 0.5, k = 16, seed = 1;
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, [['gauss', 'Gaussian noise'], ['mask', 'Mask pixels (set to 0)']], mode, v => { mode = v; draw(); });
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Corruption level', min: 0, max: 1, step: 0.05, value: lvl, fmt: v => fx(v), onInput: v => { lvl = v; draw(); } });
  UI.slider(ctr, { label: 'Code size k', min: 1, max: 64, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Other image', () => { img = (img + 5) % BARS.N; draw(); });
  UI.button(btns, 'New corruption', () => { seed++; draw(); });
  el.append(seg, ctr, btns);
  const rows = h('div', { class: 'stack' }), stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(rows, stats, calc);
  function draw() {
    rows.innerHTML = '';
    const x = BARS.X[img], rng = mulberry32(seed * 31 + img);
    const sg = lvl * 0.7, pm = lvl * 0.8;
    const xt = x.map(v => mode === 'gauss' ? clamp(v + sg * gaussian(rng), 0, 1) : (rng() < pm ? 0 : v));
    const r = BARS.recon(xt, k);
    const eIn = BARS.mse(xt, x), eOut = BARS.mse(r, x);
    rows.append(rowBox(gw('clean x (the target)', imgGrid(x)), gop('corrupt →'), gw(`x̃ · MSE ${fx(eIn, 4)}`, imgGrid(xt)), gop('→ f → g →'), gw(`g(f(x̃)) · MSE ${fx(eOut, 4)}`, imgGrid(r))));
    stats.innerHTML = '';
    const removed = eIn > 1e-9 ? 100 * (1 - eOut / eIn) : 0;
    stats.append(stat('error before (x̃ vs x)', fx(eIn, 4)), stat('error after (g(f(x̃)) vs x)', fx(eOut, 4)), stat('damage removed', (removed > 0 ? '' : '') + removed.toFixed(0) + '%'));
    calc.innerHTML = `x̃ = ${mode === 'gauss' ? `x + ε,  ε ~ N(0, σ²) with σ = ${fx(sg)}` : `x with a fraction p = ${fx(pm)} of the pixels set to 0`}\nL(x, g(f(x̃))) compares the output with the CLEAN x:   ‖x̃ − x‖²/64 = ${fx(eIn, 4)} before,   ‖g(f(x̃)) − x‖²/64 = <b>${fx(eOut, 4)}</b> after\n<span class="c">${k >= 48 ? 'With a huge code the autoencoder passes the noise straight through (back in the identity trap): denoising needs the code to prefer clean structure.' : removed > 30 ? 'To undo the corruption, f and g must know what clean data look like (bars). That knowledge is the structure of p_data(x) — learned as a by-product of minimizing reconstruction error.' : 'Little damage to remove here — raise the corruption level to see the denoising at work.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 10) the vector field a DAE learns ---------- */
Widgets.manifoldField = (el) => {
  UI.head(el, 'What a denoising autoencoder learns: a field that points back to the data', 'Green curve = where the data live (a 1-D manifold inside 2-D space). Each arrow shows r(x̃) − x̃: the reconstruction moves a corrupted point back toward the manifold. Click anywhere to inspect one arrow.');
  let noise = 0.1, showPairs = true, seed = 1, pick = [0.3, 0.8];
  const curve = s => [0.08 + 0.84 * s, 0.5 + 0.28 * Math.sin(2 * Math.PI * s)];
  function nearest(x) { let best = null, bd = 1e9; for (let i = 0; i <= 300; i++) { const q = curve(i / 300), d = sq(q[0] - x[0]) + sq(q[1] - x[1]); if (d < bd) { bd = d; best = q; } } return best; }
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Corruption size σ', min: 0.02, max: 0.25, step: 0.01, value: noise, fmt: v => fx(v), onInput: v => { noise = v; draw(); } });
  const tog = h('div', { class: 'toggles' });
  UI.chip(tog, 'show training pairs (clean x → corrupted x̃)', '--s2', showPairs, v => { showPairs = v; draw(); });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Resample the corruptions', () => { seed++; draw(); });
  el.append(ctr, h('div', { class: 'controls' }, tog, btns));
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart clickable', role: 'img', 'aria-label': 'vector field pointing toward the data manifold' });
  wrap.append(cv);
  const calc = calcBox();
  el.append(wrap, calc);
  let P = null;
  cv.addEventListener('click', e => { if (!P) return; const r = cv.getBoundingClientRect(); pick = [clamp(P.invX(e.clientX - r.left), 0, 1), clamp(P.invY(e.clientY - r.top), 0, 1)]; draw(); });
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 320, ctx = fitCanvas(cv, W, H);
    const p = makePlot(W, H, [0, 1], [0, 1], { l: 28, r: 10, t: 10, b: 24 }); P = p;
    drawAxes(ctx, p, { grid: false });
    for (let i = 0; i < 12; i++) for (let j = 0; j < 8; j++) {
      const g = [0.06 + 0.88 * (i + .5) / 12, 0.06 + 0.88 * (j + .5) / 8], n = nearest(g);
      const dx = n[0] - g[0], dy = n[1] - g[1], L = Math.hypot(dx, dy);
      if (L < 0.02) continue;
      const f = Math.min(1, 0.09 / L) * 0.85;
      arrow(ctx, p.X(g[0]), p.Y(g[1]), p.X(g[0] + dx * f), p.Y(g[1] + dy * f), alpha(c.s1, .55), 1.2, 5);
    }
    ctx.strokeStyle = c.s3; ctx.lineWidth = 3; ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const q = curve(i / 200), X = p.X(q[0]), Y = p.Y(q[1]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); }
    ctx.stroke();
    const rng = mulberry32(seed * 17 + 3);
    for (let i = 0; i < 24; i++) {
      const x = curve((i + .5) / 24);
      if (showPairs) {
        const xt = [x[0] + noise * gaussian(rng), x[1] + noise * gaussian(rng)];
        ctx.strokeStyle = alpha(c.s2, .5); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.X(x[0]), p.Y(x[1])); ctx.lineTo(p.X(xt[0]), p.Y(xt[1])); ctx.stroke();
        dot(ctx, p.X(xt[0]), p.Y(xt[1]), 3, c.s2);
      }
      dot(ctx, p.X(x[0]), p.Y(x[1]), 3, c.s3, c.surface);
    }
    const n = nearest(pick);
    arrow(ctx, p.X(pick[0]), p.Y(pick[1]), p.X(n[0]), p.Y(n[1]), c.text, 2.2, 8);
    dot(ctx, p.X(pick[0]), p.Y(pick[1]), 5.5, c.text, c.surface);
    label(ctx, 'x̃', p.X(pick[0]) + 9, p.Y(pick[1]) - 9, { size: 12, bold: true, color: c.text });
    label(ctx, 'r(x̃)', p.X(n[0]) + 9, p.Y(n[1]) + 10, { size: 11, bold: true, color: c.s3 });
    const d = [n[0] - pick[0], n[1] - pick[1]];
    calc.innerHTML = `clicked point x̃ = ${vec(pick)}\nideal reconstruction r(x̃) = g(f(x̃)) = the closest point of the data manifold = ${vec(n)}\nr(x̃) − x̃ = ${vec(d)}   length ${fx(Math.hypot(d[0], d[1]), 3)}  → points straight back toward the data\n<span class="c">Training pairs (orange → green): the network learns “from x̃, output x”. Averaged over many pairs, it learns this whole vector field — an implicit description of the structure of p_data(x). It is only reliable near the data (within about σ of the curve).</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 11) contraction of a neighbourhood ---------- */
Widgets.contractDemo = (el) => {
  UI.head(el, 'Contraction: a small input neighbourhood → a smaller code neighbourhood', 'Encoder h = f(x) = tanh(Wx) with W = gain × a fixed 2×2 matrix. A circle of inputs around x₀ is mapped to an ellipse of codes. The penalty is the squared Frobenius norm of the Jacobian ∂f/∂x at x₀.');
  let gain = 1, x0 = [0.6, -0.3], lam = 0.1;
  const W0 = [[1, 0.5], [-0.3, 1]];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Gain of W', min: 0.2, max: 3, step: 0.05, value: gain, fmt: v => fx(v), onInput: v => { gain = v; draw(); } });
  UI.slider(ctr, { label: 'x₀₁', min: -2, max: 2, step: 0.1, value: x0[0], fmt: v => fx(v, 1), onInput: v => { x0[0] = v; draw(); } });
  UI.slider(ctr, { label: 'x₀₂', min: -2, max: 2, step: 0.1, value: x0[1], fmt: v => fx(v, 1), onInput: v => { x0[1] = v; draw(); } });
  UI.slider(ctr, { label: 'λ', min: 0, max: 1, step: 0.05, value: lam, fmt: v => fx(v), onInput: v => { lam = v; draw(); } });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'input neighbourhood and its image under the encoder' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  const f = x => { const W = W0.map(r => r.map(v => v * gain)); return [Math.tanh(W[0][0] * x[0] + W[0][1] * x[1]), Math.tanh(W[1][0] * x[0] + W[1][1] * x[1])]; };
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 250, ctx = fitCanvas(cv, W, H);
    const half = (W - 24) / 2;
    const pL = makePlot(half, H, [-2.2, 2.2], [-2.2, 2.2], { l: 30, r: 6, t: 20, b: 22 });
    const pR = makePlot(half, H, [-1.2, 1.2], [-1.2, 1.2], { l: 30, r: 6, t: 20, b: 22 });
    const Wm = W0.map(r => r.map(v => v * gain));
    const hv = f(x0);
    const J = [[(1 - sq(hv[0])) * Wm[0][0], (1 - sq(hv[0])) * Wm[0][1]], [(1 - sq(hv[1])) * Wm[1][0], (1 - sq(hv[1])) * Wm[1][1]]];
    const frob = sum(J.flat().map(sq)), det = Math.abs(J[0][0] * J[1][1] - J[0][1] * J[1][0]);
    const rad = 0.35;
    const circ = []; for (let i = 0; i <= 60; i++) { const a = i / 60 * 2 * Math.PI; circ.push([x0[0] + rad * Math.cos(a), x0[1] + rad * Math.sin(a)]); }
    const drawPanel = (p, ox, pts, center, title, col) => {
      ctx.save(); ctx.translate(ox, 0);
      drawAxes(ctx, p, { grid: false });
      label(ctx, title, p.pad.l + 4, 8, { size: 11, bold: true, color: col });
      ctx.save(); p.clip(ctx);
      ctx.fillStyle = alpha(col, .18); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath();
      pts.forEach((q, i) => { const X = p.X(q[0]), Y = p.Y(q[1]); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
      ctx.closePath(); ctx.fill(); ctx.stroke();
      // a few marked points on the boundary (to see the mapping)
      [0, 15, 30, 45].forEach((i, k) => dot(ctx, p.X(pts[i][0]), p.Y(pts[i][1]), 4, [c.s1, c.s2, c.s3, c.s7][k], c.surface));
      dot(ctx, p.X(center[0]), p.Y(center[1]), 4.5, c.text, c.surface);
      ctx.restore(); ctx.restore();
    };
    drawPanel(pL, 0, circ, x0, 'input space: circle of radius 0.35 around x₀', c.text2);
    drawPanel(pR, half + 24, circ.map(f), hv, 'code space: its image under f', c.s4);
    arrow(ctx, half + 2, H / 2, half + 22, H / 2, c.muted, 2, 7);
    stats.innerHTML = '';
    stats.append(stat('‖∂f/∂x‖²_F', fx(frob, 3)), stat('penalty λ‖·‖²_F', fx(lam * frob, 3)), stat('area ratio |det J|', fx(det, 3)), stat('verdict', det < 1 ? 'contracted' : 'expanded'));
    calc.innerHTML = `W = gain × [[1, 0.5], [−0.3, 1]] = [[${fx(Wm[0][0])}, ${fx(Wm[0][1])}], [${fx(Wm[1][0])}, ${fx(Wm[1][1])}]]\nh = f(x₀) = tanh(W x₀) = ${vec(hv)}\nJacobian ∂f/∂x = diag(1 − h²)·W = [[${fx(J[0][0])}, ${fx(J[0][1])}], [${fx(J[1][0])}, ${fx(J[1][1])}]]   (row i = ∇ₓhᵢ)\n‖∂f/∂x‖²_F = ${J.flat().map(v => fx(v) + '²').join(' + ')} = <b>${fx(frob, 3)}</b>;   Ω = λ‖·‖²_F = ${fx(lam)} × ${fx(frob, 3)} = <b>${fx(lam * frob, 3)}</b>\n<span class="c">${frob < 0.5 ? 'Small Jacobian: nearby inputs get nearly the same code — the neighbourhood is contracted. That is what the CAE penalty asks for at training points.' : 'Large Jacobian: the code reacts strongly to tiny input changes. Lower the gain, or move x₀ into the saturated region of tanh, to contract.'} The encoder must still change ALONG the data (to tell examples apart), so it contracts mostly in directions across the data manifold.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 12) small noise turns denoising into a contractive penalty ---------- */
Widgets.daeCaeCompare = (el) => {
  UI.head(el, 'Small Gaussian noise turns denoising into a contractive penalty', 'One training point x₀ = 0. Near it the reconstruction function is a line r(x) = b + s·x. Corrupt x₀ with ε ~ N(0, σ²) and ask the output to be x₀: the average error is b² + σ²s² — the second term penalizes the slope (derivative) of r, i.e. it is a contractive penalty on g∘f.');
  let b = 0.2, s = 0.6, sig = 0.6, seed = 2;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Offset b = r(x₀) − x₀', min: -1, max: 1, step: 0.05, value: b, fmt: v => fx(v), onInput: v => { b = v; draw(); } });
  UI.slider(ctr, { label: 'Slope s = r′(x₀)', min: -1.5, max: 1.5, step: 0.05, value: s, fmt: v => fx(v), onInput: v => { s = v; draw(); } });
  UI.slider(ctr, { label: 'Noise size σ', min: 0.05, max: 1.2, step: 0.05, value: sig, fmt: v => fx(v), onInput: v => { sig = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Resample the noise', () => { seed++; draw(); });
  el.append(ctr, btns);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'reconstruction line and noisy samples' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 260, ctx = fitCanvas(cv, W, H);
    const p = makePlot(W, H, [-3, 3], [-2.5, 2.5], { l: 34, r: 10, t: 12, b: 26 });
    drawAxes(ctx, p, { xlabel: 'input x', ylabel: 'output r(x)' });
    ctx.save(); p.clip(ctx);
    ctx.setLineDash([4, 4]); ctx.strokeStyle = c.muted; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.X(-3), p.Y(-3)); ctx.lineTo(p.X(3), p.Y(3)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = c.good; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(p.X(-3), p.Y(0)); ctx.lineTo(p.X(3), p.Y(0)); ctx.stroke();
    ctx.strokeStyle = c.s2; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(p.X(-3), p.Y(b - 3 * s)); ctx.lineTo(p.X(3), p.Y(b + 3 * s)); ctx.stroke();
    const rng = mulberry32(seed * 101 + 7);
    let mc = 0; const N = 300, shown = [];
    for (let i = 0; i < N; i++) { const e = sig * gaussian(rng), r = b + s * e; mc += sq(r); if (i < 28) shown.push([e, r]); }
    mc /= N;
    shown.forEach(([e, r]) => { ctx.strokeStyle = alpha(c.critical, .45); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.X(e), p.Y(r)); ctx.lineTo(p.X(e), p.Y(0)); ctx.stroke(); dot(ctx, p.X(e), p.Y(-2.4), 3, alpha(c.s1, .6)); dot(ctx, p.X(e), p.Y(r), 3.5, c.s2, c.surface); });
    dot(ctx, p.X(0), p.Y(0), 6, c.good, c.surface);
    ctx.restore();
    label(ctx, 'target: output = x₀ = 0 (green line)', p.pad.l + 6, p.Y(0) - 10, { size: 10, color: c.good });
    label(ctx, 'r(x) = b + s·x', p.X(2.2), p.Y(b + 2.2 * s) - 12, { size: 11, bold: true, color: c.s2, align: 'center' });
    label(ctx, 'identity (dashed)', p.X(2.4), p.Y(2.4) + 12, { size: 10, color: c.muted, align: 'center' });
    label(ctx, 'blue ticks: corrupted inputs x̃ = x₀ + ε', p.pad.l + 6, p.Y(-2.4) - 10, { size: 10, color: c.s1 });
    const th = b * b + sig * sig * s * s;
    stats.innerHTML = '';
    stats.append(stat('b²', fx(b * b, 3)), stat('σ²·s²', fx(sig * sig * s * s, 3)), stat('theory: b² + σ²s²', fx(th, 3)), stat('average of 300 samples', fx(mc, 3)));
    calc.innerHTML = `E_ε[(r(x₀ + ε) − x₀)²] = E[(b + sε)²] = b² + 2bs·E[ε] + s²·E[ε²] = b² + σ²s² = ${fx(b * b, 3)} + ${fx(sig * sig, 3)}×${fx(s * s, 3)} = <b>${fx(th, 3)}</b>\n<b>Denoising AE</b>: finite perturbations of size σ; penalizes the slope of the RECONSTRUCTION function r = g∘f (through σ²·r′²)\n<b>Contractive AE</b>: λ·‖∂f/∂x‖² — penalizes the slope of the FEATURE function f at x₀ itself (infinitesimal perturbations)\n<span class="c">Both say: near the data, the function should barely react to small input changes. ${Math.abs(s) < 0.1 ? 'Slope ≈ 0 — the reconstruction ignores the noise completely.' : 'Flatten the slope s to see the noise term vanish.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 13) shallow or deep ---------- */
Widgets.depthBuilder = (el) => {
  UI.head(el, 'Shallow or deep autoencoder?', 'Build the encoder (the decoder mirrors it). Count the parameters and see what each depth can and cannot do.');
  let layers = 1, width = 32, k = 8;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Hidden layers in the encoder', min: 0, max: 3, step: 1, value: layers, fmt: v => String(v), onInput: v => { layers = v; draw(); } });
  UI.slider(ctr, { label: 'Hidden width', min: 8, max: 128, step: 8, value: width, fmt: v => String(v), onInput: v => { width = v; draw(); } });
  UI.slider(ctr, { label: 'Code size k', min: 1, max: 64, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'layer sizes of the autoencoder' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox(), note = h('div', { class: 'callout info' });
  el.append(wrap, stats, calc, note);
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 190, ctx = fitCanvas(cv, W, H);
    const sizes = [64, ...Array(layers).fill(width), k, ...Array(layers).fill(width), 64];
    const n = sizes.length, X = i => 30 + (i + .5) * (W - 60) / n, maxH = 120;
    let params = 0; const parts = [];
    for (let i = 1; i < n; i++) { const pm = sizes[i - 1] * sizes[i] + sizes[i]; params += pm; parts.push(`${sizes[i - 1]}×${sizes[i]} + ${sizes[i]} = ${pm.toLocaleString('en-US')}`); }
    for (let i = 0; i < n; i++) {
      const hgt = 14 + maxH * Math.sqrt(sizes[i] / 128), x = X(i), y = 25 + (maxH + 14 - hgt) / 2;
      const col = i === (n - 1) / 2 ? c.s4 : i < (n - 1) / 2 ? c.s1 : c.s2;
      ctx.fillStyle = alpha(col, .3); ctx.strokeStyle = col; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x - 16, y, 32, hgt, 6) : ctx.rect(x - 16, y, 32, hgt); ctx.fill(); ctx.stroke();
      label(ctx, String(sizes[i]), x, y + hgt / 2, { align: 'center', size: 11, bold: true, color: c.text });
      if (i < n - 1) arrow(ctx, x + 18, 25 + (maxH + 14) / 2, X(i + 1) - 18, 25 + (maxH + 14) / 2, c.muted, 1.2, 6);
    }
    label(ctx, 'x (64)', X(0), 12, { align: 'center', size: 10, color: c.text2 });
    label(ctx, 'code h', X((n - 1) / 2), 12, { align: 'center', size: 10, bold: true, color: c.s4 });
    label(ctx, 'r (64)', X(n - 1), 12, { align: 'center', size: 10, color: c.text2 });
    label(ctx, 'encoder f', X((n - 1) / 4), H - 10, { align: 'center', size: 10, color: c.s1 });
    label(ctx, 'decoder g', X(3 * (n - 1) / 4), H - 10, { align: 'center', size: 10, color: c.s2 });
    stats.innerHTML = '';
    stats.append(stat('weight layers', String(n - 1)), stat('parameters', params.toLocaleString('en-US')), stat('type', layers === 0 ? 'linear' : layers === 1 ? 'shallow (1 hidden layer per side)' : 'deep'));
    calc.innerHTML = `parameters per weight layer (in × out + biases):\n${parts.join('\n')}\ntotal = <b>${params.toLocaleString('en-US')}</b>`;
    const msgs = {
      0: ['L', 'Linear encoder and decoder: with MSE this is PCA when k < 64, and the useless identity when k ≥ 64. It can only find linear structure.'],
      1: ['1', 'One hidden layer per side: by the universal approximation theorem (L4) it can represent the identity along the data arbitrarily well — but it is hard to enforce extra constraints such as sparsity with a single layer.'],
      2: ['D', 'Deep encoder and decoder: depth can reduce the computational cost and the amount of data needed to represent some functions exponentially (the depth argument of L4). Experimentally, deep autoencoders compress much better than shallow or linear ones.'],
      3: ['D', 'Deep encoder and decoder: depth can reduce the computational cost and the amount of data needed to represent some functions exponentially (the depth argument of L4). Experimentally, deep autoencoders compress much better than shallow or linear ones.']
    };
    note.innerHTML = `<span class="ico">${msgs[layers][0]}</span><div>${msgs[layers][1]}</div>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 14) the code organizes the dataset ---------- */
Widgets.codeMap = (el) => {
  UI.head(el, 'Two code numbers are enough to organize the whole dataset', 'Every training image plotted by its first two code numbers (h₁, h₂). Colour = what kind of image it is — the autoencoder never saw these labels. Click a point to see its image and its 3 nearest neighbours in code space.');
  let sel = 10;
  const codes = BARS.X.map(x => BARS.encode(x, 2));
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart clickable', role: 'img', 'aria-label': 'scatter of images in the 2-D code space' });
  wrap.append(cv);
  const lg = UI.legend(el, []);
  const rows = h('div', { class: 'stack' }), calc = calcBox();
  el.append(wrap, rows, calc);
  let P = null;
  cv.addEventListener('click', e => {
    if (!P) return;
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    let best = 0, bd = 1e9;
    codes.forEach((hv, i) => { const d = sq(P.X(hv[0]) - px) + sq(P.Y(hv[1]) - py); if (d < bd) { bd = d; best = i; } });
    sel = best; draw();
  });
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 300, ctx = fitCanvas(cv, W, H);
    const xs = codes.map(v => v[0]), ys = codes.map(v => v[1]);
    const padv = 0.3;
    const p = makePlot(W, H, [Math.min(...xs) - padv, Math.max(...xs) + padv], [Math.min(...ys) - padv, Math.max(...ys) + padv], { l: 36, r: 10, t: 10, b: 26 }); P = p;
    drawAxes(ctx, p, { xlabel: 'h₁ (first code number)', ylabel: 'h₂' });
    const cols = [c.s1, c.s2, c.s7];
    const dist = codes.map((hv, i) => [Math.hypot(hv[0] - codes[sel][0], hv[1] - codes[sel][1]), i]).sort((a, b) => a[0] - b[0]);
    const nn = dist.slice(1, 4).map(d => d[1]);
    codes.forEach((hv, i) => {
      const X = p.X(hv[0]), Y = p.Y(hv[1]);
      if (nn.includes(i)) { ctx.strokeStyle = c.text; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.X(codes[sel][0]), p.Y(codes[sel][1])); ctx.lineTo(X, Y); ctx.stroke(); }
      dot(ctx, X, Y, i === sel ? 7 : 4.5, cols[BARS.fam[i]], i === sel ? c.text : c.surface);
    });
    lg.set(BARS.FAM.map((f, i) => ({ color: cols[i], label: f, kind: 'dot' })));
    rows.innerHTML = '';
    rows.append(rowBox(gw(`selected #${sel + 1}`, imgGrid(BARS.X[sel])), gop('nearest in code space →'), ...nn.map((i, k) => gw(`#${i + 1} · d = ${fx(dist[k + 1][0])}`, imgGrid(BARS.X[i])))));
    calc.innerHTML = `image #${sel + 1} (${BARS.FAM[BARS.fam[sel]]}):  h = (h₁, h₂) = (${fx(codes[sel][0])}, ${fx(codes[sel][1])})\nnearest neighbours by ‖h − h′‖: ${nn.map((i, k) => `#${i + 1} (d = ${fx(dist[k + 1][0])}, ${BARS.FAM[BARS.fam[i]]})`).join(', ')}\n<span class="c">Semantically related examples end up near each other in the small code space, although no labels were used. A classifier (or a search) working in this 2-D space has an easier job than one working on 64 raw pixels — that is why dimensionality reduction can improve generalization.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 15) semantic hashing ---------- */
Widgets.hashRetrieval = (el) => {
  UI.head(el, 'Semantic hashing: binary codes as a hash table', 'Threshold the first m code numbers at 0 to get m bits. Images with the same bits land in the same bucket, so finding look-alikes is one table lookup instead of comparing the query with every image.');
  let m = 3, q = 7;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Bits m', min: 1, max: 6, step: 1, value: m, fmt: v => String(v), onInput: v => { m = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Random query image', () => { q = Math.floor(Math.random() * BARS.N); draw(); });
  el.append(ctr, btns);
  const head = h('div', { class: 'stack' }), table = h('div', { class: 'grids', style: 'align-items:flex-start' }), stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(head, table, stats, calc);
  const codes = BARS.X.map(x => BARS.encode(x, 6));
  function draw() {
    const bits = hv => hv.slice(0, m).map(v => (v > 0 ? 1 : 0)).join('');
    const buckets = new Map();
    codes.forEach((hv, i) => { const b = bits(hv); if (!buckets.has(b)) buckets.set(b, []); buckets.get(b).push(i); });
    const qb = bits(codes[q]);
    head.innerHTML = '';
    head.append(rowBox(gw('query', imgGrid(BARS.X[q])), gop('→ f → sign →'), gw('its bits', h('div', { class: 'io' }, h('span', { class: 'box', style: 'font-weight:700' }, qb))), gop('→ bucket →'), gw(`${buckets.get(qb).length} images returned`, h('div', { class: 'grids', style: 'gap:4px' }, buckets.get(qb).filter(i => i !== q).slice(0, 8).map(i => imgGrid(BARS.X[i], { cs: 9, onClick: () => { q = i; draw(); }, title: 'click to make this the query' }))))));
    table.innerHTML = '';
    [...buckets.keys()].sort().forEach(b => {
      const list = buckets.get(b);
      const col = h('div', { class: 'gwrap', style: `padding:6px;border-radius:8px;border:1px solid var(--border);${b === qb ? 'background:var(--accent-soft);outline:2px solid var(--accent)' : ''}` },
        h('div', { class: 'gcap', style: 'font-family:var(--mono)' }, `${b} (${list.length})`),
        h('div', { class: 'grids', style: 'gap:3px;max-width:120px' }, list.slice(0, 12).map(i => imgGrid(BARS.X[i], { cs: 5, onClick: () => { q = i; draw(); }, title: `#${i + 1} — click to query` }))));
      table.append(col);
    });
    stats.innerHTML = '';
    stats.append(stat('possible buckets 2^m', String(2 ** m)), stat('buckets in use', String(buckets.size)), stat('linear scan', '72 comparisons'), stat('hash lookup', `1 lookup + ${buckets.get(qb).length - 1} candidates`));
    calc.innerHTML = `query code (first ${m} numbers): (${codes[q].slice(0, m).map(v => fx(v)).join(', ')})\nbits = [hⱼ > 0] = <b>${qb}</b>   → hash-table key\nbucket ${qb} holds ${buckets.get(qb).length} images (the query itself included) — returned without computing a single distance\n<span class="c">More bits = finer buckets (more selective, fewer results); fewer bits = coarser. Hamming-distance search (buckets whose key differs by one bit) extends it to “slightly less similar” items. Low-dimensional binary codes make retrieval extremely efficient.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- quiz ---------- */
window.QUIZ = [
  { q: 'An autoencoder is trained to…', opts: ['predict a class label from x', 'copy its input to its output, imperfectly', 'generate random images', 'minimize the number of layers'], a: 1, why: 'It attempts to copy x to r = g(f(x)); it is designed so that it cannot copy perfectly, which forces it to prioritize useful aspects of the data.' },
  { q: 'An autoencoder whose code h has fewer dimensions than x is called…', opts: ['overcomplete', 'undercomplete', 'contractive', 'stochastic'], a: 1, why: 'Undercomplete = code dimension < input dimension. The small code forces the model to capture the most salient features.' },
  { q: 'A linear decoder with mean squared error learns…', opts: ['the same subspace as PCA', 'a sparse code', 'a denoising function', 'nothing useful'], a: 0, why: 'An undercomplete autoencoder with a linear decoder and MSE learns to span the principal subspace of the training data, as a side effect of the copying task.' },
  { q: 'If the code is allowed to be as large as the input, a plain linear autoencoder…', opts: ['always learns PCA', 'can copy the input without learning anything about the data', 'cannot be trained', 'becomes sparse'], a: 1, why: 'With code dimension ≥ input dimension even a linear encoder/decoder can learn the identity function, which carries no information about the data distribution.' },
  { q: 'Regularized autoencoders prevent trivial copying by…', opts: ['keeping the code tiny only', 'adding a loss term or changing the input so that other properties (sparsity, robustness, small derivatives) are needed', 'using fewer training examples', 'removing the decoder'], a: 1, why: 'They use a loss that encourages extra properties, so even a nonlinear overcomplete model learns something useful.' },
  { q: 'The sparse autoencoder penalty Ω(h) = λΣ|hᵢ| corresponds to…', opts: ['a Gaussian prior on h', 'a Laplace prior on h', 'a uniform prior on x', 'no prior at all'], a: 1, why: '−log of the Laplace density (λ/2)e^(−λ|h|) is λ|h| plus a constant: the L¹ penalty is the negative log of a Laplace prior.' },
  { q: 'A denoising autoencoder minimizes…', opts: ['L(x, g(f(x)))', 'L(x̃, g(f(x̃)))', 'L(x, g(f(x̃))) with x̃ a corrupted copy of x', 'L(x̃, x)'], a: 2, why: 'The input is corrupted but the target is the clean x, so the model must undo the corruption rather than copy.' },
  { q: 'The contractive penalty is…', opts: ['the squared Frobenius norm of the Jacobian of the encoder', 'the L¹ norm of the weights', 'the number of hidden units', 'the reconstruction error'], a: 0, why: 'Ω(h) = λ‖∂f(x)/∂x‖²_F: the sum of squared partial derivatives of the code with respect to the input.' },
  { q: 'Why is the CAE called “contractive”?', opts: ['it shrinks the weights to zero', 'it maps a neighbourhood of inputs to a smaller neighbourhood of codes', 'it reduces the number of layers', 'it compresses the training set'], a: 1, why: 'Resisting perturbations means nearby inputs get nearly the same code: the input neighbourhood is contracted.' },
  { q: 'In the limit of small Gaussian input noise, the denoising error is equivalent to…', opts: ['a sparsity penalty', 'a contractive penalty on the reconstruction function g∘f', 'weight decay', 'early stopping'], a: 1, why: 'The expected denoising error becomes a penalty on the derivative of the reconstruction function. DAE: finite perturbations of r; CAE: infinitesimal perturbations of f.' },
  { q: 'Deep autoencoders, compared with shallow ones,…', opts: ['always overfit', 'experimentally give much better compression', 'cannot be trained with backprop', 'are always linear'], a: 1, why: 'Depth brings the usual advantages (exponential efficiency for some functions) and, experimentally, much better compression.' },
  { q: 'Semantic hashing works by…', opts: ['sorting images by file size', 'training a low-dimensional binary code and using it as a hash-table key', 'comparing the query with every database entry', 'storing images in a tree'], a: 1, why: 'Binary codes index a hash table; retrieval returns every entry with the same code (or a code within a small Hamming distance) — extremely efficient.' }
];
