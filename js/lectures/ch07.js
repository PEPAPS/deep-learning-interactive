/* =========================================================
   Lecture 6 · Ch 7 · Regularization
   ========================================================= */
'use strict';

/* ---------- 1) Why we do not penalize the bias ---------- */
Widgets.biasPenalty = (el) => {
  UI.head(el, 'Penalize the weights — but not the bias', 'Data y = 5 + 0.5x + noise. Both fits use the same penalty strength α; one also penalizes the bias b.');
  const rng = mulberry32(4);
  const pts = Array.from({ length: 12 }, () => { const x = 4 * rng() - 2; return [x, 5 + 0.5 * x + 0.4 * gaussian(rng)]; });
  let lg = 0;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Penalty strength α', min: -2, max: 2, step: 0.05, value: lg, fmt: v => (10 ** v).toFixed(2), onInput: v => { lg = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [-2.2, 2.2], yr: [0, 7.5], xlabel: 'x', ylabel: 'y', xname: 'x', aspect: .5, maxH: 280 });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function fit(penB) {
    const a = 10 ** lg, m = pts.length;
    const sx = pts.reduce((s, p) => s + p[0], 0), sxx = pts.reduce((s, p) => s + p[0] * p[0], 0), sy = pts.reduce((s, p) => s + p[1], 0), sxy = pts.reduce((s, p) => s + p[0] * p[1], 0);
    return solveLinear([[m / m + (penB ? a : 0), sx / m], [sx / m, sxx / m + a]], [sy / m, sxy / m]);
  }
  function draw() {
    const c = Colors.get();
    const A = fit(false), B = fit(true);
    const line = s => [[-2.2, s[0] - 2.2 * s[1]], [2.2, s[0] + 2.2 * s[1]]];
    chart.set([
      { name: 'weights only', color: 's3', data: line(A) },
      { name: 'weights + bias', color: 's8', data: line(B), dash: [6, 4] },
      { name: 'data', color: 's1', style: 'dots', r: 4, noHover: true, data: pts }
    ]);
    lgd.set([{ label: 'penalty on w only (recommended)', color: c.s3 }, { label: 'penalty on w and b', color: c.s8, kind: 'dash' }, { label: 'data', color: c.s1, kind: 'dot' }]);
    const mseOf = s => mean(pts.map(([x, y]) => (s[0] + s[1] * x - y) ** 2));
    calc.innerHTML = `<span class="c">J̃ = MSE + α·w²   vs   J̃ = MSE + α·(w² + b²),   α = ${(10 ** lg).toFixed(2)}</span>
w only:  b = ${A[0].toFixed(3)}, w = ${A[1].toFixed(3)}, training MSE = <b>${mseOf(A).toFixed(3)}</b>
w and b: b = ${B[0].toFixed(3)}, w = ${B[1].toFixed(3)}, training MSE = <b>${mseOf(B).toFixed(3)}</b>  <span class="c">← the bias is pulled toward 0 → underfitting</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) L2 shrinks along Hessian eigen-directions (fig. 7.1) ---------- */
Widgets.l2Shrink = (el) => {
  UI.head(el, 'Weight decay pulls w* toward 0 — mostly along “unimportant” directions', 'Ellipses: contours of the unregularized loss around its minimum w*. The regularized solution w̃ is where the loss and the L2 penalty balance.');
  let l1 = 0.4, l2 = 4, al = 1;
  const ws = [2, 1.5];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Curvature λ₁ along w₁ (eigenvalue)', min: 0.05, max: 6, step: 0.05, value: l1, fmt: v => v.toFixed(2), onInput: v => { l1 = v; draw(); } });
  UI.slider(ctr, { label: 'Curvature λ₂ along w₂', min: 0.05, max: 6, step: 0.05, value: l2, fmt: v => v.toFixed(2), onInput: v => { l2 = v; draw(); } });
  UI.slider(ctr, { label: 'Weight decay α', min: 0, max: 6, step: 0.05, value: al, fmt: v => v.toFixed(2), onInput: v => { al = v; draw(); } });
  const wrap = h('div', { class: 'canvas-wrap', style: 'max-width:420px;margin:0 auto' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Contours of loss and regularizer' });
  wrap.append(cv);
  el.append(ctr, wrap);
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const W = Math.min(420, boxWidth(wrap));
    const ctx = fitCanvas(cv, W, W);
    const dom = [-0.6, 3, -0.6, 3];
    const P = makePlot(W, W, [dom[0], dom[1]], [dom[2], dom[3]], { l: 0, r: 0, t: 0, b: 0 });
    const f = (x, y) => 0.5 * (l1 * (x - ws[0]) ** 2 + l2 * (y - ws[1]) ** 2);
    ctx.drawImage(contourImage(f, dom, Math.round(W / 2), Math.round(W / 2), 14), 0, 0, W, W);
    const wt = [l1 / (l1 + al) * ws[0], l2 / (l2 + al) * ws[1]];
    ctx.strokeStyle = c.s3; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    for (const r of [0.5, 1, Math.hypot(...wt)]) { ctx.beginPath(); ctx.arc(P.X(0), P.Y(0), P.X(r) - P.X(0), 0, Math.PI * 2); ctx.stroke(); }
    ctx.setLineDash([]);
    ctx.strokeStyle = c.s2; ctx.lineWidth = 2; ctx.beginPath();
    for (let k = 0; k <= 100; k++) { const a = k === 100 ? 1e6 : (k / 100) ** 3 * 200; const p = [l1 / (l1 + a) * ws[0], l2 / (l2 + a) * ws[1]]; k ? ctx.lineTo(P.X(p[0]), P.Y(p[1])) : ctx.moveTo(P.X(p[0]), P.Y(p[1])); }
    ctx.stroke();
    ctx.strokeStyle = c.axis; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(P.X(0), 0); ctx.lineTo(P.X(0), W); ctx.moveTo(0, P.Y(0)); ctx.lineTo(W, P.Y(0)); ctx.stroke();
    dot(ctx, P.X(ws[0]), P.Y(ws[1]), 6, c.text, c.surface); label(ctx, 'w*', P.X(ws[0]) + 9, P.Y(ws[1]) - 9, { color: c.text, bold: true });
    dot(ctx, P.X(wt[0]), P.Y(wt[1]), 7, c.s2, c.surface); label(ctx, 'w̃', P.X(wt[0]) + 9, P.Y(wt[1]) + 10, { color: c.text, bold: true });
    lgd.set([{ label: 'loss contours (ellipses)', color: c.muted }, { label: 'L2 penalty contours (circles)', color: c.s3, kind: 'dash' }, { label: 'path of w̃ as α grows from 0 to ∞', color: c.s2 }]);
    calc.innerHTML = `<span class="c">w̃ = (H + αI)⁻¹ H w* = Q(Λ + αI)⁻¹ΛQᵀw*  → component i is scaled by λᵢ/(λᵢ + α)   (7.10, 7.13)</span>
direction 1: λ₁/(λ₁ + α) = ${l1.toFixed(2)}/${(l1 + al).toFixed(2)} = <b>${(l1 / (l1 + al)).toFixed(3)}</b>  → w̃₁ = ${wt[0].toFixed(3)}  (w₁* = 2)
direction 2: λ₂/(λ₂ + α) = ${l2.toFixed(2)}/${(l2 + al).toFixed(2)} = <b>${(l2 / (l2 + al)).toFixed(3)}</b>  → w̃₂ = ${wt[1].toFixed(3)}  (w₂* = 1.5)
<span class="c">${l1 < l2 ? 'low curvature along w₁ = the loss barely cares about w₁, so the penalty wins there and pulls w₁ toward 0' : 'now w₂ is the low-curvature direction, so it gets shrunk more'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) Ridge path: weights vs α ---------- */
Widgets.ridgePath = (el) => {
  UI.head(el, 'Ridge regression: (XᵀX + αI)⁻¹Xᵀy', 'Four features with different usefulness: y = 3x₁ + 1.5x₂ + 0.5x₃ + 0·x₄ + noise. As α grows, weights shrink — features weakly related to y go first.');
  const rng = mulberry32(6), m = 60;
  const X = Array.from({ length: m }, () => [gaussian(rng), gaussian(rng), gaussian(rng), gaussian(rng)]);
  const y = X.map(r => 3 * r[0] + 1.5 * r[1] + 0.5 * r[2] + 1.0 * gaussian(rng));
  const XtX = [0, 1, 2, 3].map(i => [0, 1, 2, 3].map(j => X.reduce((s, r) => s + r[i] * r[j], 0)));
  const Xty = [0, 1, 2, 3].map(i => X.reduce((s, r, k) => s + r[i] * y[k], 0));
  const sol = a => solveLinear(XtX.map((r, i) => r.map((v, j) => v + (i === j ? a : 0))), Xty);
  let la = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'α (log scale)', min: -2, max: 4, step: 0.05, value: la, fmt: v => (10 ** v).toFixed(v < 0 ? 3 : 1), onInput: v => { la = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [-2, 4], yr: [-0.5, 3.3], xlabel: 'log₁₀ α', ylabel: 'weight', xname: 'log₁₀ α', aspect: .5, maxH: 280, vfmt: v => v.toFixed(3), xfmt: v => '10' + sup(v) });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const path = Array.from({ length: 121 }, (_, i) => { const a = -2 + i * 0.05; return [a, sol(10 ** a)]; });
  function draw() {
    const c = Colors.get();
    const w = sol(10 ** la);
    chart.set([0, 1, 2, 3].map(k => ({ name: 'w' + (k + 1), color: ['s1', 's2', 's3', 's7'][k], data: path.map(([a, s]) => [a, s[k]]) }))
      .concat([{ name: 'now', color: 'text', style: 'dots', r: 4, noHover: true, data: w.map(v => [la, v]) }]));
    lgd.set([{ label: 'w₁ (true 3)', color: c.s1 }, { label: 'w₂ (true 1.5)', color: c.s2 }, { label: 'w₃ (true 0.5)', color: c.s3 }, { label: 'w₄ (true 0, pure noise)', color: c.s7 }]);
    calc.innerHTML = `<span class="c">normal equations become (XᵀX + αI)w = Xᵀy   (7.17): α is added to the diagonal = the “variance” of each feature</span>
XᵀX diagonal ≈ (${XtX.map((r, i) => r[i].toFixed(0)).join(', ')})  + α = ${(10 ** la).toFixed(2)}
w = (${w.map(v => v.toFixed(3)).join(', ')})     ‖w‖² = ${dotv(w, w).toFixed(3)}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) L1 vs L2: soft-threshold vs scaling ---------- */
Widgets.l1vsl2 = (el) => {
  UI.head(el, 'L1 makes weights exactly zero; L2 only shrinks them', 'With a diagonal Hessian each weight is solved separately. Left: the regularized weight as a function of the unregularized w*. Right: eight weights before and after.');
  let a = 0.5;
  const ws = [2.5, -1.8, 1.2, 0.6, -0.4, 0.25, -0.15, 0.05];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'α / Hᵢᵢ (penalty relative to curvature)', min: 0, max: 2, step: 0.01, value: a, fmt: v => v.toFixed(2), onInput: v => { a = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [-3, 3], yr: [-3, 3], xlabel: 'w* (unregularized)', ylabel: 'regularized w', xname: 'w*', aspect: .55, maxH: 260, vfmt: v => v.toFixed(3) });
  const lgd = UI.legend(el, []);
  const bars = h('div');
  const calc = h('div', { class: 'calcbox' });
  el.append(bars, calc);
  const L1 = w => Math.sign(w) * Math.max(Math.abs(w) - a, 0), L2 = w => w / (1 + a);
  function draw() {
    const c = Colors.get();
    const xs = Array.from({ length: 121 }, (_, i) => -3 + i * 0.05);
    chart.set([
      { name: 'no penalty', color: 'muted', data: xs.map(x => [x, x]), dash: [4, 4] },
      { name: 'L2: w*/(1 + α/H)', color: 's1', data: xs.map(x => [x, L2(x)]) },
      { name: 'L1: soft threshold', color: 's2', data: xs.map(x => [x, L1(x)]) }
    ]);
    lgd.set([{ label: 'L2 (scales everything a little)', color: c.s1 }, { label: 'L1 (dead zone |w*| ≤ α/H → exactly 0)', color: c.s2 }, { label: 'no penalty', color: c.muted, kind: 'dash' }]);
    bars.innerHTML = '';
    const grid = h('div', { style: 'display:grid;grid-template-columns:90px repeat(8,1fr);gap:4px;align-items:center;font-size:.72rem' });
    const row = (lab, arr, col) => { grid.append(h('div', {}, lab)); arr.forEach(v => grid.append(h('div', { style: `height:26px;border-radius:5px;display:flex;align-items:center;justify-content:center;font-family:var(--mono);background:${v === 0 ? 'var(--surface-2)' : alpha(col, Math.min(1, 0.2 + Math.abs(v) / 3))};${v === 0 ? 'outline:2px dashed var(--critical)' : ''}` }, v === 0 ? '0' : v.toFixed(2)))); };
    row('w*', ws, c.text2); row('after L2', ws.map(L2), c.s1); row('after L1', ws.map(L1), c.s2);
    bars.append(grid);
    const zeros = ws.map(L1).filter(v => v === 0).length;
    calc.innerHTML = `<span class="c">L1 solution (7.23):  wᵢ = sign(wᵢ*)·max{|wᵢ*| − α/Hᵢᵢ, 0}      L2: wᵢ = Hᵢᵢ/(Hᵢᵢ + α)·wᵢ*</span>
example w* = 1.2:  L1 → sign(1.2)·max{1.2 − ${a.toFixed(2)}, 0} = <b>${L1(1.2).toFixed(3)}</b>     L2 → 1.2/(1 + ${a.toFixed(2)}) = <b>${L2(1.2).toFixed(3)}</b>
L1 set <b>${zeros} of 8</b> weights exactly to zero (sparse → feature selection, as in LASSO); L2 set none.`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Data augmentation on a digit ---------- */
Widgets.augmentDigit = (el) => {
  UI.head(el, 'Dataset augmentation: new examples from transformations', 'A handwritten “6”. Transform it and check whether the label is still correct.');
  const img = (u, v) => { // continuous ink intensity of a "6" in [0,1]²
    const loop = Math.abs(Math.hypot(u - 0.5, v - 0.64) - 0.2);
    let stem = 9;
    for (let t = 0; t <= 1; t += 0.02) { const bx = (1 - t) ** 2 * 0.31 + 2 * (1 - t) * t * 0.36 + t * t * 0.62, by = (1 - t) ** 2 * 0.62 + 2 * (1 - t) * t * 0.2 + t * t * 0.14; stem = Math.min(stem, Math.hypot(u - bx, v - by)); }
    const d = Math.min(loop, stem);
    return clamp(1 - (d - 0.035) / 0.035, 0, 1);
  };
  let dx = 0, dy = 0, rot = 0, sc = 1, flip = false, r180 = false, noise = 0, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Shift x (pixels)', min: -4, max: 4, step: 1, value: 0, fmt: v => String(v), onInput: v => { dx = v; draw(); } });
  UI.slider(ctr, { label: 'Shift y (pixels)', min: -4, max: 4, step: 1, value: 0, fmt: v => String(v), onInput: v => { dy = v; draw(); } });
  UI.slider(ctr, { label: 'Rotate (°)', min: -30, max: 30, step: 1, value: 0, fmt: v => v + '°', onInput: v => { rot = v; draw(); } });
  UI.slider(ctr, { label: 'Scale', min: 0.7, max: 1.3, step: 0.01, value: 1, fmt: v => v.toFixed(2) + '×', onInput: v => { sc = v; draw(); } });
  UI.slider(ctr, { label: 'Input noise σ', min: 0, max: 0.6, step: 0.02, value: 0, fmt: v => v.toFixed(2), onInput: v => { noise = v; draw(); } });
  const tg = h('div', { class: 'toggles' });
  UI.chip(tg, 'Flip horizontally', '--s8', false, v => { flip = v; draw(); });
  UI.chip(tg, 'Rotate 180°', '--s8', false, v => { r180 = v; draw(); });
  UI.button(tg, 'Random augmentations', () => { seed++; drawGallery(); });
  const row = h('div', { style: 'display:flex;gap:16px;align-items:center;flex-wrap:wrap' });
  const c0 = h('canvas', { class: 'chart', style: 'width:120px;image-rendering:pixelated' }), c1 = h('canvas', { class: 'chart', style: 'width:120px;image-rendering:pixelated' });
  const verdict = h('div', { style: 'flex:1;min-width:200px' });
  row.append(h('div', {}, h('div', { class: 'note' }, 'original (label 6)'), c0), h('div', {}, h('div', { class: 'note' }, 'augmented'), c1), verdict);
  const gallery = h('div', { style: 'display:flex;gap:6px;flex-wrap:wrap' });
  el.append(ctr, tg, row, h('div', { class: 'note' }, 'Safe random augmentations (shift ±3 px, rotate ±15°, scale ±10%) — all still labelled 6:'), gallery);
  const N = 20;
  function render(cv, tf, rng) {
    const c = Colors.get();
    const ctx = fitCanvas(cv, 120, 120);
    const ink = hexToRgb(c.text), bg = hexToRgb(c.surface2);
    const cell = 120 / N;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      let u = (i + 0.5) / N - 0.5, v = (j + 0.5) / N - 0.5;
      u -= tf.dx / N; v -= tf.dy / N;
      const a = -tf.rot * Math.PI / 180; [u, v] = [u * Math.cos(a) - v * Math.sin(a), u * Math.sin(a) + v * Math.cos(a)];
      u /= tf.sc; v /= tf.sc;
      if (tf.r180) { u = -u; v = -v; }
      if (tf.flip) u = -u;
      let val = img(u + 0.5, v + 0.5);
      if (tf.noise) val = clamp(val + tf.noise * gaussian(rng), 0, 1);
      ctx.fillStyle = `rgb(${lerp(bg[0], ink[0], val)},${lerp(bg[1], ink[1], val)},${lerp(bg[2], ink[2], val)})`;
      ctx.fillRect(i * cell, j * cell, cell + 0.5, cell + 0.5);
    }
  }
  function draw() {
    render(c0, { dx: 0, dy: 0, rot: 0, sc: 1 }, mulberry32(1));
    render(c1, { dx, dy, rot, sc, flip, r180, noise }, mulberry32(seed));
    verdict.innerHTML = '';
    const bad = flip || r180;
    verdict.append(h('div', { class: 'callout ' + (bad ? 'bad' : 'good') }, h('span', { class: 'ico' }, bad ? '✗' : '✓'), h('div', {},
      r180 && !flip ? 'Rotated 180°, a 6 looks like a 9 — keeping the label “6” would teach the model something false.' :
        flip ? 'A mirrored 6 is not a valid digit (and for letters, flips turn b into d). Not safe for character recognition.' :
          'Still clearly a 6: a valid new training example (x′, y) for free. The classifier learns to be invariant to these changes.')));
    drawGallery();
  }
  function drawGallery() {
    gallery.innerHTML = '';
    const rng = mulberry32(seed * 7);
    for (let k = 0; k < 8; k++) {
      const cv = h('canvas', { class: 'chart', style: 'width:60px;image-rendering:pixelated' });
      gallery.append(cv);
      render(cv, { dx: Math.round(6 * rng() - 3), dy: Math.round(6 * rng() - 3), rot: 30 * rng() - 15, sc: 0.9 + 0.2 * rng(), noise: 0 }, rng);
      cv.style.height = '60px';
    }
  }
  draw();
  return { redraw: draw };
};

/* ---------- 6) Weight noise prefers flat minima ---------- */
Widgets.flatMinima = (el) => {
  UI.head(el, 'Noise on the weights favours flat minima', 'A loss with a sharp, deeper minimum (left) and a wide, shallower one (right). With weight noise ε ~ 𝒩(0, σ²) the expected loss is the loss averaged over nearby weights.');
  let s = 0.15;
  const J = w => 1 + 0.03 * w * w - 1.0 * Math.exp(-((w + 1.5) ** 2) / (2 * 0.08 ** 2)) - 0.8 * Math.exp(-((w - 1.5) ** 2) / (2 * 0.6 ** 2));
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Weight-noise std σ (√η)', min: 0, max: 0.6, step: 0.01, value: s, fmt: v => v.toFixed(2), onInput: v => { s = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [-3, 3.5], yr: [-0.1, 1.4], xlabel: 'weight w', ylabel: 'loss', xname: 'w', aspect: .5, maxH: 280, vfmt: v => v.toFixed(3) });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const EJ = w => { if (s < 1e-3) return J(w); let a = 0, z = 0; for (let k = -40; k <= 40; k++) { const u = k / 10, g = Math.exp(-u * u / 2); a += g * J(w + u * s); z += g; } return a / z; };
  function draw() {
    const c = Colors.get();
    const xs = Array.from({ length: 261 }, (_, i) => -3 + i * 0.025);
    const ej = xs.map(EJ);
    let bi = 0; ej.forEach((v, i) => { if (v < ej[bi]) bi = i; });
    chart.set([
      { name: 'J(w)', color: 'text', data: xs.map(w => [w, J(w)]) },
      { name: 'E[J(w + ε)]', color: 's2', data: xs.map((w, i) => [w, ej[i]]), width: 2.6 },
      { name: 'best under noise', color: 's2', style: 'dots', r: 6, noHover: true, data: [[xs[bi], ej[bi]]] }
    ]);
    lgd.set([{ label: 'loss J(w)', color: c.text }, { label: 'expected loss with weight noise', color: c.s2 }]);
    const sharp = J(-1.5), wide = J(1.5);
    calc.innerHTML = `<span class="c">for small η:  J̃ ≈ J + η·E[‖∇_W ŷ(x)‖²]  (7.32) — and here E[J(w+ε)] ≈ J(w) + ½σ²·J″(w)</span>
sharp minimum w = −1.5:  J = ${sharp.toFixed(3)},  expected with noise = <b>${EJ(-1.5).toFixed(3)}</b>
wide  minimum w = +1.5:  J = ${wide.toFixed(3)},  expected with noise = <b>${EJ(1.5).toFixed(3)}</b>
<span class="c">→ the noisy objective prefers the ${EJ(-1.5) < EJ(1.5) ? 'sharp' : 'WIDE'} minimum: a region where small weight changes barely change the output</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) Label smoothing ---------- */
Widgets.labelSmooth = (el) => {
  UI.head(el, 'Label smoothing: targets ε/(k−1) and 1 − ε', 'Logits: correct class d, all others 0. With hard targets the loss keeps falling as d → ∞ (weights grow forever); smoothed targets have a finite best d.');
  let k = 5, eps = 0.1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Number of classes k', min: 2, max: 10, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  UI.slider(ctr, { label: 'Smoothing ε', min: 0.01, max: 0.4, step: 0.01, value: eps, fmt: v => v.toFixed(2), onInput: v => { eps = v; draw(); } });
  const bars = h('div');
  el.append(ctr, bars);
  const chart = new LineChart(el, { xr: [-2, 12], yr: [0, 3], xlabel: 'logit margin d', ylabel: 'cross-entropy', xname: 'd', aspect: .45, maxH: 250, vfmt: v => v.toFixed(4) });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const pc = d => Math.exp(d) / (Math.exp(d) + k - 1), po = d => 1 / (Math.exp(d) + k - 1);
    const hard = d => -Math.log(pc(d)), soft = d => -(1 - eps) * Math.log(pc(d)) - eps * Math.log(po(d));
    const ds = Array.from({ length: 141 }, (_, i) => -2 + i * 0.1);
    const dstar = Math.log((1 - eps) * (k - 1) / eps);
    chart.set([
      { name: 'hard targets', color: 's8', data: ds.map(d => [d, hard(d)]) },
      { name: 'smoothed targets', color: 's3', data: ds.map(d => [d, soft(d)]) },
      { name: 'optimum', color: 's3', style: 'dots', r: 6, noHover: true, data: [[dstar, soft(dstar)]] }
    ]);
    lgd.set([{ label: 'hard 0/1 targets: always pushes d higher', color: c.s8 }, { label: 'label smoothing: minimum at finite d*', color: c.s3 }]);
    bars.innerHTML = '';
    const g = h('div', { style: `display:grid;grid-template-columns:110px repeat(${k},1fr);gap:4px;align-items:end;font-size:.72rem` });
    const mk = (lab, arr, col) => { g.append(h('div', {}, lab)); arr.forEach(v => g.append(h('div', { style: 'height:56px;position:relative' }, h('div', { style: `position:absolute;bottom:0;left:10%;right:10%;height:${v * 54}px;background:${col};border-radius:3px` }), h('div', { style: 'position:absolute;top:0;width:100%;text-align:center;font-family:var(--mono)' }, v.toFixed(2))))); };
    mk('hard target', Array.from({ length: k }, (_, i) => (i ? 0 : 1)), c.s8);
    mk('smoothed target', Array.from({ length: k }, (_, i) => (i ? eps / (k - 1) : 1 - eps)), c.s3);
    bars.append(g);
    calc.innerHTML = `<span class="c">smoothed target: correct class 1 − ε = ${(1 - eps).toFixed(2)}, each other class ε/(k − 1) = ${(eps / (k - 1)).toFixed(4)}</span>
best margin: softmax(correct) = 1 − ε  ⇒  e^d/(e^d + k − 1) = 1 − ε  ⇒  d* = log((1 − ε)(k − 1)/ε) = <b>${dstar.toFixed(3)}</b>
hard-target loss at d = 10: ${hard(10).toExponential(2)} and still decreasing — the softmax never reaches exactly 1.`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 8) Early stopping on a real gradient-descent run ---------- */
Widgets.earlyStop = (el) => {
  UI.head(el, 'Early stopping: keep the parameters with the best validation error', 'Gradient descent on a degree-12 polynomial with only 20 noisy training points. Training error keeps falling; validation error falls, then rises. (log-scale time axis)');
  const rng = mulberry32(21);
  const mk = n => Array.from({ length: n }, () => { const x = rng(); return [x, TRUE_F(x) + 0.3 * gaussian(rng)]; });
  const train = mk(20), val = mk(40);
  const D = 12;
  const Xt = train.map(([x]) => polyRow(x, D)), Xv = val.map(([x]) => polyRow(x, D));
  const yt = train.map(p => p[1]), yv = val.map(p => p[1]);
  // largest eigenvalue of (2/m)XᵀX by power iteration → safe step size
  const G = Array.from({ length: D + 1 }, (_, i) => Array.from({ length: D + 1 }, (_, j) => 2 * Xt.reduce((s, r) => s + r[i] * r[j], 0) / Xt.length));
  let v = Array(D + 1).fill(1); for (let it = 0; it < 200; it++) { const nv = G.map(r => dotv(r, v)); const n = Math.hypot(...nv); v = nv.map(t => t / n); }
  const lmax = dotv(v, G.map(r => dotv(r, v)));
  const eps = 1 / lmax;
  const curve = [];
  const n1 = D + 1, Gf = new Float64Array(n1 * n1), bf = new Float64Array(n1);
  for (let i = 0; i < n1; i++) { for (let j = 0; j < n1; j++) Gf[i * n1 + j] = G[i][j]; bf[i] = 2 * Xt.reduce((s, r, k) => s + r[i] * yt[k], 0) / Xt.length; }
  let w = new Float64Array(n1), t = 0;
  const err = (X, yy) => mean(X.map((r, i) => (dotv(r, w) - yy[i]) ** 2));
  let next = 1;
  const g = new Float64Array(n1);
  while (t <= 200000) {
    if (t >= next || t === 0) { curve.push([t, err(Xt, yt), err(Xv, yv), Array.from(w)]); next = Math.ceil(next * 1.12); }
    for (let i = 0; i < n1; i++) { let s = -bf[i]; for (let j = 0; j < n1; j++) s += Gf[i * n1 + j] * w[j]; g[i] = s; }
    for (let i = 0; i < n1; i++) w[i] -= eps * g[i];
    t++;
  }
  let pat = 8;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Patience p (checks without improvement)', min: 1, max: 30, step: 1, value: pat, fmt: v => String(v), onInput: v => { pat = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 5.3], yr: [0.03, 2], logY: true, xlabel: 'training step (log₁₀)', ylabel: 'MSE', xname: 'log₁₀ step', aspect: .5, maxH: 280, xfmt: v => '10' + sup(v), hxfmt: v => Math.round(10 ** v).toLocaleString() });
  const lgd = UI.legend(el, []);
  const fitChart = new LineChart(el, { xr: [0, 1], yr: [-2, 2], xlabel: 'x', xname: 'x', aspect: .32, maxH: 170, minH: 130, vfmt: v => v.toFixed(2) });
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    let best = 0, bestV = Infinity, since = 0, stopAt = curve.length - 1;
    for (let i = 0; i < curve.length; i++) {
      if (curve[i][2] < bestV) { bestV = curve[i][2]; best = i; since = 0; } else if (++since >= pat) { stopAt = i; break; }
    }
    const lx = t0 => Math.log10(Math.max(1, t0));
    chart.set([
      { name: 'training error', color: 's1', data: curve.map(r => [lx(r[0]), Math.max(0.031, r[1])]) },
      { name: 'validation error', color: 's2', data: curve.map(r => [lx(r[0]), Math.min(1.99, r[2])]) },
      { name: 'best (returned)', color: 's3', style: 'dots', r: 7, noHover: true, data: [[lx(curve[best][0]), curve[best][2]]] },
      { name: 'stopped here', color: 'text', style: 'dots', r: 5, noHover: true, data: [[lx(curve[stopAt][0]), Math.min(1.99, curve[stopAt][2])]] }
    ]);
    lgd.set([{ label: 'training error', color: c.s1 }, { label: 'validation error', color: c.s2 }, { label: 'parameters returned', color: c.s3, kind: 'dot' }, { label: 'algorithm stops', color: c.text, kind: 'dot' }]);
    const xs = Array.from({ length: 101 }, (_, i) => i / 100);
    const wb = curve[best][3], wl = curve[curve.length - 1][3];
    fitChart.set([
      { name: 'early-stopped fit', color: 's3', data: xs.map(x => [x, dotv(wb, polyRow(x, D))]) },
      { name: 'trained to the end', color: 's8', data: xs.map(x => [x, clamp(dotv(wl, polyRow(x, D)), -2, 2)]), dash: [5, 4] },
      { name: 'train', color: 's1', style: 'dots', r: 3.5, noHover: true, data: train }
    ]);
    calc.innerHTML = `<span class="c">store a copy of θ every time validation error improves; stop after p = ${pat} checks with no improvement</span>
best validation MSE ${curve[best][2].toFixed(4)} at step ${curve[best][0].toLocaleString()} (training MSE ${curve[best][1].toFixed(4)})
stopped at step ${curve[stopAt][0].toLocaleString()};  if we trained to step ${curve[curve.length - 1][0].toLocaleString()}: validation MSE ${curve[curve.length - 1][2].toFixed(4)}
<span class="c">the number of steps acts like a hyper-parameter with a U-shaped validation curve; ‖θ‖ at the best point = ${Math.sqrt(dotv(wb, wb)).toFixed(2)} vs ${Math.sqrt(dotv(wl, wl)).toFixed(2)} at the end — early stopping keeps weights small, like L2.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 9) Early stopping ≈ L2 regularization ---------- */
Widgets.earlyVsL2 = (el) => {
  UI.head(el, 'Early stopping and weight decay shrink the same directions', 'For a quadratic loss started at w = 0, after τ steps of size ε the component along eigenvalue λ is scaled by 1 − (1 − ελ)^τ. Weight decay scales it by λ/(λ + α). Set α = 1/(τε).');
  let lt = 2, eps = 0.1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Training steps τ', min: 0, max: 4, step: 0.05, value: lt, fmt: v => String(Math.round(10 ** v)), onInput: v => { lt = v; draw(); } });
  UI.slider(ctr, { label: 'Learning rate ε', min: 0.01, max: 0.5, step: 0.01, value: eps, fmt: v => v.toFixed(2), onInput: v => { eps = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [-4, 1], yr: [0, 1.05], xlabel: 'log₁₀ λ (curvature of a direction)', ylabel: 'fraction of w* kept', xname: 'log₁₀ λ', aspect: .5, maxH: 270, vfmt: v => v.toFixed(3), xfmt: v => '10' + sup(v) });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const tau = Math.round(10 ** lt), al = 1 / (tau * eps), top = Math.log10(1 / eps);
    const ls = Array.from({ length: 201 }, (_, i) => -4 + i * (top + 4) / 200);
    chart.set([
      { name: 'early stopping', color: 's2', data: ls.map(l => [l, 1 - Math.pow(1 - eps * 10 ** l, tau)]) },
      { name: 'L2 with α = 1/(τε)', color: 's1', data: ls.map(l => [l, 10 ** l / (10 ** l + al)]), dash: [6, 4] }
    ], { xr: [-4, top] });
    lgd.set([{ label: `early stopping after τ = ${tau} steps`, color: c.s2 }, { label: `weight decay with α = 1/(τε) = ${al.toPrecision(3)}`, color: c.s1, kind: 'dash' }]);
    const lam = 0.05;
    calc.innerHTML = `<span class="c">Qᵀw⁽τ⁾ = [I − (I − εΛ)^τ]Qᵀw*   (7.40)        Qᵀw̃ = [I − (Λ + αI)⁻¹α]Qᵀw*   (7.42)</span>
equal when (I − εΛ)^τ = (Λ + αI)⁻¹α  (7.43);  for small ελ:  τ ≈ 1/(εα),  α ≈ 1/(τε)   (7.44–7.45)
example λ = ${lam}:  early stopping keeps 1 − (1 − ${(eps * lam).toFixed(4)})^${tau} = <b>${(1 - Math.pow(1 - eps * lam, tau)).toFixed(3)}</b>;  L2 keeps ${lam}/(${lam} + ${al.toPrecision(3)}) = <b>${(lam / (lam + al)).toFixed(3)}</b>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 10) Ensembles & bagging ---------- */
Widgets.ensembleError = (el) => {
  UI.head(el, 'Why averaging models helps (and when it does not)', 'k models, each with error variance v = 1 and pairwise covariance c (correlation ρ = c/v). Expected squared error of the averaged prediction:');
  let k = 5, rho = 0.3, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Ensemble size k', min: 1, max: 50, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  UI.slider(ctr, { label: 'Error correlation ρ', min: 0, max: 1, step: 0.01, value: rho, fmt: v => v.toFixed(2), onInput: v => { rho = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [1, 50], yr: [0, 1.05], xlabel: 'k', ylabel: 'E[(average error)²]', xname: 'k', aspect: .45, maxH: 240, vfmt: v => v.toFixed(3), snapX: x => Math.round(x) });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  const boot = h('div');
  el.append(calc, h('div', { class: 'note', style: 'margin-top:6px' }, 'Bagging: each model gets a bootstrap resample (same size, drawn WITH replacement):'), boot);
  const btn = UI.button(h('div'), 'New bootstrap samples', () => { seed++; drawBoot(); });
  el.append(btn);
  const E = (kk, r) => 1 / kk + (kk - 1) / kk * r;
  function draw() {
    const c = Colors.get();
    const ks = Array.from({ length: 50 }, (_, i) => i + 1);
    chart.set([
      { name: 'ρ = 0 (independent)', color: 's3', data: ks.map(x => [x, E(x, 0)]), dash: [4, 4] },
      { name: 'ρ = 1 (identical)', color: 's8', data: ks.map(x => [x, 1]), dash: [4, 4] },
      { name: `ρ = ${rho.toFixed(2)}`, color: 's1', data: ks.map(x => [x, E(x, rho)]) },
      { name: 'now', color: 'text', style: 'dots', r: 5, noHover: true, data: [[k, E(k, rho)]] }
    ]);
    lgd.set([{ label: 'independent errors: 1/k', color: c.s3, kind: 'dash' }, { label: 'perfectly correlated: no gain', color: c.s8, kind: 'dash' }, { label: 'your ρ', color: c.s1 }]);
    calc.innerHTML = `<span class="c">E[(1/k Σᵢ εᵢ)²] = (1/k)·v + ((k − 1)/k)·c     (7.50)</span>
= (1/${k})·1 + (${k - 1}/${k})·${rho.toFixed(2)} = ${(1 / k).toFixed(3)} + ${((k - 1) / k * rho).toFixed(3)} = <b>${E(k, rho).toFixed(3)}</b>   (one model alone: 1.000)`;
    drawBoot();
  }
  function drawBoot() {
    const c = Colors.get();
    const rng = mulberry32(seed * 31);
    const items = 'ABCDEFGH'.split('');
    const cols = [c.s1, c.s2, c.s3, c.s4, c.s5, c.s6, c.s7, c.s8];
    boot.innerHTML = '';
    for (let mdl = 0; mdl < 3; mdl++) {
      const s = Array.from({ length: 8 }, () => Math.floor(rng() * 8)).sort();
      const uniq = new Set(s).size;
      boot.append(h('div', { style: 'display:flex;gap:4px;align-items:center;margin:3px 0;font-size:.78rem' }, h('span', { style: 'width:70px' }, `model ${mdl + 1}`),
        ...s.map(i => h('span', { style: `width:24px;height:24px;border-radius:6px;display:inline-flex;align-items:center;justify-content:center;color:#fff;font-weight:700;background:${cols[i]}` }, items[i])),
        h('span', { class: 'muted', style: 'margin-left:6px' }, `${uniq}/8 unique, missing: ${items.filter((_, i) => !s.includes(i)).join(' ') || '—'}`)));
    }
    boot.append(h('div', { class: 'note' }, 'On average a bootstrap sample contains 1 − (1 − 1/m)^m ≈ 63.2% of the distinct examples, so every model sees a slightly different dataset → partly independent errors.'));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 11) Dropout masks ---------- */
Widgets.dropoutNet = (el) => {
  UI.head(el, 'Dropout: a new random sub-network for every example', 'Every input unit is kept with probability p_in, every hidden unit with p_hidden. Dropped units (and their connections) vanish for this one forward/backward pass.');
  let pin = 0.8, ph = 0.5, seed = 1;
  const sizes = [4, 6, 6, 1];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Keep probability, inputs', min: 0.5, max: 1, step: 0.05, value: pin, fmt: v => v.toFixed(2), onInput: v => { pin = v; draw(); } });
  UI.slider(ctr, { label: 'Keep probability, hidden', min: 0.2, max: 1, step: 0.05, value: ph, fmt: v => v.toFixed(2), onInput: v => { ph = v; draw(); } });
  const btns = h('div', { class: 'btns' }); UI.button(btns, 'Next example → new mask', () => { seed++; draw(); }, 'btn primary'); ctr.append(btns);
  const svgBox = h('div'), calc = h('div', { class: 'calcbox' });
  el.append(ctr, svgBox, calc);
  function draw() {
    const c = Colors.get();
    const rng = mulberry32(seed * 101);
    const mask = sizes.map((n, l) => Array.from({ length: n }, () => (l === sizes.length - 1 ? 1 : rng() < (l === 0 ? pin : ph) ? 1 : 0)));
    const ns = 'http://www.w3.org/2000/svg', W = 560, H = 240;
    const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('class', 'diagram');
    const mkE = (t, a, txt) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.append(e); return e; };
    const X = l => 60 + l * (W - 120) / (sizes.length - 1), Y = (l, i) => H / 2 + (i - (sizes[l] - 1) / 2) * 34;
    for (let l = 0; l < sizes.length - 1; l++) for (let i = 0; i < sizes[l]; i++) for (let j = 0; j < sizes[l + 1]; j++) {
      const on = mask[l][i] && mask[l + 1][j];
      mkE('line', { x1: X(l), y1: Y(l, i), x2: X(l + 1), y2: Y(l + 1, j), stroke: on ? c.s1 : c.border, 'stroke-width': on ? 1.4 : 0.6, opacity: on ? .8 : .4 });
    }
    sizes.forEach((n, l) => { for (let i = 0; i < n; i++) { const on = mask[l][i]; mkE('circle', { cx: X(l), cy: Y(l, i), r: 12, fill: on ? c.surface : c.surface2, stroke: on ? c.text : c.border, 'stroke-width': 2, 'stroke-dasharray': on ? '' : '3 3' }); if (!on) mkE('text', { x: X(l), y: Y(l, i) + 5, 'text-anchor': 'middle', 'font-size': 14, fill: c.critical, 'font-weight': 700 }, '×'); } });
    ['inputs', 'hidden 1', 'hidden 2', 'output'].forEach((t, l) => mkE('text', { x: X(l), y: 18, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: c.text2 }, t));
    svgBox.innerHTML = ''; svgBox.append(svg);
    const droppable = sizes.slice(0, -1).reduce((a, b) => a + b, 0);
    let pm = 1; mask.slice(0, -1).forEach((L, l) => L.forEach(v => { const p = l === 0 ? pin : ph; pm *= v ? p : 1 - p; }));
    calc.innerHTML = `<span class="c">mask μ for this example: inputs ${mask[0].join('')}, hidden ${mask[1].join('')} | ${mask[2].join('')}</span>
possible sub-networks = 2^(number of droppable units) = 2^${droppable} = <b>${(2 ** droppable).toLocaleString()}</b>, all sharing the same weights
probability of this exact mask = Πᵢ (p or 1 − p) = ${pm.toExponential(2)}
<span class="c">training minimizes E_μ J(θ, μ): one random mask per example, then ordinary forward prop, backprop and update. Test time: keep all units, multiply outgoing weights by p.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 12) Weight-scaling inference: exact for softmax regression ---------- */
Widgets.weightScaling = (el) => {
  UI.head(el, 'Ensemble prediction vs. the weight-scaling rule', 'Softmax regression, 6 inputs, 3 classes, dropout on the inputs with p = ½. We compute the ensemble over ALL 2⁶ = 64 masks and compare it with one forward pass using W/2.');
  let seed = 1, nmc = 10;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Monte Carlo samples', min: 1, max: 200, step: 1, value: nmc, fmt: v => String(v), onInput: v => { nmc = v; draw(); } });
  const btns = h('div', { class: 'btns' }); UI.button(btns, 'New example & weights', () => { seed++; draw(); }, 'btn primary'); ctr.append(btns);
  const tbl = h('div', { class: 'scroll-x' }), calc = h('div', { class: 'calcbox' });
  el.append(ctr, tbl, calc);
  const softmax = z => { const m = Math.max(...z), e = z.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };
  function draw() {
    const rng = mulberry32(seed * 13);
    const n = 6, K = 3;
    const W = Array.from({ length: n }, () => Array.from({ length: K }, () => 1.2 * gaussian(rng))), b = Array.from({ length: K }, () => 0.3 * gaussian(rng));
    const x = Array.from({ length: n }, () => gaussian(rng));
    const pred = d => softmax(b.map((bk, k) => bk + x.reduce((s, xi, i) => s + d[i] * xi * W[i][k], 0)));
    const masks = Array.from({ length: 64 }, (_, m) => Array.from({ length: n }, (_, i) => (m >> i) & 1));
    const preds = masks.map(pred);
    const geo = softmax([0, 1, 2].map(k => preds.reduce((s, p) => s + Math.log(p[k]), 0) / 64));
    const ari = [0, 1, 2].map(k => mean(preds.map(p => p[k])));
    const ws = pred(Array(n).fill(0.5));
    const r2 = mulberry32(seed * 7 + nmc);
    const mc = [0, 0, 0]; for (let s = 0; s < nmc; s++) { const p = pred(Array.from({ length: n }, () => (r2() < 0.5 ? 1 : 0))); p.forEach((v, k) => mc[k] += v / nmc); }
    const row = (lab, p, note) => h('tr', {}, h('td', {}, lab), ...p.map(v => h('td', { class: 'num' }, v.toFixed(5))), h('td', { class: 'note' }, note));
    tbl.innerHTML = '';
    tbl.append(h('table', { class: 'readout' }, h('thead', {}, h('tr', {}, h('th', {}, 'method'), h('th', {}, 'P(class 1)'), h('th', {}, 'P(class 2)'), h('th', {}, 'P(class 3)'), h('th', {}, ''))),
      h('tbody', {}, row('geometric mean over 64 masks (renormalized)', geo, 'the dropout ensemble'), row('weight scaling: softmax(½Wᵀx + b)', ws, 'ONE forward pass'), row('arithmetic mean over 64 masks', ari, 'bagging-style average'), row(`Monte Carlo, ${nmc} random masks`, mc, 'sampling approximation'))));
    const diff = Math.max(...geo.map((v, k) => Math.abs(v - ws[k])));
    calc.innerHTML = `<span class="c">p̃_ensemble(y|x) = (Π_d P(y | x; d))^(1/2ⁿ) → renormalize   (7.56–7.57)</span>
max |geometric − weight scaling| = <b>${diff.toExponential(2)}</b>  → exactly equal (up to rounding), as derived in (7.58–7.66)
<span class="c">reason: log softmax is linear in the masked inputs, and the average of d over all masks is ½ — so the geometric mean is softmax(Wᵀx/2 + b).</span>`;
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   Quiz
   ========================================================= */
window.QUIZ = [
  { q: 'Regularization is any modification intended to reduce…', opts: ['training error', 'generalization error but not training error', 'the number of layers', 'computation time'], a: 1, why: 'Regularizers often increase training error slightly in exchange for lower test error. The best model is usually a large model that is regularized appropriately.' },
  { q: 'Why are biases usually left unregularized?', opts: ['They have no gradient', 'Each bias controls a single variable, needs little data, and penalizing it can cause underfitting', 'They are always zero', 'Weight decay cannot be applied to them'], a: 1, why: 'Weights describe interactions between two variables and need lots of data; biases do not, so penalizing them adds little and hurts fit.' },
  { q: 'With L2 weight decay, a single gradient step becomes…', opts: ['w ← w − ε∇J', 'w ← (1 − εα)w − ε∇J', 'w ← w + αw', 'w ← sign(w)'], a: 1, why: 'The gradient αw + ∇J gives w ← w − ε(αw + ∇J) = (1 − εα)w − ε∇J: the weights are shrunk by a constant factor every step.' },
  { q: 'Under a quadratic approximation, L2 scales the component of w* along eigenvector i by…', opts: ['α', 'λᵢ/(λᵢ + α)', 'α/λᵢ', '1 − α'], a: 1, why: 'Directions with large curvature (λᵢ ≫ α) are kept; directions with small curvature (unimportant for the loss) are shrunk toward zero.' },
  { q: 'L1 regularization tends to produce…', opts: ['larger weights', 'sparse weights (many exactly zero)', 'only negative weights', 'identical weights'], a: 1, why: 'Its solution is a soft threshold: wᵢ = sign(wᵢ*)·max{|wᵢ*| − α/Hᵢᵢ, 0}. Weights with small |wᵢ*| become exactly 0 → feature selection (LASSO).' },
  { q: 'Which augmentation is NOT safe for digit recognition?', opts: ['Shifting by 2 pixels', 'Small rotations', 'Rotating by 180°', 'Adding slight noise'], a: 2, why: 'A 180° rotation turns 6 into 9 (and flips turn b into d), changing the correct label.' },
  { q: 'Label smoothing replaces the hard targets with…', opts: ['random labels', 'ε/(k − 1) for wrong classes and 1 − ε for the correct class', 'all zeros', 'the model’s own predictions'], a: 1, why: 'With hard targets a softmax never reaches 0/1 and the weights keep growing; soft targets have a finite optimum without discouraging correct classification.' },
  { q: 'Under a quadratic approximation, early stopping after τ steps with learning rate ε is equivalent to L2 with…', opts: ['α = τε', 'α ≈ 1/(τε)', 'α = ε/τ', 'no regularization'], a: 1, why: 'Comparing (7.40) and (7.42): (1 − εΛ)^τ ≈ (Λ + αI)⁻¹α when τ ≈ 1/(εα). Training longer = weaker regularization.' },
  { q: 'Averaging k models with independent errors (c = 0) reduces the expected squared error to…', opts: ['v', 'v/k', 'k·v', '0'], a: 1, why: 'E[(1/k Σεᵢ)²] = v/k + (k−1)/k·c. With c = 0 the error falls linearly with k; with c = v (perfectly correlated) there is no gain.' },
  { q: 'At test time, the weight-scaling inference rule for dropout…', opts: ['samples 1 000 masks', 'uses all units but multiplies each unit’s outgoing weights by its inclusion probability', 'drops half the units', 'removes the output layer'], a: 1, why: 'It approximates the geometric-mean ensemble with one forward pass. It is exact for softmax regression and some linear models, and works well empirically for deep nets.' }
];
