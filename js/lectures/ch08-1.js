/* =========================================================
   Lecture 7 · Ch 8.1–8.2 · Optimization I: why training is hard
   ========================================================= */
'use strict';

/* reusable: contour map + paths */
function landscape(el, f, dom, opts = {}) {
  const wrap = h('div', { class: 'canvas-wrap', style: `max-width:${opts.maxW || 440}px;margin:0 auto` });
  const cv = h('canvas', { class: 'chart' + (opts.click ? ' clickable' : ''), role: 'img', 'aria-label': opts.aria || 'loss landscape' });
  wrap.append(cv); el.append(wrap);
  let bg = null, key = '', P = null;
  const api = {
    wrap, cv,
    draw(paths, marks) {
      const c = Colors.get();
      const W = Math.min(opts.maxW || 440, boxWidth(wrap)), H = Math.round(W * (dom[3] - dom[2]) / (dom[1] - dom[0]));
      const ctx = fitCanvas(cv, W, H);
      P = makePlot(W, H, [dom[0], dom[1]], [dom[2], dom[3]], { l: 0, r: 0, t: 0, b: 0 });
      api.P = P;
      const k = W + Theme.current(); if (k !== key) { bg = contourImage(f, dom, Math.round(W / 2), Math.round(H / 2), opts.levels || 16); key = k; }
      ctx.drawImage(bg, 0, 0, W, H);
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.clip();
      (paths || []).forEach(p => {
        const col = c[p.color] || p.color;
        ctx.strokeStyle = c.surface; ctx.lineWidth = 4; ctx.beginPath(); p.pts.forEach((q, i) => i ? ctx.lineTo(P.X(q[0]), P.Y(q[1])) : ctx.moveTo(P.X(q[0]), P.Y(q[1]))); ctx.stroke();
        ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(p.dash || []); ctx.beginPath(); p.pts.forEach((q, i) => i ? ctx.lineTo(P.X(q[0]), P.Y(q[1])) : ctx.moveTo(P.X(q[0]), P.Y(q[1]))); ctx.stroke(); ctx.setLineDash([]);
        if (p.dots) p.pts.forEach(q => dot(ctx, P.X(q[0]), P.Y(q[1]), 2.4, col));
        const l = p.pts[p.pts.length - 1]; dot(ctx, P.X(l[0]), P.Y(l[1]), 5, col, c.surface);
      });
      (marks || []).forEach(m => { ctx.fillStyle = c.text; ctx.font = '600 15px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(m.sym || '★', P.X(m.at[0]), P.Y(m.at[1])); if (m.label) label(ctx, m.label, P.X(m.at[0]) + 10, P.Y(m.at[1]) - 10, { color: c.text, bold: true }); });
      ctx.restore();
    },
    reset() { key = ''; }
  };
  if (opts.click) cv.addEventListener('click', e => { if (!P) return; const r = cv.getBoundingClientRect(); opts.click(P.invX(e.clientX - r.left), P.invY(e.clientY - r.top)); });
  return api;
}

/* ---------- 1) Risk vs empirical risk ---------- */
Widgets.riskVsEmpirical = (el) => {
  UI.head(el, 'The risk we want vs. the empirical risk we can compute', 'Model ŷ = θx, data y = 1.5x + noise. The true risk uses the (normally unknown) data distribution; the empirical risk averages over our m training examples only.');
  let m = 8, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Training examples m', min: 2, max: 200, step: 1, value: m, fmt: v => String(v), onInput: v => { m = v; draw(); } });
  const btns = h('div', { class: 'btns' }); UI.button(btns, 'Draw a new training set', () => { seed++; draw(); }, 'btn primary'); ctr.append(btns);
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 3], yr: [0, 3], xlabel: 'θ', ylabel: 'risk', xname: 'θ', aspect: .5, maxH: 270, vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const EX2 = 1 / 3, s2 = 0.25; // x ~ U(-1,1), noise std 0.5
  function draw() {
    const c = Colors.get();
    const rng = mulberry32(seed * 31 + m);
    const data = Array.from({ length: m }, () => { const x = 2 * rng() - 1; return [x, 1.5 * x + 0.5 * gaussian(rng)]; });
    const ts = Array.from({ length: 121 }, (_, i) => i * 0.025);
    const R = t => (t - 1.5) ** 2 * EX2 + s2;
    const E = t => mean(data.map(([x, y]) => (t * x - y) ** 2));
    const th = data.reduce((s, [x, y]) => s + x * y, 0) / data.reduce((s, [x]) => s + x * x, 0);
    chart.set([
      { name: 'true risk J*(θ)', color: 's1', data: ts.map(t => [t, R(t)]) },
      { name: 'empirical risk', color: 's2', data: ts.map(t => [t, E(t)]), dash: [5, 4] },
      { name: 'θ minimizing empirical risk', color: 's2', style: 'dots', r: 6, noHover: true, data: [[th, E(th)]] },
      { name: 'true best θ', color: 's1', style: 'dots', r: 6, noHover: true, data: [[1.5, R(1.5)]] }
    ]);
    lg.set([{ label: 'true risk (expectation over p_data)', color: c.s1 }, { label: `empirical risk on ${m} examples`, color: c.s2, kind: 'dash' }]);
    calc.innerHTML = `<span class="c">J*(θ) = E_(x,y)~p_data (θx − y)² = (θ − 1.5)²·E[x²] + σ² = (θ − 1.5)²/3 + 0.25</span>
empirical: (1/m)Σ(θx⁽ⁱ⁾ − y⁽ⁱ⁾)²  minimized at θ̂ = Σxy/Σx² = <b>${th.toFixed(3)}</b>
true risk of θ̂: J*(θ̂) = <b>${R(th).toFixed(4)}</b>   vs. best possible J*(1.5) = 0.2500   (empirical risk at θ̂: ${E(th).toFixed(4)} — optimistic)`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) Surrogate loss keeps learning after 0-1 loss hits zero ---------- */
Widgets.surrogateLoss = (el) => {
  UI.head(el, 'Training with a surrogate loss', 'Logistic regression trained by gradient descent on the negative log-likelihood (the surrogate). We also track the 0-1 loss we really care about.');
  const rng = mulberry32(5);
  const gen = n => Array.from({ length: n }, (_, i) => { const y = i % 2; return [(y ? 1 : -1) * (0.9 + 0.5 * Math.abs(gaussian(rng))), 1.4 * gaussian(rng), y]; });
  const train = gen(40), test = Array.from({ length: 400 }, (_, i) => { const y = i % 2; return [(y ? 1 : -1) * 1.1 + 0.55 * gaussian(rng), 1.4 * gaussian(rng), y]; });
  const recs = [];
  let w = [0, 0], b = 0;
  const nll = D => mean(D.map(([x1, x2, y]) => { const p = sigmoid(w[0] * x1 + w[1] * x2 + b); return -(y ? Math.log(p + 1e-12) : Math.log(1 - p + 1e-12)); }));
  const err = D => mean(D.map(([x1, x2, y]) => ((w[0] * x1 + w[1] * x2 + b > 0 ? 1 : 0) !== y ? 1 : 0)));
  for (let t = 0; t <= 3000; t++) {
    if (t % 25 === 0) recs.push([t, nll(train), err(train), err(test), Math.hypot(...w)]);
    let g0 = 0, g1 = 0, gb = 0;
    train.forEach(([x1, x2, y]) => { const e = sigmoid(w[0] * x1 + w[1] * x2 + b) - y; g0 += e * x1; g1 += e * x2; gb += e; });
    w = [w[0] - 0.5 * g0 / 40, w[1] - 0.5 * g1 / 40]; b -= 0.5 * gb / 40;
  }
  const chart = new LineChart(el, { xr: [0, 3000], yr: [0, 0.75], xlabel: 'training step', xname: 'step', aspect: .5, maxH: 280, vfmt: v => v.toFixed(4) });
  const lg = UI.legend(el, []);
  const el2 = h('div', { class: 'note' }, 'The 0-1 loss itself as a function of the score z·(2y − 1): flat everywhere (zero gradient) except a jump at 0. The NLL is smooth and keeps rewarding larger margins.');
  el.append(el2);
  const chart2 = new LineChart(el, { xr: [-4, 4], yr: [0, 4], xlabel: 'margin (2y−1)·z', xname: 'margin', aspect: .35, maxH: 190, minH: 140, vfmt: v => v.toFixed(3) });
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    chart.set([
      { name: 'training NLL (surrogate)', color: 's1', data: recs.map(r => [r[0], r[1]]) },
      { name: 'training 0-1 loss', color: 's2', data: recs.map(r => [r[0], r[2]]) },
      { name: 'test 0-1 loss', color: 's3', data: recs.map(r => [r[0], r[3]]) }
    ]);
    lg.set([{ label: 'training NLL (what we optimize)', color: c.s1 }, { label: 'training 0-1 loss', color: c.s2 }, { label: 'test 0-1 loss (what we care about)', color: c.s3 }]);
    const ms = Array.from({ length: 161 }, (_, i) => -4 + i * 0.05);
    chart2.set([{ name: '0-1 loss', color: 's2', data: ms.map(v => [v, v > 0 ? 0 : 1]) }, { name: 'NLL = log(1 + e^(−margin))', color: 's1', data: ms.map(v => [v, Math.log1p(Math.exp(-v))]) }]);
    const zeroAt = recs.find(r => r[2] === 0);
    const last = recs[recs.length - 1];
    calc.innerHTML = `<span class="c">0-1 loss: derivative is 0 (or undefined) everywhere → gradient descent gets no signal. The NLL is a smooth surrogate.</span>
${zeroAt ? `training 0-1 loss first reaches its minimum at step ${zeroAt[0]}` : `training 0-1 loss stays at ${(Math.min(...recs.map(r => r[2])) * 100).toFixed(1)}% (classes overlap)`};  after that the NLL keeps falling: ${recs[Math.floor(recs.length / 4)][1].toFixed(3)} → ${last[1].toFixed(3)}
‖w‖ grows ${recs[Math.floor(recs.length / 4)][4].toFixed(2)} → ${last[4].toFixed(2)}: the classes are pushed further apart (more confident, more robust)`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) Minibatch gradient noise ---------- */
Widgets.batchNoise = (el) => {
  UI.head(el, 'How noisy is a minibatch gradient?', 'Linear regression on m = 10 000 examples. At a fixed θ, each dot is the gradient (∂J/∂w, ∂J/∂b) estimated from one random minibatch; the star is the exact full-batch gradient.');
  const rng = mulberry32(9);
  const data = Array.from({ length: 10000 }, () => { const x = 2 * rng() - 1; return [x, 2 * x + 1 + 0.8 * gaussian(rng)]; });
  const sorted = data.slice().sort((a, b) => a[0] - b[0]);
  const th = [0.5, 0];
  const gradOf = S => { let gw = 0, gb = 0; S.forEach(([x, y]) => { const e = th[0] * x + th[1] - y; gw += 2 * e * x; gb += 2 * e; }); return [gw / S.length, gb / S.length]; };
  const full = gradOf(data);
  let lb = 1, shuffle = true, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Minibatch size', min: 0, max: 4, step: 0.05, value: lb, fmt: v => String(Math.round(10 ** v)), onInput: v => { lb = v; draw(); } });
  const tg = h('div', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', {}, 'How minibatches are drawn')));
  UI.seg(tg, [['shuf', 'Random (shuffled)'], ['seq', 'Consecutive, data sorted by x']], 'shuf', v => { shuffle = v === 'shuf'; draw(); });
  ctr.append(tg);
  const btns = h('div', { class: 'btns' }); UI.button(btns, 'Resample', () => { seed++; draw(); }); ctr.append(btns);
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap', style: 'max-width:420px;margin:0 auto' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Cloud of minibatch gradient estimates' });
  wrap.append(cv); el.append(wrap);
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const B = Math.round(10 ** lb), r2 = mulberry32(seed * 7 + B);
    const ests = [];
    for (let k = 0; k < 60; k++) {
      let S;
      if (shuffle) S = Array.from({ length: B }, () => data[Math.floor(r2() * data.length)]);
      else { const st = Math.floor(r2() * (sorted.length - B)); S = sorted.slice(st, st + B); }
      ests.push(gradOf(S));
    }
    const W = Math.min(420, boxWidth(wrap));
    const ctx = fitCanvas(cv, W, W);
    const P = makePlot(W, W, [full[0] - 4, full[0] + 4], [full[1] - 4, full[1] + 4], { l: 36, b: 26 });
    drawAxes(ctx, P, { xlabel: '∂J/∂w', ylabel: '∂J/∂b', nx: 4, ny: 4 });
    ctx.save(); P.clip(ctx);
    ests.forEach(g => dot(ctx, P.X(g[0]), P.Y(g[1]), 3.5, alpha(c.s1, .7)));
    ctx.fillStyle = c.text; ctx.font = '600 18px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', P.X(full[0]), P.Y(full[1]));
    ctx.restore();
    lgd.set([{ label: '60 minibatch estimates', color: c.s1, kind: 'dot' }, { label: 'exact gradient ★', color: c.text, kind: 'dot' }]);
    const mu = [mean(ests.map(g => g[0])), mean(ests.map(g => g[1]))];
    const sd = Math.sqrt(mean(ests.map(g => (g[0] - mu[0]) ** 2 + (g[1] - mu[1]) ** 2)));
    const sd1 = (() => { const one = Array.from({ length: 400 }, () => gradOf([data[Math.floor(r2() * data.length)]])); const m0 = [mean(one.map(g => g[0])), mean(one.map(g => g[1]))]; return Math.sqrt(mean(one.map(g => (g[0] - m0[0]) ** 2 + (g[1] - m0[1]) ** 2))); })();
    calc.innerHTML = `<span class="c">gradient = expectation over the data:  ∇θJ = E_(x,y)~p̂_data ∇θ log p_model(y|x; θ)   (8.5) → estimate it from B samples</span>
batch size B = ${B}:  measured spread of the estimates = <b>${sd.toFixed(3)}</b>   theory σ/√B = ${sd1.toFixed(2)}/√${B} = ${(sd1 / Math.sqrt(B)).toFixed(3)}
cost per estimate ∝ B.  100 → 10 000 examples = 100× the work for only 10× less noise.
${shuffle ? '' : '<b>Consecutive minibatches from sorted data are BIASED — the cloud is not centred on the true gradient. Shuffle first!</b>'}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) Ill-conditioning: big gradients, tiny progress ---------- */
Widgets.illcondTrace = (el) => {
  UI.head(el, 'Ill-conditioning: a strong gradient but slow progress', 'Gradient descent on a 10-dimensional quadratic whose curvatures (Hessian eigenvalues) range from 0.01 up to κ·0.01. The step must satisfy ε < 2/λ_max.');
  let lk = 3, frac = 0.95;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Condition number κ', min: 0, max: 4, step: 0.1, value: lk, fmt: v => fmt(10 ** v, 0), onInput: v => { lk = v; draw(); } });
  UI.slider(ctr, { label: 'Step ε as a fraction of 2/λ_max', min: 0.1, max: 1.05, step: 0.01, value: frac, fmt: v => v.toFixed(2), onInput: v => { frac = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 300], yr: [1e-6, 1e3], logY: true, xlabel: 'step', xname: 'step', aspect: .5, maxH: 280, snapX: x => Math.round(x) });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const kappa = 10 ** lk, n = 10;
    const lam = Array.from({ length: n }, (_, i) => 0.01 * kappa ** (i / (n - 1)));
    const lmax = lam[n - 1], eps = frac * 2 / lmax;
    let w = Array(n).fill(1);
    const Jr = [], Gr = [], Cr = [];
    for (let t = 0; t <= 300; t++) {
      const g = w.map((v, i) => lam[i] * v);
      const gg = dotv(g, g), gHg = g.reduce((s, v, i) => s + lam[i] * v * v, 0);
      Jr.push([t, Math.max(1e-6, 0.5 * w.reduce((s, v, i) => s + lam[i] * v * v, 0))]); Gr.push([t, Math.max(1e-6, gg)]); Cr.push([t, Math.max(1e-6, 0.5 * eps * eps * gHg / (eps * gg + 1e-300))]);
      w = w.map((v, i) => v - eps * g[i]);
      if (!isFinite(w[0]) || Math.abs(w[n - 1]) > 1e6) break;
    }
    chart.set([
      { name: 'cost J', color: 's1', data: Jr.map(([t, v]) => [t, Math.min(999, v)]) },
      { name: 'gradient norm² gᵀg', color: 's2', data: Gr.map(([t, v]) => [t, Math.min(999, v)]) },
      { name: 'curvature penalty ÷ gain', color: 's3', data: Cr.map(([t, v]) => [t, Math.min(999, v)]), dash: [5, 4] }
    ]);
    lgd.set([{ label: 'cost J(θ)', color: c.s1 }, { label: 'gᵀg (strength of the gradient)', color: c.s2 }, { label: '(½ε²gᵀHg) / (εgᵀg): > 1 means the step goes uphill', color: c.s3, kind: 'dash' }]);
    const diverged = Jr.length < 301;
    calc.innerHTML = `<span class="c">2nd-order Taylor: a step −εg changes the cost by  ½ε²gᵀHg − εgᵀg   (8.10) — trouble when ½ε²gᵀHg > εgᵀg</span>
eigenvalues 0.01 … ${fmt(lmax, 2)} (κ = ${fmt(kappa, 0)}),  ε = ${frac.toFixed(2)} × 2/λ_max = ${eps.toExponential(2)}
${diverged ? '<b>ε > 2/λ_max: the steep direction explodes — the cost increases even though the gradient is large.</b>' : `after 300 steps: cost ${Jr[300][1].toExponential(2)}; flattest direction kept ${((1 - eps * lam[0]) ** 300 * 100).toFixed(1)}% of its error — ε is limited by the steepest curvature, so learning is slow.`}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Weight-space symmetry & scaling ---------- */
Widgets.weightSymmetry = (el) => {
  UI.head(el, 'Many different weights, exactly the same network', 'A 2 → 3 → 1 ReLU network. Swap hidden units, or rescale a unit (incoming ×α, outgoing ×1/α): the parameters change, the function does not.');
  const base = { W: [[0.8, -0.5], [-0.3, 0.9], [0.6, 0.4]], c: [0.1, -0.2, 0.05], w: [1.2, -0.7, 0.9], b: 0.1 };
  let net = JSON.parse(JSON.stringify(base)), al = 1, n = 3, mL = 2;
  const ctr = h('div', { class: 'controls' });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Swap hidden units 1 ↔ 2', () => { [net.W[0], net.W[1]] = [net.W[1], net.W[0]]; [net.c[0], net.c[1]] = [net.c[1], net.c[0]]; [net.w[0], net.w[1]] = [net.w[1], net.w[0]]; draw(); }, 'btn primary');
  UI.button(btns, 'Reset', () => { net = JSON.parse(JSON.stringify(base)); al = 1; aS.set(1); draw(); });
  const aS = UI.slider(ctr, { label: 'Rescale unit 3 by α (in) and 1/α (out)', min: 0.2, max: 5, step: 0.05, value: 1, fmt: v => v.toFixed(2), onInput: v => { al = v; draw(); } });
  ctr.append(btns);
  const ctr2 = h('div', { class: 'controls' });
  UI.slider(ctr2, { label: 'units per layer n', min: 2, max: 20, step: 1, value: n, fmt: v => String(v), onInput: v => { n = v; draw(); } });
  UI.slider(ctr2, { label: 'hidden layers m', min: 1, max: 10, step: 1, value: mL, fmt: v => String(v), onInput: v => { mL = v; draw(); } });
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, calc, h('div', { class: 'note' }, 'How many equivalent arrangements just from permuting units:'), ctr2);
  const calc2 = h('div', { class: 'calcbox' });
  el.append(calc2);
  const lgamma1 = k => { let s = 0; for (let i = 2; i <= k; i++) s += Math.log10(i); return s; };
  function draw() {
    const W = net.W.map((r, i) => (i === 2 ? r.map(v => v * al) : r)), c = net.c.map((v, i) => (i === 2 ? v * al : v)), w = net.w.map((v, i) => (i === 2 ? v / al : v));
    const f = x => { const hh = W.map((r, i) => Math.max(0, r[0] * x[0] + r[1] * x[1] + c[i])); return { hh, y: dotv(w, hh) + net.b }; };
    const xs = [[1, 0.5], [-0.4, 1.2], [0.3, -0.8]];
    calc.innerHTML = `<span class="c">current parameters (rows = hidden units):  incoming W | bias c | outgoing w</span>
${W.map((r, i) => `unit ${i + 1}: (${r.map(v => v.toFixed(2)).join(', ')}) | ${c[i].toFixed(2)} | ${w[i].toFixed(3)}`).join('\n')}
outputs on 3 test inputs: <b>${xs.map(x => f(x).y.toFixed(5)).join(', ')}</b>   ← never change
<span class="c">ReLU is positively homogeneous: max(0, α·z) = α·max(0, z), so (1/α)·max(0, α·z) = max(0, z) for α > 0.</span>`;
    const lg = mL * lgamma1(n);
    calc2.innerHTML = `(n!)^m = (${n}!)^${mL} ≈ <b>10^${lg.toFixed(1)}</b> equivalent weight settings for every solution — plus a continuum from rescaling (an (m·n)-dimensional hyperbola of equal minima).
<span class="c">these minima all have the SAME cost, so this kind of non-convexity is harmless.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 6) Saddle points dominate in high dimensions ---------- */
Widgets.eigenSign = (el) => {
  UI.head(el, 'Why saddle points outnumber minima in high dimensions', 'At a critical point, it is a minimum only if ALL n Hessian eigenvalues are positive. We sample random symmetric Hessians H = A + Aᵀ + shift·I and count how often that happens.');
  let shift = 0;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Shift (higher = lower-cost region)', min: -2, max: 4, step: 0.1, value: shift, fmt: v => v.toFixed(1), onInput: v => { shift = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [1, 12], yr: [1e-4, 1], logY: true, xlabel: 'dimension n', ylabel: 'fraction that are minima', xname: 'n', aspect: .5, maxH: 270, snapX: x => Math.round(x), vfmt: v => (v * 100).toFixed(2) + '%' });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function isPosDef(A) { // all eigenvalues > 0  ⇔  Cholesky factorization succeeds
    const n = A.length, Lm = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= Lm[i][k] * Lm[j][k];
      if (i === j) { if (s <= 0) return false; Lm[i][i] = Math.sqrt(s); } else Lm[i][j] = s / Lm[j][j];
    }
    return true;
  }
  function draw() {
    const c = Colors.get();
    const rng = mulberry32(17);
    const rows = [];
    for (let n = 1; n <= 12; n++) {
      let mins = 0; const N = 600;
      for (let s = 0; s < N; s++) {
        const A = Array.from({ length: n }, () => Array.from({ length: n }, () => gaussian(rng)));
        const Hm = A.map((r, i) => r.map((v, j) => (v + A[j][i]) / 2 + (i === j ? shift : 0)));
        if (isPosDef(Hm)) mins++;
      }
      rows.push([n, Math.max(1.1e-4, mins / N)]);
    }
    chart.set([
      { name: 'random Hessians', color: 's1', data: rows },
      { name: 'coin-flip model 2⁻ⁿ', color: 's2', data: rows.map(([n]) => [n, Math.max(1.1e-4, 2 ** -n)]), dash: [5, 4] }
    ]);
    lgd.set([{ label: `fraction of critical points that are minima (shift ${shift.toFixed(1)})`, color: c.s1 }, { label: 'if each eigenvalue sign were a coin flip: 2⁻ⁿ', color: c.s2, kind: 'dash' }]);
    calc.innerHTML = `<span class="c">minimum ⇔ all eigenvalues > 0;  saddle ⇔ mixed signs.  With n independent “coin flips”, P(all positive) = 2⁻ⁿ.</span>
n = 12, coin model: 2⁻¹² = ${(2 ** -12).toExponential(2)}  →  saddles outnumber minima ~ ${(2 ** 12).toLocaleString()} : 1
<span class="c">larger shift ≈ lower cost: eigenvalues become more positive, so low-cost critical points are more likely to be minima, high-cost ones are saddles.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) Cliffs and gradient clipping ---------- */
Widgets.cliffClip = (el) => {
  UI.head(el, 'A cliff in the loss, and gradient clipping', 'The loss has a sheer cliff (from multiplying large weights: w₁·w₂ > 1). Near it the gradient is enormous; one plain step can throw the parameters far away. Clipping limits the step length.');
  const f = (x, y) => 0.15 * (x * x + y * y) + 3 * sigmoid(25 * (x * y - 1));
  const g = (x, y) => { const s = sigmoid(25 * (x * y - 1)), d = 3 * 25 * s * (1 - s); return [0.3 * x + d * y, 0.3 * y + d * x]; };
  let eps = 0.3, v = 0.5, start = [1.5, 0.72];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Learning rate ε', min: 0.05, max: 1, step: 0.01, value: eps, fmt: t => t.toFixed(2), onInput: t => { eps = t; draw(); } });
  UI.slider(ctr, { label: 'Clipping threshold v', min: 0.1, max: 3, step: 0.05, value: v, fmt: t => t.toFixed(2), onInput: t => { v = t; draw(); } });
  el.append(ctr, h('div', { class: 'note' }, 'Click the map to choose a start near the cliff.'));
  const L = landscape(el, f, [-0.5, 3, -0.5, 3], { click: (x, y) => { start = [x, y]; draw(); }, aria: 'Loss with a cliff' });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function run(clip) {
    let p = start.slice(); const pts = [p.slice()]; let maxStep = 0;
    for (let t = 0; t < 40; t++) {
      let gr = g(...p); const n = Math.hypot(...gr);
      if (clip && n > v) gr = gr.map(q => q * v / n);
      const step = gr.map(q => -eps * q); maxStep = Math.max(maxStep, Math.hypot(...step));
      p = [p[0] + step[0], p[1] + step[1]]; pts.push(p.slice());
      if (Math.abs(p[0]) > 50 || Math.abs(p[1]) > 50) break;
    }
    return { pts, maxStep };
  }
  function draw() {
    const A = run(false), B = run(true);
    L.draw([{ pts: A.pts, color: 's8', dots: true }, { pts: B.pts, color: 's3', dots: true }], [{ at: [0, 0] }]);
    const c = Colors.get();
    lgd.set([{ label: 'plain gradient descent', color: c.s8 }, { label: 'with gradient clipping', color: c.s3 }, { label: 'minimum ★', color: c.text, kind: 'dot' }]);
    const g0 = g(...start), n0 = Math.hypot(...g0);
    calc.innerHTML = `<span class="c">gradient at the start: g = (${g0[0].toFixed(2)}, ${g0[1].toFixed(2)}), ‖g‖ = ${n0.toFixed(2)}</span>
plain step length ε‖g‖ = <b>${(eps * n0).toFixed(2)}</b>   (largest step in the run: ${A.maxStep.toFixed(2)})
clipped: if ‖g‖ > v then g ← g·v/‖g‖  →  step length ≤ εv = <b>${(eps * v).toFixed(2)}</b>   (largest: ${B.maxStep.toFixed(2)})`;
  }
  draw();
  return { redraw: () => { L.reset(); draw(); } };
};

/* ---------- 8) Repeated multiplication by W ---------- */
Widgets.powerEigen = (el) => {
  UI.head(el, 'Multiplying by the same matrix t times', 'A path that repeatedly multiplies by W scales each eigen-direction by λᵗ. Anything not close to |λ| = 1 explodes or vanishes — and so do the gradients flowing along that path.');
  let l1 = 0.9, l2 = 1.1, t = 30;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'eigenvalue λ₁', min: 0.5, max: 1.5, step: 0.01, value: l1, fmt: v => v.toFixed(2), onInput: v => { l1 = v; draw(); } });
  UI.slider(ctr, { label: 'eigenvalue λ₂', min: 0.5, max: 1.5, step: 0.01, value: l2, fmt: v => v.toFixed(2), onInput: v => { l2 = v; draw(); } });
  UI.slider(ctr, { label: 'steps t', min: 1, max: 100, step: 1, value: t, fmt: v => String(v), onInput: v => { t = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 100], yr: [1e-10, 1e10], logY: true, xlabel: 't', ylabel: 'λᵗ', xname: 't', aspect: .5, maxH: 270, refY: 1, snapX: x => Math.round(x) });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const ts = Array.from({ length: 101 }, (_, i) => i);
    const cl = v => clamp(v, 1.1e-10, 9e9);
    chart.set([
      { name: 'λ₁ᵗ', color: 's1', data: ts.map(k => [k, cl(l1 ** k)]) },
      { name: 'λ₂ᵗ', color: 's2', data: ts.map(k => [k, cl(l2 ** k)]) },
      { name: 'now', color: 'text', style: 'dots', r: 5, noHover: true, data: [[t, cl(l1 ** t)], [t, cl(l2 ** t)]] }
    ]);
    lgd.set([{ label: `λ₁ = ${l1.toFixed(2)}`, color: c.s1 }, { label: `λ₂ = ${l2.toFixed(2)}`, color: c.s2 }]);
    const verdict = v => (v > 1.01 ? 'explodes' : v < 0.99 ? 'vanishes' : 'stays ~1');
    calc.innerHTML = `<span class="c">W = V diag(λ) V⁻¹  ⇒  Wᵗ = (V diag(λ) V⁻¹)ᵗ = V diag(λ)ᵗ V⁻¹   (8.11)</span>
λ₁ᵗ = ${l1.toFixed(2)}^${t} = <b>${(l1 ** t).toExponential(2)}</b> (${verdict(l1)})      λ₂ᵗ = ${l2.toFixed(2)}^${t} = <b>${(l2 ** t).toExponential(2)}</b> (${verdict(l2)})
<span class="c">vanishing gradient: no idea which way to move; exploding gradient: unstable learning (the cliffs on the previous slide).</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 9) Noisy and biased gradients ---------- */
Widgets.noisyGD = (el) => {
  UI.head(el, 'Optimizing with inexact gradients', 'Three runs on the same bowl: exact gradient, gradient + random noise (variance), and gradient + a constant error (bias).');
  const f = (x, y) => 0.5 * (x * x + 4 * y * y);
  const g = (x, y) => [x, 4 * y];
  let sn = 1.5, bias = 0.6, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Noise std (variance of the estimate)', min: 0, max: 4, step: 0.1, value: sn, fmt: v => v.toFixed(1), onInput: v => { sn = v; draw(); } });
  UI.slider(ctr, { label: 'Bias (systematic error)', min: 0, max: 2, step: 0.05, value: bias, fmt: v => v.toFixed(2), onInput: v => { bias = v; draw(); } });
  const btns = h('div', { class: 'btns' }); UI.button(btns, 'New noise', () => { seed++; draw(); }); ctr.append(btns);
  el.append(ctr);
  const L = landscape(el, f, [-3, 3, -3, 3], { aria: 'Paths with exact, noisy and biased gradients' });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function run(kind) {
    const rng = mulberry32(seed * 3 + (kind === 'noise' ? 1 : 2));
    let p = [-2.6, 2.2]; const pts = [p.slice()];
    for (let t = 0; t < 60; t++) {
      let gr = g(...p);
      if (kind === 'noise') gr = gr.map(q => q + sn * gaussian(rng));
      if (kind === 'bias') gr = [gr[0] - bias, gr[1] - bias];
      p = [p[0] - 0.12 * gr[0], p[1] - 0.12 * gr[1]]; pts.push(p.slice());
    }
    return pts;
  }
  function draw() {
    const c = Colors.get();
    const A = run('exact'), B = run('noise'), C = run('bias');
    L.draw([{ pts: A, color: 's1' }, { pts: B, color: 's2' }, { pts: C, color: 's7' }], [{ at: [0, 0] }]);
    lgd.set([{ label: 'exact gradient', color: c.s1 }, { label: 'noisy (unbiased) gradient', color: c.s2 }, { label: 'biased gradient', color: c.s7 }]);
    const end = p => p[p.length - 1];
    calc.innerHTML = `final distance to the minimum:  exact ${Math.hypot(...end(A)).toFixed(3)},  noisy ${Math.hypot(...end(B)).toFixed(3)} (keeps jittering),  biased ${Math.hypot(...end(C)).toFixed(3)}
<span class="c">biased estimate → converges to the WRONG point (where ∇f = bias: x = ${bias.toFixed(2)}, y = ${(bias / 4).toFixed(3)}); noise → wanders around the right point (smaller steps / averaging help)</span>`;
  }
  draw();
  return { redraw: () => { L.reset(); draw(); } };
};

/* ---------- 10) Wrong side of the mountain ---------- */
Widgets.wrongSide = (el) => {
  UI.head(el, 'Local descent can lead away from the solution', 'Deep valley on the left, a mountain in the middle, and on the right a slope that keeps going down gently forever (no minimum). Click to start somewhere.');
  const f = (x, y) => 2 - 0.6 * Math.tanh(0.6 * x) + 2.2 * Math.exp(-(x * x) / 0.8 - (y * y) / 2.2) - 2.6 * Math.exp(-((x + 2.6) ** 2) / 0.6 - (y * y) / 1.2) + 0.04 * y * y;
  const grad = (x, y) => { const e = 1e-5; return [(f(x + e, y) - f(x - e, y)) / (2 * e), (f(x, y + e) - f(x, y - e)) / (2 * e)]; };
  let starts = [[0.6, 0.2], [0.4, 2.6], [-1.0, 0.5]];
  const L = landscape(el, f, [-4.5, 4.5, -3.5, 3.5], { maxW: 520, click: (x, y) => { starts = [[x, y], ...starts].slice(0, 3); draw(); }, aria: 'Mountain between start and valley' });
  const lgd = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function run(s) { let p = s.slice(); const pts = [p.slice()]; let len = 0; for (let t = 0; t < 400; t++) { const g = grad(...p); const np = [p[0] - 0.08 * g[0], p[1] - 0.08 * g[1]]; len += Math.hypot(np[0] - p[0], np[1] - p[1]); p = np; pts.push(p.slice()); if (Math.abs(p[0]) > 4.4) break; } return { pts, len, end: p }; }
  function draw() {
    const c = Colors.get();
    const runs = starts.map(run), cols = ['s1', 's2', 's7'];
    L.draw(runs.map((r, i) => ({ pts: r.pts, color: cols[i] })), [{ at: [-2.6, 0], label: 'deep valley' }, { at: [0, 0], sym: '▲' }]);
    lgd.set(runs.map((r, i) => ({ label: `start (${starts[i][0].toFixed(1)}, ${starts[i][1].toFixed(1)}) → ${r.end[0] < -1.5 ? 'reaches the valley' : 'slides away (no minimum there)'}`, color: c[cols[i]] })));
    calc.innerHTML = runs.map((r, i) => `run ${i + 1}: path length ${r.len.toFixed(2)}, ends at (${r.end[0].toFixed(2)}, ${r.end[1].toFixed(2)}), cost ${f(...r.end).toFixed(3)}`).join('\n') +
      `\n<span class="c">no local minimum or saddle is to blame: the locally best direction simply points away from the good region. Good initialization matters.</span>`;
  }
  draw();
  return { redraw: () => { L.reset(); draw(); } };
};

/* =========================================================
   Quiz
   ========================================================= */
window.QUIZ = [
  { q: 'The risk J*(θ) is an expectation over…', opts: ['the training set', 'the true data-generating distribution p_data', 'the validation set', 'the parameters'], a: 1, why: 'We can’t compute it, so we minimize the empirical risk (average over the training set) and hope the true risk goes down too.' },
  { q: 'Why do we rarely minimize the empirical 0-1 loss directly?', opts: ['It is too accurate', 'It has zero or undefined derivatives everywhere, and exact minimization is intractable', 'It is always zero', 'It needs labels'], a: 1, why: 'We optimize a surrogate such as the negative log-likelihood, which is smooth and even keeps improving robustness after the training 0-1 loss reaches zero.' },
  { q: 'How does training in deep learning usually halt?', opts: ['At an exact local minimum', 'When an early-stopping criterion on the validation set is met, often while gradients are still large', 'When the gradient is exactly zero', 'After one epoch'], a: 1, why: 'Unlike pure optimization, the stopping rule is based on the true loss on held-out data.' },
  { q: 'Increasing the minibatch from 100 to 10 000 examples reduces the standard error of the gradient by…', opts: ['100×', '10×', '2×', 'nothing'], a: 1, why: 'Standard error ∝ 1/√n: √(10 000/100) = 10. So 100× the computation buys only 10× less noise.' },
  { q: 'Why must minibatches be drawn randomly (e.g. from a shuffled dataset)?', opts: ['To use less memory', 'So that each gradient estimate is unbiased and consecutive estimates are independent', 'Because GPUs require it', 'It does not matter'], a: 1, why: 'Datasets are often ordered (by class, time, patient…); consecutive examples are correlated and give biased estimates.' },
  { q: 'Ill-conditioning shows up when…', opts: ['the gradient is zero', '½ε²gᵀHg exceeds εgᵀg, so even small steps increase the cost', 'the network is too small', 'the data are shuffled'], a: 1, why: 'Learning then becomes slow despite a strong gradient: ε must shrink to compensate for the strong curvature.' },
  { q: 'Why are the many local minima caused by weight-space symmetry NOT a problem?', opts: ['They have high cost', 'They all have exactly the same cost (they represent the same function)', 'Gradient descent avoids them', 'There are only a few'], a: 1, why: 'Swapping hidden units or rescaling ReLU units gives equivalent networks. Only local minima with much higher cost than the global minimum would hurt — and for large nets most local minima have low cost.' },
  { q: 'In high-dimensional non-convex functions, critical points are most often…', opts: ['global minima', 'saddle points', 'local maxima', 'flat regions only'], a: 1, why: 'A minimum needs all n Hessian eigenvalues positive — exponentially unlikely. Saddle points are a problem for Newton’s method; gradient descent usually escapes them.' },
  { q: 'Gradient clipping is a remedy for…', opts: ['vanishing gradients', 'cliffs / exploding gradients that would throw the parameters far away', 'saddle points', 'overfitting'], a: 1, why: 'When the gradient is huge, clipping caps the step so we stay where the gradient is still a reasonable guide.' },
  { q: 'Repeatedly multiplying by W = V diag(λ) V⁻¹ for t steps scales eigen-directions by…', opts: ['tλ', 'λᵗ', 'λ/t', 'λ²'], a: 1, why: 'Wᵗ = V diag(λ)ᵗ V⁻¹. |λ| < 1 vanishes, |λ| > 1 explodes — the vanishing/exploding gradient problem, worst in RNNs.' },
  { q: 'Which statement about local vs. global structure is correct?', opts: ['Training always ends at a critical point', 'Local descent may point away from distant low-cost regions; good initialization in a well-behaved region can avoid this', 'Saddle points are the only difficulty', 'Softmax NLL has a global minimum at 0 that is always reached'], a: 1, why: 'Neural nets often never reach a critical point (e.g. softmax NLL only approaches 0). Much research therefore looks for good initial points rather than non-local moves.' }
];
