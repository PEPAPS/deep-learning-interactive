/* =========================================================
   second-order.js — Part A demos (Chapter 8.6)
   empirical risk · Taylor approximations · optimizer playground · cost calculator
   ========================================================= */
'use strict';

/* ---------- 2-D linear algebra ---------- */
const V2 = {
  dot: (a, b) => a[0] * b[0] + a[1] * b[1],
  norm: a => Math.hypot(a[0], a[1]),
  add: (a, b) => [a[0] + b[0], a[1] + b[1]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1]],
  scale: (a, s) => [a[0] * s, a[1] * s],
  neg: a => [-a[0], -a[1]],
  mv: (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]],
  mm: (A, B) => [
    [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
    [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]],
  T: A => [[A[0][0], A[1][0]], [A[0][1], A[1][1]]],
  addI: (M, a) => [[M[0][0] + a, M[0][1]], [M[1][0], M[1][1] + a]],
  inv(M) {
    const d = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    if (Math.abs(d) < 1e-12) return null;
    return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]];
  },
  solve(M, b) { const I = V2.inv(M); return I ? V2.mv(I, b) : null; },
  eig(M) { // symmetric 2x2
    const a = M[0][0], b = (M[0][1] + M[1][0]) / 2, d = M[1][1];
    const m = (a + d) / 2, r = Math.sqrt(((a - d) / 2) ** 2 + b * b);
    return [m - r, m + r];
  }
};

/* =========================================================
   1) Empirical risk demo — J(θ) is an average of per-example losses
   ========================================================= */
Widgets.riskDemo = (el) => {
  UI.head(el, 'Your first loss landscape', 'Model: <b>ŷ = θ·x</b> (one weight). Each dot is a training example. Move θ and watch the per-example losses and their average J(θ).');
  const xs = [-2.2, -1.4, -0.6, 0.2, 0.9, 1.6, 2.3];
  const noise = [0.5, -0.6, 0.35, -0.3, 0.55, -0.7, 0.25];
  const ys = xs.map((x, i) => 1.5 * x + noise[i]);
  const m = xs.length;
  const Li = (th, i) => (th * xs[i] - ys[i]) ** 2;
  const J = th => xs.reduce((s, _, i) => s + Li(th, i), 0) / m;
  const dJ = th => xs.reduce((s, x, i) => s + 2 * (th * x - ys[i]) * x, 0) / m;
  const thStar = xs.reduce((s, x, i) => s + x * ys[i], 0) / xs.reduce((s, x) => s + x * x, 0);
  let th = 0.2, anim = null;

  const ctr = h('div', { class: 'controls' });
  const sl = UI.slider(ctr, { label: 'Weight θ', min: -1, max: 4, step: 0.01, value: th, fmt: v => v.toFixed(2), onInput: v => { th = v; stop(); draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Take 1 gradient step', () => { stop(); step(); }, 'btn primary');
  UI.button(btns, 'Train (20 steps)', () => { stop(); let k = 0; anim = setInterval(() => { step(); if (++k >= 20) stop(); }, 110); });
  ctr.append(btns);
  const two = h('div', { class: 'row2' });
  const w1 = h('div', { class: 'canvas-wrap' }), w2 = h('div', { class: 'canvas-wrap' });
  const c1 = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Data and fitted line' });
  const c2 = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Loss curve J of theta' });
  w1.append(c1); w2.append(c2); two.append(w1, w2);
  el.append(ctr, two);
  const lg = UI.legend(el, []);
  const read = h('div', { class: 'stat-row' });
  el.append(read);
  function stop() { if (anim) { clearInterval(anim); anim = null; } }
  function step() { th = th - 0.12 * dJ(th); sl.set(th); draw(); }
  function draw() {
    const c = Colors.get();
    lg.set([
      { label: 'Training examples', color: c.s1, kind: 'dot' },
      { label: 'Error of each example', color: c.s2 },
      { label: 'J(θ) = average loss', color: c.text },
      { label: 'Per-example loss curves', color: c.muted }
    ]);
    // left: data
    let W = boxWidth(w1), H = Math.round(W * .8);
    let ctx = fitCanvas(c1, W, H);
    let p = makePlot(W, H, [-2.8, 2.8], [-5, 5], { l: 34, b: 26 });
    drawAxes(ctx, p, { xlabel: 'x', ylabel: 'y', nx: 5 });
    ctx.save(); p.clip(ctx);
    xs.forEach((x, i) => {
      ctx.strokeStyle = c.s2; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(p.X(x), p.Y(ys[i])); ctx.lineTo(p.X(x), p.Y(th * x)); ctx.stroke();
    });
    ctx.strokeStyle = c.text; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(p.X(-3), p.Y(-3 * th)); ctx.lineTo(p.X(3), p.Y(3 * th)); ctx.stroke();
    xs.forEach((x, i) => dot(ctx, p.X(x), p.Y(ys[i]), 4.5, c.s1, c.surface));
    ctx.restore();
    label(ctx, `ŷ = ${th.toFixed(2)}·x`, p.X(-2.6), p.Y(4.3), { color: c.text, bold: true });
    // right: loss curves
    W = boxWidth(w2); H = Math.round(W * .8);
    ctx = fitCanvas(c2, W, H);
    p = makePlot(W, H, [-1, 4], [0, 24], { l: 34, b: 26 });
    drawAxes(ctx, p, { xlabel: 'θ', ylabel: 'loss', nx: 5 });
    ctx.save(); p.clip(ctx);
    for (let i = 0; i < m; i++) {
      ctx.strokeStyle = alpha(c.muted, .45); ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = 0; k <= 120; k++) { const t = -1 + 5 * k / 120; const y = Li(t, i); k ? ctx.lineTo(p.X(t), p.Y(y)) : ctx.moveTo(p.X(t), p.Y(y)); }
      ctx.stroke();
      dot(ctx, p.X(th), p.Y(Li(th, i)), 2.5, c.muted);
    }
    ctx.strokeStyle = c.text; ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let k = 0; k <= 160; k++) { const t = -1 + 5 * k / 160; k ? ctx.lineTo(p.X(t), p.Y(J(t))) : ctx.moveTo(p.X(t), p.Y(J(t))); }
    ctx.stroke();
    ctx.setLineDash([3, 4]); ctx.strokeStyle = c.axis; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(p.X(th), p.Y(0)); ctx.lineTo(p.X(th), p.Y(J(th))); ctx.stroke(); ctx.setLineDash([]);
    dot(ctx, p.X(th), p.Y(J(th)), 6, c.s2, c.surface);
    ctx.restore();
    read.innerHTML = '';
    read.append(
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'J(θ) now'), h('div', { class: 'v' }, J(th).toFixed(3))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'slope ∇J(θ)'), h('div', { class: 'v' }, dJ(th).toFixed(3))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'best θ*'), h('div', { class: 'v' }, thStar.toFixed(3), h('small', {}, ` (J = ${J(thStar).toFixed(3)})`)))
    );
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   2) Taylor approximations in 1-D: gradient step vs Newton step
   ========================================================= */
Widgets.taylor1d = (el) => {
  UI.head(el, 'Slope-only step vs. curvature-aware step',
    'Drag on the plot (or use the slider) to choose where you stand, θ₀. The dashed line is what the <b>gradient</b> sees; the orange parabola is what <b>Newton</b> sees.');
  const J = t => 0.05 * t ** 4 - 0.3 * t * t + 0.2 * t + 2;
  const d1 = t => 0.2 * t ** 3 - 0.6 * t + 0.2;
  const d2 = t => 0.6 * t * t - 0.6;
  let th = 2.7, lr = 0.25;
  const ctr = h('div', { class: 'controls' });
  const s1 = UI.slider(ctr, { label: 'Where you stand θ₀', min: -3.3, max: 3.3, step: 0.01, value: th, fmt: v => v.toFixed(2), onInput: v => { th = v; draw(); } });
  UI.slider(ctr, { label: 'Learning rate ε (gradient step)', min: 0.02, max: 1.2, step: 0.01, value: lr, fmt: v => v.toFixed(2), onInput: v => { lr = v; draw(); } });
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart clickable', role: 'img', 'aria-label': 'Function with tangent line and quadratic approximation' });
  wrap.append(cv);
  el.append(ctr, wrap);
  const lg = UI.legend(el, []);
  const out = h('div', { class: 'stack', style: 'gap:8px' });
  el.append(out);
  let p = null, dragging = false;
  const setFromEvent = e => {
    if (!p) return;
    const r = cv.getBoundingClientRect();
    th = clamp(p.invX(e.clientX - r.left), -3.3, 3.3);
    s1.set(th); draw();
  };
  cv.addEventListener('pointerdown', e => { dragging = true; cv.setPointerCapture(e.pointerId); setFromEvent(e); });
  cv.addEventListener('pointermove', e => { if (dragging) setFromEvent(e); });
  cv.addEventListener('pointerup', () => { dragging = false; });

  function draw() {
    const c = Colors.get();
    lg.set([
      { label: 'True loss J(θ)', color: c.text },
      { label: 'Gradient view: tangent line', color: c.s1, kind: 'dash' },
      { label: "Newton view: 2nd-order parabola", color: c.s2 }
    ]);
    const W = boxWidth(wrap), H = clamp(Math.round(W * .62), 240, 380);
    const ctx = fitCanvas(cv, W, H);
    p = makePlot(W, H, [-3.6, 3.6], [0, 7.5], { l: 34 });
    drawAxes(ctx, p, { xlabel: 'θ', ylabel: 'J(θ)' });
    const J0 = J(th), g = d1(th), cur = d2(th);
    const thG = th - lr * g;
    const thN = Math.abs(cur) > 1e-6 ? th - g / cur : NaN;
    ctx.save(); p.clip(ctx);
    const path = (fn, color, w, dash) => {
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash || []);
      ctx.beginPath();
      for (let k = 0; k <= 240; k++) { const t = -3.6 + 7.2 * k / 240; const y = fn(t); k ? ctx.lineTo(p.X(t), p.Y(y)) : ctx.moveTo(p.X(t), p.Y(y)); }
      ctx.stroke(); ctx.setLineDash([]);
    };
    path(J, c.text, 2.5);
    path(t => J0 + g * (t - th), c.s1, 2, [6, 5]);
    path(t => J0 + g * (t - th) + 0.5 * cur * (t - th) ** 2, c.s2, 2);
    // landing points
    const yB = p.Y(0) - 10;
    if (isFinite(thG)) {
      arrow(ctx, p.X(th), yB, p.X(clamp(thG, -3.6, 3.6)), yB, c.s1, 2);
      dot(ctx, p.X(thG), p.Y(J(thG)), 6, c.s1, c.surface);
    }
    if (isFinite(thN)) {
      arrow(ctx, p.X(th), yB - 14, p.X(clamp(thN, -3.6, 3.6)), yB - 14, c.s2, 2);
      const vy = J0 + g * (thN - th) + 0.5 * cur * (thN - th) ** 2;
      ctx.setLineDash([2, 3]); ctx.strokeStyle = c.s2; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(p.X(thN), p.Y(vy)); ctx.lineTo(p.X(thN), p.Y(J(thN))); ctx.stroke(); ctx.setLineDash([]);
      dot(ctx, p.X(thN), p.Y(vy), 4, c.surface, c.s2);
      dot(ctx, p.X(thN), p.Y(J(thN)), 6, c.s2, c.surface);
    }
    dot(ctx, p.X(th), p.Y(J0), 7, c.text, c.surface);
    ctx.restore();
    label(ctx, 'θ₀', p.X(th) + 9, p.Y(J0) - 10, { color: c.text, bold: true });
    label(ctx, 'GD', p.X(clamp(thG, -3.5, 3.5)), yB - 1, { color: c.text2, align: 'center', base: 'bottom', size: 10 });
    if (isFinite(thN)) label(ctx, 'Newton', p.X(clamp(thN, -3.4, 3.4)), yB - 15, { color: c.text2, align: 'center', base: 'bottom', size: 10 });

    out.innerHTML = '';
    out.append(h('div', { class: 'stat-row' },
      h('div', { class: 'stat' }, h('div', { class: 'k' }, "Slope J′(θ₀)"), h('div', { class: 'v' }, fmt(g))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, "Curvature J″(θ₀)"), h('div', { class: 'v' }, fmt(cur))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'GD lands at'), h('div', { class: 'v' }, fmt(thG), h('small', {}, ` J=${fmt(J(thG))}`))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Newton lands at'), h('div', { class: 'v' }, isFinite(thN) ? fmt(thN) : '—', h('small', {}, isFinite(thN) ? ` J=${fmt(J(thN))}` : '')))
    ));
    let msg;
    if (cur < 0) msg = h('div', { class: 'callout bad' }, h('span', { class: 'ico' }, '!'), h('div', {}, h('b', { class: 'tag' }, 'Negative curvature.'), 'The parabola opens downward, so its flat point is a ', h('b', {}, 'maximum'), '. Newton jumps to where the slope is zero — here that is a hilltop, so it goes the wrong way. This is exactly the saddle-point problem from the lecture.'));
    else if (J(thN) < J(thG)) msg = h('div', { class: 'callout good' }, h('span', { class: 'ico' }, '✓'), h('div', {}, h('b', { class: 'tag' }, 'Curvature helps.'), 'Newton used J″ to pick its own step size and landed lower than the gradient step. Gradient descent needs you to guess ε.'));
    else msg = h('div', { class: 'callout info' }, h('span', { class: 'ico' }, 'i'), h('div', {}, h('b', { class: 'tag' }, 'Approximations are local.'), 'The parabola only matches J near θ₀, so Newton is not always better after one step — that is why it is iterated.'));
    out.append(msg);
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   3) Optimizer playground (2-D)
   ========================================================= */
function makeA(kappa, deg = 35) {
  const t = deg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const l1 = 1, l2 = kappa;
  return [[l1 * c * c + l2 * s * s, (l1 - l2) * c * s], [(l1 - l2) * c * s, l1 * s * s + l2 * c * c]];
}
const FUNCS = {
  bowl: {
    label: 'Stretched bowl (quadratic)', dom: [-3, 3, -3, 3], start: [-2.5, 1.2], usesKappa: true, lr: 0.15,
    build(P) {
      const A = makeA(P.kappa);
      return {
        f: (x, y) => 0.5 * (A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y),
        g: (x, y) => [A[0][0] * x + A[0][1] * y, A[1][0] * x + A[1][1] * y],
        H: () => A, minima: [[0, 0]], saddles: [], quad: true
      };
    }
  },
  convex: {
    label: 'Curvy bowl (convex, not quadratic)', dom: [-3, 3, -3, 3], start: [-2.6, 2.2], usesKappa: true, lr: 0.06,
    build(P) {
      const A = makeA(P.kappa);
      return {
        f: (x, y) => 0.5 * (A[0][0] * x * x + 2 * A[0][1] * x * y + A[1][1] * y * y) + 0.1 * (x ** 4 + y ** 4),
        g: (x, y) => [A[0][0] * x + A[0][1] * y + 0.4 * x ** 3, A[1][0] * x + A[1][1] * y + 0.4 * y ** 3],
        H: (x, y) => [[A[0][0] + 1.2 * x * x, A[0][1]], [A[1][0], A[1][1] + 1.2 * y * y]],
        minima: [[0, 0]], saddles: []
      };
    }
  },
  saddle: {
    label: 'Saddle point', dom: [-3, 3, -3, 3], start: [2.4, 0.35], lr: 0.3,
    build() {
      return {
        f: (x, y) => 0.5 * x * x - 0.5 * y * y + 0.05 * y ** 4,
        g: (x, y) => [x, -y + 0.2 * y ** 3],
        H: (x, y) => [[1, 0], [0, -1 + 0.6 * y * y]],
        minima: [[0, Math.sqrt(5)], [0, -Math.sqrt(5)]], saddles: [[0, 0]]
      };
    }
  },
  banana: {
    label: 'Banana valley (Rosenbrock)', dom: [-2, 2, -1, 3], start: [-1.4, 2.4], lr: 0.012,
    build() {
      return {
        f: (x, y) => (1 - x) ** 2 + 5 * (y - x * x) ** 2,
        g: (x, y) => [-2 * (1 - x) - 20 * x * (y - x * x), 10 * (y - x * x)],
        H: (x, y) => [[2 - 20 * y + 60 * x * x, -20 * x], [-20 * x, 10]],
        minima: [[1, 1]], saddles: []
      };
    }
  }
};

/* exact-ish line search: bracket, then golden-section on φ(t) = f(θ + t d), t ≥ 0 */
function lineSearch(F, th, d, mode) {
  if (mode === 'rough') { // backtracking (Armijo): accept the first step that decreases J enough
    const slope = V2.dot(F.g(...th), d), f0 = F.f(...th);
    let t = 1;
    for (let k = 0; k < 60 && !(F.f(th[0] + t * d[0], th[1] + t * d[1]) <= f0 + 1e-4 * t * slope); k++) t *= 0.5;
    return t;
  }
  if (F.quad) { // quadratic: the exact minimizer along the line has a closed form
    const g = F.g(...th), den = V2.dot(d, V2.mv(F.H(), d));
    if (den > 0) return -V2.dot(g, d) / den;
  }
  const phi = t => F.f(th[0] + t * d[0], th[1] + t * d[1]);
  const f0 = phi(0);
  let t = 1e-3;
  let guard = 0;
  while (phi(t) >= f0 && t > 1e-12 && guard++ < 60) t /= 2;
  guard = 0;
  while (phi(2 * t) < phi(t) && t < 1e4 && guard++ < 60) t *= 2;
  let lo = 0, hi = 2 * t;
  const gr = (Math.sqrt(5) - 1) / 2;
  let a = hi - gr * (hi - lo), b = lo + gr * (hi - lo), fa = phi(a), fb = phi(b);
  for (let k = 0; k < 90; k++) {
    if (fa < fb) { hi = b; b = a; fb = fa; a = hi - gr * (hi - lo); fa = phi(a); }
    else { lo = a; a = b; fa = fb; b = lo + gr * (hi - lo); fb = phi(b); }
  }
  return (lo + hi) / 2;
}

const METHODS = {
  gd: {
    label: 'Gradient descent', color: '--s1', key: 's1',
    step(r, F, P) { const g = F.g(...r.th); r.th = V2.sub(r.th, V2.scale(g, P.lr)); }
  },
  newton: {
    label: "Newton's method", color: '--s2', key: 's2',
    step(r, F) {
      const g = F.g(...r.th), H = F.H(...r.th);
      const p = V2.solve(H, V2.neg(g));
      if (!p) { r.done = true; r.msg = 'Hessian is singular — cannot invert'; return; }
      r.th = V2.add(r.th, p);
    }
  },
  regnewton: {
    label: 'Regularized Newton (H + αI)', color: '--s3', key: 's3',
    step(r, F, P) {
      const g = F.g(...r.th), H = V2.addI(F.H(...r.th), P.alpha);
      const p = V2.solve(H, V2.neg(g));
      if (!p) { r.done = true; r.msg = 'H + αI is singular'; return; }
      r.th = V2.add(r.th, p);
    }
  },
  steepest: {
    label: 'Steepest descent + line search', color: '--s7', key: 's7',
    step(r, F, P) {
      const d = V2.neg(F.g(...r.th));
      const t = lineSearch(F, r.th, d, P.ls);
      r.th = V2.add(r.th, V2.scale(d, t));
    }
  },
  cg: {
    label: 'Conjugate gradients', color: '--s6', key: 's6',
    step(r, F, P) {
      const g = F.g(...r.th);
      let d;
      const reset = !r.d || (P.restart && r.k % 2 === 0);
      if (reset) { d = V2.neg(g); r.beta = 0; }
      else {
        const gp = r.gPrev, den = V2.dot(gp, gp);
        let beta = P.cgVariant === 'FR' ? V2.dot(g, g) / den : Math.max(0, V2.dot(V2.sub(g, gp), g) / den);
        d = V2.add(V2.neg(g), V2.scale(r.d, beta));
        if (V2.dot(d, g) >= 0) { d = V2.neg(g); beta = 0; }
        r.beta = beta;
      }
      const t = lineSearch(F, r.th, d, P.ls);
      r.th = V2.add(r.th, V2.scale(d, t));
      r.d = d; r.gPrev = g;
    }
  },
  bfgs: {
    label: 'BFGS', color: '--s8', key: 's8',
    init(r) { r.M = [[1, 0], [0, 1]]; },
    step(r, F, P) {
      const g = F.g(...r.th);
      let d = V2.neg(V2.mv(r.M, g));
      if (V2.dot(d, g) >= 0) { r.M = [[1, 0], [0, 1]]; d = V2.neg(g); }
      const t = lineSearch(F, r.th, d, P.ls);
      const s = V2.scale(d, t);
      const thN = V2.add(r.th, s);
      const y = V2.sub(F.g(...thN), g);
      const sy = V2.dot(s, y);
      if (sy > 1e-12) {
        const q = 1 / sy;
        const A = [[1 - q * s[0] * y[0], -q * s[0] * y[1]], [-q * s[1] * y[0], 1 - q * s[1] * y[1]]];
        const M = V2.mm(V2.mm(A, r.M), V2.T(A));
        r.M = [[M[0][0] + q * s[0] * s[0], M[0][1] + q * s[0] * s[1]], [M[1][0] + q * s[1] * s[0], M[1][1] + q * s[1] * s[1]]];
      }
      r.th = thN;
    }
  },
  lbfgs: {
    label: 'L-BFGS', color: '--s5', key: 's5',
    init(r) { r.mem = []; },
    step(r, F, P) {
      const g = F.g(...r.th);
      let q = g.slice();
      const al = [];
      for (let i = r.mem.length - 1; i >= 0; i--) {
        const { s, y, rho } = r.mem[i];
        al[i] = rho * V2.dot(s, q);
        q = V2.sub(q, V2.scale(y, al[i]));
      }
      let z = q; // start from M = identity, as in the lecture
      for (let i = 0; i < r.mem.length; i++) {
        const { s, y, rho } = r.mem[i];
        const b = rho * V2.dot(y, z);
        z = V2.add(z, V2.scale(s, al[i] - b));
      }
      let d = V2.neg(z);
      if (V2.dot(d, g) >= 0) d = V2.neg(g);
      const t = lineSearch(F, r.th, d, P.ls);
      const s = V2.scale(d, t);
      const thN = V2.add(r.th, s);
      const y = V2.sub(F.g(...thN), g);
      const sy = V2.dot(s, y);
      if (sy > 1e-12) { r.mem.push({ s, y, rho: 1 / sy }); while (r.mem.length > P.m) r.mem.shift(); }
      r.th = thN;
    }
  }
};

const PRESETS = {
  newton: { funcs: ['bowl', 'convex'], methods: ['gd', 'newton'], on: ['gd', 'newton'], kappa: 10 },
  saddle: { funcs: ['saddle'], methods: ['gd', 'newton', 'regnewton'], on: ['gd', 'newton', 'regnewton'], alpha: 1.5, showEigen: true },
  steepest: { funcs: ['bowl'], methods: ['gd', 'steepest'], on: ['steepest'], kappa: 12, rightAngles: true, start: [-2.23, -0.83] },
  cg: { funcs: ['bowl', 'convex'], methods: ['steepest', 'cg'], on: ['steepest', 'cg'], kappa: 12, rightAngles: true, start: [-2.23, -0.83] },
  ncg: { funcs: ['banana', 'convex', 'bowl'], methods: ['steepest', 'cg'], on: ['steepest', 'cg'], showRestart: true, showLS: true },
  bfgs: { funcs: ['banana', 'convex', 'bowl'], methods: ['cg', 'bfgs', 'newton'], on: ['cg', 'bfgs'], showM: true, showLS: true },
  lbfgs: { funcs: ['banana', 'convex', 'bowl'], methods: ['bfgs', 'lbfgs', 'cg'], on: ['bfgs', 'lbfgs'], m: 1, showLS: true, ls: 'rough' },
  all: { funcs: ['bowl', 'convex', 'saddle', 'banana'], methods: ['gd', 'newton', 'regnewton', 'steepest', 'cg', 'bfgs', 'lbfgs'], on: ['gd', 'newton', 'cg', 'bfgs'], showRestart: true, showLS: true }
};

Widgets.playground = (el, ds) => {
  const cfg = PRESETS[ds.preset || 'all'];
  const P = {
    func: cfg.funcs[0], kappa: cfg.kappa || 8, lr: FUNCS[cfg.funcs[0]].lr, alpha: cfg.alpha ?? 1.5,
    m: cfg.m || 2, cgVariant: 'PR', restart: false, ls: cfg.ls || 'exact',
    mom: cfg.mom ?? 0.9, lrA: cfg.lrA ?? 0.1
  };
  const FIRST = ['gd', 'momentum', 'nesterov'], ADAPT = ['adagrad', 'rmsprop', 'adam'];
  if (cfg.lr) P.lr = cfg.lr;
  const enabled = new Set(cfg.on);
  let start = (cfg.start || FUNCS[P.func].start).slice();
  let F = FUNCS[P.func].build(P);
  let runs = {};
  let timer = null;
  let bg = null, bgKey = '';

  UI.head(el, ds.title || 'Optimizer playground',
    (ds.sub || 'Press <b>Run</b> or <b>Step</b>. <b>Click the map</b> to choose a new starting point.') + ' Darker shading = lower loss.');

  // controls
  const top = h('div', { class: 'controls' });
  let funcSel = null;
  if (cfg.funcs.length > 1) funcSel = UI.select(top, 'Loss surface', cfg.funcs.map(k => [k, FUNCS[k].label]), P.func, v => {
    P.func = v; P.lr = (cfg.lrBy && cfg.lrBy[v]) || FUNCS[v].lr; lrS && lrS.set(Math.log10(P.lr)); start = FUNCS[v].start.slice(); rebuild();
  });
  el.append(top);
  const chips = h('div', { class: 'toggles' });
  cfg.methods.forEach(k => UI.chip(chips, METHODS[k].label, METHODS[k].color, enabled.has(k), on => { on ? enabled.add(k) : enabled.delete(k); syncControls(); reset(); }));
  el.append(chips);

  const params = h('div', { class: 'controls' });
  const kS = UI.slider(params, { label: 'Stretch of the bowl (κ)', min: 1, max: 20, step: 1, value: P.kappa, fmt: v => v + '×', onInput: v => { P.kappa = v; rebuild(); } });
  const lrS = cfg.methods.some(k => FIRST.includes(k)) ? UI.slider(params, { label: 'Learning rate ε (GD / momentum)', min: -3, max: 0, step: 0.01, value: Math.log10(P.lr), fmt: v => (10 ** v).toPrecision(2), onInput: v => { P.lr = 10 ** v; reset(); } }) : null;
  const momS = cfg.methods.some(k => k === 'momentum' || k === 'nesterov') ? UI.slider(params, { label: 'Momentum α', min: 0, max: 0.99, step: 0.01, value: P.mom, fmt: v => v.toFixed(2), onInput: v => { P.mom = v; reset(); } }) : null;
  const lrAS = cfg.methods.some(k => ADAPT.includes(k)) ? UI.slider(params, { label: 'Step size ε (AdaGrad / RMSProp / Adam)', min: -3, max: 0, step: 0.01, value: Math.log10(P.lrA), fmt: v => (10 ** v).toPrecision(2), onInput: v => { P.lrA = 10 ** v; reset(); } }) : null;
  const aS = cfg.methods.includes('regnewton') ? UI.slider(params, { label: 'Regularization α', min: 0, max: 4, step: 0.05, value: P.alpha, fmt: v => v.toFixed(2), onInput: v => { P.alpha = v; reset(); } }) : null;
  const mS = cfg.methods.includes('lbfgs') ? UI.slider(params, { label: 'L-BFGS memory m (pairs kept)', min: 1, max: 4, step: 1, value: P.m, fmt: v => String(v), onInput: v => { P.m = v; reset(); } }) : null;
  const lsSel = cfg.showLS ? UI.select(params, 'Line search', [['exact', 'Exact (precise, costly)'], ['rough', 'Rough (backtracking, cheap)']], P.ls, v => { P.ls = v; reset(); }) : null;
  const cgSel = cfg.methods.includes('cg') ? UI.select(params, 'CG formula for β', [['PR', 'Polak–Ribière'], ['FR', 'Fletcher–Reeves']], P.cgVariant, v => { P.cgVariant = v; reset(); }) : null;
  let rsChip = null;
  if (cfg.methods.includes('cg') && cfg.showRestart) {
    const box = h('div', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', {}, 'CG resets')));
    rsChip = UI.chip(box, 'Reset to −gradient every 2 steps', '--s6', P.restart, v => { P.restart = v; reset(); });
    params.append(box);
  }
  el.append(params);

  const btns = h('div', { class: 'btns' });
  const runBtn = UI.button(btns, 'Run ▶', () => (timer ? pause() : play()), 'btn primary');
  UI.button(btns, 'Step', () => { pause(); stepAll(); draw(); });
  UI.button(btns, 'Reset', () => { pause(); reset(); });
  el.append(btns);

  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart clickable', role: 'img', 'aria-label': 'Contour map of the loss with optimizer paths' });
  wrap.append(cv);
  el.append(wrap);
  const lg = UI.legend(el, []);
  const tableBox = h('div', { class: 'scroll-x' });
  el.append(tableBox);
  const extra = h('div', { class: 'stack', style: 'gap:8px' });
  el.append(extra);

  let p = null;
  cv.addEventListener('click', e => {
    if (!p) return;
    const r = cv.getBoundingClientRect();
    const x = p.invX(e.clientX - r.left), y = p.invY(e.clientY - r.top);
    const d = FUNCS[P.func].dom;
    if (x < d[0] || x > d[1] || y < d[2] || y > d[3]) return;
    pause(); start = [x, y]; reset();
  });

  function syncControls() {
    kS.show(!!FUNCS[P.func].usesKappa);
    lrS && lrS.show(FIRST.some(k => enabled.has(k)));
    momS && momS.show(enabled.has('momentum') || enabled.has('nesterov'));
    lrAS && lrAS.show(ADAPT.some(k => enabled.has(k)));
    aS && aS.show(enabled.has('regnewton'));
    mS && mS.show(enabled.has('lbfgs'));
    cgSel && cgSel.show(enabled.has('cg'));
    lsSel && lsSel.show(['steepest', 'cg', 'bfgs', 'lbfgs'].some(k => enabled.has(k)));
    rsChip && (rsChip.el.parentElement.hidden = !enabled.has('cg'));
  }
  function rebuild() { F = FUNCS[P.func].build(P); bgKey = ''; pause(); syncControls(); reset(); }
  function reset() {
    runs = {};
    for (const k of cfg.methods) {
      if (!enabled.has(k)) continue;
      const r = { th: start.slice(), path: [start.slice()], done: false, msg: '', k: 0 };
      METHODS[k].init && METHODS[k].init(r);
      runs[k] = r;
    }
    checkDone();
    draw();
  }
  function classify(th) {
    const ev = V2.eig(F.H(...th));
    if (ev[0] > 1e-9) return ['good', 'Reached a minimum'];
    if (ev[1] < -1e-9) return ['bad', 'Reached a MAXIMUM'];
    return ['bad', 'Stuck at a saddle point'];
  }
  function checkDone() {
    const d = FUNCS[P.func].dom, span = Math.max(d[1] - d[0], d[3] - d[2]);
    for (const k in runs) {
      const r = runs[k];
      if (r.done) continue;
      const g = F.g(...r.th);
      if (!isFinite(r.th[0]) || !isFinite(r.th[1]) || Math.abs(r.th[0]) > 4 * span || Math.abs(r.th[1]) > 4 * span) {
        r.done = true; r.msg = 'Diverged (flew off the map)'; r.status = 'bad';
      } else if (V2.norm(g) < 1e-6) {
        r.done = true; [r.status, r.msg] = classify(r.th);
      } else if (r.k >= 300) { r.done = true; r.msg = 'Stopped after 300 steps'; r.status = 'warn'; }
    }
  }
  function stepAll() {
    for (const k in runs) {
      const r = runs[k];
      if (r.done) continue;
      METHODS[k].step(r, F, P);
      r.k++;
      r.path.push(r.th.slice());
    }
    checkDone();
  }
  function play() {
    if (Object.values(runs).every(r => r.done)) reset();
    runBtn.textContent = 'Pause ❚❚';
    timer = setInterval(() => {
      stepAll(); draw();
      if (Object.values(runs).every(r => r.done)) pause();
    }, 170);
  }
  function pause() { if (timer) clearInterval(timer); timer = null; runBtn.textContent = 'Run ▶'; }

  function draw() {
    const c = Colors.get();
    const d = FUNCS[P.func].dom;
    const W = Math.min(boxWidth(wrap), 620);
    const H = Math.round(W * (d[3] - d[2]) / (d[1] - d[0]));
    cv.style.maxWidth = W + 'px';
    cv.style.margin = '0 auto';
    const ctx = fitCanvas(cv, W, H);
    p = makePlot(W, H, [d[0], d[1]], [d[2], d[3]], { l: 0, r: 0, t: 0, b: 0 });
    const key = [P.func, P.kappa, W, H, Theme.current()].join('|');
    if (key !== bgKey) { bg = contourImage(F.f, d, Math.round(W / 2), Math.round(H / 2)); bgKey = key; }
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(bg, 0, 0, W, H);
    // landmarks
    for (const m of F.minima) { drawStar(ctx, p.X(m[0]), p.Y(m[1]), c.text); }
    for (const s of F.saddles) { label(ctx, '✕ saddle', p.X(s[0]) + 6, p.Y(s[1]) - 10, { color: c.text, bold: true, size: 11 }); }
    for (const s of F.saddles) { ctx.strokeStyle = c.text; ctx.lineWidth = 2; const X = p.X(s[0]), Y = p.Y(s[1]); ctx.beginPath(); ctx.moveTo(X - 5, Y - 5); ctx.lineTo(X + 5, Y + 5); ctx.moveTo(X + 5, Y - 5); ctx.lineTo(X - 5, Y + 5); ctx.stroke(); }
    // paths
    const legendItems = [];
    for (const k of cfg.methods) {
      const r = runs[k];
      if (!r) continue;
      const col = c[METHODS[k].key];
      legendItems.push({ label: METHODS[k].label, color: col });
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, W, H); ctx.clip();
      ctx.strokeStyle = c.surface; ctx.lineWidth = 4.5; ctx.lineJoin = 'round';
      ctx.beginPath(); r.path.forEach((q, i) => i ? ctx.lineTo(p.X(q[0]), p.Y(q[1])) : ctx.moveTo(p.X(q[0]), p.Y(q[1]))); ctx.stroke();
      ctx.strokeStyle = col; ctx.lineWidth = 2.2;
      ctx.beginPath(); r.path.forEach((q, i) => i ? ctx.lineTo(p.X(q[0]), p.Y(q[1])) : ctx.moveTo(p.X(q[0]), p.Y(q[1]))); ctx.stroke();
      if (cfg.rightAngles && k === 'steepest') drawRightAngles(ctx, r.path, col);
      const n = r.path.length;
      const every = n > 60 ? Math.ceil(n / 60) : 1;
      r.path.forEach((q, i) => { if (i % every === 0 && i > 0 && i < n - 1) dot(ctx, p.X(q[0]), p.Y(q[1]), 2.6, col); });
      const last = r.path[n - 1];
      dot(ctx, p.X(last[0]), p.Y(last[1]), 6, col, c.surface);
      ctx.restore();
    }
    // start
    ctx.beginPath(); ctx.arc(p.X(start[0]), p.Y(start[1]), 7, 0, Math.PI * 2);
    ctx.lineWidth = 2.5; ctx.strokeStyle = c.text; ctx.stroke();
    label(ctx, 'start', p.X(start[0]) + 10, p.Y(start[1]) - 10, { color: c.text, bold: true });
    lg.set([...legendItems, { label: 'minimum ★', color: c.text, kind: 'dot' }]);
    renderTable(c);
    renderExtra(c);
  }
  function drawStar(ctx, x, y, col) {
    ctx.save(); ctx.fillStyle = col; ctx.strokeStyle = Colors.get().surface; ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < 10; i++) { const r = i % 2 ? 3.4 : 8; const a = -Math.PI / 2 + i * Math.PI / 5; ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a)); }
    ctx.closePath(); ctx.stroke(); ctx.fill(); ctx.restore();
  }
  function drawRightAngles(ctx, path, col) {
    ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.3;
    for (let i = 1; i < path.length - 1 && i < 12; i++) {
      const P0 = [p.X(path[i][0]), p.Y(path[i][1])];
      const a = [p.X(path[i - 1][0]) - P0[0], p.Y(path[i - 1][1]) - P0[1]];
      const b = [p.X(path[i + 1][0]) - P0[0], p.Y(path[i + 1][1]) - P0[1]];
      const la = Math.hypot(...a), lb = Math.hypot(...b);
      if (la < 14 || lb < 14) continue;
      const s = 7, u = [a[0] / la * s, a[1] / la * s], v = [b[0] / lb * s, b[1] / lb * s];
      ctx.beginPath();
      ctx.moveTo(P0[0] + u[0], P0[1] + u[1]); ctx.lineTo(P0[0] + u[0] + v[0], P0[1] + u[1] + v[1]); ctx.lineTo(P0[0] + v[0], P0[1] + v[1]);
      ctx.stroke();
    }
    ctx.restore();
  }
  function renderTable(c) {
    const rows = cfg.methods.filter(k => runs[k]).map(k => {
      const r = runs[k];
      const st = r.done ? (r.status === 'good' ? '✓ ' : r.status === 'warn' ? '… ' : '⚠ ') + r.msg : 'running…';
      return h('tr', {},
        h('td', {}, h('span', { class: 'sw', style: `--c:${c[METHODS[k].key]}` }), METHODS[k].label),
        h('td', { class: 'num' }, String(r.k)),
        h('td', { class: 'num' }, fmt(F.f(...r.th), 4)),
        h('td', { class: 'num' }, fmt(V2.norm(F.g(...r.th)), 4)),
        h('td', {}, st));
    });
    tableBox.innerHTML = '';
    if (!rows.length) { tableBox.append(h('p', { class: 'note' }, 'Turn on at least one method above.')); return; }
    tableBox.append(h('table', { class: 'readout' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Method'), h('th', {}, 'Steps'), h('th', {}, 'J(θ)'), h('th', {}, '‖∇J‖'), h('th', {}, 'Status'))),
      h('tbody', {}, rows)));
  }
  function matHTML(M) { return h('span', { class: 'mat' }, ...M.flat().map(v => h('span', {}, fmt(v, 2)))); }
  function renderExtra(c) {
    extra.innerHTML = '';
    if (cfg.showEigen) {
      const H0 = F.H(...start);
      const e0 = V2.eig(H0), e1 = V2.eig(V2.addI(H0, P.alpha));
      const neg = e0[0] < 0;
      extra.append(h('div', { class: 'stat-row' },
        h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Eigenvalues of H at start'), h('div', { class: 'v' }, `${fmt(e0[0])}, ${fmt(e0[1])}`)),
        h('div', { class: 'stat' }, h('div', { class: 'k' }, `Eigenvalues of H + αI (α=${P.alpha.toFixed(2)})`), h('div', { class: 'v' }, `${fmt(e1[0])}, ${fmt(e1[1])}`))));
      extra.append(neg
        ? h('div', { class: e1[0] > 0 ? 'callout good' : 'callout warn' }, h('span', { class: 'ico' }, e1[0] > 0 ? '✓' : '!'),
          h('div', {}, 'H has a ', h('b', {}, 'negative eigenvalue'), ' here, so plain Newton treats the downhill direction as uphill and is pulled toward the saddle. ',
            e1[0] > 0 ? 'With this α every eigenvalue of H + αI is positive, so the regularized step points downhill.' : 'α is still too small: H + αI keeps a negative eigenvalue. Increase α past ' + fmt(-e0[0]) + '.'))
        : h('div', { class: 'callout info' }, h('span', { class: 'ico' }, 'i'), h('div', {}, 'At this start point H is positive definite, so Newton behaves. Click near the middle row (y ≈ 0) to start in the negative-curvature zone.')));
    }
    if (cfg.showM && runs.bfgs) {
      const r = runs.bfgs;
      const Hi = V2.inv(F.H(...r.th));
      extra.append(h('div', { class: 'note' }, 'BFGS never computes H⁻¹. It builds its own estimate M from how the gradient changes. Compare it with the true H⁻¹ at BFGS\'s current point:'));
      extra.append(h('div', { class: 'mat-row' },
        h('span', {}, 'BFGS estimate M = '), matHTML(r.M),
        h('span', {}, 'true H⁻¹ = '), Hi ? matHTML(Hi) : h('span', {}, 'not invertible here')));
    }
    if (enabled.has('cg') && runs.cg && runs.cg.k > 0 && !cfg.showM) {
      extra.append(h('div', { class: 'note' }, `Last β used by CG: ${fmt(runs.cg.beta || 0, 3)} — the share of the previous direction mixed into the new one.`));
    }
  }
  syncControls();
  reset();
  return { redraw: () => { bgKey = ''; draw(); } };
};

/* =========================================================
   4) Why exact Newton does not scale — memory & time calculator
   ========================================================= */
Widgets.newtonCost = (el) => {
  UI.head(el, 'How big is the Hessian for a real network?', 'Pick a model size. Bars use a log scale (each tick = 1000× more). Assumes 4 bytes per number (float32).');
  const presets = [['Tiny MLP', 1e4], ['LeNet-5', 6e4], ['ResNet-50', 2.56e7], ['BERT-base', 1.1e8], ['GPT-3', 1.75e11]];
  let lk = Math.log10(2.56e7);
  const ctr = h('div', { class: 'controls' });
  const sl = UI.slider(ctr, { label: 'Number of parameters k', min: 1, max: 12, step: 0.05, value: lk, fmt: v => bigNum(10 ** v), onInput: v => { lk = v; draw(); } });
  const pb = h('div', { class: 'btns' });
  presets.forEach(([n, k]) => UI.button(pb, n, () => { lk = Math.log10(k); sl.set(lk); draw(); }));
  ctr.append(pb);
  const bars = h('div', { class: 'hbars' });
  const facts = h('div', { class: 'stack', style: 'gap:8px' });
  el.append(ctr, bars, facts);
  function bigNum(n) {
    const u = [[1e12, ' trillion'], [1e9, ' billion'], [1e6, ' million'], [1e3, ' thousand']];
    for (const [v, s] of u) if (n >= v) return (n / v).toPrecision(3).replace(/\.?0+$/, '') + s;
    return String(Math.round(n));
  }
  function bytes(b) {
    const u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
    const i = clamp(Math.floor(Math.log10(Math.max(b, 1)) / 3), 0, u.length - 1);
    const v = b / 10 ** (3 * i);
    return (v >= 1000 ? fmt(v, 0) : v.toPrecision(3)) + ' ' + u[i];
  }
  function time(s) {
    if (s < 1e-3) return 'under a millisecond';
    if (s < 1) return (s * 1e3).toPrecision(2) + ' milliseconds';
    if (s < 60) return s.toPrecision(2) + ' seconds';
    if (s < 3600) return (s / 60).toPrecision(2) + ' minutes';
    if (s < 86400) return (s / 3600).toPrecision(2) + ' hours';
    const y = s / 3.156e7;
    if (y < 1) return (s / 86400).toPrecision(2) + ' days';
    if (y < 1e4) return y.toPrecision(2) + ' years';
    if (y < 1.38e10) return fmt(y, 0) + ' years';
    return fmt(y / 1.38e10, 1) + '× the age of the universe';
  }
  function draw() {
    const k = 10 ** lk;
    const c = Colors.get();
    const rows = [
      ['Weights θ itself (k numbers)', 4 * k, c.s1],
      ['Gradient ∇J (k numbers)', 4 * k, c.s1],
      ['L-BFGS history, m = 10 (2mk numbers)', 80 * k, c.s5],
      ['Hessian H or BFGS matrix M (k² numbers)', 4 * k * k, c.s2]
    ];
    const maxL = 26, gpu = 8e10;
    bars.innerHTML = '';
    rows.forEach(([lab, b, col]) => {
      const w = clamp(Math.log10(Math.max(b, 1)) / maxL * 100, 1, 100);
      bars.append(h('div', { class: 'hbar' }, h('div', { class: 'lab' }, lab),
        h('div', { class: 'track' }, h('div', { class: 'fill', style: `width:${w}%;--c:${col}` }), h('div', { class: 'ref', style: `left:${Math.log10(gpu) / maxL * 100}%`, title: '80 GB: memory of a large GPU' }), h('span', { class: 'val' }, bytes(b)))));
    });
    bars.append(h('div', { class: 'legend' }, h('span', {}, h('i', { class: 'dash', style: `--c:${c.critical}` }), '80 GB — memory of one large data-centre GPU')));
    const flops = k ** 3;
    const hBytes = 4 * k * k;
    facts.innerHTML = '';
    facts.append(h('div', { class: 'stat-row' },
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Hessian entries'), h('div', { class: 'v' }, fmt(k * k, 0))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Ops to invert (≈k³)'), h('div', { class: 'v' }, fmt(flops, 0))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Time per Newton step*'), h('div', { class: 'v', style: 'font-size:.95rem' }, time(flops / 1e14)))));
    const fits = hBytes <= gpu;
    facts.append(h('div', { class: fits ? 'callout good' : 'callout bad' }, h('span', { class: 'ico' }, fits ? '✓' : '✗'),
      h('div', {}, fits ? 'At this size the Hessian still fits in GPU memory — exact Newton is possible (but only for small models like this).'
        : `The Hessian alone needs ${bytes(hBytes)} — about ${fmt(hBytes / gpu, 0)} large GPUs just to store it, and it must be rebuilt and inverted again at every single step because θ keeps changing.`)));
    facts.append(h('p', { class: 'note' }, '*Assuming a perfect 100 trillion operations per second. Real inversions are slower.'));
  }
  draw();
  return { redraw: draw };
};
