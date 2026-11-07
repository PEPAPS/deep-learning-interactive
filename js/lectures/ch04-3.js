/* =========================================================
   Lecture 3 · Ch 4.3–4.5 & 5.9 · Numerical optimization & SGD
   (the 2-D optimizer playground comes from ch08-6a.js)
   ========================================================= */
'use strict';

/* ---------- 1) Gradient descent in 1-D ---------- */
Widgets.gd1d = (el) => {
  UI.head(el, 'Gradient descent on a 1-D function', 'f(x) = 0.05x⁴ − 0.5x² + 0.2x + 3. Each step moves against the slope: x ← x − ε·f′(x). Try different starts and learning rates.');
  const f = x => 0.05 * x ** 4 - 0.5 * x * x + 0.2 * x + 3;
  const d1 = x => 0.2 * x ** 3 - x + 0.2;
  const d2 = x => 0.6 * x * x - 1;
  let x0 = 3.2, eps = 0.3, path = [x0], timer = null;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Start x₀', min: -3.4, max: 3.4, step: 0.05, value: x0, fmt: v => v.toFixed(2), onInput: v => { x0 = v; reset(); } });
  UI.slider(ctr, { label: 'Learning rate ε', min: 0.02, max: 2, step: 0.01, value: eps, fmt: v => v.toFixed(2), onInput: v => { eps = v; reset(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Step', () => { step(); draw(); }, 'btn primary');
  UI.button(btns, 'Run 30 steps', () => { if (timer) return; let n = 0; timer = setInterval(() => { step(); draw(); if (++n >= 30) { clearInterval(timer); timer = null; } }, 120); });
  UI.button(btns, 'Reset', () => reset());
  ctr.append(btns);
  el.append(ctr);
  const chart = new LineChart(el, { xr: [-3.5, 3.5], yr: [0.5, 5.5], xlabel: 'x', ylabel: 'f(x)', xname: 'x', aspect: .5, maxH: 300 });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  const verdict = h('div');
  el.append(calc, verdict);
  // critical points: roots of f' found by bisection
  const crit = [];
  for (let x = -3.5; x < 3.5; x += 0.01) if (Math.sign(d1(x)) !== Math.sign(d1(x + 0.01))) { let a = x, b = x + 0.01; for (let k = 0; k < 50; k++) { const m = (a + b) / 2; Math.sign(d1(m)) === Math.sign(d1(a)) ? a = m : b = m; } crit.push((a + b) / 2); }
  function step() { const x = path[path.length - 1]; const nx = x - eps * d1(x); if (Math.abs(nx) < 50) path.push(nx); }
  function reset() { path = [x0]; draw(); }
  function draw() {
    const c = Colors.get();
    const xs = Array.from({ length: 141 }, (_, i) => -3.5 + i * 0.05);
    const gmin = crit.reduce((a, x) => (d2(x) > 0 && (a == null || f(x) < f(a)) ? x : a), null);
    chart.set([
      { name: 'f(x)', color: 'text', data: xs.map(x => [x, f(x)]) },
      { name: 'path', color: 's2', data: path.filter(x => Math.abs(x) <= 3.6).map(x => [x, f(x)]), width: 1.5 },
      { name: 'iterates', color: 's2', style: 'dots', r: 4, noHover: true, data: path.filter(x => Math.abs(x) <= 3.6).map(x => [x, f(x)]) },
      { name: 'critical points', color: 's1', style: 'dots', r: 6, noHover: true, data: crit.map(x => [x, f(x)]) }
    ]);
    lg.set([{ label: 'f(x)', color: c.text }, { label: 'gradient-descent iterates', color: c.s2, kind: 'dot' }, { label: `critical points: global min x≈${gmin.toFixed(2)}, local min x≈${crit.find(x => d2(x) > 0 && x !== gmin).toFixed(2)}, local max x≈${crit.find(x => d2(x) < 0).toFixed(2)}`, color: c.s1, kind: 'dot' }]);
    const x = path[path.length - 1], g = d1(x);
    calc.innerHTML = `<span class="c">step ${path.length - 1}:  f′(x) = 0.2x³ − x + 0.2</span>
x = ${x.toFixed(4)}   f(x) = ${f(x).toFixed(4)}   f′(x) = ${g.toFixed(4)}
next x = x − ε·f′(x) = ${x.toFixed(4)} − ${eps.toFixed(2)} × ${g.toFixed(4)} = <b>${(x - eps * g).toFixed(4)}</b>
<span class="c">first-order prediction: f(x − εf′) ≈ f(x) − ε·f′(x)² = ${(f(x) - eps * g * g).toFixed(4)}   actual: ${f(x - eps * g).toFixed(4)}</span>`;
    verdict.innerHTML = '';
    const near = crit.find(cp => Math.abs(cp - x) < 1e-3);
    if (Math.abs(x) > 3.5) verdict.append(h('div', { class: 'callout bad' }, h('span', { class: 'ico' }, '✗'), h('div', {}, 'Diverged: the learning rate is so large that each step overshoots further.')));
    else if (near != null && Math.abs(g) < 1e-3) verdict.append(h('div', { class: 'callout ' + (Math.abs(near - gmin) < 1e-3 ? 'good' : 'warn') }, h('span', { class: 'ico' }, Math.abs(near - gmin) < 1e-3 ? '✓' : '!'),
      h('div', {}, Math.abs(near - gmin) < 1e-3 ? 'Converged to the global minimum.' : d2(near) > 0 ? 'Converged to a LOCAL minimum: f′ = 0 and f″ > 0, but the other valley is lower. Gradient descent only sees the local slope.' : 'Stopped at a maximum (f′ = 0 exactly) — a critical point that is not a minimum.')));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) Second-derivative test in 2-D ---------- */
Widgets.criticalTest = (el) => {
  UI.head(el, 'Classify a critical point with the Hessian', 'f(x, y) = a·x² + c·xy + b·y² has a critical point at the origin (∇f = 0 there). Its Hessian decides what kind of point it is.');
  let a = 1, b = -0.6, c0 = 0;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'a', min: -2, max: 2, step: 0.05, value: a, fmt: v => v.toFixed(2), onInput: v => { a = v; draw(); } });
  UI.slider(ctr, { label: 'b', min: -2, max: 2, step: 0.05, value: b, fmt: v => v.toFixed(2), onInput: v => { b = v; draw(); } });
  UI.slider(ctr, { label: 'c (mixing term)', min: -2, max: 2, step: 0.05, value: c0, fmt: v => v.toFixed(2), onInput: v => { c0 = v; draw(); } });
  const wrap = h('div', { class: 'canvas-wrap', style: 'max-width:380px;margin:0 auto' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Contour plot around the critical point' });
  wrap.append(cv);
  const calc = h('div', { class: 'calcbox' });
  const verdict = h('div');
  el.append(ctr, wrap, calc, verdict);
  function draw() {
    const c = Colors.get();
    const W = Math.min(380, boxWidth(wrap));
    const ctx = fitCanvas(cv, W, W);
    const f = (x, y) => a * x * x + c0 * x * y + b * y * y;
    ctx.drawImage(contourImage(f, [-2, 2, -2, 2], Math.round(W / 2), Math.round(W / 2), 14), 0, 0, W, W);
    const H = [[2 * a, c0], [c0, 2 * b]];
    const tr = (H[0][0] + H[1][1]) / 2, r = Math.sqrt(((H[0][0] - H[1][1]) / 2) ** 2 + c0 * c0);
    const l1 = tr - r, l2 = tr + r;
    const vec = l => { const v = Math.abs(c0) > 1e-9 ? [l - H[1][1], c0] : (Math.abs(l - H[0][0]) < 1e-9 ? [1, 0] : [0, 1]); const n = Math.hypot(...v); return [v[0] / n, v[1] / n]; };
    [[l1, vec(l1), c.s1], [l2, vec(l2), c.s2]].forEach(([l, v, col]) => {
      const X = t => W / 2 + t * v[0] * W / 4, Y = t => W / 2 - t * v[1] * W / 4;
      ctx.strokeStyle = col; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(-1.6), Y(-1.6)); ctx.lineTo(X(1.6), Y(1.6)); ctx.stroke();
      label(ctx, `λ = ${l.toFixed(2)}`, X(1.65), Y(1.65), { color: c.text, bold: true });
    });
    dot(ctx, W / 2, W / 2, 6, c.text, c.surface);
    const kind = (l1 > 1e-9 && l2 > 1e-9) ? ['good', '✓', 'Positive definite (all eigenvalues > 0) → LOCAL MINIMUM: it curves up in every direction.']
      : (l1 < -1e-9 && l2 < -1e-9) ? ['bad', '✗', 'Negative definite (all eigenvalues < 0) → LOCAL MAXIMUM.']
        : (l1 < -1e-9 && l2 > 1e-9) ? ['warn', '!', 'Eigenvalues of both signs → SADDLE POINT: a minimum along one eigenvector (orange-ish line), a maximum along the other.']
          : ['info', 'i', 'An eigenvalue is zero → the test is INCONCLUSIVE (could be a flat valley or a saddle).'];
    calc.innerHTML = `<span class="c">Hessian H = [[∂²f/∂x², ∂²f/∂x∂y], [∂²f/∂y∂x, ∂²f/∂y²]] = [[2a, c], [c, 2b]]</span>
H = [[${(2 * a).toFixed(2)}, ${c0.toFixed(2)}], [${c0.toFixed(2)}, ${(2 * b).toFixed(2)}]]   (symmetric)
eigenvalues  λ = (tr H)/2 ± √(((H₁₁ − H₂₂)/2)² + H₁₂²) = ${tr.toFixed(2)} ± ${r.toFixed(2)}  →  <b>${l1.toFixed(2)}, ${l2.toFixed(2)}</b>
<span class="c">directional 2nd derivative along unit d: dᵀHd — equals λ along each eigenvector (coloured lines)</span>`;
    verdict.innerHTML = '';
    verdict.append(h('div', { class: 'callout ' + kind[0] }, h('span', { class: 'ico' }, kind[1]), h('div', {}, kind[2])));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) Directional derivative ---------- */
Widgets.dirDeriv = (el) => {
  UI.head(el, 'Which direction goes downhill fastest?', 'f(x, y) = ½x² + 1.5y². Click to move the point; turn the direction u. The slope along u is uᵀ∇f = ‖∇f‖·cos α.');
  let p = [1.6, 1.0], ang = 120;
  const ctr = h('div', { class: 'controls' });
  const aS = UI.slider(ctr, { label: 'Direction u (degrees)', min: 0, max: 359, step: 1, value: ang, fmt: v => v + '°', onInput: v => { ang = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Point u along −∇f', () => { const g = [p[0], 3 * p[1]]; ang = Math.round(((Math.atan2(-g[1], -g[0]) * 180 / Math.PI) + 360) % 360); aS.set(ang); draw(); }, 'btn primary');
  ctr.append(btns);
  const wrap = h('div', { class: 'canvas-wrap', style: 'max-width:420px;margin:0 auto' });
  const cv = h('canvas', { class: 'chart clickable', role: 'img', 'aria-label': 'Contours with gradient and direction arrows' });
  wrap.append(cv);
  el.append(ctr, wrap);
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const f = (x, y) => 0.5 * x * x + 1.5 * y * y;
  let P = null, bg = null, bgKey = '';
  cv.addEventListener('click', e => { if (!P) return; const r = cv.getBoundingClientRect(); p = [P.invX(e.clientX - r.left), P.invY(e.clientY - r.top)]; draw(); });
  function draw() {
    const c = Colors.get();
    const W = Math.min(420, boxWidth(wrap));
    const ctx = fitCanvas(cv, W, W);
    P = makePlot(W, W, [-2.5, 2.5], [-2.5, 2.5], { l: 0, r: 0, t: 0, b: 0 });
    const key = W + Theme.current(); if (key !== bgKey) { bg = contourImage(f, [-2.5, 2.5, -2.5, 2.5], Math.round(W / 2), Math.round(W / 2), 14); bgKey = key; }
    ctx.drawImage(bg, 0, 0, W, W);
    const g = [p[0], 3 * p[1]], gn = Math.hypot(...g);
    const u = [Math.cos(ang * Math.PI / 180), Math.sin(ang * Math.PI / 180)];
    const s = 0.6 / Math.max(gn, 1e-9);
    arrow(ctx, P.X(p[0]), P.Y(p[1]), P.X(p[0] + g[0] * s), P.Y(p[1] + g[1] * s), c.s2, 3);
    arrow(ctx, P.X(p[0]), P.Y(p[1]), P.X(p[0] - g[0] * s), P.Y(p[1] - g[1] * s), c.s3, 3);
    arrow(ctx, P.X(p[0]), P.Y(p[1]), P.X(p[0] + u[0] * 0.8), P.Y(p[1] + u[1] * 0.8), c.s1, 3);
    dot(ctx, P.X(p[0]), P.Y(p[1]), 6, c.text, c.surface);
    lg.set([{ label: '∇f — points straight uphill', color: c.s2 }, { label: '−∇f — steepest descent', color: c.s3 }, { label: 'your direction u', color: c.s1 }]);
    const dd = u[0] * g[0] + u[1] * g[1], cosA = dd / Math.max(gn, 1e-12);
    calc.innerHTML = `<span class="c">at x = (${p[0].toFixed(2)}, ${p[1].toFixed(2)}):  ∇f = (∂f/∂x, ∂f/∂y) = (x, 3y) = (${g[0].toFixed(2)}, ${g[1].toFixed(2)}),  ‖∇f‖ = ${gn.toFixed(3)}</span>
u = (cos ${ang}°, sin ${ang}°) = (${u[0].toFixed(3)}, ${u[1].toFixed(3)})
uᵀ∇f = ${u[0].toFixed(3)}×${g[0].toFixed(2)} + ${u[1].toFixed(3)}×${g[1].toFixed(2)} = <b>${dd.toFixed(3)}</b>  = ‖∇f‖·cos α = ${gn.toFixed(3)} × ${cosA.toFixed(3)}
<span class="c">${dd < -0.999 * gn ? '→ the most negative possible value: this is the steepest-descent direction (α = 180°)' : dd < 0 ? '→ negative: moving along u goes downhill' : '→ positive: moving along u goes uphill'}</span>`;
  }
  draw();
  return { redraw: () => { bgKey = ''; draw(); } };
};

/* ---------- 4) How curvature changes what a gradient step does ---------- */
Widgets.stepCurvature = (el) => {
  UI.head(el, 'The second-order view of one gradient step', 'f(x) = ½xᵀHx with Hessian eigenvalues λ₁, λ₂, starting at x⁽⁰⁾ = (1.5, 1). Predicted change after a step of size ε: −ε gᵀg + ½ε² gᵀHg.');
  let l1 = 1, l2 = 6, eps = 0.15;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Eigenvalue λ₁ (curvature along x)', min: -2, max: 10, step: 0.1, value: l1, fmt: v => v.toFixed(1), onInput: v => { l1 = v; draw(); } });
  UI.slider(ctr, { label: 'Eigenvalue λ₂ (curvature along y)', min: -2, max: 10, step: 0.1, value: l2, fmt: v => v.toFixed(1), onInput: v => { l2 = v; draw(); } });
  UI.slider(ctr, { label: 'Step size ε', min: 0.005, max: 1, step: 0.005, value: eps, fmt: v => v.toFixed(3), onInput: v => { eps = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 1], yr: [-4, 2], xlabel: 'step size ε', ylabel: 'change in f', xname: 'ε', aspect: .5, maxH: 280, refY: 0, vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const x0 = [1.5, 1], g = [l1 * x0[0], l2 * x0[1]];
    const gg = g[0] ** 2 + g[1] ** 2, gHg = l1 * g[0] ** 2 + l2 * g[1] ** 2;
    const es = Array.from({ length: 201 }, (_, i) => i / 200);
    const change = e => -e * gg + 0.5 * e * e * gHg;
    const opt = gHg > 0 ? gg / gHg : null;
    const lmax = Math.max(l1, l2);
    chart.set([
      { name: 'first-order prediction −ε gᵀg', color: 's1', data: es.map(e => [e, -e * gg]), dash: [5, 4] },
      { name: 'actual change (= 2nd-order Taylor here)', color: 's2', data: es.map(e => [e, change(e)]) },
      { name: 'your ε', color: 'text', style: 'dots', r: 5, noHover: true, data: [[eps, change(eps)]] }
    ].concat(opt && opt <= 1 ? [{ name: 'ε*', color: 's3', style: 'dots', r: 7, noHover: true, data: [[opt, change(opt)]] }] : []),
    { yr: [Math.min(-0.5, change(opt && opt <= 1 ? opt : 1) * 1.2, -gg * 0.3), Math.max(1, Math.min(8, change(1)) + 0.5)] });
    lg.set([{ label: 'what the slope alone predicts', color: c.s1, kind: 'dash' }, { label: 'actual change in f', color: c.s2 }, { label: 'optimal step ε* = gᵀg / gᵀHg', color: c.s3, kind: 'dot' }]);
    calc.innerHTML = `<span class="c">g = Hx⁽⁰⁾ = (${g[0].toFixed(2)}, ${g[1].toFixed(2)})    gᵀg = ${gg.toFixed(3)}    gᵀHg = λ₁g₁² + λ₂g₂² = ${gHg.toFixed(3)}</span>
f(x⁽⁰⁾ − εg) ≈ f(x⁽⁰⁾) − ε·gᵀg + ½ε²·gᵀHg
          change = −${eps.toFixed(3)}×${gg.toFixed(3)} + ½×${eps.toFixed(3)}²×${gHg.toFixed(3)} = ${(-eps * gg).toFixed(3)} + ${(0.5 * eps * eps * gHg).toFixed(3)} = <b>${change(eps).toFixed(3)}</b>
${opt ? `ε* = gᵀg / gᵀHg = ${gg.toFixed(3)} / ${gHg.toFixed(3)} = <b>${opt.toFixed(4)}</b>   (worst case 1/λ_max = ${(1 / lmax).toFixed(4)});  ε > 2ε* = ${(2 * opt).toFixed(3)} goes UPHILL` : '<b>gᵀHg ≤ 0: the quadratic model says a bigger step always helps (negative curvature) — in practice the Taylor model breaks down for large ε.</b>'}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Projected gradient descent with ‖x‖ ≤ 1 ---------- */
Widgets.projectedGD = (el) => {
  UI.head(el, 'Constrained optimization: stay inside the circle', 'Minimize f(x) = (x₁ − 2)² + 2(x₂ − 1.5)² subject to ‖x‖₂ ≤ 1. Projected gradient descent: take a step, then pull the point back onto the feasible set.');
  const f = (x, y) => (x - 2) ** 2 + 2 * (y - 1.5) ** 2;
  const grad = (x, y) => [2 * (x - 2), 4 * (y - 1.5)];
  let eps = 0.12, start = [-0.6, -0.5], path = [], free = [], timer = null;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Learning rate ε', min: 0.02, max: 0.24, step: 0.01, value: eps, fmt: v => v.toFixed(2), onInput: v => { eps = v; reset(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Run ▶', () => { if (timer) return; reset(); let n = 0; timer = setInterval(() => { step(); draw(); if (++n >= 40) { clearInterval(timer); timer = null; } }, 90); }, 'btn primary');
  UI.button(btns, 'Step', () => { step(); draw(); });
  UI.button(btns, 'Reset', () => reset());
  ctr.append(btns);
  const wrap = h('div', { class: 'canvas-wrap', style: 'max-width:420px;margin:0 auto' });
  const cv = h('canvas', { class: 'chart clickable', role: 'img', 'aria-label': 'Constrained optimization path' });
  wrap.append(cv);
  el.append(ctr, wrap);
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  let P = null, bg = null, bgKey = '';
  cv.addEventListener('click', e => { if (!P) return; const r = cv.getBoundingClientRect(); const x = P.invX(e.clientX - r.left), y = P.invY(e.clientY - r.top); const n = Math.hypot(x, y); start = n > 1 ? [x / n, y / n] : [x, y]; reset(); });
  const proj = v => { const n = Math.hypot(...v); return n > 1 ? [v[0] / n, v[1] / n] : v; };
  function reset() { path = [start.slice()]; free = [start.slice()]; draw(); }
  function step() {
    const x = path[path.length - 1], g = grad(...x);
    path.push(proj([x[0] - eps * g[0], x[1] - eps * g[1]]));
    const y = free[free.length - 1], h2 = grad(...y);
    free.push([y[0] - eps * h2[0], y[1] - eps * h2[1]]);
  }
  function draw() {
    const c = Colors.get();
    const W = Math.min(420, boxWidth(wrap));
    const ctx = fitCanvas(cv, W, W);
    P = makePlot(W, W, [-1.6, 2.8], [-1.6, 2.8], { l: 0, r: 0, t: 0, b: 0 });
    const key = W + Theme.current(); if (key !== bgKey) { bg = contourImage(f, [-1.6, 2.8, -1.6, 2.8], Math.round(W / 2), Math.round(W / 2), 16); bgKey = key; }
    ctx.drawImage(bg, 0, 0, W, W);
    ctx.fillStyle = alpha(c.s1, .12); ctx.strokeStyle = c.s1; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(P.X(0), P.Y(0), P.X(1) - P.X(0), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    const line = (arr, col, dash) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash || []); ctx.beginPath(); arr.forEach((q, i) => i ? ctx.lineTo(P.X(q[0]), P.Y(q[1])) : ctx.moveTo(P.X(q[0]), P.Y(q[1]))); ctx.stroke(); ctx.setLineDash([]); };
    line(free, alpha(c.s3, .9), [5, 4]);
    line(path, c.s2);
    path.forEach(q => dot(ctx, P.X(q[0]), P.Y(q[1]), 3, c.s2));
    ctx.fillStyle = c.text; ctx.font = '600 15px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', P.X(2), P.Y(1.5));
    const x = path[path.length - 1], g = grad(...x);
    arrow(ctx, P.X(x[0]), P.Y(x[1]), P.X(x[0] - 0.12 * g[0]), P.Y(x[1] - 0.12 * g[1]), c.s3, 2.5);
    lg.set([{ label: 'feasible set ‖x‖ ≤ 1', color: c.s1 }, { label: 'projected gradient descent', color: c.s2 }, { label: 'unconstrained descent (leaves the set)', color: c.s3, kind: 'dash' }, { label: 'unconstrained minimum ★', color: c.text, kind: 'dot' }]);
    const prev = path[Math.max(0, path.length - 2)], gp = grad(...prev), raw = [prev[0] - eps * gp[0], prev[1] - eps * gp[1]], rn = Math.hypot(...raw);
    const lam = Math.hypot(...x) > 0.999 ? -(g[0] * x[0] + g[1] * x[1]) / 2 : 0;
    calc.innerHTML = `<span class="c">one step: x' = x − ε∇f(x),  then project:  x ← x' / max(1, ‖x'‖)</span>
x' = (${raw[0].toFixed(3)}, ${raw[1].toFixed(3)}),  ‖x'‖ = ${rn.toFixed(3)} ${rn > 1 ? '> 1 → divide by it' : '≤ 1 → keep'}  →  x = (${x[0].toFixed(3)}, ${x[1].toFixed(3)}),  f = ${f(...x).toFixed(4)}
<span class="c">KKT at the boundary: ∇f(x) + λ·∇(xᵀx − 1) = 0 with λ ≥ 0  →  λ = −(∇f·x)/2 = <b>${lam.toFixed(3)}</b> ${lam > 0 ? '(constraint ACTIVE: it pushes back)' : '(inactive)'}</span>`;
  }
  reset();
  return { redraw: () => { bgKey = ''; draw(); } };
};

/* ---------- 6) Least squares with a norm constraint: the λ path ---------- */
Widgets.lsqConstrained = (el) => {
  UI.head(el, 'Least squares with xᵀx ≤ 1: the Lagrange multiplier at work', 'f(x) = ½‖Ax − b‖². For each λ, minimizing the Lagrangian gives x(λ) = (AᵀA + 2λI)⁻¹Aᵀb. Increase λ until the solution lands on the circle.');
  const A = [[1, 0.4], [0.2, 0.8]], bv = [1.6, 1.4];
  const AtA = [[A[0][0] ** 2 + A[1][0] ** 2, A[0][0] * A[0][1] + A[1][0] * A[1][1]], [0, A[0][1] ** 2 + A[1][1] ** 2]]; AtA[1][0] = AtA[0][1];
  const Atb = [A[0][0] * bv[0] + A[1][0] * bv[1], A[0][1] * bv[0] + A[1][1] * bv[1]];
  const xOf = l => solveLinear([[AtA[0][0] + 2 * l, AtA[0][1]], [AtA[1][0], AtA[1][1] + 2 * l]], Atb);
  const f = (x, y) => 0.5 * ((A[0][0] * x + A[0][1] * y - bv[0]) ** 2 + (A[1][0] * x + A[1][1] * y - bv[1]) ** 2);
  let lg10 = -2, timer = null;
  let lo = 0, hi = 50; for (let k = 0; k < 60; k++) { const m = (lo + hi) / 2; Math.hypot(...xOf(m)) > 1 ? lo = m : hi = m; } const lamStar = (lo + hi) / 2;
  const ctr = h('div', { class: 'controls' });
  const lS = UI.slider(ctr, { label: 'KKT multiplier λ', min: -3, max: 1.3, step: 0.01, value: lg10, fmt: v => (10 ** v).toFixed(3), onInput: v => { lg10 = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Gradient ascent on λ ▶', () => { if (timer) return; let lam = 10 ** lg10; timer = setInterval(() => { const x = xOf(lam); lam = Math.max(1e-3, lam + 0.6 * (x[0] ** 2 + x[1] ** 2 - 1)); lg10 = Math.log10(lam); lS.set(lg10); draw(); if (Math.abs(Math.hypot(...x) - 1) < 1e-4) { clearInterval(timer); timer = null; } }, 80); }, 'btn primary');
  ctr.append(btns);
  const wrap = h('div', { class: 'canvas-wrap', style: 'max-width:420px;margin:0 auto' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Path of solutions as lambda grows' });
  wrap.append(cv);
  el.append(ctr, wrap);
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  let bg = null, bgKey = '';
  function draw() {
    const c = Colors.get();
    const W = Math.min(420, boxWidth(wrap));
    const ctx = fitCanvas(cv, W, W);
    const P = makePlot(W, W, [-1.4, 2.4], [-1.4, 2.4], { l: 0, r: 0, t: 0, b: 0 });
    const key = W + Theme.current(); if (key !== bgKey) { bg = contourImage(f, [-1.4, 2.4, -1.4, 2.4], Math.round(W / 2), Math.round(W / 2), 16); bgKey = key; }
    ctx.drawImage(bg, 0, 0, W, W);
    ctx.strokeStyle = c.s1; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(P.X(0), P.Y(0), P.X(1) - P.X(0), 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = c.s2; ctx.lineWidth = 2; ctx.setLineDash([4, 3]); ctx.beginPath();
    for (let k = 0; k <= 120; k++) { const l = 10 ** (-3 + k * 4.5 / 120); const x = xOf(l); k ? ctx.lineTo(P.X(x[0]), P.Y(x[1])) : ctx.moveTo(P.X(x[0]), P.Y(x[1])); }
    ctx.stroke(); ctx.setLineDash([]);
    const x0 = xOf(0), xs = xOf(lamStar), xl = xOf(10 ** lg10);
    ctx.fillStyle = c.text; ctx.font = '600 15px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', P.X(x0[0]), P.Y(x0[1]));
    dot(ctx, P.X(xs[0]), P.Y(xs[1]), 6, c.s3, c.surface);
    dot(ctx, P.X(xl[0]), P.Y(xl[1]), 7, c.s2, c.surface);
    lg.set([{ label: 'constraint ‖x‖ = 1', color: c.s1 }, { label: 'path of x(λ) as λ grows (moves toward 0)', color: c.s2, kind: 'dash' }, { label: 'current x(λ)', color: c.s2, kind: 'dot' }, { label: 'constrained solution', color: c.s3, kind: 'dot' }, { label: 'unconstrained minimum ★', color: c.text, kind: 'dot' }]);
    const lam = 10 ** lg10, nrm = Math.hypot(...xl);
    calc.innerHTML = `<span class="c">Lagrangian L(x, λ) = f(x) + λ(xᵀx − 1).  ∂L/∂x = AᵀAx − Aᵀb + 2λx = 0  ⇒  x = (AᵀA + 2λI)⁻¹Aᵀb</span>
λ = ${lam.toFixed(4)}  →  x(λ) = (${xl[0].toFixed(3)}, ${xl[1].toFixed(3)}),  ‖x‖ = ${nrm.toFixed(4)}
∂L/∂λ = xᵀx − 1 = <b>${(nrm * nrm - 1).toFixed(4)}</b>  ${nrm > 1 ? '> 0 → increase λ (gradient ascent)' : nrm < 0.999 ? '< 0 → λ too big; decrease it' : '≈ 0 → done'}
<span class="c">answer: λ* = ${lamStar.toFixed(4)}, x* = (${xs[0].toFixed(3)}, ${xs[1].toFixed(3)}).  Unconstrained x = (${x0[0].toFixed(2)}, ${x0[1].toFixed(2)}) has norm ${Math.hypot(...x0).toFixed(2)} > 1, so the constraint is active.</span>`;
  }
  draw();
  return { redraw: () => { bgKey = ''; draw(); } };
};

/* ---------- 7) SGD vs full-batch gradient descent ---------- */
Widgets.sgdVsGd = (el) => {
  UI.head(el, 'SGD vs. batch gradient descent: progress per unit of work', 'Linear regression on m = 2 000 examples. The x-axis counts how many per-example gradients were computed — the real cost.');
  const rng = mulberry32(8);
  const data = Array.from({ length: 2000 }, () => { const x = 2 * rng() - 1; return [x, 2 * x + 1 + 0.5 * gaussian(rng)]; });
  const loss = (w, b) => mean(data.map(([x, y]) => (w * x + b - y) ** 2));
  let mb = 16, lr = 0.2;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Minibatch size m′', min: 0, max: 8, step: 1, value: 4, fmt: v => String(2 ** v), onInput: v => { mb = 2 ** v; draw(); } });
  UI.slider(ctr, { label: 'SGD learning rate ε', min: 0.02, max: 0.6, step: 0.01, value: lr, fmt: v => v.toFixed(2), onInput: v => { lr = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 30000], yr: [0.2, 20], logY: true, xlabel: 'per-example gradients computed (work)', ylabel: 'training MSE', xname: 'work', aspect: .5, maxH: 300, refY: 0.25, xfmt: v => (v / 1000) + 'k', hxfmt: v => Math.round(v).toLocaleString() });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const budget = 30000;
    let w = -2, b = -2; const gd = [[0, loss(w, b)]];
    for (let t = 1; t * 2000 <= budget; t++) {
      let gw = 0, gb = 0; data.forEach(([x, y]) => { const e = w * x + b - y; gw += 2 * e * x; gb += 2 * e; });
      w -= 0.5 * gw / 2000; b -= 0.5 * gb / 2000; gd.push([t * 2000, loss(w, b)]);
    }
    let w2 = -2, b2 = -2; const sg = [[0, loss(w2, b2)]]; const r2 = mulberry32(77);
    let first = null;
    const steps = Math.floor(budget / mb), every = Math.max(1, Math.floor(steps / 150));
    for (let t = 1; t <= steps; t++) {
      let gw = 0, gb = 0;
      const batch = Array.from({ length: mb }, () => data[Math.floor(r2() * data.length)]);
      batch.forEach(([x, y]) => { const e = w2 * x + b2 - y; gw += 2 * e * x; gb += 2 * e; });
      if (!first) first = { batch, gw: gw / mb, gb: gb / mb };
      w2 -= lr * gw / mb; b2 -= lr * gb / mb;
      if (t % every === 0) sg.push([t * mb, Math.min(19, loss(w2, b2))]);
    }
    chart.set([
      { name: 'batch GD (all 2 000 per step)', color: 's1', data: gd },
      { name: `SGD, m′ = ${mb}`, color: 's2', data: sg }
    ]);
    lg.set([{ label: `batch gradient descent: ${gd.length - 1} steps`, color: c.s1 }, { label: `SGD with minibatches of ${mb}: ${steps.toLocaleString()} steps for the same work`, color: c.s2 }, { label: 'noise floor 0.25 (best possible)', color: c.muted, kind: 'dash' }]);
    calc.innerHTML = `<span class="c">J(θ) = (1/m)·Σ L(x⁽ⁱ⁾, y⁽ⁱ⁾, θ):  exact gradient costs m = 2 000 example-gradients per step  (O(m))</span>
minibatch estimate:  g = (1/m′)·∇θ Σ_{i∈B} L(x⁽ⁱ⁾, y⁽ⁱ⁾, θ)   with m′ = ${mb}  (cost O(m′), independent of m)
first minibatch: g = (∂/∂w, ∂/∂b) = (${first.gw.toFixed(3)}, ${first.gb.toFixed(3)})   update θ ← θ − ε·g = (−2, −2) − ${lr.toFixed(2)}·g = (${(-2 - lr * first.gw).toFixed(3)}, ${(-2 - lr * first.gb).toFixed(3)})`;
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   Quiz
   ========================================================= */
window.QUIZ = [
  { q: 'Gradient descent updates x by…', opts: ['x ← x + ε∇f(x)', 'x ← x − ε∇f(x)', 'x ← ∇f(x)', 'x ← x − ε f(x)'], a: 1, why: 'The gradient points uphill; moving a small step ε against it decreases f (for small enough ε): f(x − ε∇f) ≈ f(x) − ε‖∇f‖².' },
  { q: 'A point where f′(x) = 0 and f″(x) > 0 is…', opts: ['a local maximum', 'a local minimum', 'a saddle point', 'impossible to classify'], a: 1, why: 'Second-derivative test: the slope increases through zero, so the function curves upward around the point.' },
  { q: 'At a critical point the Hessian has eigenvalues 3 and −2. The point is…', opts: ['a local minimum', 'a local maximum', 'a saddle point', 'a global minimum'], a: 2, why: 'Mixed signs: a minimum along one eigenvector and a maximum along the other — a saddle point.' },
  { q: 'The directional derivative of f in unit direction u is…', opts: ['‖∇f‖', 'uᵀ∇f(x)', 'uᵀHu', '∇f / ‖u‖'], a: 1, why: 'uᵀ∇f = ‖∇f‖ cos α, which is most negative when u points opposite to the gradient — steepest descent.' },
  { q: 'What does the Hessian matrix contain?', opts: ['First derivatives of a vector function', 'All second partial derivatives ∂²f/∂xᵢ∂xⱼ', 'The learning rates', 'The eigenvectors of the data'], a: 1, why: 'H is the Jacobian of the gradient. It is symmetric where the second derivatives are continuous, so it has real eigenvalues.' },
  { q: 'Along the gradient direction, the optimal step for a quadratic model is ε* = …', opts: ['gᵀg / gᵀHg', '1/‖g‖', 'gᵀHg', '2/λ_min'], a: 0, why: 'Minimizing f − εgᵀg + ½ε²gᵀHg over ε gives ε* = gᵀg/gᵀHg; in the worst case (g along the top eigenvector) ε* = 1/λ_max.' },
  { q: 'A large condition number of the Hessian means…', opts: ['the function is linear', 'curvature differs a lot between directions, so gradient descent zig-zags and must use a small step', 'Newton’s method fails', 'there is no minimum'], a: 1, why: 'The step must be small enough for the steepest direction, which makes progress in the flat directions very slow.' },
  { q: 'In the KKT approach, complementary slackness says…', opts: ['all multipliers are zero', 'α ⊙ h(x) = 0: either an inequality constraint is active or its multiplier is zero', 'λ must be negative', 'constraints are ignored'], a: 1, why: 'At the solution, an inequality constraint either touches the solution (active, multiplier can be > 0) or has no influence (multiplier 0).' },
  { q: 'For least squares f(x) = ½‖Ax − b‖², the gradient is…', opts: ['Ax − b', 'Aᵀ(Ax − b)', 'AᵀA', 'bᵀA'], a: 1, why: '∇f = AᵀAx − Aᵀb. Newton’s method solves this quadratic in one step; adding xᵀx ≤ 1 gives x = (AᵀA + 2λI)⁻¹Aᵀb.' },
  { q: 'Why is SGD the workhorse of deep learning?', opts: ['It computes the exact gradient', 'Its cost per update depends only on the minibatch size m′, not on the training-set size m', 'It never needs a learning rate', 'It always finds the global minimum'], a: 1, why: 'A minibatch gives an unbiased estimate of the gradient at O(m′) cost, so even billions of examples can be used.' }
];
