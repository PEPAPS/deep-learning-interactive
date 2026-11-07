/* =========================================================
   Lecture 5 · Ch 6.5 · Back-propagation
   ========================================================= */
'use strict';

const f3 = v => fmt(v, 3);
const matTxt = M => M.map(r => '[' + r.map(v => (v >= 0 ? ' ' : '') + v.toFixed(2)).join(' ') + ' ]').join('\n');
function matTable(title, M) {
  return h('table', { class: 'ntable' }, h('caption', {}, title + `  (${M.length}×${M[0].length})`),
    h('tbody', {}, M.map(r => h('tr', {}, r.map(v => h('td', {}, v.toFixed(2)))))));
}
const mm = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
const tr = A => A[0].map((_, j) => A.map(r => r[j]));
const madd = (A, B, s = 1) => A.map((r, i) => r.map((v, j) => v + s * B[i][j]));
const msc = (A, s) => A.map(r => r.map(v => v * s));

/* ---------- 1) Numerical gradient vs back-propagation ---------- */
Widgets.numVsBackprop = (el) => {
  UI.head(el, 'What back-propagation buys you', 'Two ways to get ∂J/∂θ for a small network: nudge each parameter and re-run the network (numerical), or one backward pass (backprop). Same numbers, very different cost.');
  let lk = 3;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Number of parameters in the network', min: 1, max: 9, step: 0.1, value: lk, fmt: v => fmt(Math.round(10 ** v), 0), onInput: v => { lk = v; draw(); } });
  const bars = h('div', { class: 'hbars' });
  const tbl = h('div', { class: 'scroll-x' });
  el.append(ctr, bars, h('div', { class: 'note' }, 'Gradient check on a tiny 2→2→1 tanh network (6 weights shown):'), tbl);
  const rng = mulberry32(3);
  const W1 = [[0.5, -0.3], [0.8, 0.2]], W2 = [0.7, -1.1], x = [1.0, -0.5], y = 0.3;
  const J = (A, B) => { const h1 = A.map(r => Math.tanh(r[0] * x[0] + r[1] * x[1])); const o = B[0] * h1[0] + B[1] * h1[1]; return 0.5 * (o - y) ** 2; };
  void rng;
  function draw() {
    const c = Colors.get();
    const n = Math.round(10 ** lk);
    const num = 2 * n, bp = 2;
    bars.innerHTML = '';
    const row = (lab, v, col, txt) => h('div', { class: 'hbar' }, h('div', { class: 'lab' }, lab), h('div', { class: 'track' }, h('div', { class: 'fill', style: `width:${clamp(Math.log10(v + 1) / 9.5 * 100, 1, 100)}%;--c:${col}` }), h('span', { class: 'val' }, txt)));
    bars.append(row('Numerical: 2 network evaluations per parameter', num, c.s8, `${fmt(num, 0)} forward passes per gradient`), row('Back-propagation: 1 forward + 1 backward', bp, c.s3, `≈ 2 passes per gradient`));
    // gradient check
    const h1 = W1.map(r => Math.tanh(r[0] * x[0] + r[1] * x[1])); const o = W2[0] * h1[0] + W2[1] * h1[1]; const g = o - y;
    const gW2 = h1.map(v => g * v);
    const gh = W2.map(w => g * w), ga = gh.map((v, i) => v * (1 - h1[i] ** 2));
    const gW1 = ga.map(v => x.map(xx => v * xx));
    const eps = 1e-5, rows = [];
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      const P = W1.map(r => r.slice()), M = W1.map(r => r.slice()); P[i][j] += eps; M[i][j] -= eps;
      rows.push([`W⁽¹⁾[${i + 1},${j + 1}]`, (J(P, W2) - J(M, W2)) / (2 * eps), gW1[i][j]]);
    }
    for (let i = 0; i < 2; i++) { const P = W2.slice(), M = W2.slice(); P[i] += eps; M[i] -= eps; rows.push([`W⁽²⁾[${i + 1}]`, (J(W1, P) - J(W1, M)) / (2 * eps), gW2[i]]); }
    tbl.innerHTML = '';
    tbl.append(h('table', { class: 'readout' }, h('thead', {}, h('tr', {}, h('th', {}, 'parameter'), h('th', {}, 'numerical (J(θ+ε) − J(θ−ε)) / 2ε'), h('th', {}, 'back-propagation'), h('th', {}, 'difference'))),
      h('tbody', {}, rows.map(([n0, a, b]) => h('tr', {}, h('td', {}, n0), h('td', { class: 'num' }, a.toFixed(6)), h('td', { class: 'num' }, b.toFixed(6)), h('td', { class: 'num' }, Math.abs(a - b).toExponential(1)))))));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) Step-through computational graphs ---------- */
const GRAPHS = {
  mul: {
    label: '(a) z = x·y', leaves: { x: 2, y: -3 },
    nodes: [
      { id: 'x', pos: [80, 70] }, { id: 'y', pos: [80, 190] },
      { id: 'z', op: '×', in: ['x', 'y'], pos: [330, 130], f: (a, b) => a * b, d: [(a, b) => b, (a, b) => a] }
    ], out: 'z'
  },
  logistic: {
    label: '(b) ŷ = σ(xᵀw + b)', leaves: { x1: 1.5, w1: 0.8, x2: -1, w2: 0.4, b: -0.2 },
    nodes: [
      { id: 'x1', pos: [50, 40] }, { id: 'w1', pos: [50, 100] }, { id: 'x2', pos: [50, 170] }, { id: 'w2', pos: [50, 230] }, { id: 'b', pos: [250, 230] },
      { id: 'u1', op: '×', in: ['x1', 'w1'], pos: [170, 70], f: (a, b) => a * b, d: [(a, b) => b, (a, b) => a] },
      { id: 'u2', op: '×', in: ['x2', 'w2'], pos: [170, 200], f: (a, b) => a * b, d: [(a, b) => b, (a, b) => a] },
      { id: 'u3', op: '+', in: ['u1', 'u2'], pos: [290, 135], f: (a, b) => a + b, d: [() => 1, () => 1] },
      { id: 'u4', op: '+', in: ['u3', 'b'], pos: [400, 175], f: (a, b) => a + b, d: [() => 1, () => 1] },
      { id: 'ŷ', op: 'σ', in: ['u4'], pos: [520, 175], f: a => sigmoid(a), d: [a => sigmoid(a) * (1 - sigmoid(a))] }
    ], out: 'ŷ'
  },
  linreg: {
    label: '(c) J = (wx + b − y)² + λw²  (two paths from w!)', leaves: { w: 1.5, x: 2, b: 0.5, y: 2, 'λ': 0.1 },
    nodes: [
      { id: 'w', pos: [50, 90] }, { id: 'x', pos: [50, 30] }, { id: 'b', pos: [170, 30] }, { id: 'y', pos: [290, 30] }, { id: 'λ', pos: [290, 240] },
      { id: 'u1', op: '×', in: ['w', 'x'], pos: [150, 100], f: (a, b) => a * b, d: [(a, b) => b, (a, b) => a] },
      { id: 'u2', op: '+', in: ['u1', 'b'], pos: [250, 100], f: (a, b) => a + b, d: [() => 1, () => 1] },
      { id: 'u3', op: '−', in: ['u2', 'y'], pos: [350, 100], f: (a, b) => a - b, d: [() => 1, () => -1] },
      { id: 'u4', op: '²', in: ['u3'], pos: [450, 100], f: a => a * a, d: [a => 2 * a] },
      { id: 'u5', op: '²', in: ['w'], pos: [150, 200], f: a => a * a, d: [a => 2 * a] },
      { id: 'u6', op: '×', in: ['λ', 'u5'], pos: [370, 200], f: (a, b) => a * b, d: [(a, b) => b, (a, b) => a] },
      { id: 'J', op: '+', in: ['u4', 'u6'], pos: [530, 150], f: (a, b) => a + b, d: [() => 1, () => 1] }
    ], out: 'J'
  }
};
Widgets.graphBackprop = (el, ds) => {
  const keys = (ds.graphs || 'mul,logistic,linreg').split(',');
  UI.head(el, 'Forward, then backward, through a computational graph', 'Each node is a variable; each edge means “is used to compute”. Step forward to fill in values (blue), then backward to fill in gradients ∂(output)/∂(node) (orange).');
  let key = keys[0], step = 0;
  const segBox = h('div');
  const short = { mul: '(a) z = x·y', logistic: '(b) logistic unit', linreg: '(c) regression + weight decay' };
  if (keys.length > 1) UI.seg(segBox, keys.map(k => [k, short[k]]), key, v => { key = v; step = 0; build(); });
  const ctr = h('div', { class: 'btns' });
  const prev = UI.button(ctr, '← Back', () => { step = Math.max(0, step - 1); draw(); });
  const next = UI.button(ctr, 'Next step →', () => { step = Math.min(maxStep(), step + 1); draw(); }, 'btn primary');
  UI.button(ctr, 'Show all', () => { step = maxStep(); draw(); });
  UI.button(ctr, 'Reset', () => { step = 0; draw(); });
  const leafBox = h('div', { class: 'controls' });
  const svgBox = h('div'), calc = h('div', { class: 'calcbox' });
  el.append(segBox, h('div', { class: 'note' }, GRAPHS[key].label), leafBox, ctr, svgBox, calc);
  const lab = el.querySelector('.note');
  let vals = {};
  const G = () => GRAPHS[key];
  const ops = () => G().nodes.filter(n => n.op);
  const maxStep = () => ops().length + ops().length + 1;
  function build() {
    lab.textContent = G().label;
    vals = { ...G().leaves };
    leafBox.innerHTML = '';
    Object.keys(G().leaves).forEach(k => UI.slider(leafBox, { label: k, min: -3, max: 3, step: 0.1, value: vals[k], fmt: v => v.toFixed(1), onInput: v => { vals[k] = v; draw(); } }));
    draw();
  }
  function compute() {
    const v = { ...vals };
    ops().forEach(n => { v[n.id] = n.f(...n.in.map(p => v[p])); });
    const g = {}; G().nodes.forEach(n => g[n.id] = 0); g[G().out] = 1;
    const back = [];
    [...ops()].reverse().forEach(n => {
      n.in.forEach((p, i) => { const local = n.d[i](...n.in.map(q => v[q])); const contrib = g[n.id] * local; back.push({ node: n, parent: p, local, contrib, up: g[n.id], before: g[p] }); g[p] += contrib; });
    });
    return { v, g, back };
  }
  function draw() {
    const c = Colors.get();
    const { v, back } = compute();
    const nOps = ops().length;
    const fwdDone = Math.min(step, nOps);
    const bwdSteps = Math.max(0, step - nOps - 1);
    const shownV = new Set(Object.keys(vals)); ops().slice(0, fwdDone).forEach(n => shownV.add(n.id));
    const order = [...ops()].reverse();
    const doneB = order.slice(0, bwdSteps).map(n => n.id);
    const shownG = {};
    if (step > nOps) { shownG[G().out] = 1; }
    back.forEach(b => { if (doneB.includes(b.node.id)) shownG[b.parent] = (shownG[b.parent] || 0) + b.contrib; });
    if (step > nOps) G().nodes.forEach(n => { if (n.id === G().out) shownG[n.id] = 1; });
    const ns = 'http://www.w3.org/2000/svg', W = 600, H = 270;
    const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('class', 'diagram');
    const mk = (t, a, txt) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.append(e); return e; };
    const defs = mk('defs', {}); defs.innerHTML = `<marker id="ga" viewBox="0 0 10 10" refX="10" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="${c.muted}"/></marker>`;
    const pos = Object.fromEntries(G().nodes.map(n => [n.id, n.pos]));
    ops().forEach(n => n.in.forEach(p => {
      const [x1, y1] = pos[p], [x2, y2] = pos[n.id], d = Math.hypot(x2 - x1, y2 - y1);
      mk('line', { x1: x1 + (x2 - x1) * 30 / d, y1: y1 + (y2 - y1) * 22 / d, x2: x2 - (x2 - x1) * 34 / d, y2: y2 - (y2 - y1) * 26 / d, stroke: c.muted, 'stroke-width': 1.6, 'marker-end': 'url(#ga)' });
    }));
    const cur = step === 0 ? null : step <= nOps ? ops()[step - 1].id : step === nOps + 1 ? G().out : order[bwdSteps - 1] ? order[bwdSteps - 1].id : null;
    G().nodes.forEach(n => {
      const [x, y] = n.pos;
      mk('rect', { x: x - 32, y: y - 24, width: 64, height: 48, rx: 10, fill: c.surface, stroke: n.id === cur ? c.text : c.border, 'stroke-width': n.id === cur ? 3 : 1.5 });
      mk('text', { x, y: y - 8, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: c.text }, n.op ? `${n.id} = ${n.op}` : n.id);
      if (shownV.has(n.id)) mk('text', { x, y: y + 7, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: c.s1 }, v[n.id].toFixed(3));
      if (shownG[n.id] != null) mk('text', { x, y: y + 20, 'text-anchor': 'middle', 'font-size': 11, 'font-weight': 700, fill: c.s2 }, '∂=' + shownG[n.id].toFixed(3));
    });
    svgBox.innerHTML = ''; svgBox.append(svg);
    svgBox.append(UI.legend(h('div'), [{ label: 'value (forward pass)', color: c.s1, kind: 'dot' }, { label: `gradient ∂${G().out}/∂node (backward pass)`, color: c.s2, kind: 'dot' }]).el);
    prev.disabled = step === 0; next.disabled = step >= maxStep();
    let txt;
    if (step === 0) txt = '<span class="c">Leaves (inputs/parameters) are known. Press “Next step” to run the forward pass.</span>';
    else if (step <= nOps) { const n = ops()[step - 1]; txt = `<span class="c">forward step ${step}: apply the operation to the parents' values</span>\n${n.id} = ${n.op}(${n.in.join(', ')}) = ${n.op}(${n.in.map(p => v[p].toFixed(3)).join(', ')}) = <b>${v[n.id].toFixed(4)}</b>`; }
    else if (step === nOps + 1) txt = `<span class="c">start the backward pass at the output:</span>\n∂${G().out}/∂${G().out} = <b>1</b>`;
    else {
      const n = order[bwdSteps - 1];
      const items = back.filter(b => b.node.id === n.id);
      txt = `<span class="c">backward step: node ${n.id} sends gradient to its parents = (gradient of ${n.id}) × (local derivative)</span>\n` + items.map(b => `∂${G().out}/∂${b.parent} += ∂${G().out}/∂${n.id} · ∂${n.id}/∂${b.parent} = ${b.up.toFixed(3)} × ${b.local.toFixed(3)} = ${b.contrib.toFixed(4)}${b.before ? `   (adds to ${b.before.toFixed(4)} from another path → ${(b.before + b.contrib).toFixed(4)})` : ''}`).join('\n');
      if (step === maxStep()) txt += `\n<span class="c">done: every node now holds ∂${G().out}/∂node — computed once each, reusing the stored values.</span>`;
    }
    calc.innerHTML = txt;
  }
  build();
  return { redraw: draw };
};

/* ---------- 3) Chain rule: scalar and vector (Jacobian) ---------- */
Widgets.chainRule = (el) => {
  UI.head(el, 'The chain rule, numerically', 'Scalar: y = g(x) = x² + 1, z = f(y) = sin(y). Vector: y = A x, z = ‖y‖². In the vector case the chain rule is a Jacobian-transpose × gradient product.');
  let x = 0.7, v = [1, -0.5];
  const A = [[1, 2], [-1, 0.5]];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'scalar x', min: -2, max: 2, step: 0.01, value: x, fmt: t => t.toFixed(2), onInput: t => { x = t; draw(); } });
  UI.slider(ctr, { label: 'vector x₁', min: -2, max: 2, step: 0.05, value: v[0], fmt: t => t.toFixed(2), onInput: t => { v[0] = t; draw(); } });
  UI.slider(ctr, { label: 'vector x₂', min: -2, max: 2, step: 0.05, value: v[1], fmt: t => t.toFixed(2), onInput: t => { v[1] = t; draw(); } });
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, calc);
  function draw() {
    const y = x * x + 1, z = Math.sin(y), dydx = 2 * x, dzdy = Math.cos(y), dzdx = dzdy * dydx;
    const num = (Math.sin((x + 1e-5) ** 2 + 1) - Math.sin((x - 1e-5) ** 2 + 1)) / 2e-5;
    const yv = [A[0][0] * v[0] + A[0][1] * v[1], A[1][0] * v[0] + A[1][1] * v[1]];
    const gy = yv.map(t => 2 * t);
    const gx = [A[0][0] * gy[0] + A[1][0] * gy[1], A[0][1] * gy[0] + A[1][1] * gy[1]];
    calc.innerHTML = `<span class="c">SCALAR  dz/dx = dz/dy · dy/dx   (6.44)</span>
y = x² + 1 = ${y.toFixed(4)}      z = sin(y) = ${z.toFixed(4)}
dy/dx = 2x = ${dydx.toFixed(4)}      dz/dy = cos(y) = ${dzdy.toFixed(4)}
dz/dx = ${dzdy.toFixed(4)} × ${dydx.toFixed(4)} = <b>${dzdx.toFixed(5)}</b>    (finite-difference check: ${num.toFixed(5)})

<span class="c">VECTOR  ∇ₓz = (∂y/∂x)ᵀ ∇ᵧz   (6.46),  Jacobian ∂y/∂x = A = [[1, 2], [−1, 0.5]]</span>
y = A x = (${yv.map(f3).join(', ')})        ∇ᵧz = 2y = (${gy.map(f3).join(', ')})
∇ₓz = Aᵀ ∇ᵧz = ( 1·${f3(gy[0])} + (−1)·${f3(gy[1])},  2·${f3(gy[0])} + 0.5·${f3(gy[1])} ) = <b>(${gx.map(f3).join(', ')})</b>
<span class="c">element form (6.45): ∂z/∂xᵢ = Σⱼ (∂z/∂yⱼ)(∂yⱼ/∂xᵢ)</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) Store vs recompute ---------- */
Widgets.chainStoreRecompute = (el) => {
  UI.head(el, 'Store intermediate values, or recompute them?', 'A chain w → x = f(w) → y = f(x) → … with f(t) = sin(t) + t applied n times. The gradient is a product of f′ at every intermediate value.');
  let n = 8, w = 0.4;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Chain length n', min: 1, max: 40, step: 1, value: n, fmt: v => String(v), onInput: v => { n = v; draw(); } });
  UI.slider(ctr, { label: 'input w', min: -2, max: 2, step: 0.05, value: w, fmt: v => v.toFixed(2), onInput: v => { w = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [1, 40], yr: [0, 900], xlabel: 'chain length n', ylabel: 'evaluations of f or f′', xname: 'n', aspect: .45, maxH: 240, snapX: x => Math.round(x) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const f = t => Math.sin(t) + t, fp = t => Math.cos(t) + 1;
  function draw() {
    const c = Colors.get();
    const ns = Array.from({ length: 40 }, (_, i) => i + 1);
    chart.set([
      { name: 'store (backprop)', color: 's3', data: ns.map(k => [k, 2 * k]) },
      { name: 'recompute', color: 's8', data: ns.map(k => [k, k * (k - 1) / 2 + k]) },
      { name: 'now', color: 'text', style: 'dots', r: 5, noHover: true, data: [[n, n * (n - 1) / 2 + n], [n, 2 * n]] }
    ]);
    lg.set([{ label: 'store each value once: n forward + n derivative evaluations, memory n', color: c.s3 }, { label: 'recompute f(f(…f(w))) for every factor: n(n−1)/2 + n, memory O(1)', color: c.s8 }]);
    const xs = [w]; for (let k = 0; k < n; k++) xs.push(f(xs[k]));
    let grad = 1; const factors = []; for (let k = 0; k < n; k++) { factors.push(fp(xs[k])); grad *= fp(xs[k]); }
    calc.innerHTML = `<span class="c">∂z/∂w = f′(y)·f′(x)·f′(w)   (6.51, stored values)   =   f′(f(f(w)))·f′(f(w))·f′(w)   (6.52, recomputed)</span>
stored values: ${xs.slice(0, 6).map(t => t.toFixed(3)).join(' → ')}${n > 5 ? ' → …' : ''}
factors f′: ${factors.slice(0, 6).map(t => t.toFixed(3)).join(' × ')}${n > 6 ? ' × …' : ''}  =  <b>${grad.toExponential(4)}</b>
cost for n = ${n}:  store → ${2 * n} evaluations (+ ${n} stored numbers);  recompute → ${n * (n - 1) / 2 + n} evaluations (no storage)`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Algorithms 6.3 / 6.4 line by line ---------- */
Widgets.mlpBackprop = (el) => {
  UI.head(el, 'Algorithm 6.3 + 6.4 on a real 2-2-1 network', 'One example x = (1, −0.5), y = 0.3. Hidden layer: tanh. Output: identity. Loss L = ½(ŷ − y)², plus λΩ with Ω = Σ W² (λ = 0.01). Step through every line.');
  const x = [1, -0.5], y = 0.3, lam = 0.01;
  const W1 = [[0.5, -0.3], [0.8, 0.2]], b1 = [0.1, -0.1], W2 = [[0.7, -1.1]], b2 = [0.05];
  let step = 0;
  const ctr = h('div', { class: 'btns' });
  const prev = UI.button(ctr, '← Back', () => { step = Math.max(0, step - 1); draw(); });
  const next = UI.button(ctr, 'Next line →', () => { step = Math.min(lines.length - 1, step + 1); draw(); }, 'btn primary');
  UI.button(ctr, 'Show all', () => { step = lines.length - 1; draw(); });
  const list = h('ol', { class: 'calc', style: 'font-family:var(--mono);font-size:.78rem' });
  el.append(ctr, list);
  const mv = (M, v) => M.map(r => r.reduce((s, w, i) => s + w * v[i], 0));
  const a1 = mv(W1, x).map((t, i) => t + b1[i]), h1 = a1.map(Math.tanh);
  const a2 = mv(W2, h1).map((t, i) => t + b2[i]), yh = a2[0];
  const L = 0.5 * (yh - y) ** 2, Om = [...W1.flat(), ...W2.flat()].reduce((s, w) => s + w * w, 0), J = L + lam * Om;
  const g0 = yh - y;
  const g2 = g0 * 1;
  const gW2 = [h1.map(t => g2 * t + 2 * lam * 0)].map((r, i) => r.map((t, j) => t + 2 * lam * W2[i][j]));
  const gh1 = W2[0].map(w => w * g2);
  const ga1 = gh1.map((t, i) => t * (1 - h1[i] ** 2));
  const gW1 = ga1.map((t, i) => x.map((xx, j) => t * xx + 2 * lam * W1[i][j]));
  const v2 = v => '(' + v.map(f3).join(', ') + ')';
  const lines = [
    ['FORWARD (Alg. 6.3)', 'h⁽⁰⁾ = x', `x = ${v2(x)}`],
    ['', 'a⁽¹⁾ = b⁽¹⁾ + W⁽¹⁾h⁽⁰⁾', `= ${v2(b1)} + [[0.5, −0.3], [0.8, 0.2]]·${v2(x)} = ${v2(a1)}`],
    ['', 'h⁽¹⁾ = f(a⁽¹⁾) = tanh(a⁽¹⁾)', `= ${v2(h1)}`],
    ['', 'a⁽²⁾ = b⁽²⁾ + W⁽²⁾h⁽¹⁾', `= 0.05 + 0.7·${f3(h1[0])} − 1.1·${f3(h1[1])} = ${f3(a2[0])}`],
    ['', 'ŷ = h⁽²⁾ = a⁽²⁾ (identity output)', `ŷ = ${f3(yh)}`],
    ['', 'J = L(ŷ, y) + λΩ(θ)', `= ½(${f3(yh)} − 0.3)² + 0.01·${f3(Om)} = ${f3(L)} + ${f3(lam * Om)} = ${f3(J)}`],
    ['BACKWARD (Alg. 6.4)', 'g ← ∇ŷ J = ∇ŷ L(ŷ, y)', `g = ŷ − y = ${f3(g0)}`],
    ['k = 2', 'g ← g ⊙ f′(a⁽²⁾)', `identity: f′ = 1 → g = ${f3(g2)}`],
    ['', '∇b⁽²⁾J = g + λ∇b⁽²⁾Ω', `= ${f3(g2)} (biases not penalized)`],
    ['', '∇W⁽²⁾J = g h⁽¹⁾ᵀ + λ∇W⁽²⁾Ω', `= ${f3(g2)}·${v2(h1)} + 0.02·W⁽²⁾ = ${v2(gW2[0])}`],
    ['', 'g ← ∇h⁽¹⁾J = W⁽²⁾ᵀ g', `= (0.7, −1.1)·${f3(g2)} = ${v2(gh1)}`],
    ['k = 1', 'g ← g ⊙ f′(a⁽¹⁾),  f′ = 1 − tanh²', `= ${v2(gh1)} ⊙ ${v2(h1.map(t => 1 - t * t))} = ${v2(ga1)}`],
    ['', '∇b⁽¹⁾J = g', `= ${v2(ga1)}`],
    ['', '∇W⁽¹⁾J = g h⁽⁰⁾ᵀ + λ∇W⁽¹⁾Ω', `= [${v2(gW1[0])}, ${v2(gW1[1])}]`],
    ['', 'g ← ∇h⁽⁰⁾J = W⁽¹⁾ᵀ g', `= ${v2([W1[0][0] * ga1[0] + W1[1][0] * ga1[1], W1[0][1] * ga1[0] + W1[1][1] * ga1[1]])}  (not needed for learning; shows the pattern)`],
    ['UPDATE (SGD, ε = 0.1)', 'θ ← θ − ε∇θJ', `e.g. W⁽²⁾ → (${f3(W2[0][0] - 0.1 * gW2[0][0])}, ${f3(W2[0][1] - 0.1 * gW2[0][1])})`]
  ];
  function draw() {
    list.innerHTML = '';
    lines.slice(0, step + 1).forEach(([tag, form, val], i) => list.append(h('li', { style: i === step ? 'background:var(--accent-soft);border-radius:8px;padding:4px 6px 4px 30px' : '' },
      tag ? h('div', { style: 'font-family:var(--font);font-size:.7rem;color:var(--muted);font-weight:700;text-transform:uppercase' }, tag) : null,
      h('div', {}, h('b', {}, form)), h('div', { style: 'color:var(--text-2)' }, val))));
    prev.disabled = step === 0; next.disabled = step >= lines.length - 1;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 6) Exponentially many paths vs. table filling ---------- */
Widgets.pathCount = (el) => {
  UI.head(el, 'Sum over paths (naive) vs. dynamic programming (backprop)', 'A graph with 2 nodes per layer, every node feeding both nodes of the next layer. The chain rule written out is a sum over ALL paths from w to z.');
  let d = 4;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Depth (layers between w and z)', min: 1, max: 30, step: 1, value: d, fmt: v => String(v), onInput: v => { d = v; draw(); } });
  el.append(ctr);
  const svgBox = h('div');
  el.append(svgBox);
  const chart = new LineChart(el, { xr: [1, 30], yr: [1, 1e10], logY: true, xlabel: 'depth', ylabel: 'multiplications', xname: 'depth', aspect: .4, maxH: 220, snapX: x => Math.round(x) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const rng = mulberry32(2);
    const layers = [[0]]; for (let k = 0; k < d; k++) layers.push([0, 1]); layers.push([0]);
    const edgeW = layers.slice(0, -1).map((L, k) => L.map(() => layers[k + 1].map(() => +(0.5 + rng()).toFixed(2))));
    // backprop: grad at z = 1, fill backwards
    const grad = layers.map(L => L.map(() => 0)); grad[layers.length - 1][0] = 1;
    for (let k = layers.length - 2; k >= 0; k--) layers[k].forEach((_, i) => { grad[k][i] = layers[k + 1].reduce((s, _, j) => s + edgeW[k][i][j] * grad[k + 1][j], 0); });
    const paths = 2 ** d, naiveMults = paths * (d + 1), bpMults = edgeW.reduce((s, L) => s + L.reduce((t, r) => t + r.length, 0), 0);
    if (d <= 6) {
      const ns = 'http://www.w3.org/2000/svg', W = 600, H = 170;
      const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.setAttribute('class', 'diagram');
      const mk = (t, a, txt) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; svg.append(e); return e; };
      const X = k => 40 + k * (W - 80) / (layers.length - 1), Y = (k, i) => layers[k].length === 1 ? H / 2 : 45 + i * 80;
      edgeW.forEach((L, k) => L.forEach((r, i) => r.forEach((wv, j) => { mk('line', { x1: X(k), y1: Y(k, i), x2: X(k + 1), y2: Y(k + 1, j), stroke: c.muted, 'stroke-width': 1.3 }); mk('text', { x: (X(k) + X(k + 1)) / 2, y: (Y(k, i) + Y(k + 1, j)) / 2 - 3, 'text-anchor': 'middle', 'font-size': 9, fill: c.text2 }, wv.toFixed(2)); })));
      layers.forEach((L, k) => L.forEach((_, i) => { mk('circle', { cx: X(k), cy: Y(k, i), r: 15, fill: c.surface, stroke: c.text, 'stroke-width': 1.5 }); mk('text', { x: X(k), y: Y(k, i) + 4, 'text-anchor': 'middle', 'font-size': 9, 'font-weight': 700, fill: c.s2 }, grad[k][i].toFixed(1)); }));
      mk('text', { x: X(0), y: H / 2 + 32, 'text-anchor': 'middle', 'font-size': 11, fill: c.text, 'font-weight': 700 }, 'w');
      mk('text', { x: X(layers.length - 1), y: H / 2 + 32, 'text-anchor': 'middle', 'font-size': 11, fill: c.text, 'font-weight': 700 }, 'z');
      svgBox.innerHTML = ''; svgBox.append(svg, h('div', { class: 'note' }, 'Edge labels = local derivatives; orange numbers in nodes = ∂z/∂node, filled from right to left (each computed once).'));
    } else { svgBox.innerHTML = ''; svgBox.append(h('div', { class: 'note' }, 'Graph too large to draw — see the counts below.')); }
    const ds = Array.from({ length: 30 }, (_, i) => i + 1);
    chart.set([
      { name: 'naive: sum over 2ᵈ paths', color: 's8', data: ds.map(k => [k, Math.min(1e10, 2 ** k * (k + 1))]) },
      { name: 'backprop: one multiply per edge', color: 's3', data: ds.map(k => [k, 4 * (k - 1) + 4]) }
    ]);
    lg.set([{ label: 'naive chain rule (exponential)', color: c.s8 }, { label: 'back-propagation (linear in edges)', color: c.s3 }]);
    calc.innerHTML = `<span class="c">∂z/∂w = Σ over paths (w → … → z) of Π (local derivatives along the path)    (6.55)</span>
paths = 2^${d} = ${fmt(paths, 0)}  →  naive ≈ ${fmt(naiveMults, 0)} multiplications
backprop: one multiply-add per edge = ${bpMults} → same answer ∂z/∂w = <b>${grad[0][0].toPrecision(6)}</b>
<span class="c">“table filling” / dynamic programming: store ∂z/∂node for every node and reuse it</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) The full minibatch example (6.5.7) ---------- */
Widgets.mlpTrainExample = (el) => {
  UI.head(el, 'The textbook MLP, on a real minibatch', 'm = 4 examples, 3 input features, 4 ReLU hidden units, 3 classes, no biases, weight decay λ = 0.01. Pick a step to see the matrices; press Train to run SGD with these gradients.');
  const rng = mulberry32(12);
  const X = [[1, 0.5, -1], [0.2, -1, 0.8], [-0.7, 0.3, 0.5], [1.2, 1, 0.1]], y = [0, 1, 2, 0];
  let W1 = Array.from({ length: 3 }, () => Array.from({ length: 4 }, () => +(0.6 * gaussian(rng)).toFixed(2)));
  let W2 = Array.from({ length: 4 }, () => Array.from({ length: 3 }, () => +(0.6 * gaussian(rng)).toFixed(2)));
  const W1_0 = W1.map(r => r.slice()), W2_0 = W2.map(r => r.slice());
  const lam = 0.01;
  let step = 0, hist = [];
  const steps = ['1 · XW⁽¹⁾', '2 · H = max{0, XW⁽¹⁾}', '3 · U⁽²⁾ = HW⁽²⁾ (logits)', '4 · softmax & J', '5 · G = ∇U⁽²⁾J_MLE', '6 · ∇W⁽²⁾ = HᵀG + 2λW⁽²⁾', '7 · ∇H = GW⁽²⁾ᵀ', '8 · G′ = ∇H ⊙ 1[XW⁽¹⁾ > 0]', '9 · ∇W⁽¹⁾ = XᵀG′ + 2λW⁽¹⁾'];
  const ctr = h('div', { class: 'controls' });
  const sel = UI.select(ctr, 'Show step', steps.map((s, i) => [String(i), s]), '0', v => { step = +v; draw(); });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Train 50 SGD steps', () => { for (let i = 0; i < 50; i++) sgd(); draw(); }, 'btn primary');
  UI.button(btns, 'Reset weights', () => { W1 = W1_0.map(r => r.slice()); W2 = W2_0.map(r => r.slice()); hist = []; draw(); });
  ctr.append(btns);
  const tables = h('div', { class: 'ntables scroll-x' });
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, tables, calc);
  const chart = new LineChart(el, { xr: [0, 50], yr: [0, 2], xlabel: 'SGD step', ylabel: 'J', xname: 'step', aspect: .35, maxH: 180, minH: 140 });
  function compute() {
    const XW = mm(X, W1), Hh = XW.map(r => r.map(v => Math.max(0, v))), U = mm(Hh, W2);
    const P = U.map(r => { const mx = Math.max(...r), e = r.map(v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); });
    const m = X.length;
    const Jmle = mean(P.map((p, i) => -Math.log(p[y[i]])));
    const Om = W1.flat().reduce((s, w) => s + w * w, 0) + W2.flat().reduce((s, w) => s + w * w, 0);
    const G = P.map((p, i) => p.map((v, j) => (v - (j === y[i] ? 1 : 0)) / m));
    const gW2 = madd(mm(tr(Hh), G), msc(W2, 2 * lam));
    const gH = mm(G, tr(W2));
    const Gp = gH.map((r, i) => r.map((v, j) => (XW[i][j] > 0 ? v : 0)));
    const gW1 = madd(mm(tr(X), Gp), msc(W1, 2 * lam));
    return { XW, Hh, U, P, Jmle, Om, J: Jmle + lam * Om, G, gW2, gH, Gp, gW1 };
  }
  function sgd() { const r = compute(); W1 = madd(W1, r.gW1, -0.5); W2 = madd(W2, r.gW2, -0.5); hist.push(r.J); }
  function draw() {
    const r = compute();
    tables.innerHTML = '';
    const T = [
      () => [matTable('X', X), matTable('W⁽¹⁾', W1), matTable('XW⁽¹⁾', r.XW)],
      () => [matTable('XW⁽¹⁾', r.XW), matTable('H = relu(XW⁽¹⁾)', r.Hh)],
      () => [matTable('H', r.Hh), matTable('W⁽²⁾', W2), matTable('U⁽²⁾ = HW⁽²⁾', r.U)],
      () => [matTable('softmax(U⁽²⁾) = P', r.P)],
      () => [matTable('P', r.P), matTable('G = (P − Y_onehot)/m', r.G)],
      () => [matTable('Hᵀ', tr(r.Hh)), matTable('∇W⁽²⁾J', r.gW2)],
      () => [matTable('∇H = GW⁽²⁾ᵀ', r.gH)],
      () => [matTable('mask 1[XW⁽¹⁾>0]', r.XW.map(rr => rr.map(v => (v > 0 ? 1 : 0)))), matTable('G′', r.Gp)],
      () => [matTable('Xᵀ', tr(X)), matTable('∇W⁽¹⁾J', r.gW1)]
    ][step]();
    tables.append(...T);
    calc.innerHTML = `<span class="c">J = J_MLE + λ(Σ W⁽¹⁾² + Σ W⁽²⁾²)   (6.56),  labels y = (${y.map(v => v + 1).join(', ')})</span>
J_MLE = mean(−log P[i, yᵢ]) = mean(${r.P.map((p, i) => '−log ' + p[y[i]].toFixed(3)).join(', ')}) = ${r.Jmle.toFixed(4)}
J = ${r.Jmle.toFixed(4)} + 0.01 × ${r.Om.toFixed(3)} = <b>${r.J.toFixed(4)}</b>
<span class="c">cost: forward ≈ m·(3·4 + 4·3) = ${4 * 24} multiply-adds; backward uses the transposes — the same order. Memory: H (m × n_h = 4×4) is stored for the backward pass.</span>`;
    void sel;
    chart.set([{ name: 'J', color: 's1', data: hist.map((v, i) => [i + 1, v]) }], { xr: [0, Math.max(50, hist.length)], yr: [0, Math.max(1.5, ...hist, 0.1) * 1.05] });
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   Quiz
   ========================================================= */
window.QUIZ = [
  { q: 'Back-propagation is…', opts: ['the whole learning algorithm', 'a method for computing the gradient; SGD (or another optimizer) then uses it to learn', 'only for convolutional networks', 'a way to choose the learning rate'], a: 1, why: 'Backprop only computes ∇θJ efficiently; the update step is done by gradient descent or another optimizer. It can compute derivatives of any function, not only MLPs.' },
  { q: 'In a computational graph, a node represents…', opts: ['a layer only', 'a variable (scalar, vector, matrix, tensor…)', 'a learning rate', 'a training example'], a: 1, why: 'Nodes are variables; an operation produces a node from its parents; an edge from x to y means y is computed from x.' },
  { q: 'For y = g(x) and z = f(y) with vectors, ∇ₓz equals…', opts: ['∇ᵧz · ∇ₓy elementwise', '(∂y/∂x)ᵀ ∇ᵧz', '∂y/∂x · ∂z/∂x', 'the Hessian'], a: 1, why: 'The chain rule in vector form is a Jacobian-transpose times gradient product. Backprop performs one such product per operation.' },
  { q: 'If a node is reached backward from the output by two different paths, backprop…', opts: ['ignores one path', 'adds the gradients arriving from both paths', 'multiplies them', 'averages them'], a: 1, why: 'Total derivative: ∂z/∂w = Σ over children of (∂z/∂child)(∂child/∂w). In the J = (wx+b−y)² + λw² graph, w gets a contribution from both terms.' },
  { q: 'Why can the naive chain-rule expansion be exponentially expensive?', opts: ['Because of floating point', 'The number of paths from a node to the output can grow exponentially with depth, and subexpressions are recomputed', 'Because Jacobians are large', 'It is not expensive'], a: 1, why: 'Backprop stores each ∂z/∂u⁽ʲ⁾ once (table filling / dynamic programming), so its cost is linear in the number of edges.' },
  { q: 'Storing intermediate values instead of recomputing them trades…', opts: ['accuracy for speed', 'memory for computation time', 'bias for variance', 'nothing'], a: 1, why: 'Storing f(w), f(f(w))… avoids recomputation (fast) but costs memory; recomputing saves memory at the cost of runtime.' },
  { q: 'In Algorithm 6.4, after computing g for layer k, the gradient on that layer’s weights is…', opts: ['g ⊙ h⁽ᵏ⁻¹⁾', 'g h⁽ᵏ⁻¹⁾ᵀ + λ∇W⁽ᵏ⁾Ω', 'W⁽ᵏ⁾ᵀ g', 'f′(a⁽ᵏ⁾)'], a: 1, why: 'Then g ← W⁽ᵏ⁾ᵀ g propagates the gradient to the layer below, and g ← g ⊙ f′(a⁽ᵏ⁻¹⁾) passes it through the non-linearity.' },
  { q: 'In the MLP example, what does the ReLU backward rule do?', opts: ['Doubles the gradient', 'Zeros out gradient entries where the ReLU input XW⁽¹⁾ was negative', 'Adds weight decay', 'Applies softmax'], a: 1, why: 'ReLU’s derivative is 0 where inactive and 1 where active, so G′ = ∇H ⊙ 1[XW⁽¹⁾ > 0].' },
  { q: 'For a chain-structured network, the cost of back-propagation is roughly…', opts: ['exponential in depth', 'the same order as the forward pass (O(number of weights))', 'O(n³)', 'independent of the network size'], a: 1, why: 'Forward multiplies by each W, backward by each Wᵀ — the same number of multiply-adds. Memory is dominated by storing activations (O(m·n_h)).' },
  { q: 'Which is a real-world complication of backprop implementations?', opts: ['Operations with multiple outputs, buffered summation, data types, undefined gradients', 'Backprop needs labels for hidden layers', 'It only works for scalar parameters', 'It requires convex losses'], a: 0, why: 'Practical libraries must support multi-output ops (e.g. max and argmax together), accumulate sums in one buffer, handle float/int types, track undefined gradients and support higher-order derivatives.' }
];
