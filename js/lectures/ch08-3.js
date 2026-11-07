/* =========================================================
   Lecture 8 · Ch 8.3–8.5 · Optimization II: SGD, momentum, Adam, initialization
   (extends the optimizer playground defined in ch08-6a.js)
   ========================================================= */
'use strict';

/* ---------- new methods for the playground ---------- */
Object.assign(METHODS, {
  momentum: {
    label: 'SGD + momentum', color: '--s4', key: 's4',
    init(r) { r.v = [0, 0]; },
    step(r, F, P) { const g = F.g(...r.th); r.v = [P.mom * r.v[0] - P.lr * g[0], P.mom * r.v[1] - P.lr * g[1]]; r.th = V2.add(r.th, r.v); }
  },
  nesterov: {
    label: 'Nesterov momentum', color: '--s3', key: 's3',
    init(r) { r.v = [0, 0]; },
    step(r, F, P) { const look = [r.th[0] + P.mom * r.v[0], r.th[1] + P.mom * r.v[1]]; const g = F.g(...look); r.v = [P.mom * r.v[0] - P.lr * g[0], P.mom * r.v[1] - P.lr * g[1]]; r.th = V2.add(r.th, r.v); }
  },
  adagrad: {
    label: 'AdaGrad', color: '--s7', key: 's7',
    init(r) { r.acc = [0, 0]; },
    step(r, F, P) { const g = F.g(...r.th); r.acc = [r.acc[0] + g[0] * g[0], r.acc[1] + g[1] * g[1]]; r.th = [r.th[0] - P.lrA * g[0] / (1e-7 + Math.sqrt(r.acc[0])), r.th[1] - P.lrA * g[1] / (1e-7 + Math.sqrt(r.acc[1]))]; }
  },
  rmsprop: {
    label: 'RMSProp', color: '--s5', key: 's5',
    init(r) { r.acc = [0, 0]; },
    step(r, F, P) { const g = F.g(...r.th), rho = 0.9; r.acc = [rho * r.acc[0] + (1 - rho) * g[0] * g[0], rho * r.acc[1] + (1 - rho) * g[1] * g[1]]; r.th = [r.th[0] - P.lrA * g[0] / Math.sqrt(1e-6 + r.acc[0]), r.th[1] - P.lrA * g[1] / Math.sqrt(1e-6 + r.acc[1])]; }
  },
  adam: {
    label: 'Adam', color: '--s8', key: 's8',
    init(r) { r.s = [0, 0]; r.rr = [0, 0]; r.t = 0; },
    step(r, F, P) {
      const g = F.g(...r.th), b1 = 0.9, b2 = 0.999; r.t++;
      r.s = r.s.map((v, i) => b1 * v + (1 - b1) * g[i]); r.rr = r.rr.map((v, i) => b2 * v + (1 - b2) * g[i] * g[i]);
      const sh = r.s.map(v => v / (1 - b1 ** r.t)), rh = r.rr.map(v => v / (1 - b2 ** r.t));
      r.th = r.th.map((v, i) => v - P.lrA * sh[i] / (Math.sqrt(rh[i]) + 1e-8));
    }
  }
});
FUNCS.axis = {
  label: 'Axis-aligned valley (0.1x² vs 10y²)', dom: [-3, 3, -3, 3], start: [-2.7, 1.6], lr: 0.15,
  build() { return { f: (x, y) => 0.5 * (0.1 * x * x + 10 * y * y), g: (x, y) => [0.1 * x, 10 * y], H: () => [[0.1, 0], [0, 10]], minima: [[0, 0]], saddles: [], quad: true }; }
};
Object.assign(PRESETS, {
  mom: { funcs: ['axis', 'bowl', 'banana'], methods: ['gd', 'momentum'], on: ['gd', 'momentum'], kappa: 15, lr: 0.15, lrBy: { bowl: 0.1, axis: 0.15, banana: 0.01 }, mom: 0.9 },
  nest: { funcs: ['bowl', 'banana', 'axis'], methods: ['gd', 'momentum', 'nesterov'], on: ['momentum', 'nesterov'], kappa: 15, lr: 0.08, lrBy: { bowl: 0.08, axis: 0.12, banana: 0.008 }, mom: 0.8 },
  adapt: { funcs: ['axis', 'bowl', 'banana'], methods: ['gd', 'adagrad', 'rmsprop', 'adam'], on: ['gd', 'adagrad', 'rmsprop', 'adam'], kappa: 10, lr: 0.15, lrBy: { axis: 0.15, bowl: 0.15, banana: 0.012 }, lrA: 0.1 },
  rms: { funcs: ['banana', 'axis', 'bowl'], methods: ['adagrad', 'rmsprop', 'adam'], on: ['adagrad', 'rmsprop'], kappa: 10, lrA: 0.05 },
  race: { funcs: ['banana', 'bowl', 'axis', 'saddle'], methods: ['gd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam'], on: ['gd', 'momentum', 'rmsprop', 'adam'], kappa: 10, lr: 0.012, lrBy: { banana: 0.012, bowl: 0.12, axis: 0.15, saddle: 0.3 }, mom: 0.9, lrA: 0.05 }
});

/* ---------- 1) Learning-rate schedules for SGD ---------- */
Widgets.lrSchedule = (el) => {
  UI.head(el, 'Why SGD needs a decaying learning rate', 'Minimize J(θ) = ½θ² with noisy gradients (noise std 1, like a minibatch). Constant ε vs. linear decay to ε_τ = 1% of ε₀ by iteration τ. Averaged over 30 runs.');
  let e0 = 0.5, tau = 200;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Initial learning rate ε₀', min: 0.02, max: 1.95, step: 0.01, value: e0, fmt: v => v.toFixed(2), onInput: v => { e0 = v; draw(); } });
  UI.slider(ctr, { label: 'Decay length τ (iterations)', min: 10, max: 500, step: 10, value: tau, fmt: v => String(v), onInput: v => { tau = v; draw(); } });
  el.append(ctr);
  const ch1 = new LineChart(el, { xr: [0, 600], yr: [0, 2], xlabel: 'iteration k', ylabel: 'ε_k', xname: 'k', aspect: .28, maxH: 150, minH: 120, vfmt: v => v.toFixed(4) });
  const ch2 = new LineChart(el, { xr: [0, 600], yr: [1e-4, 30], logY: true, xlabel: 'iteration k', ylabel: 'excess error J − J*', xname: 'k', aspect: .45, maxH: 250 });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const et = 0.01 * e0, K = 600;
    const sched = k => (k < tau ? (1 - k / tau) * e0 + (k / tau) * et : et);
    const runs = 30, cst = new Array(K + 1).fill(0), dec = new Array(K + 1).fill(0);
    for (let r = 0; r < runs; r++) {
      const rng = mulberry32(r + 1);
      let a = 5, b = 5;
      for (let k = 0; k <= K; k++) {
        cst[k] += Math.min(1e6, 0.5 * a * a) / runs; dec[k] += Math.min(1e6, 0.5 * b * b) / runs;
        const n1 = gaussian(rng);
        a -= e0 * (a + n1); b -= sched(k) * (b + n1);
      }
    }
    const ks = Array.from({ length: K + 1 }, (_, k) => k).filter(k => k % 3 === 0);
    ch1.set([{ name: 'constant', color: 's2', data: [[0, e0], [K, e0]] }, { name: 'linear decay', color: 's3', data: ks.map(k => [k, sched(k)]) }], { yr: [0, Math.max(0.1, e0 * 1.1)] });
    ch2.set([{ name: 'constant ε', color: 's2', data: ks.map(k => [k, clamp(cst[k], 1.1e-4, 29)]) }, { name: 'decaying ε', color: 's3', data: ks.map(k => [k, clamp(dec[k], 1.1e-4, 29)]) }]);
    lg.set([{ label: 'constant learning rate', color: c.s2 }, { label: 'linear decay then constant', color: c.s3 }]);
    const half = Math.round(tau / 2);
    calc.innerHTML = `<span class="c">ε_k = (1 − α)ε₀ + αε_τ,  α = k/τ   (8.14);  ε_τ = 0.01·ε₀ = ${et.toFixed(4)};  constant after k = τ</span>
k = ${half}: α = ${(half / tau).toFixed(2)} → ε = ${(1 - half / tau).toFixed(2)}×${e0.toFixed(2)} + ${(half / tau).toFixed(2)}×${et.toFixed(4)} = <b>${sched(half).toFixed(4)}</b>
final excess error (avg of 30 runs): constant <b>${cst[K].toExponential(2)}</b>  vs  decayed <b>${dec[K].toExponential(2)}</b>
<span class="c">${e0 > 1.6 ? 'ε₀ close to 2: violent oscillations at the start.' : 'with a constant ε the noise keeps the iterate bouncing around the minimum forever — its error floor ∝ ε.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) Momentum as physics: which kind of drag? ---------- */
Widgets.dragDemo = (el) => {
  UI.head(el, 'Momentum = a particle with friction', 'Simulate a ball under force −∇J(θ) plus a drag force. The momentum algorithm uses viscous drag (−v). Compare it with other kinds of drag.');
  let scen = 'bowl', cdrag = 0.4;
  const segBox = h('div');
  UI.seg(segBox, [['bowl', 'Ball released in a bowl J = ½θ²'], ['flat', 'Ball pushed on a flat plain (J = 0)']], scen, v => { scen = v; draw(); });
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Drag strength', min: 0.05, max: 2, step: 0.05, value: cdrag, fmt: v => v.toFixed(2), onInput: v => { cdrag = v; draw(); } });
  el.append(segBox, ctr);
  const chart = new LineChart(el, { xr: [0, 30], yr: [-3.5, 3.5], xlabel: 'time t', ylabel: 'position θ(t)', xname: 't', aspect: .5, maxH: 280, vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function sim(kind) {
    let th = scen === 'bowl' ? 3 : 0, v = scen === 'bowl' ? 0 : 2; const dt = 0.01, out = [];
    for (let i = 0; i <= 3000; i++) {
      if (i % 20 === 0) out.push([i * dt, th]);
      const grad = scen === 'bowl' ? th : 0;
      let fdrag = 0;
      if (kind === 'viscous') fdrag = -cdrag * v;
      else if (kind === 'turbulent') fdrag = -cdrag * v * Math.abs(v);
      else if (kind === 'dry') { if (Math.abs(v) > 1e-3) fdrag = -cdrag * Math.sign(v); else if (Math.abs(grad) <= cdrag) { v = 0; continue; } else fdrag = cdrag * Math.sign(grad); }
      v += dt * (-grad + fdrag); th += dt * v;
    }
    return out;
  }
  function draw() {
    const c = Colors.get();
    const kinds = [['none', 'no drag', 'muted'], ['viscous', 'viscous drag −c·v (momentum)', 's3'], ['turbulent', 'turbulent drag −c·v|v|', 's2'], ['dry', 'dry friction −c·sign(v)', 's7']];
    const res = kinds.map(([k]) => sim(k));
    const all = res.flat().map(p => p[1]);
    chart.set(kinds.map(([k, n, col], i) => ({ name: n, color: col, data: res[i], dash: k === 'none' ? [4, 4] : undefined })), { yr: [Math.min(-0.5, ...all) - 0.2, Math.max(0.5, ...all) + 0.2] });
    lg.set(kinds.map(([, n, col]) => ({ label: n, color: c[col], kind: col === 'muted' ? 'dash' : undefined })));
    const fin = res.map(r => r[r.length - 1][1]);
    calc.innerHTML = `<span class="c">Newton: f(t) = ∂²θ/∂t²  →  v(t) = ∂θ/∂t,  f(t) = ∂v/∂t   (8.18–8.20);  forces: −∇J(θ) and drag.  Euler steps give v ← αv − ε∇J, θ ← θ + v.</span>
final position at t = 30:  no drag ${fin[0].toFixed(3)} · viscous <b>${fin[1].toFixed(3)}</b> · turbulent ${fin[2].toFixed(3)} · dry friction ${fin[3].toFixed(3)}
${scen === 'bowl' ? `dry friction stops wherever |∇J| = |θ| ≤ c = ${cdrag.toFixed(2)} — before the minimum; turbulent drag is weak at low speed, so it keeps oscillating; viscous drag settles at 0.` : 'turbulent drag is too weak at low speed — the ball creeps away forever (θ keeps growing); viscous drag stops it after a finite distance v₀/c; dry friction stops it too.'}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) AdaGrad vs RMSProp effective learning rate ---------- */
Widgets.adagradDecay = (el) => {
  UI.head(el, 'AdaGrad’s learning rate only shrinks; RMSProp can recover', 'One parameter sees large gradients for 100 steps (a steep region), then small gradients (a gentle convex bowl). Effective step size per unit gradient: ε / √(accumulated g²).');
  let rho = 0.9;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'RMSProp decay ρ', min: 0.5, max: 0.999, step: 0.001, value: rho, fmt: v => v.toFixed(3), onInput: v => { rho = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [1, 400], yr: [1e-4, 1], logY: true, xlabel: 'step', ylabel: 'actual step size |Δθ|', xname: 'step', aspect: .5, maxH: 270, snapX: x => Math.round(x) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const eps = 0.01, rng = mulberry32(3);
    let ra = 0, rr = 0; const A = [], R = [];
    let at100 = null;
    for (let t = 1; t <= 400; t++) {
      const g = (t <= 100 ? 5 : 0.2) * (1 + 0.3 * gaussian(rng));
      ra += g * g; rr = rho * rr + (1 - rho) * g * g;
      const sa = eps * Math.abs(g) / (1e-7 + Math.sqrt(ra)), sr = eps * Math.abs(g) / Math.sqrt(1e-6 + rr);
      A.push([t, Math.max(1.1e-4, sa)]); R.push([t, Math.max(1.1e-4, sr)]);
      if (t === 300) at100 = { ra, rr, g, sa, sr };
    }
    chart.set([{ name: 'AdaGrad', color: 's7', data: A }, { name: 'RMSProp', color: 's5', data: R }]);
    lg.set([{ label: 'AdaGrad: r ← r + g⊙g', color: c.s7 }, { label: `RMSProp: r ← ρr + (1−ρ)g⊙g, ρ = ${rho}`, color: c.s5 }]);
    const s = at100;
    calc.innerHTML = `<span class="c">update for both: Δθ = −ε/(δ + √r) ⊙ g  (RMSProp: −ε/√(δ + r) ⊙ g),  ε = ${eps}</span>
at step 300 (in the gentle bowl, g ≈ ${s.g.toFixed(3)}):
AdaGrad: r = Σ g² = ${s.ra.toFixed(1)}  →  |Δθ| = 0.01 × ${Math.abs(s.g).toFixed(3)} / ${Math.sqrt(s.ra).toFixed(2)} = <b>${s.sa.toExponential(2)}</b>   (still remembers the steep start)
RMSProp: r = ${s.rr.toFixed(4)}  →  |Δθ| = 0.01 × ${Math.abs(s.g).toFixed(3)} / ${Math.sqrt(s.rr).toFixed(3)} = <b>${s.sr.toExponential(2)}</b>   (old history has faded, ~1/(1−ρ) = ${(1 / (1 - rho)).toFixed(0)} steps)`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) Adam’s bias correction ---------- */
Widgets.adamBias = (el) => {
  UI.head(el, 'Adam’s moment estimates and their bias correction', 'Gradients g_t ≈ 1 with some noise. The moving averages s (1st moment) and r (2nd moment) start at 0, so early on they are biased toward 0; dividing by 1 − βᵗ fixes it.');
  let b1 = 0.9, b2 = 0.999, sn = 0.3;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'β₁ (1st-moment decay)', min: 0.5, max: 0.99, step: 0.01, value: b1, fmt: v => v.toFixed(2), onInput: v => { b1 = v; draw(); } });
  UI.slider(ctr, { label: 'β₂ (2nd-moment decay)', min: 0.9, max: 0.9999, step: 0.0001, value: b2, fmt: v => v.toFixed(4), onInput: v => { b2 = v; draw(); } });
  UI.slider(ctr, { label: 'gradient noise', min: 0, max: 1, step: 0.05, value: sn, fmt: v => v.toFixed(2), onInput: v => { sn = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [1, 60], yr: [0, 1.6], xlabel: 'step t', xname: 't', aspect: .45, maxH: 250, snapX: x => Math.round(x), vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const rng = mulberry32(8);
    let s = 0, r = 0; const S = [], Sh = [], St = [], Sr = [];
    const first = [];
    for (let t = 1; t <= 60; t++) {
      const g = 1 + sn * gaussian(rng);
      s = b1 * s + (1 - b1) * g; r = b2 * r + (1 - b2) * g * g;
      const sh = s / (1 - b1 ** t), rh = r / (1 - b2 ** t);
      S.push([t, s]); Sh.push([t, sh]); St.push([t, sh / Math.sqrt(rh)]); Sr.push([t, Math.min(1.6, s / Math.sqrt(r))]);
      if (t <= 2) first.push({ t, g, s, r, sh, rh });
    }
    chart.set([
      { name: 's (raw 1st moment)', color: 's1', data: S, dash: [5, 4] },
      { name: 'ŝ (corrected)', color: 's1', data: Sh },
      { name: 'step ŝ/√r̂ (Adam)', color: 's8', data: St },
      { name: 'step s/√r (no correction)', color: 's4', data: Sr, dash: [5, 4] }
    ], { yr: [0, Math.max(1.6, ...Sr.map(p => p[1])) * 1.05] });
    lg.set([{ label: 'raw s — starts near 0', color: c.s1, kind: 'dash' }, { label: 'corrected ŝ = s/(1 − β₁ᵗ)', color: c.s1 }, { label: 'Adam step direction ŝ/√r̂', color: c.s8 }, { label: 'without correction s/√r (too big early on)', color: c.s4, kind: 'dash' }]);
    const f = first[0];
    calc.innerHTML = `<span class="c">s ← β₁s + (1−β₁)g     r ← β₂r + (1−β₂)g⊙g     ŝ = s/(1−β₁ᵗ)     r̂ = r/(1−β₂ᵗ)     Δθ = −ε ŝ/(√r̂ + δ)</span>
t = 1, g = ${f.g.toFixed(3)}:  s = ${(1 - b1).toFixed(2)}×${f.g.toFixed(3)} = ${f.s.toFixed(4)},  ŝ = ${f.s.toFixed(4)}/${(1 - b1).toFixed(2)} = <b>${f.sh.toFixed(4)}</b>
          r = ${(1 - b2).toFixed(4)}×${(f.g * f.g).toFixed(3)} = ${f.r.toExponential(2)},  r̂ = <b>${f.rh.toFixed(4)}</b>
first step uses ŝ/√r̂ = ${(f.sh / Math.sqrt(f.rh)).toFixed(3)} (≈ ±1: a step of size ≈ ε).  Without correction: s/√r = ${(f.s / Math.sqrt(f.r)).toFixed(3)} → ${(f.s / Math.sqrt(f.r) / (f.sh / Math.sqrt(f.rh))).toFixed(1)}× too large.`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Breaking symmetry ---------- */
Widgets.symmetryBreak = (el) => {
  UI.head(el, 'Why the initial weights must break symmetry', 'A 2 → 2 → 1 tanh network trained on XOR. If both hidden units start identical, they receive identical gradients and stay identical forever — the network behaves as if it had one hidden unit.');
  let mode = 'same';
  const segBox = h('div');
  UI.seg(segBox, [['same', 'Identical initial weights'], ['rand', 'Small random weights']], mode, v => { mode = v; draw(); });
  el.append(segBox);
  const chart = new LineChart(el, { xr: [0, 3000], yr: [-3, 3], xlabel: 'step', ylabel: 'incoming weight', xname: 'step', aspect: .4, maxH: 220, vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const chart2 = new LineChart(el, { xr: [0, 3000], yr: [1e-4, 1], logY: true, xlabel: 'step', ylabel: 'MSE', xname: 'step', aspect: .3, maxH: 170, minH: 130 });
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const X = [[0, 0], [0, 1], [1, 0], [1, 1]], Y = [0, 1, 1, 0];
    const rng = mulberry32(11);
    let W = mode === 'same' ? [[0.5, 0.5], [0.5, 0.5]] : [[gaussian(rng), gaussian(rng)], [gaussian(rng), gaussian(rng)]].map(r => r.map(v => v * 0.8));
    let bh = mode === 'same' ? [0, 0] : [0.1 * gaussian(rng), 0.1 * gaussian(rng)];
    let w = mode === 'same' ? [0.5, 0.5] : [gaussian(rng) * 0.8, gaussian(rng) * 0.8], b = 0;
    const tr = [[], [], [], []], ls = [];
    for (let t = 0; t <= 3000; t++) {
      let gW = [[0, 0], [0, 0]], gb = [0, 0], gw = [0, 0], gbo = 0, L = 0;
      X.forEach((x, i) => {
        const a = W.map((r, j) => r[0] * x[0] + r[1] * x[1] + bh[j]), hh = a.map(Math.tanh), o = w[0] * hh[0] + w[1] * hh[1] + b, e = o - Y[i];
        L += e * e / 4;
        gbo += e / 2; for (let j = 0; j < 2; j++) { gw[j] += e * hh[j] / 2; const d = e * w[j] * (1 - hh[j] ** 2) / 2; gb[j] += d; gW[j][0] += d * x[0]; gW[j][1] += d * x[1]; }
      });
      if (t % 30 === 0) { tr[0].push([t, W[0][0]]); tr[1].push([t, W[1][0]]); tr[2].push([t, W[0][1]]); tr[3].push([t, W[1][1]]); ls.push([t, Math.max(1.1e-4, L)]); }
      const lr = 0.5;
      for (let j = 0; j < 2; j++) { w[j] -= lr * gw[j]; bh[j] -= lr * gb[j]; W[j][0] -= lr * gW[j][0]; W[j][1] -= lr * gW[j][1]; }
      b -= lr * gbo;
    }
    chart.set([
      { name: 'unit 1, weight from x₁', color: 's1', data: tr[0] }, { name: 'unit 2, weight from x₁', color: 's2', data: tr[1], dash: [6, 4] },
      { name: 'unit 1, weight from x₂', color: 's3', data: tr[2] }, { name: 'unit 2, weight from x₂', color: 's7', data: tr[3], dash: [6, 4] }
    ], { yr: [Math.min(-1, ...tr.flat().map(p => p[1])) - 0.3, Math.max(1, ...tr.flat().map(p => p[1])) + 0.3] });
    lg.set([{ label: 'unit 1 weights (solid)', color: c.s1 }, { label: 'unit 2 weights (dashed)', color: c.s2, kind: 'dash' }]);
    chart2.set([{ name: 'MSE', color: 's8', data: ls }]);
    const diff = Math.hypot(W[0][0] - W[1][0], W[0][1] - W[1][1], bh[0] - bh[1], w[0] - w[1]);
    calc.innerHTML = `<span class="c">identical units ⇒ identical activations ⇒ identical gradients ⇒ identical updates (a deterministic loop that never splits them)</span>
difference between the two hidden units after 3 000 steps: ‖θ_unit1 − θ_unit2‖ = <b>${diff.toExponential(2)}</b>
final MSE = <b>${ls[ls.length - 1][1].toExponential(2)}</b>  ${mode === 'same' ? '→ stuck: one effective unit cannot represent XOR' : '→ the units specialize and XOR is solved'}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 6) Initialization scale through a deep network ---------- */
Widgets.initScale = (el) => {
  UI.head(el, 'Does the signal survive 20 layers?', 'A real random network (width 64, minibatch 32). Bars: std of activations (forward) and of gradients (backward) at each layer, on a log scale. Pick an initialization rule.');
  let scheme = 'glorot', act = 'tanh', gain = 1, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.select(ctr, 'Initialization', [['small', 'N(0, 0.01²) “small random”'], ['unif', 'U(−1/√m, 1/√m)'], ['glorot', 'Normalized (Glorot) U(±√(6/(m+n)))'], ['he', 'He: N(0, 2/m) (for ReLU)']], scheme, v => { scheme = v; draw(); });
  UI.select(ctr, 'Activation', [['tanh', 'tanh'], ['relu', 'ReLU']], act, v => { act = v; draw(); });
  UI.slider(ctr, { label: 'Extra gain factor', min: 0.5, max: 2, step: 0.05, value: gain, fmt: v => v.toFixed(2) + '×', onInput: v => { gain = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [1, 20], yr: [1e-6, 1e3], logY: true, xlabel: 'layer', ylabel: 'std', xname: 'layer', aspect: .5, maxH: 270, snapX: x => Math.round(x), vfmt: v => v.toExponential(2) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const n = 64, B = 32, L = 20, rng = mulberry32(seed * 101);
    const f = act === 'tanh' ? Math.tanh : (z => (z > 0 ? z : 0)), df = act === 'tanh' ? (z => 1 - Math.tanh(z) ** 2) : (z => (z > 0 ? 1 : 0));
    const sampler = () => {
      if (scheme === 'small') return 0.01 * gaussian(rng);
      if (scheme === 'unif') return (2 * rng() - 1) / Math.sqrt(n);
      if (scheme === 'glorot') return (2 * rng() - 1) * Math.sqrt(6 / (n + n));
      return Math.sqrt(2 / n) * gaussian(rng);
    };
    const Ws = Array.from({ length: L }, () => Float64Array.from({ length: n * n }, () => gain * sampler()));
    let hcur = Float64Array.from({ length: B * n }, () => gaussian(rng));
    const hs = [hcur], as = [], fstd = [];
    for (let l = 0; l < L; l++) {
      const a = new Float64Array(B * n), ho = new Float64Array(B * n), W = Ws[l];
      for (let b = 0; b < B; b++) for (let j = 0; j < n; j++) { let s = 0; for (let i = 0; i < n; i++) s += hcur[b * n + i] * W[i * n + j]; a[b * n + j] = s; ho[b * n + j] = f(s); }
      as.push(a); hs.push(ho); hcur = ho;
      fstd.push(Math.sqrt(mean(Array.from(ho, v => v * v))));
    }
    let g = Float64Array.from({ length: B * n }, () => gaussian(rng));
    const bstd = new Array(L);
    for (let l = L - 1; l >= 0; l--) {
      const da = new Float64Array(B * n); for (let q = 0; q < B * n; q++) da[q] = g[q] * df(as[l][q]);
      const gp = new Float64Array(B * n), W = Ws[l];
      for (let b = 0; b < B; b++) for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j < n; j++) s += da[b * n + j] * W[i * n + j]; gp[b * n + i] = s; }
      bstd[l] = Math.sqrt(mean(Array.from(gp, v => v * v))); g = gp;
    }
    const cl = v => clamp(v, 1.1e-6, 900);
    chart.set([{ name: 'activations (forward)', color: 's1', data: fstd.map((v, i) => [i + 1, cl(v)]) }, { name: 'gradients (backward)', color: 's2', data: bstd.map((v, i) => [i + 1, cl(v)]) }]);
    lg.set([{ label: 'std of activations hˡ', color: c.s1 }, { label: 'std of gradients ∂J/∂hˡ⁻¹', color: c.s2 }]);
    const varw = { small: 1e-4, unif: 1 / (3 * n), glorot: 6 / (2 * n) / 3, he: 2 / n }[scheme] * gain * gain;
    calc.innerHTML = `<span class="c">Var(z) ≈ m · Var(w) · E[h²]  — each layer multiplies the signal variance by m·Var(w) (m = fan-in = ${n})</span>
this rule: Var(w) = ${varw.toExponential(3)}  →  m·Var(w) = <b>${(n * varw).toFixed(3)}</b>  ${act === 'relu' ? '(ReLU zeroes half the inputs, so ≈ 2 is neutral)' : '(tanh ≈ identity near 0, so ≈ 1 is neutral)'}
layer 1 → 20: activation std ${fstd[0].toExponential(2)} → <b>${fstd[L - 1].toExponential(2)}</b>;  gradient std ${bstd[L - 1].toExponential(2)} → <b>${bstd[0].toExponential(2)}</b>
<span class="c">heuristics: U(−1/√m, 1/√m) (8.23);  normalized U(−√(6/(m+n)), √(6/(m+n))) (8.24) balances forward and backward variance.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) Initializing biases ---------- */
Widgets.outputBiasInit = (el) => {
  UI.head(el, 'Smart bias initialization', 'Left: an output layer for 3 very unbalanced classes. Setting b so that softmax(b) = class frequencies c gives a much lower starting loss. Right: an LSTM-style gate bias.');
  let p1 = 0.9, p2 = 0.09, bf = 1, T = 20;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Frequency of class 1', min: 0.34, max: 0.98, step: 0.01, value: p1, fmt: v => v.toFixed(2), onInput: v => { p1 = v; draw(); } });
  UI.slider(ctr, { label: 'Share of the rest that is class 2', min: 0.1, max: 0.9, step: 0.01, value: p2 / (1 - p1), fmt: v => v.toFixed(2), onInput: v => { p2 = v * (1 - p1); draw(); } });
  UI.slider(ctr, { label: 'Gate bias b_f', min: -2, max: 3, step: 0.1, value: bf, fmt: v => v.toFixed(1), onInput: v => { bf = v; draw(); } });
  UI.slider(ctr, { label: 'Time steps to remember T', min: 5, max: 100, step: 1, value: T, fmt: v => String(v), onInput: v => { T = v; draw(); } });
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, calc);
  function draw() {
    const c3 = [p1, p2, 1 - p1 - p2];
    const b = c3.map(Math.log);
    const H = -c3.reduce((s, v) => s + v * Math.log(v), 0);
    const sg = sigmoid(bf);
    calc.innerHTML = `<span class="c">OUTPUT BIAS: solve softmax(b) = c  ⇒  b = log c (+ any constant)</span>
c = (${c3.map(v => v.toFixed(3)).join(', ')})  →  b = (${b.map(v => v.toFixed(3)).join(', ')})
initial loss with b = 0:      −log(1/3) = <b>${Math.log(3).toFixed(3)}</b>  (predicts uniform)
initial loss with b = log c:  H(c) = −Σ cᵢ log cᵢ = <b>${H.toFixed(3)}</b>  (already right on average — the weights can focus on the hard part)

<span class="c">GATE BIAS: a gate u = σ(b_f + …) multiplies another unit; with small weights u ≈ σ(b_f)</span>
σ(${bf.toFixed(1)}) = <b>${sg.toFixed(3)}</b>  →  after ${T} steps the remembered fraction ≈ σ(b_f)^${T} = <b>${(sg ** T).toExponential(2)}</b>   (b_f = 0: ${(0.5 ** T).toExponential(2)})
<span class="c">ReLU hidden bias 0.1 keeps units active at the start; variance / precision parameters can simply start at 1.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   Quiz
   ========================================================= */
window.QUIZ = [
  { q: 'Why must SGD’s learning rate decrease over time, while batch GD can keep it fixed?', opts: ['SGD is slower', 'Minibatch sampling adds noise that does not vanish at the minimum; the exact gradient does become 0', 'Batch GD has no learning rate', 'To save memory'], a: 1, why: 'With a constant ε, SGD keeps bouncing around the minimum with an error floor proportional to ε. Sufficient conditions: Σε_k = ∞ and Σε_k² < ∞.' },
  { q: 'In the linear schedule ε_k = (1 − α)ε₀ + αε_τ with α = k/τ, a common choice of ε_τ is…', opts: ['10·ε₀', 'about 1% of ε₀', 'exactly 0', 'ε₀'], a: 1, why: 'τ is typically a few hundred epochs’ worth of iterations; after τ the rate is kept constant at ε_τ ≈ 0.01ε₀.' },
  { q: 'The momentum update is…', opts: ['θ ← θ − εg', 'v ← αv − εg;  θ ← θ + v', 'θ ← θ − ε g/√r', 'v ← g; θ ← v'], a: 1, why: 'v is an exponentially decaying average of past negative gradients; consistent directions accumulate speed, oscillating ones cancel.' },
  { q: 'With momentum α = 0.9 and a constant gradient g, the terminal step size is…', opts: ['ε‖g‖', 'ε‖g‖/(1 − α) = 10·ε‖g‖', '0.9·ε‖g‖', 'unbounded'], a: 1, why: 'Solving v = αv − εg gives ‖v‖ = ε‖g‖/(1 − α): α = 0.9 multiplies the maximum speed by 10.' },
  { q: 'Why does the momentum algorithm correspond to VISCOUS drag (force ∝ −v)?', opts: ['It is the only drag that exists', 'Turbulent drag is too weak at low speed (never stops), dry friction too strong (stops before the minimum); viscous drag is in between', 'It makes the math nonlinear', 'It removes the gradient'], a: 1, why: 'Viscous drag lets the gradient keep the particle moving until a minimum, but stops motion that the gradient does not justify.' },
  { q: 'What is the difference between Nesterov and standard momentum?', opts: ['Nesterov has no velocity', 'Nesterov evaluates the gradient after applying the current velocity (at θ + αv)', 'Nesterov uses second derivatives', 'There is no difference'], a: 1, why: 'It acts as a correction factor. In convex batch settings it improves convergence from O(1/k) to O(1/k²), but not in the stochastic case.' },
  { q: 'AdaGrad’s main weakness for deep networks is…', opts: ['It ignores gradients', 'Accumulating ALL past squared gradients can shrink the learning rate prematurely and excessively', 'It needs the Hessian', 'It only works for one parameter'], a: 1, why: 'RMSProp replaces the sum with an exponentially decaying average, so old history from the steep start is forgotten once it reaches a convex bowl.' },
  { q: 'What does Adam add on top of RMSProp?', opts: ['A line search', 'Momentum as a first-moment estimate, and bias correction of both moment estimates', 'Newton’s method', 'Dropout'], a: 1, why: 's and r start at 0, so early estimates are biased toward 0; dividing by 1 − β₁ᵗ and 1 − β₂ᵗ removes this. Adam is fairly robust to hyper-parameters.' },
  { q: 'Which statement about choosing an optimizer is closest to the lecture?', opts: ['Adam is proven best', 'There is no consensus; SGD, momentum, RMSProp and Adam are all popular, and familiarity with tuning matters a lot', 'Always use AdaGrad', 'Use Newton’s method'], a: 1, why: 'Benchmarks show no single winner across tasks.' },
  { q: 'Why can’t all weights be initialized to the same value?', opts: ['It is too slow', 'Units with the same inputs and the same initial parameters get the same updates and stay identical (no symmetry breaking)', 'It causes overfitting', 'It is allowed for ReLU only'], a: 1, why: 'Random initialization from a high-entropy distribution makes every unit compute a different function from the start.' },
  { q: 'The normalized (Glorot) initialization U(−√(6/(m+n)), √(6/(m+n))) aims to…', opts: ['make all weights equal', 'compromise between equal activation variance and equal gradient variance across layers', 'make the network sparse', 'set biases'], a: 1, why: 'With m inputs and n outputs, it balances the forward signal (depends on m) and the backward signal (depends on n).' },
  { q: 'For an output layer predicting very unbalanced classes with frequencies c, a good initial bias is…', opts: ['b = 0', 'b such that softmax(b) = c, i.e. b = log c', 'b = c', 'b = 1'], a: 1, why: 'Then the untrained network already predicts the marginal class distribution, and the initial loss equals the entropy H(c) instead of log k.' }
];
