/* =========================================================
   Lecture 11 · Ch 10 · Sequence modeling: recurrent and recursive nets
   ========================================================= */
'use strict';

/* ---------- helpers ---------- */
const ns = (v, d = 2) => (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(d)).replace('-', '−');
const np = (v, d = 2) => (v < -1e-12 ? '(' + ns(v, d) + ')' : ns(v, d));
const vec = (a, d = 2) => '[' + a.map(v => ns(v, d)).join(', ') + ']';
function nodeAt(ctx, x, y, r, text, fill, stroke, tcol) {
  dot(ctx, x, y, r, fill, stroke);
  label(ctx, text, x, y, { align: 'center', size: 11, bold: true, color: tcol });
}
function link(ctx, x1, y1, x2, y2, r1, r2, color, w = 1.6, txt, tcol) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  arrow(ctx, x1 + Math.cos(a) * r1, y1 + Math.sin(a) * r1, x2 - Math.cos(a) * (r2 + 1), y2 - Math.sin(a) * (r2 + 1), color, w, 7);
  if (txt) label(ctx, txt, (x1 + x2) / 2 + 9 * Math.sin(a), (y1 + y2) / 2 - 9 * Math.cos(a), { align: 'center', size: 10, bold: true, color: tcol || color });
}
function dataTable(head, rows) {
  return h('div', { class: 'scroll-x' }, h('table', { class: 'data-table' },
    h('thead', {}, h('tr', {}, head.map(t => h('th', {}, t)))),
    h('tbody', {}, rows.map(r => h('tr', r.attrs || {}, (r.cells || r).map(v => h('td', {}, v)))))));
}
const softmax = o => { const m = Math.max(...o), e = o.map(v => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };

/* ---------- 1) Sharing parameters across time ---------- */
Widgets.sharingTime = (el) => {
  UI.head(el, 'Three ways to read a sentence', '<b>Click an output</b> (top row) to see which words can influence it. Same colour = same set of weights.');
  let mode = 'rnn', sent = 0, sel = 5;
  const S = [['I', 'went', 'to', 'Nepal', 'in', '2009'], ['In', '2009', ',', 'I', 'went', 'to', 'Nepal']];
  const segA = h('div', { class: 'toggles' });
  UI.seg(segA, [['dense', 'Own weights per position'], ['conv', '1-D convolution (k = 3)'], ['rnn', 'Recurrent network']], mode, v => { mode = v; draw(); });
  const segB = h('div', { class: 'toggles' });
  UI.seg(segB, [['0', '“I went to Nepal in 2009”'], ['1', '“In 2009, I went to Nepal”']], '0', v => { sent = +v; sel = S[sent].length - 1; draw(); });
  el.append(segA, segB);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'inputs, hidden states and outputs over time' });
  wrap.append(cv); el.append(wrap);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  let geo = null;
  cv.addEventListener('click', e => {
    if (!geo) return;
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left;
    let best = 0, bd = 1e9;
    for (let t = 0; t < geo.T; t++) { const d = Math.abs(px - geo.X(t)); if (d < bd) { bd = d; best = t; } }
    sel = best; draw();
  });
  const COLS = ['s1', 's2', 's3', 's4', 's5', 's7', 's8'];
  function draw() {
    const c = Colors.get(), words = S[sent], T = words.length;
    const W = boxWidth(wrap), H = 220;
    const ctx = fitCanvas(cv, W, H);
    const X = t => 40 + (t + .5) * (W - 60) / T, yo = 30, yh = 110, yx = 190;
    geo = { X, T };
    const infl = t => mode === 'dense' ? words.map((_, s) => s) : mode === 'conv' ? words.map((_, s) => s).filter(s => Math.abs(s - t) <= 1) : words.map((_, s) => s).filter(s => s <= t);
    const on = new Set(infl(sel));
    if (mode === 'rnn') {
      for (let t = 0; t < T; t++) {
        const hot = t <= sel;
        link(ctx, X(t), yx, X(t), yh, 16, 14, hot ? c.s1 : alpha(c.muted, .5), hot ? 2 : 1.2, t === 0 ? 'U' : null);
        link(ctx, X(t), yh, X(t), yo, 14, 11, t === sel ? c.s1 : alpha(c.muted, .5), t === sel ? 2 : 1.2, t === 0 ? 'V' : null);
        if (t > 0) link(ctx, X(t - 1), yh, X(t), yh, 14, 14, t <= sel ? c.s1 : alpha(c.muted, .5), t <= sel ? 2 : 1.2, t === 1 ? 'W' : null);
        nodeAt(ctx, X(t), yh, 14, 'h', hot ? alpha(c.s1, .25) : c.surface2, c.border, c.text);
      }
    } else {
      for (let t = 0; t < T; t++) for (const s of infl(t)) {
        const hot = t === sel;
        const col = mode === 'dense' ? c[COLS[t % COLS.length]] : c.s1;
        ctx.strokeStyle = hot ? col : alpha(mode === 'dense' ? col : c.muted, .22);
        ctx.lineWidth = hot ? 2 : 1;
        ctx.beginPath(); ctx.moveTo(X(s), yx - 14); ctx.lineTo(X(t), yo + 11); ctx.stroke();
      }
    }
    for (let t = 0; t < T; t++) {
      const w = words[t], bw = Math.max(30, w.length * 8 + 12);
      ctx.fillStyle = on.has(t) ? alpha(c.s1, .22) : c.surface2; ctx.strokeStyle = c.border; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(X(t) - bw / 2, yx - 13, bw, 26, 7) : ctx.rect(X(t) - bw / 2, yx - 13, bw, 26); ctx.fill(); ctx.stroke();
      label(ctx, w, X(t), yx, { align: 'center', size: 11, bold: w === '2009', color: c.text });
      nodeAt(ctx, X(t), yo, 11, '', t === sel ? c.text : c.surface2, t === sel ? c.accent : c.border);
    }
    label(ctx, 'outputs', 6, yo, { size: 10, color: c.muted }); label(ctx, 'words', 6, yx + 22, { size: 10, color: c.muted });
    const year = words.indexOf('2009');
    const d = 50, n = 100;
    const info = {
      dense: `${T} different weight sets (one per position): the “year detector” must be learned separately at position ${year + 1} and at every other position
parameters (word vectors of size ${d}, ${n} units per position): ${T} × ${d}·${n} = <b>${(T * d * n).toLocaleString('en-US')}</b>   — and none at all for a position ${T + 1}
output ${sel + 1} can see: every word, but with position-specific rules`,
      conv: `1 weight set (kernel of 3 words) reused at every position — shares parameters, but is SHALLOW
output ${sel + 1} only sees words ${[...on].map(s => s + 1).join(', ')} (its neighbours); a year far from the verb is invisible
parameters: 3 × ${d}·${n} = <b>${(3 * d * n).toLocaleString('en-US')}</b>, for any length`,
      rnn: `1 weight set (U, W, V) applied at every time step by the same update rule h(t) = f(h(t−1), x(t); θ)
output ${sel + 1} sees words 1…${sel + 1} through a chain of depth ${sel + 1} — a very deep, shared computational graph
parameters: U ${d}·${n} + W ${n}·${n} = <b>${(d * n + n * n).toLocaleString('en-US')}</b>, for any length (the year can be found at word ${year + 1} with the same weights)`
    };
    calc.innerHTML = info[mode];
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) Unfolding a recurrence ---------- */
Widgets.unfoldDemo = (el) => {
  UI.head(el, 'Circuit view vs. unfolded view', 'A one-unit RNN: h(t) = tanh(w·h(t−1) + u·x(t)). The circuit has a loop with a one-step delay (■). Unfolding draws one copy of the same cell per time step.');
  let view = 'unfold', T = 5, w = 0.8, u = 1;
  const xs = [1, 0, 0, -1, 0, 0.5, 0, 0];
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['circuit', 'Circuit (recurrent graph)'], ['unfold', 'Unfolded graph']], view, v => { view = v; draw(); });
  el.append(segBox);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Time steps τ', min: 1, max: 8, step: 1, value: T, fmt: v => String(v), onInput: v => { T = v; draw(); } });
  UI.slider(ctr, { label: 'Recurrent weight w', min: -1.5, max: 1.5, step: 0.1, value: w, fmt: v => v.toFixed(1), onInput: v => { w = v; draw(); } });
  UI.slider(ctr, { label: 'Input weight u', min: -2, max: 2, step: 0.1, value: u, fmt: v => v.toFixed(1), onInput: v => { u = v; draw(); } });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'recurrent network drawn as a circuit or unfolded in time' });
  wrap.append(cv);
  const calc = h('div', { class: 'calcbox' });
  el.append(wrap, calc);
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = 200;
    const ctx = fitCanvas(cv, W, H);
    const hs = [0]; for (let t = 1; t <= T; t++) hs.push(Math.tanh(w * hs[t - 1] + u * xs[t - 1]));
    if (view === 'circuit') {
      const cx = W / 2, yh = 95, yx = 170;
      nodeAt(ctx, cx, yx, 17, 'x', c.surface2, c.border, c.text);
      nodeAt(ctx, cx, yh, 17, 'h', alpha(c.s1, .25), c.s1, c.text);
      link(ctx, cx, yx, cx, yh, 17, 17, c.text2, 1.8, 'u');
      ctx.strokeStyle = c.s1; ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(cx + 15, yh - 9); ctx.bezierCurveTo(cx + 80, yh - 70, cx - 80, yh - 70, cx - 15, yh - 9); ctx.stroke();
      arrow(ctx, cx - 22, yh - 20, cx - 15, yh - 10, c.s1, 1.8, 7);
      ctx.fillStyle = c.text; ctx.fillRect(cx - 5, yh - 58, 10, 10);
      label(ctx, 'w (one-step delay ■)', cx + 52, yh - 56, { size: 11, color: c.s1, bold: true });
      label(ctx, 'f(h, x; θ) with θ = (w, u)', cx, 22, { align: 'center', size: 11, color: c.text2 });
    } else {
      const X = t => 36 + t * (W - 60) / Math.max(T, 1), yh = 92, yx = 165;
      nodeAt(ctx, X(0), yh, 15, 'h₀', c.surface2, c.border, c.text);
      label(ctx, '0', X(0), yh + 26, { align: 'center', size: 10 });
      for (let t = 1; t <= T; t++) {
        nodeAt(ctx, X(t), yx, 15, 'x', c.surface2, c.border, c.text);
        label(ctx, ns(xs[t - 1], 1), X(t), yx + 25, { align: 'center', size: 10 });
        link(ctx, X(t), yx, X(t), yh, 15, 16, c.text2, 1.4, t === 1 ? 'u' : null);
        link(ctx, X(t - 1), yh, X(t), yh, t === 1 ? 15 : 16, 16, c.s1, 1.8, 'w');
        nodeAt(ctx, X(t), yh, 16, 'h' + '₁₂₃₄₅₆₇₈'[t - 1], alpha(c.s1, .22), c.s1, c.text);
        label(ctx, hs[t].toFixed(2), X(t), yh - 27, { align: 'center', size: 10, color: c.text });
      }
      label(ctx, 'same w and u at every step', W / 2, 20, { align: 'center', size: 11, color: c.text2 });
    }
    const lines = []; for (let t = 1; t <= Math.min(T, 3); t++) lines.push(`h(${t}) = tanh(${ns(w, 1)}×${np(hs[t - 1])} + ${ns(u, 1)}×${np(xs[t - 1], 1)}) = tanh(${ns(w * hs[t - 1] + u * xs[t - 1])}) = <b>${hs[t].toFixed(3)}</b>`);
    calc.innerHTML = `<span class="c">(10.5): h(t) = f(h(t−1), x(t); θ)   unfolded (10.6–10.7): h(t) = g⁽ᵗ⁾(x(t), …, x(1)) = f(h(t−1), x(t); θ)</span>
${lines.join('\n')}${T > 3 ? '\n…' : ''}
h(${T}) = <b>${ns(hs[T], 3)}</b> summarizes ${T} inputs in ONE number → a lossy summary of the past
<span class="c">${view === 'circuit' ? 'compact: one node per physical component' : `explicit: ${T} copies, but only 2 parameters (w, u) for any τ — same input size at every step`}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) Design patterns (fig 10.3–10.5) ---------- */
Widgets.rnnPatterns = (el) => {
  UI.head(el, 'Three important RNN design patterns', 'x = input, h = hidden state, o = output, L = loss, y = target. U, V, W are the shared weight matrices.');
  let pat = 'hh';
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['hh', 'Output every step, h → h (10.3)'], ['oh', 'Output every step, o → h (10.4)'], ['seq', 'Read all, one output (10.5)']], pat, v => { pat = v; draw(); });
  el.append(segBox);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'unfolded RNN design pattern' });
  wrap.append(cv);
  const tbox = h('div');
  el.append(wrap, tbox);
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = 300;
    const ctx = fitCanvas(cv, W, H);
    const X = t => 50 + t * (W - 80) / 3, R = { x: 270, h: 200, o: 135, L: 78, y: 22 };
    const has = t => pat !== 'seq' || t === 3;
    for (let t = 1; t <= 3; t++) {
      link(ctx, X(t), R.x, X(t), R.h, 14, 15, c.text2, 1.5, 'U');
      if (has(t)) {
        link(ctx, X(t), R.h, X(t), R.o, 15, 14, c.text2, 1.5, 'V');
        link(ctx, X(t), R.o, X(t), R.L, 14, 14, c.text2, 1.4);
        link(ctx, X(t) + 46, R.y, X(t) + 12, R.L - 10, 13, 2, c.muted, 1.2);
      }
      if (pat === 'oh') link(ctx, X(t - 1), R.o, X(t), R.h, 14, 15, c.s2, 2, 'W');
      else link(ctx, X(t - 1), R.h, X(t), R.h, 15, 15, c.s1, 2, 'W');
    }
    nodeAt(ctx, X(0), pat === 'oh' ? R.o : R.h, 13, '…', c.surface2, c.border, c.text);
    for (let t = 1; t <= 3; t++) {
      const sub = ['', '(t−1)', '(t)', '(t+1)'][t].replace(/\(t\)/, '(t)');
      nodeAt(ctx, X(t), R.x, 14, 'x', c.surface2, c.border, c.text);
      label(ctx, sub, X(t) + 26, R.x + 2, { size: 9, color: c.muted });
      nodeAt(ctx, X(t), R.h, 15, 'h', alpha(c.s1, .22), c.s1, c.text);
      if (has(t)) {
        nodeAt(ctx, X(t), R.o, 14, 'o', alpha(c.s3, .25), c.s3, c.text);
        nodeAt(ctx, X(t), R.L, 14, 'L', alpha(c.s8, .2), c.s8, c.text);
        nodeAt(ctx, X(t) + 46, R.y, 13, 'y', c.surface2, c.border, c.text);
      }
    }
    const rows = {
      hh: [['What is recurrent?', 'hidden → hidden (W)'], ['Outputs', 'one per time step'], ['Power', 'universal: can compute anything a Turing machine can (finite size)'], ['Training', 'BPTT, sequential: O(τ) time and memory, no parallelism over time'], ['Example use', 'tag every word, predict the next character']],
      oh: [['What is recurrent?', 'output → next hidden (W) only'], ['Outputs', 'one per time step'], ['Power', 'strictly less powerful: o must carry everything about the past'], ['Training', 'teacher forcing: time steps decouple → parallel, no BPTT needed'], ['Example use', 'when the targets fully describe the state']],
      seq: [['What is recurrent?', 'hidden → hidden (W)'], ['Outputs', 'a single output at the end'], ['Power', 'h(τ) is a fixed-size summary of the whole sequence'], ['Training', 'BPTT from the single loss back through all steps'], ['Example use', 'classify a sentence, summarize a sequence for later processing']]
    }[pat];
    tbox.innerHTML = '';
    tbox.append(dataTable(['Property', 'This pattern'], rows));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) Forward propagation with numbers ---------- */
Widgets.rnnForward = (el) => {
  UI.head(el, 'Forward propagation, step by step (2 hidden units, 3 symbols)', 'Predict the next character of “a b c a” (targets “b c a b”). Press <b>Step</b> to run one time step; <b>click a row</b> to see its arithmetic.');
  const V3 = ['a', 'b', 'c'], xs = [0, 1, 2, 0], ys = [1, 2, 0, 1];
  const U = [[1.0, -1.0, 0.5], [-0.5, 1.0, 1.0]], Wm = [[0.5, -0.3], [0.2, 0.4]], Vm = [[1.0, -1.0], [-1.0, 1.5], [0.5, 0.5]], b = [0, 0], cb = [0, 0, 0];
  let cur = 1, sel = 1;
  const ctr = h('div', { class: 'toggles' });
  UI.button(ctr, 'Step ▶', () => { if (cur < 4) { cur++; sel = cur; draw(); } });
  UI.button(ctr, 'Run all', () => { cur = 4; sel = 4; draw(); });
  UI.button(ctr, 'Reset', () => { cur = 1; sel = 1; draw(); });
  el.append(ctr);
  const tbox = h('div');
  const calc = h('div', { class: 'calcbox' });
  el.append(tbox, calc);
  const steps = []; let hp = [0, 0];
  for (let t = 0; t < 4; t++) {
    const a = [0, 1].map(i => b[i] + Wm[i][0] * hp[0] + Wm[i][1] * hp[1] + U[i][xs[t]]);
    const hh = a.map(Math.tanh);
    const o = [0, 1, 2].map(k => cb[k] + Vm[k][0] * hh[0] + Vm[k][1] * hh[1]);
    const p = softmax(o);
    steps.push({ hp, a, h: hh, o, p, L: -Math.log(p[ys[t]]) });
    hp = hh;
  }
  function draw() {
    tbox.innerHTML = '';
    let tot = 0;
    const rows = steps.slice(0, cur).map((s, t) => {
      tot += s.L;
      return { attrs: { style: 'cursor:pointer' + (t + 1 === sel ? ';background:var(--surface-2)' : ''), onclick: () => { sel = t + 1; draw(); } }, cells: [String(t + 1), V3[xs[t]], vec(s.h), vec(s.p), V3[ys[t]], s.L.toFixed(3)] };
    });
    tbox.append(dataTable(['t', 'x(t)', 'h(t)', 'ŷ(t) = P(a, b, c)', 'target y(t)', 'L(t) = −log ŷ'], rows));
    const s = steps[sel - 1], t = sel - 1;
    calc.innerHTML = `<span class="c">(10.8–10.11):  a = b + W h(t−1) + U x(t),   h = tanh(a),   o = c + V h,   ŷ = softmax(o)</span>
t = ${sel}, input “${V3[xs[t]]}” (one-hot → picks column ${xs[t] + 1} of U), h(${sel - 1}) = ${vec(s.hp, 3)}
a(${sel}) = W h(${sel - 1}) + U[:, ${V3[xs[t]]}] = [${ns(Wm[0][0], 1)}×${np(s.hp[0], 3)} + ${np(Wm[0][1], 1)}×${np(s.hp[1], 3)} + ${np(U[0][xs[t]], 1)},  ${ns(Wm[1][0], 1)}×${np(s.hp[0], 3)} + ${ns(Wm[1][1], 1)}×${np(s.hp[1], 3)} + ${np(U[1][xs[t]], 1)}] = ${vec(s.a, 3)}
h(${sel}) = tanh(a) = <b>${vec(s.h, 3)}</b>
o(${sel}) = V h = ${vec(s.o, 3)}   →   ŷ = e^o / Σe^o = <b>${vec(s.p, 3)}</b>
L(${sel}) = −log ŷ[“${V3[ys[t]]}”] = −log ${s.p[ys[t]].toFixed(3)} = <b>${s.L.toFixed(3)}</b>
total so far (10.12–10.14): L = Σₜ L(t) = <b>${tot.toFixed(3)}</b>     <span class="c">shared parameters: U 2×3, W 2×2, V 3×2, b 2, c 3 = 21 numbers for ANY length</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Teacher forcing vs free running ---------- */
Widgets.teacherForcing = (el) => {
  UI.head(el, 'Teacher forcing: feed the true previous output during training', 'A model predicts the next value from the previous two: ŷ(t) = a₁·y(t−1) + a₂·y(t−2). With teacher forcing the inputs are the TRUE values; when generating freely it must use its OWN predictions.');
  let err = 0.03;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Model error in a₁', min: 0, max: 0.15, step: 0.005, value: err, fmt: v => v.toFixed(3), onInput: v => { err = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 60], yr: [-2, 2], xlabel: 'time t', ylabel: 'value', xname: 't', aspect: .45, maxH: 250, vfmt: v => v.toFixed(3), snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const om = 0.3, T = 60;
  const y = Array.from({ length: T + 1 }, (_, t) => Math.sin(om * t));
  function draw() {
    const c = Colors.get();
    const a1 = 2 * Math.cos(om) + err, a2 = -1;
    const tf = y.map((_, t) => t < 2 ? NaN : a1 * y[t - 1] + a2 * y[t - 2]);
    const fr = [y[0], y[1]]; for (let t = 2; t <= T; t++) fr.push(a1 * fr[t - 1] + a2 * fr[t - 2]);
    const ymax = Math.max(1.3, ...fr.map(Math.abs)) * 1.05;
    chart.set([
      { name: 'true y(t)', color: 'muted', data: y.map((v, t) => [t, v]), dash: [4, 4] },
      { name: 'free running', color: 's2', data: fr.map((v, t) => [t, v]) },
      { name: 'teacher-forced', color: 's1', data: tf.map((v, t) => [t, v]).filter(p => isFinite(p[1])), style: 'dots', r: 3 }
    ], { yr: [-ymax, ymax] });
    lg.set([{ label: 'true sequence', color: c.muted, kind: 'dash' }, { label: 'teacher forcing (inputs = true y)', color: c.s1 }, { label: 'free running (inputs = own predictions)', color: c.s2 }]);
    const e1 = mean(tf.slice(2).map((v, i) => Math.abs(v - y[i + 2])));
    const e2 = mean(fr.slice(40).map((v, i) => Math.abs(v - y[i + 40])));
    calc.innerHTML = `true rule: y(t) = 2cos(0.3)·y(t−1) − y(t−2) = ${(2 * Math.cos(om)).toFixed(3)}·y(t−1) − y(t−2);  model a₁ = ${a1.toFixed(3)}
t = 2 with teacher forcing: ŷ(2) = ${a1.toFixed(3)}×${y[1].toFixed(3)} − ${y[0].toFixed(3)} = ${tf[2].toFixed(3)}   (true ${y[2].toFixed(3)})
average |error|, one step ahead (teacher forced): <b>${e1.toFixed(4)}</b>     free running, steps 40–60: <b>${e2.toFixed(4)}</b>
<span class="c">(10.15–10.16): log p(y(1), y(2) | x(1), x(2)) = log p(y(2) | y(1), x(1), x(2)) + log p(y(1) | x(1), x(2)) — the TRUE y(1) is the input at t = 2,
so every time step's loss can be computed in isolation (parallel training). But at test time the model sees its own outputs: small errors compound.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 6) Back-propagation through time ---------- */
Widgets.bpttDemo = (el) => {
  UI.head(el, 'Back-propagation through time (one hidden unit)', 'a(t) = w·h(t−1) + u·x(t), h(t) = tanh(a(t)), o(t) = v·h(t), L(t) = ½(o(t) − y(t))². The gradient flows backward from the loss, through every time step.');
  let T = 12, w = 0.9, mode = 'last';
  const u = 0.8, v = 1.0;
  const rng = mulberry32(17);
  const XS = Array.from({ length: 30 }, () => rng() * 2 - 1), YS = Array.from({ length: 30 }, (_, t) => 0.5 * Math.sin(0.7 * t));
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['last', 'Loss only at the last step'], ['all', 'Loss at every step']], mode, vv => { mode = vv; draw(); });
  el.append(segBox);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Sequence length τ', min: 3, max: 30, step: 1, value: T, fmt: vv => String(vv), onInput: vv => { T = vv; draw(); } });
  UI.slider(ctr, { label: 'Recurrent weight w', min: -2, max: 2, step: 0.05, value: w, fmt: vv => vv.toFixed(2), onInput: vv => { w = vv; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [1, 30], yr: [1e-6, 10], logY: true, xlabel: 'time step t', ylabel: 'magnitude', xname: 't', aspect: .45, maxH: 240, snapX: x => Math.round(x), hxfmt: vv => String(Math.round(vv)), vfmt: vv => vv.toExponential(2) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function run(ww) {
    const hs = [0], os = [], Ls = [];
    for (let t = 1; t <= T; t++) { hs.push(Math.tanh(ww * hs[t - 1] + u * XS[t - 1])); os.push(v * hs[t]); }
    let L = 0; const go = new Array(T + 1).fill(0);
    for (let t = 1; t <= T; t++) if (mode === 'all' || t === T) { const d = os[t - 1] - YS[t - 1]; L += 0.5 * d * d; go[t] = d; }
    return { hs, go, L };
  }
  function draw() {
    const c = Colors.get();
    const { hs, go, L } = run(w);
    const gh = new Array(T + 2).fill(0);
    for (let t = T; t >= 1; t--) gh[t] = v * go[t] + (t < T ? w * (1 - hs[t + 1] ** 2) * gh[t + 1] : 0);
    const contrib = Array.from({ length: T + 1 }, (_, t) => t === 0 ? 0 : (1 - hs[t] ** 2) * gh[t] * hs[t - 1]);
    const dW = contrib.reduce((a, b) => a + b, 0);
    const dU = Array.from({ length: T }, (_, i) => (1 - hs[i + 1] ** 2) * gh[i + 1] * XS[i]).reduce((a, b) => a + b, 0);
    const eps = 1e-6, num = (run(w + eps).L - run(w - eps).L) / (2 * eps);
    const pts = (arr) => arr.map((val, t) => [t, Math.max(1e-12, Math.abs(val))]).filter(p => p[0] >= 1 && p[0] <= T);
    const g1 = pts(gh.slice(0, T + 1)), g2 = pts(contrib);
    const vals = [...g1, ...g2].map(p => p[1]).filter(x => x > 1e-11);
    const lo = vals.length ? Math.max(1e-9, Math.min(...vals)) : 1e-6, hi = vals.length ? Math.max(...vals) : 1;
    chart.set([{ name: '|∂L/∂h(t)|', color: 's1', data: g1 }, { name: '|contribution of W(t)|', color: 's2', data: g2, style: 'dots', r: 3.5 }],
      { xr: [1, Math.max(T, 3)], yr: [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi) + 1e-9)] });
    lg.set([{ label: '|∂L/∂h(t)| (gradient arriving at step t)', color: c.s1 }, { label: '|(1 − h(t)²)·∂L/∂h(t)·h(t−1)| = share of ∂L/∂w from step t', color: c.s2 }]);
    const t2 = T - 1;
    calc.innerHTML = `start (10.19): ∂L/∂h(${T}) = v·∂L/∂o(${T}) = ${ns(v, 1)}×${np(go[T], 3)} = <b>${ns(gh[T], 4)}</b>     <span class="c">(∂L/∂o(t) = o(t) − y(t) here; with softmax it is ŷ − 1_y (10.18))</span>
${T > 1 ? `one step back (10.21): ∂L/∂h(${t2}) = w·(1 − h(${T})²)·∂L/∂h(${T}) + v·∂L/∂o(${t2}) = ${ns(w, 2)}×${(1 - hs[T] ** 2).toFixed(3)}×${np(gh[T], 4)} + ${np(v * go[t2], 4)} = <b>${ns(gh[t2], 4)}</b>` : ''}
each step back multiplies by w·(1 − h²): here between ${Math.min(...hs.slice(2, T + 1).map(x => Math.abs(w * (1 - x * x)))).toFixed(3)} and ${Math.max(...hs.slice(2, T + 1).map(x => Math.abs(w * (1 - x * x)))).toFixed(3)}   → ∂L/∂h(1) = ${gh[1].toExponential(2)}
∂L/∂w = Σₜ (1 − h(t)²)·∂L/∂h(t)·h(t−1)  (10.26, sum over the copies W(t)) = <b>${ns(dW, 5)}</b>   numerical check: ${ns(num, 5)}
∂L/∂u = Σₜ (1 − h(t)²)·∂L/∂h(t)·x(t) = ${ns(dU, 5)}      loss L = ${L.toFixed(4)}
<span class="c">cost: forward then backward through all ${T} steps — O(τ) time, O(τ) memory to store every h(t), and no parallelism across time</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) Bidirectional RNN: what each output can see ---------- */
Widgets.birnnDemo = (el) => {
  UI.head(el, 'Bidirectional RNN: past and future context', 'How strongly the output at time t can depend on the input at time s, if each recurrent step keeps a fraction α of its memory. Forward RNN: only s ≤ t. Backward RNN: only s ≥ t. Bidirectional: both.');
  let t0 = 8, a = 0.7;
  const T = 15;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Output time t', min: 1, max: T, step: 1, value: t0, fmt: v => String(v), onInput: v => { t0 = v; draw(); } });
  UI.slider(ctr, { label: 'Memory kept per step α', min: 0.3, max: 0.95, step: 0.05, value: a, fmt: v => v.toFixed(2), onInput: v => { a = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [1, T], yr: [0, 2.1], xlabel: 'input time s', ylabel: 'influence on o(t)', xname: 's', aspect: .45, maxH: 240, snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)), vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const ss = Array.from({ length: T }, (_, i) => i + 1);
    const fw = ss.map(s => [s, s <= t0 ? a ** (t0 - s) : 0]), bw = ss.map(s => [s, s >= t0 ? a ** (s - t0) : 0]);
    const bi = ss.map((s, i) => [s, fw[i][1] + bw[i][1]]);
    chart.set([{ name: 'bidirectional', color: 's3', data: bi, width: 3 }, { name: 'forward h(t)', color: 's1', data: fw, style: 'dots', r: 3.5 }, { name: 'backward g(t)', color: 's2', data: bw, style: 'dots', r: 3.5 }]);
    lg.set([{ label: 'bidirectional o(t) = f(h(t), g(t))', color: c.s3 }, { label: 'forward state h(t): x(1) … x(t)', color: c.s1 }, { label: 'backward state g(t): x(t) … x(τ)', color: c.s2 }]);
    calc.innerHTML = `forward only: o(${t0}) sees x(1…${t0}); influence of x(${Math.max(1, t0 - 2)}) ≈ α^${t0 - Math.max(1, t0 - 2)} = ${(a ** (t0 - Math.max(1, t0 - 2))).toFixed(3)};  of x(${Math.min(T, t0 + 2)}) = <b>0</b> (it hasn't happened yet)
bidirectional: also x(${Math.min(T, t0 + 2)}) via g(${t0}): α^${Math.min(T, t0 + 2) - t0} = ${(a ** (Math.min(T, t0 + 2) - t0)).toFixed(3)}
→ most sensitive to inputs around t = ${t0}, without a fixed-size window   <span class="c">(speech: the right phoneme depends on the next few sounds — coarticulation)</span>
<span class="c">2-D extension: four RNNs (up, down, left, right) give each pixel mostly local but potentially long-range context</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 8) Deep RNNs and path lengths ---------- */
Widgets.deepRnnPaths = (el) => {
  UI.head(el, 'Where to add depth in an RNN', 'Three ways to make an RNN deep (figure 10.13). The highlighted path is the shortest route from h(t) to h(t+1).');
  let arch = 'a';
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['a', '(a) Stacked hidden layers'], ['b', '(b) Deep MLP in each block'], ['c', '(c) Deep MLP + skip']], arch, v => { arch = v; draw(); });
  el.append(segBox);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'deep recurrent network architecture' });
  wrap.append(cv);
  const calc = h('div', { class: 'calcbox' });
  el.append(wrap, calc);
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = 260;
    const ctx = fitCanvas(cv, W, H);
    const cx = [W * 0.3, W * 0.72];
    const N = {}, E = [];
    const add = (id, t, y, txt, col) => { N[id + t] = { x: cx[t], y, txt, col }; };
    for (let t = 0; t < 2; t++) {
      add('x', t, 235, 'x', 'surface2');
      add('o', t, 25, 'o', 's3');
      if (arch === 'a') { add('h1', t, 165, 'h¹', 's1'); add('h2', t, 95, 'h²', 's1'); E.push(['x' + t, 'h1' + t], ['h1' + t, 'h2' + t], ['h2' + t, 'o' + t]); }
      else { add('z', t, 185, 'z', 'surface2'); add('h', t, 125, 'h', 's1'); add('m', t, 70, 'm', 'surface2'); E.push(['x' + t, 'z' + t], ['z' + t, 'h' + t], ['h' + t, 'm' + t], ['m' + t, 'o' + t]); }
    }
    let path = [];
    if (arch === 'a') { E.push(['h10', 'h11', 'W'], ['h20', 'h21', 'W']); path = [['h20', 'h21']]; }
    else {
      N.r0 = { x: (cx[0] + cx[1]) / 2, y: 150, txt: 'r', col: 'surface2' };
      E.push(['h0', 'r0'], ['r0', 'h1']);
      path = [['h0', 'r0'], ['r0', 'h1']];
      if (arch === 'c') { E.push(['h0', 'h1', 'skip']); path = [['h0', 'h1']]; }
    }
    const isPath = (a, b) => path.some(([p, q]) => p === a && q === b);
    for (const [a, b, txt] of E) {
      const A = N[a], B = N[b];
      const hot = isPath(a, b);
      link(ctx, A.x, A.y, B.x, B.y, 15, 15, hot ? c.s2 : c.text2, hot ? 2.6 : 1.4, txt, hot ? c.s2 : c.text2);
    }
    for (const id in N) { const n = N[id]; nodeAt(ctx, n.x, n.y, 15, n.txt, n.col === 'surface2' ? c.surface2 : alpha(c[n.col], .25), n.col === 'surface2' ? c.border : c[n.col], c.text); }
    label(ctx, 'time t', cx[0], H - 6, { align: 'center', size: 10, color: c.muted });
    label(ctx, 'time t + 1', cx[1], H - 6, { align: 'center', size: 10, color: c.muted });
    const per = arch === 'b' ? 2 : 1;
    calc.innerHTML = `shortest path h(t) → h(t+1): <b>${per}</b> edge${per > 1 ? 's' : ''};  over k steps: ${per > 1 ? per : ''}k edges
${[1, 5, 10, 20].map(k => `k = ${k}: ${per * k} edge${per * k > 1 ? 's' : ''}, gradient factor if each edge multiplies by 0.9 → 0.9^${per * k} = ${(0.9 ** (per * k)).toExponential(2)}`).join('\n')}
<span class="c">${arch === 'a' ? 'lower layers turn the raw input into a better representation for the higher layers' : arch === 'b' ? 'more capacity in each of the 3 blocks (input→hidden, hidden→hidden, hidden→output), but the transition is twice as deep → harder to optimize' : 'skip connections in the hidden-to-hidden path restore a short path while keeping the extra capacity'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 9) Recursive (tree) vs recurrent (chain) ---------- */
Widgets.treeVsChain = (el) => {
  UI.head(el, 'Recursive nets: a tree instead of a chain', 'Combine n inputs into one vector, two at a time. A chain (RNN) needs depth n − 1; a balanced binary tree needs only ⌈log₂ n⌉.');
  let n = 8;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Sequence length n', min: 2, max: 64, step: 1, value: n, fmt: v => String(v), onInput: v => { n = v; draw(); } });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'chain and tree computation graphs' });
  wrap.append(cv);
  el.append(wrap);
  const chart = new LineChart(el, { xr: [2, 64], yr: [0, 64], xlabel: 'sequence length n', ylabel: 'depth', xname: 'n', aspect: .35, maxH: 180, minH: 140, snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)), vfmt: v => String(v) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = 200;
    const ctx = fitCanvas(cv, W, H);
    const half = W / 2, m = Math.min(n, 16);
    const leafX = (i, x0) => x0 + 14 + i * (half - 34) / Math.max(m - 1, 1);
    // chain
    const lh = (H - 50) / Math.max(m - 1, 1);
    let px = leafX(0, 0), py = H - 18;
    for (let i = 0; i < m; i++) dot(ctx, leafX(i, 0), H - 18, 4.5, c.text2);
    for (let k = 1; k < m; k++) {
      const nx = (px + leafX(k, 0)) / 2 + 4, ny = H - 18 - k * lh;
      ctx.strokeStyle = c.s1; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(nx, ny); ctx.moveTo(leafX(k, 0), H - 18); ctx.lineTo(nx, ny); ctx.stroke();
      dot(ctx, nx, ny, 3.5, c.s1); px = nx; py = ny;
    }
    label(ctx, `chain: depth ${n - 1}${n > 16 ? ' (first 16 shown)' : ''}`, 6, 12, { size: 11, bold: true, color: c.text2 });
    // tree
    let level = Array.from({ length: m }, (_, i) => [leafX(i, half), H - 18]);
    for (const [x, y] of level) dot(ctx, x, y, 4.5, c.text2);
    const depthT = Math.ceil(Math.log2(m));
    const th = (H - 50) / Math.max(depthT, 1);
    let d = 0;
    while (level.length > 1) {
      d++;
      const nxt = [];
      for (let i = 0; i < level.length; i += 2) {
        if (i + 1 >= level.length) { nxt.push(level[i]); continue; }
        const nx = (level[i][0] + level[i + 1][0]) / 2, ny = H - 18 - d * th;
        ctx.strokeStyle = c.s3; ctx.lineWidth = 1.4;
        ctx.beginPath(); ctx.moveTo(level[i][0], level[i][1]); ctx.lineTo(nx, ny); ctx.moveTo(level[i + 1][0], level[i + 1][1]); ctx.lineTo(nx, ny); ctx.stroke();
        dot(ctx, nx, ny, 3.5, c.s3); nxt.push([nx, ny]);
      }
      level = nxt;
    }
    label(ctx, `tree: depth ${Math.ceil(Math.log2(n))}`, half + 6, 12, { size: 11, bold: true, color: c.text2 });
    const ns2 = Array.from({ length: 63 }, (_, i) => i + 2);
    chart.set([{ name: 'chain', color: 's1', data: ns2.map(k => [k, k - 1]) }, { name: 'tree', color: 's3', data: ns2.map(k => [k, Math.ceil(Math.log2(k))]) }, { name: 'you', color: 'text', data: [[n, n - 1], [n, Math.ceil(Math.log2(n))]], style: 'dots', r: 4, noHover: true }]);
    lg.set([{ label: 'chain (recurrent): n − 1', color: c.s1 }, { label: 'balanced tree (recursive): ⌈log₂ n⌉', color: c.s3 }]);
    const dt = Math.ceil(Math.log2(n));
    calc.innerHTML = `n = ${n}: chain depth ${n - 1}, tree depth ⌈log₂ ${n}⌉ = <b>${dt}</b>
if each level multiplies the gradient by 0.8: chain 0.8^${n - 1} = ${(0.8 ** (n - 1)).toExponential(2)}   vs   tree 0.8^${dt} = <b>${(0.8 ** dt).toFixed(3)}</b>
<span class="c">open question: which tree? a fixed balanced tree, a parse tree from a language parser, or a structure the learner discovers itself</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 10) Long-term dependencies ---------- */
Widgets.longTermDemo = (el) => {
  UI.head(el, 'Why long-term dependencies are hard', 'Repeating the same map τ times raises its eigenvalues to the power τ, and composing the same nonlinearity many times gives extremely nonlinear functions.');
  let mode = 'eig', l1 = 0.9, l2 = 1.1, wv = 2.5;
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['eig', 'Powers of eigenvalues'], ['compose', 'Repeated composition']], mode, v => { mode = v; s1.show(mode === 'eig'); s2.show(mode === 'eig'); s3.show(mode === 'compose'); draw(); });
  el.append(segBox);
  const ctr = h('div', { class: 'controls' });
  const s1 = UI.slider(ctr, { label: 'Eigenvalue λ₁', min: 0, max: 1.5, step: 0.01, value: l1, fmt: v => v.toFixed(2), onInput: v => { l1 = v; draw(); } });
  const s2 = UI.slider(ctr, { label: 'Eigenvalue λ₂', min: 0, max: 1.5, step: 0.01, value: l2, fmt: v => v.toFixed(2), onInput: v => { l2 = v; draw(); } });
  const s3 = UI.slider(ctr, { label: 'Weight w in f(x) = tanh(w·x)', min: 0.5, max: 4, step: 0.1, value: wv, fmt: v => v.toFixed(1), onInput: v => { wv = v; draw(); } });
  s3.show(false);
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 60], yr: [1e-4, 1e4], logY: true, xlabel: 't', ylabel: '|λ|ᵗ', xname: 't', aspect: .5, maxH: 260, snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)), vfmt: v => v.toExponential(2) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    if (mode === 'eig') {
      const ts = Array.from({ length: 61 }, (_, t) => t);
      const pw = l => ts.map(t => [t, clamp(l ** t, 1.01e-4, 0.99e4)]);
      chart.set([{ name: 'λ₁ᵗ', color: 's1', data: pw(l1) }, { name: 'λ₂ᵗ', color: 's2', data: pw(l2) }], { xr: [0, 60], yr: [1e-4, 1e4], logY: true, xlabel: 't', ylabel: '|λ|ᵗ', xname: 't' });
      lg.set([{ label: `λ₁ = ${l1.toFixed(2)}`, color: c.s1 }, { label: `λ₂ = ${l2.toFixed(2)}`, color: c.s2 }]);
      calc.innerHTML = `<span class="c">(10.36–10.39): h(t) = Wᵀh(t−1) = (Wᵗ)ᵀ h(0);  W = QΛQᵀ (Q orthogonal)  ⇒  h(t) = Qᵀ Λᵗ Q h(0)</span>
component along eigenvector 1 is multiplied by λ₁ᵗ:  ${l1.toFixed(2)}^20 = ${(l1 ** 20).toExponential(2)},  ${l1.toFixed(2)}^60 = <b>${(l1 ** 60).toExponential(2)}</b>
component along eigenvector 2 is multiplied by λ₂ᵗ:  ${l2.toFixed(2)}^20 = ${(l2 ** 20).toExponential(2)},  ${l2.toFixed(2)}^60 = <b>${(l2 ** 60).toExponential(2)}</b>
|λ| &lt; 1 → vanishes, |λ| &gt; 1 → explodes; only |λ| ≈ 1 survives many steps
<span class="c">a NON-recurrent net with independent weights w(t) (mean 0, variance v) gives Var(Π w(t)) = vⁿ, so choosing v = ⁿ√v* hits any desired variance v* — sharing W removes that freedom</span>`;
    } else {
      const xs = Array.from({ length: 161 }, (_, i) => -2 + i * 0.025);
      const comp = k => xs.map(x => { let z = x; for (let i = 0; i < k; i++) z = Math.tanh(wv * z + 0.4 * Math.sin(2 * z)); return [x, z]; });
      const ks = [1, 2, 3, 5, 10];
      const colors = ['s1', 's3', 's4', 's7', 's2'];
      chart.set(ks.map((k, i) => ({ name: `${k}×`, color: colors[i], data: comp(k) })), { xr: [-2, 2], yr: [-1.1, 1.1], logY: false, xlabel: 'input x', ylabel: 'f(f(…f(x)))', xname: 'x' });
      lg.set(ks.map((k, i) => ({ label: `composed ${k}×`, color: c[colors[i]] })));
      const slope = k => { let z = 0, d = 1; for (let i = 0; i < k; i++) { const a = wv * z + 0.4 * Math.sin(2 * z); d *= (1 - Math.tanh(a) ** 2) * (wv + 0.8 * Math.cos(2 * z)); z = Math.tanh(a); } return d; };
      calc.innerHTML = `f(x) = tanh(w·x + 0.4·sin 2x) with w = ${wv.toFixed(1)}, applied k times (like k time steps with no input)
slope at x = 0 after k compositions (chain rule = product of k derivatives): k = 1: ${slope(1).toFixed(2)},  k = 5: ${slope(5).toExponential(2)},  k = 10: <b>${slope(10).toExponential(2)}</b>
<span class="c">the composed function becomes almost flat (derivative ≈ 0) in most places and extremely steep in a few — the gradient either vanishes or explodes</span>`;
    }
  }
  draw();
  return { redraw: draw };
};

/* ---------- 11) Echo state network: memory of a reservoir ---------- */
Widgets.esnDemo = (el) => {
  UI.head(el, 'Echo state network: fixed random recurrence, learned linear readout', '40 tanh units with random fixed weights, scaled to spectral radius ρ. Only the output weights are trained (linear regression, a convex problem). Task: output the input from d steps ago.');
  let rho = 0.9, d = 5;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Spectral radius ρ', min: 0.1, max: 1.6, step: 0.05, value: rho, fmt: v => v.toFixed(2), onInput: v => { rho = v; draw(); } });
  UI.slider(ctr, { label: 'Delay d to remember', min: 1, max: 30, step: 1, value: d, fmt: v => String(v), onInput: v => { d = v; draw(); } });
  el.append(ctr);
  const ch1 = new LineChart(el, { xr: [300, 360], yr: [-0.6, 0.6], xlabel: 't (test data)', ylabel: 'u(t − d)', xname: 't', aspect: .32, maxH: 170, minH: 130, snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)), vfmt: v => v.toFixed(3) });
  const ch2 = new LineChart(el, { xr: [1, 30], yr: [0, 1], xlabel: 'delay d', ylabel: 'R² on test data', xname: 'd', aspect: .32, maxH: 170, minH: 130, snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)), vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const N = 40, T = 420, W0 = 60, TR = 300;
  const rng = mulberry32(8);
  const G = Array.from({ length: N }, () => Array.from({ length: N }, () => gaussian(rng) / Math.sqrt(N)));
  const Uin = Array.from({ length: N }, () => rng() - 0.5);
  const u = Array.from({ length: T }, () => rng() - 0.5);
  function states() {
    const H = []; let hcur = new Array(N).fill(0);
    for (let t = 0; t < T; t++) {
      const nx = new Array(N);
      for (let i = 0; i < N; i++) { let s = Uin[i] * u[t] * 2; for (let j = 0; j < N; j++) s += rho * G[i][j] * hcur[j]; nx[i] = Math.tanh(s); }
      H.push(nx); hcur = nx;
    }
    return H;
  }
  function draw() {
    const c = Colors.get();
    const H = states();
    const rows = []; for (let t = W0; t < TR; t++) rows.push([1, ...H[t]]);
    const D = N + 1;
    const A = Array.from({ length: D }, () => new Array(D).fill(0));
    rows.forEach(r => { for (let p = 0; p < D; p++) for (let q = 0; q < D; q++) A[p][q] += r[p] * r[q]; });
    for (let p = 1; p < D; p++) A[p][p] += 1e-4 * rows.length;
    const fit = dd => {
      const bvec = new Array(D).fill(0);
      rows.forEach((r, k) => { const y = u[W0 + k - dd]; for (let p = 0; p < D; p++) bvec[p] += r[p] * y; });
      const wv = solveLinear(A.map(r => r.slice()), bvec);
      const pred = [], tgt = [];
      for (let t = TR; t < T; t++) { pred.push(dotv(wv, [1, ...H[t]])); tgt.push(u[t - dd]); }
      const mt = mean(tgt), ssr = tgt.reduce((s, y, i) => s + (y - pred[i]) ** 2, 0), sst = tgt.reduce((s, y) => s + (y - mt) ** 2, 0);
      return { r2: Math.max(0, 1 - ssr / sst), pred, tgt };
    };
    const cur = fit(d);
    const curve = Array.from({ length: 30 }, (_, i) => [i + 1, fit(i + 1).r2]);
    ch1.set([{ name: 'target u(t − d)', color: 'muted', data: cur.tgt.map((v, i) => [TR + i, v]).slice(0, 61), dash: [4, 4] }, { name: 'readout', color: 's1', data: cur.pred.map((v, i) => [TR + i, v]).slice(0, 61) }]);
    ch2.set([{ name: 'R²', color: 's3', data: curve }, { name: 'this d', color: 'text', data: [[d, cur.r2]], style: 'dots', r: 5, noHover: true }]);
    lg.set([{ label: 'target: the input d steps ago', color: c.muted, kind: 'dash' }, { label: 'linear readout of the reservoir state', color: c.s1 }, { label: 'memory curve: R² for each delay', color: c.s3 }]);
    const cap = curve.reduce((s, p) => s + p[1], 0);
    calc.innerHTML = `h(t) = tanh(ρ·W̃ h(t−1) + U u(t))   (W̃ random, spectral radius ≈ 1 → W has spectral radius ≈ ρ = ${rho.toFixed(2)})
readout ŷ(t) = w·[1, h(t)] fitted by least squares on t = ${W0}…${TR - 1}, tested on t = ${TR}…${T - 1}
delay ${d}: R² = <b>${cur.r2.toFixed(3)}</b>      total memory capacity Σ_d R² = <b>${cap.toFixed(2)}</b> (out of at most ${N})
<span class="c">${rho < 0.5 ? 'small ρ: the state forgets within a few steps' : rho > 1.2 ? 'large ρ: the dynamics become chaotic-ish and saturate; precise memory degrades' : 'near the edge of stability (ρ ≈ 1): the richest memory of past inputs'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 12) Leaky units & skip connections ---------- */
Widgets.leakyDemo = (el) => {
  UI.head(el, 'Leaky units: running averages with different time scales', 'μ(t) ← α·μ(t−1) + (1 − α)·v(t). The input has a pulse at t = 5 and small noise. α near 1 = long memory, α near 0 = short memory.');
  let a = 0.9, dly = 4, T = 40;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Your unit’s α', min: 0, max: 0.99, step: 0.01, value: a, fmt: v => v.toFixed(2), onInput: v => { a = v; draw(); } });
  UI.slider(ctr, { label: 'Skip-connection delay d', min: 1, max: 10, step: 1, value: dly, fmt: v => String(v), onInput: v => { dly = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 60], yr: [-0.2, 1.05], xlabel: 'time t', ylabel: 'μ(t)', xname: 't', aspect: .45, maxH: 240, snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)), vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const rng = mulberry32(4);
  const v = Array.from({ length: 61 }, (_, t) => (t === 5 ? 1 : 0) + 0.08 * gaussian(rng));
  const run = al => { const m = [0]; for (let t = 1; t <= 60; t++) m.push(al * m[t - 1] + (1 - al) * v[t]); return m; };
  function draw() {
    const c = Colors.get();
    const series = [{ name: 'input v(t)', color: 'muted', data: v.map((x, t) => [t, x]), style: 'dots', r: 2.5 },
      { name: 'α = 0.5', color: 's2', data: run(0.5).map((x, t) => [t, x]) },
      { name: `α = ${a.toFixed(2)}`, color: 's1', data: run(a).map((x, t) => [t, x]), width: 3 }];
    const mx = Math.max(...series.flatMap(s => s.data.map(p => p[1])));
    chart.set(series, { yr: [-0.25, Math.max(0.3, mx * 1.08)] });
    lg.set([{ label: 'input v(t)', color: c.muted }, { label: 'α = 0.5 (forgets fast)', color: c.s2 }, { label: `your unit, α = ${a.toFixed(2)}`, color: c.s1 }]);
    const ma = run(a);
    calc.innerHTML = `μ(5) = α·μ(4) + (1 − α)·v(5) = ${a.toFixed(2)}×${np(ma[4], 3)} + ${(1 - a).toFixed(2)}×${np(v[5], 3)} = <b>${ma[5].toFixed(3)}</b>
the pulse's trace k steps later ≈ (1 − α)·αᵏ:  k = 10 → ${((1 - a) * a ** 10).toExponential(2)},  k = 30 → ${((1 - a) * a ** 30).toExponential(2)}      time constant ≈ 1/(1 − α) = <b>${a >= 0.995 ? '∞' : (1 / (1 - a)).toFixed(1)}</b> steps
skip connections with delay d = ${dly}: a dependency ${T} steps back crosses ⌈${T}/${dly}⌉ = ${Math.ceil(T / dly)} delayed edges instead of ${T}
→ with factor 0.9 per edge: 0.9^${T} = ${(0.9 ** T).toExponential(2)}  vs  0.9^${Math.ceil(T / dly)} = <b>${(0.9 ** Math.ceil(T / dly)).toExponential(2)}</b>
<span class="c">α is a real number (adjusted smoothly) — a skip length is an integer; both give paths whose derivative product stays near 1</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 13) An LSTM cell storing and recalling a value ---------- */
Widgets.lstmCell = (el) => {
  UI.head(el, 'One LSTM cell remembering a value', 'At t = 3 a “store” signal opens the input gate; at t = 12 a “recall” signal opens the output gate. In between, distractor values arrive. <b>Click a row</b> for its arithmetic.');
  let bf = 3, vstore = 0.8, noGate = false, sel = 3;
  const T = 12, vals = [0.3, 0, -0.6, 0.9, -0.2, 0.5, -0.8, 0.1, 0.7, -0.4, 0.2, 0];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Forget-gate bias b_f', min: -2, max: 6, step: 0.25, value: bf, fmt: v => v.toFixed(2), onInput: v => { bf = v; draw(); } });
  UI.slider(ctr, { label: 'Value to store', min: -1, max: 1, step: 0.05, value: vstore, fmt: v => v.toFixed(2), onInput: v => { vstore = v; draw(); } });
  el.append(ctr);
  const tg = h('div', { class: 'toggles' });
  UI.chip(tg, 'Remove the input gate (always open)', '--s2', false, v => { noGate = v; draw(); });
  el.append(tg);
  const tbox = h('div');
  const chart = new LineChart(el, { xr: [1, T], yr: [-1, 1], xlabel: 't', ylabel: 'value', xname: 't', aspect: .32, maxH: 170, minH: 130, snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)), vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(tbox, calc);
  el.insertBefore(tbox, chart.wrap);
  function run() {
    const out = []; let s = 0;
    for (let t = 1; t <= T; t++) {
      const v = t === 3 ? vstore : vals[t - 1], st = t === 3 ? 1 : 0, rc = t === T ? 1 : 0;
      const f = sigmoid(bf - 10 * st), g = noGate ? 1 : sigmoid(-6 + 12 * st), cand = Math.tanh(2 * v), q = sigmoid(-6 + 12 * rc);
      const sp = s; s = f * s + g * cand;
      out.push({ t, v, st, rc, f, g, cand, sp, s, q, h: Math.tanh(s) * q });
    }
    return out;
  }
  function draw() {
    const c = Colors.get(), R = run();
    tbox.innerHTML = '';
    tbox.append(dataTable(['t', 'input v', 'flags', 'forget f', 'input gate g', 'candidate', 'state s', 'output gate q', 'output h'],
      R.map(r => ({ attrs: { style: 'cursor:pointer' + (r.t === sel ? ';background:var(--surface-2)' : ''), onclick: () => { sel = r.t; draw(); } }, cells: [String(r.t), ns(r.v), r.st ? 'store' : r.rc ? 'recall' : '', r.f.toFixed(3), r.g.toFixed(3), r.cand.toFixed(3), r.s.toFixed(3), r.q.toFixed(3), r.h.toFixed(3)] }))));
    chart.set([{ name: 'state s(t)', color: 's1', data: R.map(r => [r.t, r.s]) }, { name: 'output h(t)', color: 's3', data: R.map(r => [r.t, r.h]), style: 'dots', r: 3.5 }, { name: 'stored target', color: 'muted', data: [[1, Math.tanh(2 * vstore)], [T, Math.tanh(2 * vstore)]], dash: [4, 4] }]);
    lg.set([{ label: 'cell state s(t)', color: c.s1 }, { label: 'output h(t)', color: c.s3 }, { label: 'value written at t = 3', color: c.muted, kind: 'dash' }]);
    const r = R[sel - 1];
    const prodF = R.slice(3).reduce((p, x) => p * x.f, 1);
    calc.innerHTML = `t = ${sel}:  forget f = σ(b_f + …) = σ(${ns(bf - 10 * r.st)}) = ${r.f.toFixed(3)}    input gate g = ${noGate ? '1 (removed)' : `σ(${ns(-6 + 12 * r.st)}) = ${r.g.toFixed(3)}`}    output gate q = σ(${ns(-6 + 12 * r.rc)}) = ${r.q.toFixed(3)}
state (10.41):  s(t) = f·s(t−1) + g·candidate = ${r.f.toFixed(3)}×${np(r.sp, 3)} + ${r.g.toFixed(3)}×${np(r.cand, 3)} = <b>${r.s.toFixed(3)}</b>
output (10.43): h(t) = tanh(s(t))·q = tanh(${np(r.s, 3)})×${r.q.toFixed(3)} = <b>${r.h.toFixed(3)}</b>
memory from t = 3 to t = 12 is multiplied by Π f = σ(b_f)⁹ = <b>${prodF.toFixed(3)}</b>   (∂s(12)/∂s(3) = Π f: no tanh squashing on this path)
recalled ${R[T - 1].h.toFixed(3)} vs ideal tanh(written) = ${Math.tanh(Math.tanh(2 * vstore)).toFixed(3)}  <span class="c">(candidate uses tanh here; the book writes σ — same idea)</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 14) GRU update ---------- */
Widgets.gruUpdate = (el) => {
  UI.head(el, 'GRU: one update gate decides copy vs. replace', 'h(t) = u·h(t−1) + (1 − u)·h̃, with the “target state” h̃ = σ(b + U·x + W·(r·h(t−1))). Here u and r are set by hand to see their effect.');
  let u = 0.8, r = 1, hp = 0.6, x = 0.5;
  const Wg = 2, Ug = 1.5, bg = -0.5;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Update gate u', min: 0, max: 1, step: 0.01, value: u, fmt: v => v.toFixed(2), onInput: v => { u = v; draw(); } });
  UI.slider(ctr, { label: 'Reset gate r', min: 0, max: 1, step: 0.01, value: r, fmt: v => v.toFixed(2), onInput: v => { r = v; draw(); } });
  UI.slider(ctr, { label: 'Previous state h(t−1)', min: 0, max: 1, step: 0.01, value: hp, fmt: v => v.toFixed(2), onInput: v => { hp = v; draw(); } });
  UI.slider(ctr, { label: 'Input x(t)', min: -1, max: 1, step: 0.05, value: x, fmt: v => v.toFixed(2), onInput: v => { x = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 40], yr: [0, 1], xlabel: 'time t', ylabel: 'h(t)', xname: 't', aspect: .4, maxH: 220, snapX: x2 => Math.round(x2), hxfmt: v => String(Math.round(v)), vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const xin = Array.from({ length: 41 }, (_, t) => (t >= 5 && t < 8 ? 1.5 : t >= 22 && t < 25 ? -1.5 : 0));
  const run = (uu, rr) => { const hs = [0.2]; for (let t = 1; t <= 40; t++) { const ht = sigmoid(bg + Ug * xin[t] + Wg * rr * hs[t - 1]); hs.push(uu * hs[t - 1] + (1 - uu) * ht); } return hs; };
  function draw() {
    const c = Colors.get();
    const ht = sigmoid(bg + Ug * x + Wg * r * hp), hn = u * hp + (1 - u) * ht;
    chart.set([{ name: 'u = 0 (always replace)', color: 's2', data: run(0, r).map((v, t) => [t, v]) }, { name: `u = ${u.toFixed(2)}`, color: 's1', data: run(u, r).map((v, t) => [t, v]), width: 3 }]);
    lg.set([{ label: 'u = 0: no memory, follows the input', color: c.s2 }, { label: `your u = ${u.toFixed(2)} (input pulses at t = 5 and 22)`, color: c.s1 }]);
    calc.innerHTML = `<span class="c">(10.45–10.47): h(t) = u·h(t−1) + (1 − u)·σ(b + U x(t) + W (r ⊙ h(t−1)));   u = σ(bᵘ + Uᵘx + Wᵘh),  r = σ(bʳ + Uʳx + Wʳh)</span>
target h̃ = σ(${ns(bg, 1)} + ${ns(Ug, 1)}×${np(x)} + ${ns(Wg, 1)}×(${r.toFixed(2)}×${hp.toFixed(2)})) = σ(${(bg + Ug * x + Wg * r * hp).toFixed(3)}) = <b>${ht.toFixed(3)}</b>
new state h(t) = ${u.toFixed(2)}×${hp.toFixed(2)} + ${(1 - u).toFixed(2)}×${ht.toFixed(3)} = <b>${hn.toFixed(3)}</b>
<span class="c">u = 1: copy the old state (perfect memory) · u = 0: replace it by the target · r = 0: ignore the past when computing the target (a fresh start)</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 15) Gradient clipping ---------- */
Widgets.clipDemo = (el) => {
  UI.head(el, 'Clipping an exploding gradient', 'Norm clipping rescales the whole vector (direction kept). Element-wise clipping cuts each component separately (direction can change). Circle = threshold v.');
  let gx = 12, gy = 3, v = 5;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Gradient g₁', min: -20, max: 20, step: 0.5, value: gx, fmt: z => z.toFixed(1), onInput: z => { gx = z; draw(); } });
  UI.slider(ctr, { label: 'Gradient g₂', min: -20, max: 20, step: 0.5, value: gy, fmt: z => z.toFixed(1), onInput: z => { gy = z; draw(); } });
  UI.slider(ctr, { label: 'Threshold v', min: 0.5, max: 15, step: 0.5, value: v, fmt: z => z.toFixed(1), onInput: z => { v = z; draw(); } });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'gradient vector before and after clipping' });
  wrap.append(cv);
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.insertBefore(wrap, lg.el);
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = Math.min(300, Math.round(W * 0.62));
    const ctx = fitCanvas(cv, W, H);
    const nrm = Math.hypot(gx, gy), sc = nrm > v ? v / nrm : 1;
    const gn = [gx * sc, gy * sc], ge = [clamp(gx, -v, v), clamp(gy, -v, v)];
    const R = Math.max(nrm, v) * 1.12 || 1;
    const p = makePlot(W, H, [-R * W / H, R * W / H], [-R, R], { l: 10, r: 10, t: 10, b: 10 });
    ctx.strokeStyle = c.grid; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(p.X(p.xr[0]), p.Y(0)); ctx.lineTo(p.X(p.xr[1]), p.Y(0)); ctx.moveTo(p.X(0), p.Y(-R)); ctx.lineTo(p.X(0), p.Y(R)); ctx.stroke();
    ctx.strokeStyle = c.muted; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.arc(p.X(0), p.Y(0), Math.abs(p.Y(v) - p.Y(0)), 0, Math.PI * 2); ctx.stroke();
    ctx.strokeRect(p.X(-v), p.Y(v), p.X(v) - p.X(-v), p.Y(-v) - p.Y(v));
    ctx.setLineDash([]);
    arrow(ctx, p.X(0), p.Y(0), p.X(gx), p.Y(gy), c.text2, 2, 9);
    arrow(ctx, p.X(0), p.Y(0), p.X(ge[0]), p.Y(ge[1]), c.s2, 3, 10);
    arrow(ctx, p.X(0), p.Y(0), p.X(gn[0]), p.Y(gn[1]), c.s1, 3, 10);
    lg.set([{ label: 'original g', color: c.text2 }, { label: 'norm-clipped (same direction)', color: c.s1 }, { label: 'element-wise clipped (box)', color: c.s2 }]);
    const ang = a => Math.atan2(a[1], a[0]) * 180 / Math.PI;
    const dAng = Math.abs(((ang(ge) - ang([gx, gy]) + 540) % 360) - 180);
    calc.innerHTML = `‖g‖ = √(${np(gx, 1)}² + ${np(gy, 1)}²) = <b>${nrm.toFixed(3)}</b> ${nrm > v ? '&gt;' : '≤'} v = ${v.toFixed(1)}
(10.48–10.49) if ‖g‖ &gt; v:  g ← g·v/‖g‖ = [${ns(gx, 1)}, ${ns(gy, 1)}] × ${sc.toFixed(3)} = <b>[${gn[0].toFixed(3)}, ${gn[1].toFixed(3)}]</b>  (direction unchanged)
element-wise: clip each to [−v, v] → <b>[${ge[0].toFixed(2)}, ${ge[1].toFixed(2)}]</b>, direction turned by ${dAng.toFixed(1)}°
<span class="c">both work similarly in practice; if the gradient is Inf/NaN, even a random step of size v moves away from the unstable region. Clipping fixes EXPLODING, not vanishing gradients.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 16) Regularizing to encourage information flow ---------- */
Widgets.infoFlow = (el) => {
  UI.head(el, 'Keep the back-propagated gradient’s size constant', 'If every step back multiplies the gradient norm by j, the gradient from a loss at the end of a 30-step sequence decays (j &lt; 1) or explodes (j &gt; 1). The regularizer Ω pushes each ratio towards 1.');
  let j = 0.8, timer = null;
  const T = 30;
  const ctr = h('div', { class: 'controls' });
  const jS = UI.slider(ctr, { label: 'Per-step factor ‖(∇ₕL)·∂h(t)/∂h(t−1)‖ / ‖∇ₕL‖', min: 0.5, max: 1.3, step: 0.01, value: j, fmt: v => v.toFixed(2), onInput: v => { j = v; draw(); } });
  UI.button(ctr, 'Apply the regularizer (gradient steps on Ω)', () => {
    if (timer) clearInterval(timer);
    let k = 0;
    timer = setInterval(() => {
      if (!el.isConnected || k++ > 40) { clearInterval(timer); timer = null; return; }
      j = j - 0.002 * 2 * (T - 1) * (j - 1); jS.set(j); draw();
    }, 60);
  });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [1, T], yr: [1e-3, 1e3], logY: true, xlabel: 'time step t (loss at t = 30)', ylabel: '‖∇h(t) L‖', xname: 't', aspect: .45, maxH: 230, snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)), vfmt: v => v.toExponential(2) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const ts = Array.from({ length: T }, (_, i) => i + 1);
    chart.set([{ name: 'ideal (ratio 1)', color: 'muted', data: ts.map(t => [t, 1]), dash: [4, 4] }, { name: 'gradient norm', color: 's1', data: ts.map(t => [t, clamp(j ** (T - t), 1.01e-3, 0.99e3)]) }]);
    lg.set([{ label: `gradient norm with factor ${j.toFixed(3)}`, color: c.s1 }, { label: 'constant norm (what Ω asks for)', color: c.muted, kind: 'dash' }]);
    const om = (T - 1) * (j - 1) ** 2;
    calc.innerHTML = `(10.50) Ω = Σₜ ( ‖(∇_{h(t)}L)·∂h(t)/∂h(t−1)‖ / ‖∇_{h(t)}L‖ − 1 )² = ${T - 1} × (${j.toFixed(3)} − 1)² = <b>${om.toFixed(4)}</b>
gradient reaching t = 1: ${j.toFixed(3)}^${T - 1} = <b>${(j ** (T - 1)).toExponential(2)}</b> of the gradient at t = 30
<span class="c">trick: treat the back-propagated vectors ∇_{h(t)}L as constants when differentiating Ω. Works best with gradient clipping (it keeps the dynamics on the edge of explosion); less effective than LSTMs when data is abundant</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- quiz ---------- */
window.QUIZ = [
  { q: 'What key idea lets a recurrent network handle sequences of any length?', opts: ['A separate weight matrix for every time step', 'Sharing the same parameters across time steps (the same update rule at every step)', 'Using only convolution', 'Padding all sequences to the same length'], a: 1, why: 'The same transition function f with the same θ is applied at every step, so the model has the same input size and parameters regardless of length.' },
  { q: 'Why is the hidden state h(t) called a “lossy summary” of the past?', opts: ['Because it is computed with dropout', 'It maps an arbitrary-length history into a fixed-length vector, so some information must be lost', 'Because the loss function is applied to it', 'Because it is always zero'], a: 1, why: 'A fixed number of units cannot store an arbitrarily long sequence exactly; training decides which aspects to keep.' },
  { q: 'In the forward equations, which matrix connects h(t−1) to h(t)?', opts: ['U', 'V', 'W', 'c'], a: 2, why: 'a(t) = b + W h(t−1) + U x(t). U: input-to-hidden, W: hidden-to-hidden, V: hidden-to-output.' },
  { q: 'Why can’t back-propagation through time be parallelized over time steps?', opts: ['Because tanh is slow', 'The forward pass is inherently sequential: each step needs the previous state; all states must be stored (O(τ) memory)', 'Because the loss is a sum', 'It can be fully parallelized'], a: 1, why: 'Runtime O(τ) cannot be reduced by parallelization, and memory is O(τ) because states are reused in the backward pass.' },
  { q: 'What is teacher forcing?', opts: ['Training with a larger learning rate', 'During training, feeding the ground-truth output y(t) as the input at time t + 1 instead of the model’s own output', 'Clipping gradients', 'Freezing the recurrent weights'], a: 1, why: 'It follows from maximum likelihood for models with output-to-hidden recurrence and decouples the time steps, allowing parallel training.' },
  { q: 'In BPTT, the gradient for the shared weight W is…', opts: ['the gradient at the last step only', 'the sum over time steps of the contributions of the copies W(t)', 'the average of the inputs', 'always zero'], a: 1, why: '∇_W L = Σₜ diag(1 − h(t)²)(∇_{h(t)} L) h(t−1)ᵀ — each time step contributes because W is used at every step.' },
  { q: 'Why use a bidirectional RNN for speech recognition?', opts: ['It is cheaper', 'The correct label at time t may depend on future inputs (coarticulation), and a backward RNN supplies that context', 'It avoids back-propagation', 'It has fewer parameters'], a: 1, why: 'o(t) combines the forward state (past) and the backward state (future), being most sensitive to inputs around t.' },
  { q: 'For a sequence of length 64, a balanced recursive (tree) network has depth about…', opts: ['64', '32', '6', '1'], a: 2, why: 'log₂ 64 = 6, compared with 63 for a chain — shorter paths help with long-term dependencies.' },
  { q: 'Repeating h(t) = Wᵀ h(t−1) with an eigenvalue λ = 0.9 for 50 steps multiplies that component by about…', opts: ['0.45', '0.005', '45', '1'], a: 1, why: '0.9⁵⁰ ≈ 0.005: components with |λ| < 1 vanish, with |λ| > 1 explode.' },
  { q: 'What do echo state networks learn?', opts: ['All weights by BPTT', 'Only the output weights; the recurrent and input weights are fixed (set near the edge of stability)', 'Only the recurrent weights', 'Nothing'], a: 1, why: 'Reservoir computing: a fixed random dynamical system plus a linear readout, so training is convex.' },
  { q: 'What does the LSTM forget gate control?', opts: ['The learning rate', 'The weight of the cell state’s self-loop: how much of s(t−1) is kept', 'The number of hidden units', 'The output vocabulary'], a: 1, why: 's(t) = f(t) s(t−1) + g(t)·(input). With f ≈ 1 information and gradients survive many steps; a forget-gate bias of 1 is a crucial trick.' },
  { q: 'Gradient norm clipping with threshold v = 5 applied to g = (6, 8) gives…', opts: ['(5, 5)', '(3, 4)', '(6, 8)', '(0.6, 0.8)'], a: 1, why: '‖g‖ = 10 > 5, so g ← g·5/10 = (3, 4): same direction, norm 5. Element-wise clipping would give (5, 5).' }
];
