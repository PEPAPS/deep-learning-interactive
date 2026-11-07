/* =========================================================
   Lecture 10 · Ch 9 · Convolutional networks
   ========================================================= */
'use strict';

/* ---------- small helpers for number grids ---------- */
function shadeVal(v, scale) {
  const c = Colors.get();
  if (!scale || v == null || !isFinite(v) || Math.abs(v) < 1e-12) return '';
  const t = clamp(Math.abs(v) / scale, 0, 1);
  return alpha(v > 0 ? c.s1 : c.s2, 0.12 + 0.5 * t);
}
function numStr(v, d = 1) {
  if (v == null) return '';
  if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v)).replace('-', '−');
  return v.toFixed(d).replace('-', '−');
}
function cellGrid(M, o = {}) {
  const cols = M[0].length, cs = o.cs || 26;
  const g = h('div', { class: 'cgrid' + (o.onClick ? ' click' : ''), style: `grid-template-columns:repeat(${cols}, ${cs}px);--cs:${cs}px` });
  const flat = M.flat().filter(v => v != null && isFinite(v));
  const scale = o.scale ?? Math.max(1e-9, ...flat.map(Math.abs));
  M.forEach((row, i) => row.forEach((v, j) => {
    const cls = ['cell'];
    const isPad = o.pad && o.pad(i, j);
    if (isPad) cls.push('pad');
    if (o.hl && o.hl(i, j)) cls.push('hl');
    if (o.hl2 && o.hl2(i, j)) cls.push('hl2');
    if (o.dim && o.dim(i, j)) cls.push('dim');
    const cell = h('div', { class: cls.join(' '), title: o.title ? o.title(i, j, v) : null }, v == null ? '' : (o.fmt ? o.fmt(v, i, j) : numStr(v)));
    if (!isPad) { const bg = o.color ? o.color(i, j, v) : shadeVal(v, scale); if (bg) cell.style.background = bg; }
    if (o.onClick) cell.addEventListener('click', () => o.onClick(i, j));
    g.append(cell);
  }));
  return g;
}
const pn = (v, d = 1) => (v < -1e-12 ? '(' + numStr(v, d) + ')' : numStr(v, d)); // bracket negatives in products
const gwrap = (cap, node) => h('div', { class: 'gwrap' }, h('div', { class: 'gcap' }, cap), node);
const gop = s => h('div', { class: 'gop' }, s);
const zeros2 = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));

/* the small test image used on several slides: a filled block and a diagonal stroke */
const IMG8 = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 2, 2, 2, 0, 0, 0, 0],
  [0, 2, 2, 2, 0, 0, 1, 0],
  [0, 2, 2, 2, 0, 1, 0, 0],
  [0, 0, 0, 0, 1, 0, 0, 0],
  [0, 0, 0, 1, 0, 0, 0, 0],
  [0, 0, 1, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0]
];
const KERNELS = {
  vedge: { name: 'Vertical edges', K: [[1, 0, -1], [1, 0, -1], [1, 0, -1]] },
  hedge: { name: 'Horizontal edges', K: [[1, 1, 1], [0, 0, 0], [-1, -1, -1]] },
  blur: { name: 'Blur (average)', K: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map(r => r.map(v => v / 9)) },
  sharpen: { name: 'Sharpen', K: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  shift: { name: 'Shift (shows the flip)', K: [[0, 0, 0], [0, 0, 1], [0, 0, 0]] }
};
function xcorr2(I, K) { // valid cross-correlation
  const kh = K.length, kw = K[0].length, oh = I.length - kh + 1, ow = I[0].length - kw + 1;
  return Array.from({ length: oh }, (_, i) => Array.from({ length: ow }, (_, j) => {
    let s = 0; for (let m = 0; m < kh; m++) for (let n = 0; n < kw; n++) s += I[i + m][j + n] * K[m][n]; return s;
  }));
}
const flip2 = K => K.slice().reverse().map(r => r.slice().reverse());

/* ---------- 1) 1-D convolution: smoothing a noisy sensor ---------- */
Widgets.conv1dSmooth = (el) => {
  UI.head(el, 'Smoothing a noisy sensor with a convolution', 'Noisy laser readings x(t) of a spaceship’s position. The kernel w(a) gives a weight to a reading of age a; the output s(t) is their weighted average.');
  let K = 5, decay = 0.5, t0 = 40, causal = true;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Kernel length (number of weights)', min: 1, max: 12, step: 1, value: K, fmt: v => String(v), onInput: v => { K = v; draw(); } });
  UI.slider(ctr, { label: 'How fast old readings fade', min: 0.05, max: 1.5, step: 0.05, value: decay, fmt: v => v.toFixed(2), onInput: v => { decay = v; draw(); } });
  UI.slider(ctr, { label: 'Show the calculation at time t', min: 0, max: 59, step: 1, value: t0, fmt: v => String(v), onInput: v => { t0 = v; draw(); } });
  const tg = h('div', { class: 'toggles' });
  UI.chip(tg, 'Use future readings too (centered kernel)', '--s2', false, v => { causal = !v; draw(); });
  el.append(ctr, tg);
  const chart = new LineChart(el, { xr: [0, 59], yr: [-1, 11], xlabel: 'time t (s)', ylabel: 'position', xname: 't', aspect: .5, maxH: 250, vfmt: v => v.toFixed(2), snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)) });
  const lg = UI.legend(el, []);
  const kbox = h('div', { class: 'grids' });
  const calc = h('div', { class: 'calcbox' });
  el.append(kbox, calc);
  const rng = mulberry32(7), T = 60;
  const truth = Array.from({ length: T }, (_, t) => 2 + 6 / (1 + Math.exp(-(t - 30) / 6)) + 0.8 * Math.sin(t / 5));
  const x = truth.map(v => v + 0.9 * gaussian(rng));
  function draw() {
    const c = Colors.get();
    const offs = causal ? Array.from({ length: K }, (_, a) => a) : Array.from({ length: K }, (_, i) => i - Math.floor(K / 2));
    const raw = offs.map(a => Math.exp(-decay * Math.abs(a)));
    const Z = raw.reduce((s, v) => s + v, 0);
    const w = raw.map(v => v / Z);
    const valid = t => offs.every(a => t - a >= 0 && t - a < T);
    const s = x.map((_, t) => valid(t) ? offs.reduce((acc, a, i) => acc + x[t - a] * w[i], 0) : NaN);
    chart.set([
      { name: 'true position', color: 'muted', data: truth.map((v, t) => [t, v]), dash: [4, 4] },
      { name: 'reading x(t)', color: 's2', data: x.map((v, t) => [t, v]), style: 'dots', r: 3 },
      { name: 'output s(t)', color: 's1', data: s.map((v, t) => [t, v]) }
    ]);
    lg.set([{ label: 'noisy readings x(t)', color: c.s2 }, { label: 'output s(t) = (x ∗ w)(t)', color: c.s1 }, { label: 'true position', color: c.muted, kind: 'dash' }]);
    kbox.innerHTML = '';
    kbox.append(gwrap('kernel w(a), for age a = ' + offs.join(', '), cellGrid([w], { fmt: v => v.toFixed(2), cs: 36, scale: Math.max(...w) })));
    let tc = t0; let guard = 0; while (!valid(tc) && guard++ < 80) tc += tc < T / 2 ? 1 : -1;
    const show = offs.length > 6 ? 5 : offs.length;
    const more = offs.length > show ? ' + …' : '';
    const terms = offs.slice(0, show).map(a => `x(${tc - a})·w(${a})`).join(' + ') + more;
    const nums = offs.slice(0, show).map((a, i) => `${x[tc - a].toFixed(2)}×${w[i].toFixed(2)}`).join(' + ') + more;
    const rms = arr => { let e = 0, n = 0; arr.forEach((v, t) => { if (isFinite(v)) { e += (v - truth[t]) ** 2; n++; } }); return Math.sqrt(e / n); };
    calc.innerHTML = `<span class="c">discrete convolution (9.3): s(t) = Σₐ x(a)·w(t − a)  =  Σ over ages a of  x(t − a)·w(a)</span>
s(${tc}) = ${terms}
      = ${nums}
      = <b>${s[tc].toFixed(3)}</b>     (true ${truth[tc].toFixed(3)}, raw reading ${x[tc].toFixed(3)})
the weights sum to ${w.reduce((a, b) => a + b, 0).toFixed(2)} → a weighted average${causal ? ';  w(a) = 0 for a &lt; 0 → never looks into the future' : ' — but it uses FUTURE readings (impossible in real time)'}
average error vs. the truth (RMS):  raw readings <b>${rms(x).toFixed(3)}</b>  →  smoothed <b>${rms(s).toFixed(3)}</b>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) 2-D convolution explorer ---------- */
Widgets.conv2dExplorer = (el) => {
  UI.head(el, '2-D convolution, one output at a time', 'Slide a 3×3 kernel over the 8×8 image. Each output is the sum of (image patch × kernel), element by element. <b>Click any output cell</b> to see its calculation.');
  let kname = 'vedge', op = 'xcorr', sel = [1, 3], timer = null;
  const top = h('div', { class: 'controls' });
  UI.select(top, 'Kernel', Object.entries(KERNELS).map(([k, v]) => [k, v.name]), kname, v => { kname = v; draw(); });
  el.append(top);
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['xcorr', 'Cross-correlation (what libraries do)'], ['conv', 'True convolution (kernel flipped)']], op, v => { op = v; draw(); });
  UI.button(segBox, 'Sweep ▶', () => sweep());
  el.append(segBox);
  const grids = h('div', { class: 'grids' });
  const calc = h('div', { class: 'calcbox' });
  el.append(grids, calc);
  function sweep() {
    if (timer) { clearInterval(timer); timer = null; return; }
    let k = 0;
    timer = setInterval(() => {
      if (!el.isConnected || k >= 36) { clearInterval(timer); timer = null; return; }
      sel = [Math.floor(k / 6), k % 6]; k++; draw();
    }, 260);
  }
  function draw() {
    const K0 = KERNELS[kname].K;
    const K = op === 'conv' ? flip2(K0) : K0;
    const S = xcorr2(IMG8, K);
    const [r, cc] = sel;
    const kf = v => Math.abs(v - Math.round(v)) < 1e-9 ? numStr(v) : v.toFixed(2).replace('0.', '.');
    grids.innerHTML = '';
    grids.append(
      gwrap('input image I (8×8)', cellGrid(IMG8, { cs: 24, scale: 2.5, hl: (i, j) => i >= r && i < r + 3 && j >= cc && j < cc + 3 })),
      gop('⋆'),
      gwrap(op === 'conv' ? 'kernel, flipped' : 'kernel K (3×3)', cellGrid(K, { cs: 30, fmt: kf, scale: Math.max(...K.flat().map(Math.abs)) })),
      gop('='),
      gwrap('output S (6×6) — click', cellGrid(S, { cs: 26, fmt: v => numStr(v, 1), hl: (i, j) => i === r && j === cc, onClick: (i, j) => { if (timer) { clearInterval(timer); timer = null; } sel = [i, j]; draw(); } }))
    );
    const lines = [0, 1, 2].map(m => [0, 1, 2].map(n => `${numStr(IMG8[r + m][cc + n])}×${K[m][n] < 0 ? '(' + kf(K[m][n]) + ')' : kf(K[m][n])}`).join(' + ')).join('\n           + ');
    calc.innerHTML = `<span class="c">${op === 'conv' ? 'convolution (9.4–9.5): S(i,j) = Σₘ Σₙ I(i − m, j − n) K(m, n)  — same as cross-correlation with the kernel flipped' : 'cross-correlation (9.6): S(i,j) = Σₘ Σₙ I(i + m, j + n) K(m, n)   (rows/cols counted from 0)'}</span>
S(${r},${cc}) = ${lines}
           = <b>${numStr(S[r][cc], 2)}</b>
<span class="c">${kname === 'shift' ? (op === 'conv' ? 'flipped “shift” kernel copies the pixel on the LEFT → the picture moves right' : 'unflipped “shift” kernel copies the pixel on the RIGHT → the picture moves left') : kname === 'vedge' ? 'big |S| where brightness changes from left to right (vertical edges); 0 on flat areas' : kname === 'hedge' ? 'big |S| where brightness changes from top to bottom (horizontal edges)' : kname === 'blur' ? 'each output = average of its 3×3 neighbourhood' : 'centre ×5 minus the 4 neighbours: boosts local contrast'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) Convolution as a (Toeplitz) matrix ---------- */
Widgets.convMatrix = (el) => {
  UI.head(el, 'Convolution = multiplying by a special sparse matrix', 'A 1-D kernel (a, b, c) = (1, 2, −1) applied to an input of length n. Each matrix row is the row above shifted by one (a <b>Toeplitz</b> matrix). Compare with an ordinary dense layer.');
  let n = 7, mode = 'conv';
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Input length n', min: 4, max: 10, step: 1, value: n, fmt: v => String(v), onInput: v => { n = v; draw(); } });
  el.append(ctr);
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['conv', 'Convolution matrix'], ['dense', 'Dense (fully connected) matrix']], mode, v => { mode = v; draw(); });
  el.append(segBox);
  const grids = h('div', { class: 'grids' });
  const calc = h('div', { class: 'calcbox' });
  el.append(grids, calc);
  const kern = [1, 2, -1], cols = ['s1', 's3', 's7'];
  const xs = [2, 1, 0, 3, 1, 2, 0, 1, 3, 2];
  function draw() {
    const c = Colors.get();
    const rows = n - 2, x = xs.slice(0, n);
    const rng = mulberry32(5);
    const M = Array.from({ length: rows }, (_, i) => Array.from({ length: n }, (_, j) => {
      if (mode === 'dense') return Math.round((rng() * 4 - 2) * 10) / 10;
      const d = j - i; return d >= 0 && d < 3 ? kern[d] : 0;
    }));
    const y = M.map(r => r.reduce((s, v, j) => s + v * x[j], 0));
    grids.innerHTML = '';
    grids.append(
      gwrap(`matrix (${rows}×${n})`, cellGrid(M, {
        cs: 28, fmt: v => numStr(v, 1),
        color: (i, j, v) => { if (mode === 'dense') return shadeVal(v, 2); const d = j - i; return d >= 0 && d < 3 ? alpha(c[cols[d]], .35) : ''; },
        dim: (i, j) => mode === 'conv' && !(j - i >= 0 && j - i < 3), hl: i => i === 0
      })),
      gop('×'),
      gwrap('x', cellGrid(x.map(v => [v]), { cs: 28, scale: 3 })),
      gop('='),
      gwrap('y', cellGrid(y.map(v => [v]), { cs: 34, fmt: v => numStr(v, 1), hl: i => i === 0 }))
    );
    const row0 = M[0].map((v, j) => `${pn(v)}×${x[j]}`).join(' + ');
    const dense = rows * n, nz = rows * 3;
    calc.innerHTML = `y₀ = ${row0} = <b>${numStr(y[0], 1)}</b>
${mode === 'conv' ? `<span class="c">colours = which kernel weight (a, b or c); grey cells are forced to 0</span>
stored numbers: <b>3</b> (just the kernel)   ·   non-zero entries: ${rows}×3 = <b>${nz}</b>   ·   multiplications: ${nz}
same output size, dense layer: ${rows}×${n} = <b>${dense}</b> free weights, ${dense} multiplications
2-D image: a doubly block circulant matrix — for an m×m image and k×k kernel: (m−k+1)² × m² entries, only k² distinct values` : `<span class="c">every entry is its own free parameter — no sharing, no zeros</span>
free weights: ${rows}×${n} = <b>${dense}</b>   (the convolution needs only 3, with ${nz} non-zeros)`}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) Sparse connectivity & receptive field ---------- */
Widgets.receptiveField = (el) => {
  UI.head(el, 'Sparse interactions and the receptive field', '<b>Click any unit.</b> Blue = units that influence it (its receptive field, below). Orange = units it influences (above). Deeper units see a wider part of the input even though every connection is local.');
  let L = 2, k = 3, mode = 'sparse', sel = [2, 6];
  const N = 13;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Number of layers above the input', min: 1, max: 4, step: 1, value: L, fmt: v => String(v), onInput: v => { L = v; sel = [Math.min(sel[0], L), sel[1]]; draw(); } });
  UI.slider(ctr, { label: 'Kernel width k', min: 1, max: 7, step: 2, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  el.append(ctr);
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['sparse', 'Convolution (sparse)'], ['dense', 'Fully connected']], mode, v => { mode = v; draw(); });
  el.append(segBox);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'layers of units with connections' });
  wrap.append(cv); el.append(wrap);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  let geo = null;
  const connected = (a, b) => mode === 'dense' || Math.abs(a - b) <= (k - 1) / 2;
  cv.addEventListener('click', e => {
    if (!geo) return;
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    let best = null, bd = 1e9;
    for (let l = 0; l <= L; l++) for (let i = 0; i < N; i++) { const d = Math.hypot(px - geo.X(i), py - geo.Y(l)); if (d < bd) { bd = d; best = [l, i]; } }
    if (best && bd < 30) { sel = best; draw(); }
  });
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = 70 + L * 62;
    const ctx = fitCanvas(cv, W, H);
    const X = i => 70 + (i + .5) * (W - 84) / N, Y = l => H - 26 - l * 62;
    geo = { X, Y };
    // ancestors (below) and descendants (above)
    const anc = Array.from({ length: L + 1 }, () => new Set()), desc = Array.from({ length: L + 1 }, () => new Set());
    anc[sel[0]].add(sel[1]); desc[sel[0]].add(sel[1]);
    for (let l = sel[0]; l > 0; l--) for (const i of anc[l]) for (let j = 0; j < N; j++) if (connected(i, j)) anc[l - 1].add(j);
    for (let l = sel[0]; l < L; l++) for (const i of desc[l]) for (let j = 0; j < N; j++) if (connected(i, j)) desc[l + 1].add(j);
    ctx.lineWidth = 1;
    for (let l = 1; l <= L; l++) for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      if (!connected(i, j)) continue;
      const hiA = l <= sel[0] && anc[l].has(i) && anc[l - 1].has(j);
      const hiD = l > sel[0] && desc[l].has(i) && desc[l - 1].has(j);
      ctx.strokeStyle = hiA ? alpha(c.s1, .85) : hiD ? alpha(c.s2, .85) : alpha(c.muted, mode === 'dense' ? .12 : .35);
      ctx.lineWidth = hiA || hiD ? 1.6 : 1;
      ctx.beginPath(); ctx.moveTo(X(j), Y(l - 1)); ctx.lineTo(X(i), Y(l)); ctx.stroke();
    }
    for (let l = 0; l <= L; l++) {
      label(ctx, l === 0 ? 'input x' : `layer ${l}`, 8, Y(l), { color: c.text2, size: 11, bold: true });
      for (let i = 0; i < N; i++) {
        const isSel = sel[0] === l && sel[1] === i;
        const fill = isSel ? c.text : anc[l].has(i) && l < sel[0] ? c.s1 : desc[l].has(i) && l > sel[0] ? c.s2 : c.surface2;
        dot(ctx, X(i), Y(l), 8, fill, isSel ? c.accent : c.border);
      }
    }
    const rf = sel[0] === 0 ? 1 : anc[0].size;
    const rfFormula = mode === 'dense' ? N : Math.min(N, 1 + sel[0] * (k - 1));
    calc.innerHTML = `selected: ${sel[0] === 0 ? 'input' : 'layer ' + sel[0]} unit ${sel[1] + 1}
receptive field in the input: <b>${rf}</b> units   <span class="c">(${mode === 'dense' ? 'fully connected: everything after one layer' : `formula 1 + depth×(k − 1) = 1 + ${sel[0]}×${k - 1} = ${1 + sel[0] * (k - 1)}, cut at the edges, ≤ ${N}`})</span>
per layer, m = n = ${N} units:  dense  m×n = <b>${N * N}</b> weights, O(m×n) = ${N * N} multiplies
                         sparse k×n = <b>${k * N}</b> connections, O(k×n) = ${k * N} multiplies${mode === 'dense' ? '' : `  → ${(N * N / (k * N)).toFixed(1)}× fewer`}
<span class="c">${rfFormula >= N && mode !== 'dense' ? 'deep enough: this unit indirectly sees the whole input' : 'add layers or widen k to grow the receptive field'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Parameter sharing: edge detection & the cost table ---------- */
Widgets.edgeEfficiency = (el) => {
  UI.head(el, 'Parameter sharing: one tiny kernel, the whole image', 'Kernel (−1, 1): each output pixel = this pixel minus its left neighbour. Two shared weights find every vertical edge. Then compare the cost with a dense matrix.');
  let Wd = 320, Hd = 280, k = 2;
  const row = h('div', { style: 'display:flex;gap:12px;flex-wrap:wrap' });
  const c0 = h('canvas', { class: 'chart', style: 'width:200px;image-rendering:pixelated' }), c1 = h('canvas', { class: 'chart', style: 'width:200px;image-rendering:pixelated' });
  row.append(gwrap('input image', c0), gwrap('output: I(i, j) − I(i, j − 1)', c1));
  el.append(row);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Image width (pixels)', min: 32, max: 1024, step: 8, value: Wd, fmt: v => String(v), onInput: v => { Wd = v; table(); } });
  UI.slider(ctr, { label: 'Image height (pixels)', min: 32, max: 1024, step: 8, value: Hd, fmt: v => String(v), onInput: v => { Hd = v; table(); } });
  UI.slider(ctr, { label: 'Kernel width k', min: 2, max: 11, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; table(); } });
  el.append(ctr);
  const tbox = h('div');
  const calc = h('div', { class: 'calcbox' });
  el.append(tbox, calc);
  const IW = 80, IH = 64;
  const img = Array.from({ length: IH }, (_, y) => Array.from({ length: IW }, (_, x) => {
    let v = 0.25 + 0.35 * (y / IH); // sky gradient
    if (Math.hypot(x - 64, y - 14) < 8) v = 0.95; // sun
    if (y > 46) v = 0.15; // ground
    if (x > 14 && x < 42 && y > 28 && y <= 46) v = 0.75; // house wall
    if (y > 12 && y <= 28 && Math.abs(x - 28) < (y - 12) * 0.95) v = 0.45; // roof
    if (x > 24 && x < 31 && y > 36 && y <= 46) v = 0.3; // door
    return v;
  }));
  function paint() {
    const c = Colors.get();
    const draw1 = (cv, f) => {
      const ctx = fitCanvas(cv, 200, 160);
      const off = document.createElement('canvas'); off.width = IW; off.height = IH;
      const ox = off.getContext('2d'), im = ox.createImageData(IW, IH);
      for (let y = 0; y < IH; y++) for (let x = 0; x < IW; x++) { const [r, g, b] = f(x, y); const k4 = (y * IW + x) * 4; im.data[k4] = r; im.data[k4 + 1] = g; im.data[k4 + 2] = b; im.data[k4 + 3] = 255; }
      ox.putImageData(im, 0, 0);
      ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, 200, 160);
    };
    const bg = hexToRgb(c.surface2), ink = hexToRgb(c.text), pos = hexToRgb(c.s1), neg = hexToRgb(c.s2);
    draw1(c0, (x, y) => { const v = img[y][x]; return [0, 1, 2].map(i => lerp(bg[i], ink[i], v)); });
    draw1(c1, (x, y) => {
      const d = x > 0 ? img[y][x] - img[y][x - 1] : 0, t = clamp(Math.abs(d) / 0.5, 0, 1), col = d > 0 ? pos : neg;
      return [0, 1, 2].map(i => lerp(bg[i], col[i], t));
    });
  }
  function table() {
    const ow = Wd - k + 1, nIn = Wd * Hd, nOut = ow * Hd;
    const dense = nIn * nOut, sparse = k * nOut, ops = (2 * k - 1) * nOut;
    const big = v => v >= 1e6 ? fmt(v, 1) : v.toLocaleString('en-US');
    tbox.innerHTML = '';
    tbox.append(h('div', { class: 'scroll-x' }, h('table', { class: 'data-table' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Same linear map stored as…'), h('th', {}, 'Numbers to store'), h('th', {}, 'Operations'))),
      h('tbody', {},
        h('tr', {}, h('td', {}, 'Dense matrix'), h('td', {}, big(dense)), h('td', {}, big(2 * dense))),
        h('tr', {}, h('td', {}, 'Sparse matrix (no sharing)'), h('td', {}, big(sparse)), h('td', {}, big(ops))),
        h('tr', {}, h('td', {}, h('b', {}, 'Convolution')), h('td', {}, h('b', {}, String(k))), h('td', {}, big(ops)))
      ))));
    calc.innerHTML = `input ${Wd}×${Hd} = ${big(nIn)} pixels → output ${ow}×${Hd} = ${big(nOut)} pixels   <span class="c">(valid: width shrinks by k − 1)</span>
dense:  ${big(nIn)} × ${big(nOut)} = <b>${big(dense)}</b> matrix entries
conv:   each output needs ${k} multiplies + ${k - 1} add${k - 1 === 1 ? '' : 's'} = ${2 * k - 1} ops → ${2 * k - 1} × ${big(nOut)} = <b>${big(ops)}</b> ops
→ about <b>${big(Math.round(2 * dense / ops))}×</b> less computation and ${big(Math.round(dense / k))}× less storage than the dense matrix`;
  }
  paint(); table();
  return { redraw: () => { paint(); table(); } };
};

/* ---------- 6) Equivariance to translation ---------- */
Widgets.equivariance = (el) => {
  UI.head(el, 'Equivariance: shift the input → the output shifts the same way', 'f = convolution with a fixed kernel (a detector for the bump shape). g = a transformation of the input. Equivariance means f(g(x)) = g(f(x)).');
  let tf = 'shift', s = 12;
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['shift', 'g = shift (translation)'], ['stretch', 'g = stretch ×1.8 (scale)']], tf, v => { tf = v; sS.show(tf === 'shift'); draw(); });
  el.append(segBox);
  const ctr = h('div', { class: 'controls' });
  const sS = UI.slider(ctr, { label: 'Shift (time steps)', min: -14, max: 20, step: 1, value: s, fmt: v => String(v), onInput: v => { s = v; draw(); } });
  el.append(ctr);
  const ch1 = new LineChart(el, { xr: [0, 79], yr: [-0.6, 1.3], xlabel: 't', ylabel: 'input', xname: 't', aspect: .3, maxH: 150, minH: 120, vfmt: v => v.toFixed(3), snapX: x => Math.round(x) });
  const ch2 = new LineChart(el, { xr: [0, 79], yr: [-1.2, 2.6], xlabel: 't', ylabel: 'feature map', xname: 't', aspect: .38, maxH: 190, minH: 140, vfmt: v => v.toFixed(3), snapX: x => Math.round(x) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const T = 80;
  const bump = (t, c0, wd) => Math.exp(-0.5 * ((t - c0) / wd) ** 2) - 0.35 * Math.exp(-0.5 * ((t - c0) / (2.2 * wd)) ** 2);
  const x = Array.from({ length: T }, (_, t) => bump(t, 26, 2.5));
  const ker = Array.from({ length: 13 }, (_, j) => bump(j, 6, 2.5));
  const conv = a => a.map((_, t) => { let v = 0; for (let j = 0; j < 13; j++) { const i = t + j - 6; if (i >= 0 && i < T) v += a[i] * ker[j]; } return v; });
  const shift = (a, d) => a.map((_, t) => { const i = t - d; return i >= 0 && i < T ? a[i] : 0; });
  const stretch = (a, f) => a.map((_, t) => { const u = 26 + (t - 26) / f, i0 = Math.floor(u), fr = u - i0; return i0 >= 0 && i0 + 1 < T ? a[i0] * (1 - fr) + a[i0 + 1] * fr : 0; });
  function draw() {
    const c = Colors.get();
    const g = a => tf === 'shift' ? shift(a, s) : stretch(a, 1.8);
    const gx = g(x), fgx = conv(gx), gfx = g(conv(x)), fx = conv(x);
    ch1.set([{ name: 'x', color: 'muted', data: x.map((v, t) => [t, v]), dash: [4, 4] }, { name: 'g(x)', color: 's1', data: gx.map((v, t) => [t, v]) }]);
    ch2.set([{ name: 'f(x)', color: 'muted', data: fx.map((v, t) => [t, v]), dash: [4, 4] }, { name: 'f(g(x))', color: 's1', data: fgx.map((v, t) => [t, v]) }, { name: 'g(f(x))', color: 's2', data: gfx.map((v, t) => [t, v]).filter((_, t) => t % 2 === 0), style: 'dots', r: 3 }]);
    lg.set([{ label: 'original (x and f(x))', color: c.muted, kind: 'dash' }, { label: 'transform first, then convolve: f(g(x))', color: c.s1 }, { label: 'convolve first, then transform: g(f(x))', color: c.s2 }]);
    const am = a => a.reduce((b, v, i) => v > a[b] ? i : b, 0);
    const diff = Math.max(...fgx.map((v, i) => Math.abs(v - gfx[i])));
    calc.innerHTML = `peak of f(x) at t = ${am(fx)}  (height ${fx[am(fx)].toFixed(3)})
f(g(x)): peak at t = <b>${am(fgx)}</b>, height ${fgx[am(fgx)].toFixed(3)}      g(f(x)): peak at t = <b>${am(gfx)}</b>, height ${gfx[am(gfx)].toFixed(3)}
largest difference |f(g(x)) − g(f(x))| = <b>${diff < 1e-9 ? '0 (exactly equal)' : diff.toFixed(3)}</b>
<span class="c">${tf === 'shift' ? `shifting by ${s}: the same feature appears ${s} steps ${s >= 0 ? 'later' : 'earlier'} — convolution is equivariant to translation: f(g(x)) = g(f(x))` : 'a stretched bump no longer matches the fixed-size kernel: the response is weaker and differently shaped — convolution is NOT equivariant to scale (or rotation)'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) A full convolutional layer: conv → detector → pool ---------- */
Widgets.layerPipeline = (el) => {
  UI.head(el, 'The three stages of a convolutional layer', 'Convolution (+ bias) → detector stage (ReLU) → pooling (2×2, stride 2). <b>Click a pooled cell</b> to trace where it comes from.');
  let kname = 'vedge', b = 0, ptype = 'max', sel = [0, 1];
  const ctr = h('div', { class: 'controls' });
  UI.select(ctr, 'Kernel', [['vedge', 'Vertical edges'], ['hedge', 'Horizontal edges'], ['sharpen', 'Sharpen']], kname, v => { kname = v; draw(); });
  UI.slider(ctr, { label: 'Bias b (shared by the whole map)', min: -3, max: 3, step: 0.5, value: b, fmt: v => numStr(v, 1), onInput: v => { b = v; draw(); } });
  el.append(ctr);
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['max', 'Max pooling'], ['avg', 'Average pooling']], ptype, v => { ptype = v; draw(); });
  el.append(segBox);
  const grids = h('div', { class: 'grids' });
  const calc = h('div', { class: 'calcbox' });
  el.append(grids, calc);
  function draw() {
    const K = KERNELS[kname].K;
    const Z = xcorr2(IMG8, K).map(r => r.map(v => v + b));
    const Hd = Z.map(r => r.map(v => Math.max(0, v)));
    const P = [0, 1, 2].map(i => [0, 1, 2].map(j => {
      const vals = [Hd[2 * i][2 * j], Hd[2 * i][2 * j + 1], Hd[2 * i + 1][2 * j], Hd[2 * i + 1][2 * j + 1]];
      return ptype === 'max' ? Math.max(...vals) : vals.reduce((a, v) => a + v, 0) / 4;
    }));
    const [pi, pj] = sel;
    const inPool = (i, j) => Math.floor(i / 2) === pi && Math.floor(j / 2) === pj;
    const inRF = (i, j) => i >= 2 * pi && i < 2 * pi + 4 && j >= 2 * pj && j < 2 * pj + 4;
    grids.innerHTML = '';
    grids.append(
      gwrap('input (8×8)', cellGrid(IMG8, { cs: 22, scale: 2.5, hl: inRF })),
      gop('→'),
      gwrap('1. conv + b (6×6)', cellGrid(Z, { cs: 25, hl: inPool })),
      gop('→'),
      gwrap('2. ReLU (6×6)', cellGrid(Hd, { cs: 25, hl: inPool, dim: (i, j) => Hd[i][j] === 0 })),
      gop('→'),
      gwrap(`3. ${ptype} pool (3×3)`, cellGrid(P, { cs: 30, fmt: v => numStr(v, 2), hl: (i, j) => i === pi && j === pj, onClick: (i, j) => { sel = [i, j]; draw(); } }))
    );
    const zs = [[0, 0], [0, 1], [1, 0], [1, 1]].map(([a, d]) => [2 * pi + a, 2 * pj + d]);
    calc.innerHTML = `pooled unit (${pi},${pj}) looks at ReLU cells ${zs.map(([i, j]) => `(${i},${j})`).join(' ')}
conv:  z = (I ⋆ K) + b  →  ${b ? zs.map(([i, j]) => numStr(Z[i][j] - b, 1) + ` ${b > 0 ? '+' : '−'} ${Math.abs(b)}`).join(',  ') + '  =  ' : ''}${zs.map(([i, j]) => numStr(Z[i][j], 1)).join(', ')}${b ? '' : '   (b = 0)'}
ReLU:  h = max(0, z)  →  ${zs.map(([i, j]) => numStr(Hd[i][j], 1)).join(', ')}
pool:  ${ptype === 'max' ? 'max' : 'average'}(${zs.map(([i, j]) => numStr(Hd[i][j], 1)).join(', ')}) = <b>${numStr(P[pi][pj], 2)}</b>
<span class="c">its receptive field in the input = the highlighted 4×4 patch (3×3 kernel + 2×2 pool); 36 detector units → 9 pooled units (4× fewer for the next layer)</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 8) Pooling: invariance to small shifts & downsampling ---------- */
Widgets.poolingShift = (el) => {
  UI.head(el, 'Pooling makes the output barely notice small shifts', 'Top: detector-stage values and their pooled summary. Bottom: the same after shifting the input right. Orange outline = value changed.');
  let w = 3, s = 1, ptype = 'max', shiftN = 1;
  const base = [0.1, 1.0, 0.2, 0.1, 0.3, 0.1, 0.9, 0.2, 0.1, 0.6, 0.1, 0.2, 0.8, 0.1];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Pool width', min: 1, max: 5, step: 1, value: w, fmt: v => String(v), onInput: v => { w = v; draw(); } });
  UI.slider(ctr, { label: 'Stride between pools', min: 1, max: 3, step: 1, value: s, fmt: v => String(v), onInput: v => { s = v; draw(); } });
  UI.slider(ctr, { label: 'Shift the input right by', min: 0, max: 3, step: 1, value: shiftN, fmt: v => v + ' px', onInput: v => { shiftN = v; draw(); } });
  el.append(ctr);
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['max', 'Max'], ['avg', 'Average'], ['l2', 'L² norm']], ptype, v => { ptype = v; draw(); });
  el.append(segBox);
  const box = h('div', { class: 'stack', style: 'gap:8px' });
  const calc = h('div', { class: 'calcbox' });
  el.append(box, calc);
  const pool = a => { const out = []; for (let i = 0; i + w <= a.length; i += s) { const win = a.slice(i, i + w); out.push(ptype === 'max' ? Math.max(...win) : ptype === 'avg' ? win.reduce((p, v) => p + v, 0) / w : Math.sqrt(win.reduce((p, v) => p + v * v, 0))); } return out; };
  function draw() {
    const sh = base.map((_, i) => i - shiftN >= 0 ? base[i - shiftN] : 0.1);
    const p0 = pool(base), p1 = pool(sh);
    const chD = sh.filter((v, i) => Math.abs(v - base[i]) > 1e-9).length, chP = p1.filter((v, i) => Math.abs(v - p0[i]) > 1e-9).length;
    const rowOf = (lab, arr, other) => h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, lab), cellGrid([arr], { cs: 30, fmt: v => v.toFixed(1).replace(/^0\./, '.'), scale: 1, hl2: other ? (i, j) => Math.abs(arr[j] - other[j]) > 1e-9 : null }));
    box.innerHTML = '';
    box.append(rowOf('pooled', p0), rowOf('detector', base), h('div', { class: 'note' }, `after shifting ${shiftN} px to the right:`), rowOf('pooled', p1, p0), rowOf('detector', sh, base));
    const win = base.slice(0, w).map(v => v.toFixed(1));
    const f = ptype === 'max' ? `max(${win.join(', ')})` : ptype === 'avg' ? `(${win.join(' + ')}) / ${w}` : `√(${win.map(v => v + '²').join(' + ')})`;
    calc.innerHTML = `first pooled unit = ${f} = <b>${p0[0].toFixed(3)}</b>
detector values changed: <b>${chD} / ${base.length}</b>     pooled values changed: <b>${chP} / ${p0.length}</b>
number of pooled outputs: ⌊(n − width) / stride⌋ + 1 = ⌊(${base.length} − ${w}) / ${s}⌋ + 1 = <b>${p0.length}</b>  <span class="c">${s > 1 ? `(stride ${s} → the next layer gets ≈ ${s}× fewer inputs)` : '(stride 1: no downsampling)'}</span>
<span class="c">${ptype === 'max' ? 'max pooling only reports the largest value in each window, not where it is → approximately invariant to small translations' : 'average / L² pooling change a little whenever any value in the window changes'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 9) Convolution + pooling as an infinitely strong prior ---------- */
Widgets.priorTest = (el) => {
  UI.head(el, 'A strong prior: a huge help when right, a handicap when wrong', 'Sequences of 16 noisy values containing one 3-value motif. Two models trained with gradient descent: a dense linear classifier (17 weights) and conv (kernel 3) + global max-pool (4 weights).');
  let task = 'which', ntr = 30, seed = 1;
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['which', 'Task A: which motif? (position irrelevant)'], ['where', 'Task B: motif in the left half? (position matters)']], task, v => { task = v; draw(); });
  el.append(segBox);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Training examples', min: 10, max: 200, step: 10, value: ntr, fmt: v => String(v), onInput: v => { ntr = v; draw(); } });
  UI.button(ctr, 'New random training set', () => { seed++; draw(); });
  el.append(ctr);
  const out = h('div');
  const wbox = h('div', { class: 'stack', style: 'gap:8px' });
  const calc = h('div', { class: 'calcbox' });
  el.append(out, wbox, calc);
  const L = 16, A = [1, 0, 1], B = [1, 1, 0];
  function makeSet(n, sd) {
    const rng = mulberry32(sd), X = [], Y = [];
    for (let e = 0; e < n; e++) {
      const x = Array.from({ length: L }, () => 0.25 * gaussian(rng));
      const p = Math.floor(rng() * (L - 2));
      let y;
      if (task === 'which') { y = rng() < 0.5 ? 1 : 0; const m = y ? A : B; for (let j = 0; j < 3; j++) x[p + j] += m[j]; }
      else { for (let j = 0; j < 3; j++) x[p + j] += A[j]; y = p + 1 < L / 2 ? 1 : 0; }
      X.push(x); Y.push(y);
    }
    return [X, Y];
  }
  function trainDense(X, Y) {
    const w = new Array(L).fill(0); let b = 0;
    for (let it = 0; it < 500; it++) {
      const gw = new Array(L).fill(0); let gb = 0;
      X.forEach((x, e) => { const p = sigmoid(dotv(w, x) + b), d = p - Y[e]; for (let i = 0; i < L; i++) gw[i] += d * x[i]; gb += d; });
      for (let i = 0; i < L; i++) w[i] -= 0.5 * (gw[i] / X.length + 1e-3 * w[i]);
      b -= 0.5 * gb / X.length;
    }
    return { w, b, f: x => dotv(w, x) + b };
  }
  function convMax(k, x) { let best = -Infinity, at = 0; for (let t = 0; t + 3 <= L; t++) { const v = k[0] * x[t] + k[1] * x[t + 1] + k[2] * x[t + 2]; if (v > best) { best = v; at = t; } } return [best, at]; }
  function trainConv(X, Y) {
    const rng = mulberry32(99); const k = [0, 1, 2].map(() => 0.1 * gaussian(rng)); let b = 0;
    for (let it = 0; it < 500; it++) {
      const gk = [0, 0, 0]; let gb = 0;
      X.forEach((x, e) => { const [m, at] = convMax(k, x); const d = sigmoid(m + b) - Y[e]; for (let j = 0; j < 3; j++) gk[j] += d * x[at + j]; gb += d; });
      for (let j = 0; j < 3; j++) k[j] -= 0.5 * (gk[j] / X.length + 1e-3 * k[j]);
      b -= 0.5 * gb / X.length;
    }
    return { k, b, f: x => convMax(k, x)[0] + b };
  }
  const acc = (m, X, Y) => X.reduce((s, x, e) => s + ((m.f(x) > 0 ? 1 : 0) === Y[e] ? 1 : 0), 0) / X.length;
  function draw() {
    const [Xtr, Ytr] = makeSet(ntr, 1000 + seed), [Xte, Yte] = makeSet(500, 7);
    const D = trainDense(Xtr, Ytr), C = trainConv(Xtr, Ytr);
    const r = [[acc(D, Xtr, Ytr), acc(D, Xte, Yte)], [acc(C, Xtr, Ytr), acc(C, Xte, Yte)]];
    const pc = v => (100 * v).toFixed(0) + '%';
    out.innerHTML = '';
    out.append(h('div', { class: 'scroll-x' }, h('table', { class: 'data-table' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Model'), h('th', {}, 'Free weights'), h('th', {}, 'Train accuracy'), h('th', {}, 'Test accuracy (500 new)'))),
      h('tbody', {},
        h('tr', {}, h('td', {}, 'Dense linear'), h('td', {}, '16 + 1'), h('td', {}, pc(r[0][0])), h('td', {}, h('b', {}, pc(r[0][1])))),
        h('tr', {}, h('td', {}, 'Conv k = 3 + global max-pool'), h('td', {}, '3 + 1'), h('td', {}, pc(r[1][0])), h('td', {}, h('b', {}, pc(r[1][1]))))
      ))));
    wbox.innerHTML = '';
    wbox.append(
      h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, 'dense weights'), cellGrid([D.w], { cs: 26, fmt: v => v.toFixed(1).replace('-', '−').replace(/^(−?)0\./, '$1.') })),
      h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, 'conv kernel'), cellGrid([C.k], { cs: 34, fmt: v => v.toFixed(2).replace('-', '−') })),
      h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, 'motifs'), h('span', {}, task === 'which' ? 'class 1: (1, 0, 1)   class 0: (1, 1, 0) — anywhere' : 'always (1, 0, 1); class 1 if it sits in the left half'))
    );
    calc.innerHTML = task === 'which'
      ? `the right answer does not depend on position → convolution's prior (same detector everywhere, local, then max over positions) is CORRECT
conv + max-pool: ${pc(r[1][1])} test accuracy from ${ntr} examples with 4 weights
dense: must learn the motif separately at all 14 positions — no single linear weight vector can do that for these motifs → ${pc(r[0][1])}
<span class="c">the infinitely strong prior removes the impossible/irrelevant solutions → better statistical efficiency</span>`
      : `now WHERE the motif is decides the label → “invariant to translation” is the WRONG prior
conv + global max-pool throws the position away → stuck near chance: <b>${pc(r[1][1])}</b> (underfitting — even the training set is ${pc(r[1][0])})
dense: positive weights on the left, negative on the right → <b>${pc(r[0][1])}</b>
<span class="c">fix in practice: pool only some channels, pool locally instead of globally, or keep position information</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 10b) Multi-channel convolution (9.7) ---------- */
Widgets.multiChannel = (el) => {
  UI.head(el, 'Many channels in, many channels out', 'Input V: 2 channels of 4×4. Kernel K: for each of the 2 output channels, one 2×2 filter per input channel. Every output value sums over <b>all</b> input channels. <b>Click an output cell.</b>');
  let sel = [0, 1, 1];
  const V = [[[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], [[0, 1, 1, 0], [2, 0, 1, 1], [1, 1, 0, 2], [0, 2, 1, 0]]];
  const Kt = [[[[1, 0], [0, -1]], [[0, 1], [1, 0]]], [[[1, 1], [0, 0]], [[-1, 0], [0, 1]]]];
  const Z = Kt.map(Ki => Array.from({ length: 3 }, (_, j) => Array.from({ length: 3 }, (_, k) => {
    let s = 0; for (let l = 0; l < 2; l++) for (let m = 0; m < 2; m++) for (let n = 0; n < 2; n++) s += V[l][j + m][k + n] * Ki[l][m][n]; return s;
  })));
  const grids = h('div', { class: 'grids' });
  const calc = h('div', { class: 'calcbox' });
  el.append(grids, calc);
  function draw() {
    const [i, j, k] = sel;
    const win = (r, c) => r >= j && r < j + 2 && c >= k && c < k + 2;
    grids.innerHTML = '';
    grids.append(
      h('div', { class: 'stack', style: 'gap:8px' }, gwrap('input channel 0', cellGrid(V[0], { cs: 24, scale: 3, hl: win })), gwrap('input channel 1', cellGrid(V[1], { cs: 24, scale: 3, hl: win }))),
      gop('⋆'),
      h('div', { class: 'stack', style: 'gap:8px' }, gwrap(`K(${i}, 0, ·, ·)`, cellGrid(Kt[i][0], { cs: 26 })), gwrap(`K(${i}, 1, ·, ·)`, cellGrid(Kt[i][1], { cs: 26 }))),
      gop('='),
      h('div', { class: 'stack', style: 'gap:8px' }, ...[0, 1].map(o => gwrap(`output channel ${o} — click`, cellGrid(Z[o], { cs: 28, hl: (r, c) => o === i && r === j && c === k, onClick: (r, c) => { sel = [o, r, c]; draw(); } }))))
    );
    const part = l => { const t = []; let s = 0; for (let m = 0; m < 2; m++) for (let n = 0; n < 2; n++) { t.push(`${V[l][j + m][k + n]}×${pn(Kt[i][l][m][n])}`); s += V[l][j + m][k + n] * Kt[i][l][m][n]; } return [t.join(' + '), s]; };
    const [t0, s0] = part(0), [t1, s1] = part(1);
    calc.innerHTML = `<span class="c">(9.7, counted from 0): Z(i, j, k) = Σₗ Σₘ Σₙ V(l, j + m, k + n) · K(i, l, m, n)    l = input channel, i = output channel</span>
Z(${i}, ${j}, ${k}) = [channel 0] ${t0}
           + [channel 1] ${t1}
           = ${numStr(s0)} + ${numStr(s1)} = <b>${numStr(Z[i][j][k])}</b>
kernel tensor K has C_out × C_in × k × k = 2 × 2 × 2 × 2 = <b>16</b> weights (+ 2 biases, one per output channel)
<span class="c">in software a 4th axis indexes the examples of the minibatch: V is (batch, channels, rows, cols)</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 10) Convolution arithmetic: padding & stride ---------- */
Widgets.convArithmetic = (el, ds) => {
  const isStride = ds.mode === 'stride';
  UI.head(el, isStride ? 'Strided convolution = convolution, then keep every s-th output' : 'Zero padding: valid, same and full', isStride
    ? 'Move the kernel s positions at a time. The result equals the unit-stride output sampled at every s-th position. <b>Click an output.</b>'
    : 'Dashed cells are implicit zeros. The row below the input counts how many outputs use each input pixel. <b>Click an output.</b>');
  let m = isStride ? 9 : 7, k = 3, s = isStride ? 2 : 1, pm = 'valid', sel = 0;
  const xs = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];
  const kv = [1, 0, -1, 2, 1, -2, 1];
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Input width m', min: 4, max: 12, step: 1, value: m, fmt: v => String(v), onInput: v => { m = v; sel = 0; draw(); } });
  UI.slider(ctr, { label: 'Kernel width k', min: 1, max: 7, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; sel = 0; draw(); } });
  UI.slider(ctr, { label: 'Stride s', min: 1, max: 4, step: 1, value: s, fmt: v => String(v), onInput: v => { s = v; sel = 0; draw(); } });
  el.append(ctr);
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['valid', 'Valid (no padding)'], ['same', 'Same'], ['full', 'Full']], pm, v => { pm = v; sel = 0; draw(); });
  el.append(segBox);
  const box = h('div', { class: 'stack', style: 'gap:8px' });
  el.append(box);
  const chart = isStride ? null : new LineChart(el, { xr: [0, 10], yr: [0, 40], xlabel: 'number of layers', ylabel: 'width', xname: 'layers', aspect: .32, maxH: 170, minH: 130, vfmt: v => String(v), snapX: x => Math.round(x), hxfmt: v => String(Math.round(v)) });
  const lg = isStride ? null : UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const x = xs.slice(0, m), K = kv.slice(0, k);
    const pl = pm === 'valid' ? 0 : pm === 'same' ? Math.floor((k - 1) / 2) : k - 1;
    const pr = pm === 'valid' ? 0 : pm === 'same' ? Math.ceil((k - 1) / 2) : k - 1;
    const P = [...new Array(pl).fill(0), ...x, ...new Array(pr).fill(0)];
    const full1 = []; for (let t = 0; t + k <= P.length; t++) full1.push(K.reduce((a, v, j) => a + v * P[t + j], 0));
    const outIdx = full1.map((_, t) => t).filter(t => t % s === 0);
    const Z = outIdx.map(t => full1[t]);
    if (sel >= Z.length) sel = 0;
    const st = outIdx[sel] ?? 0;
    const uses = P.map((_, i) => outIdx.filter(t => i >= t && i < t + k).length);
    box.innerHTML = '';
    if (!Z.length) { box.append(h('div', { class: 'note' }, 'Kernel wider than the (padded) input: no valid position — the output is empty.')); }
    box.append(
      h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, 'input (padded)'), cellGrid([P], { cs: 28, pad: (i, j) => j < pl || j >= pl + m, scale: 9, hl: (i, j) => Z.length && j >= st && j < st + k })),
      h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, 'times used'), cellGrid([uses.map((u, j) => j < pl || j >= pl + m ? null : u)], { cs: 28, color: () => '', fmt: v => String(v) })),
      h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, 'kernel'), cellGrid([K], { cs: 28 })),
      Z.length ? h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, `output (${Z.length})`), cellGrid([Z], { cs: 30, hl: (i, j) => j === sel, onClick: (i, j) => { sel = j; draw(); } })) : null,
      isStride && s > 1 ? h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, 'unit stride'), cellGrid([full1], { cs: 30, hl2: (i, j) => j % s === 0, dim: (i, j) => j % s !== 0 })) : null
    );
    const nOut = Math.floor((m + pl + pr - k) / s) + 1;
    const winTerms = K.map((v, j) => `${pn(P[st + j])}×${pn(v)}`).join(' + ');
    let extra = '';
    if (chart) {
      const L = 10, W0 = 32;
      const data = (fn) => Array.from({ length: L + 1 }, (_, l) => [l, Math.max(0, fn(l))]);
      const valid = data(l => W0 - l * (k - 1)), same = data(() => W0), full = data(l => W0 + l * (k - 1));
      chart.set([{ name: 'valid', color: 's2', data: valid }, { name: 'same', color: 's1', data: same }, { name: 'full', color: 's3', data: full }], { yr: [0, Math.max(40, W0 + L * (k - 1) + 2)] });
      lg.set([{ label: `valid: 32 − L(k − 1)`, color: c.s2 }, { label: 'same: stays 32', color: c.s1 }, { label: 'full: 32 + L(k − 1)', color: c.s3 }]);
      const lim = k > 1 ? Math.floor((W0 - 1) / (k - 1)) : Infinity;
      extra = `\n<span class="c">chart (32-pixel input, k = ${k}): valid convolution can stack at most ${isFinite(lim) ? lim : '∞'} layers before the width hits 1</span>`;
    }
    calc.innerHTML = `output width = ⌊(m + p_left + p_right − k) / s⌋ + 1 = ⌊(${m} + ${pl} + ${pr} − ${k}) / ${s}⌋ + 1 = <b>${Math.max(0, nOut)}</b>
${pm === 'valid' ? `valid: no zeros; width m − k + 1 = ${m - k + 1} at stride 1 — every output sees k real pixels` : pm === 'same' ? `same: k − 1 = ${k - 1} zeros in total → width stays m = ${m} at stride 1; border pixels are used fewer times` : `full: k − 1 = ${k - 1} zeros on each side → width m + k − 1 = ${m + k - 1}; every pixel is used k = ${k} times, but border outputs see mostly zeros`}
${Z.length ? `output ${sel} = ${winTerms} = <b>${numStr(Z[sel])}</b>` : ''}${isStride ? `\n<span class="c">strided (9.8): Z(i) = Σⱼ V((i − 1)·s + j)·K(j) — only every ${s}${s === 2 ? 'nd' : s === 3 ? 'rd' : 'th'} position is computed: ${full1.length} → ${Z.length} outputs (${full1.length - Z.length} fewer kernel applications)</span>` : ''}${extra}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 11) Locally connected, tiled and standard convolution ---------- */
Widgets.connectivityCompare = (el) => {
  UI.head(el, 'How many different weights? Four ways to connect two layers', 'Lines with the same colour use the same set of weights. The table lists which weight each connection uses.');
  let mode = 'conv', t = 2;
  const N = 8, k = 3, O = N - k + 1;
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['full', 'Fully connected'], ['local', 'Locally connected'], ['tiled', 'Tiled'], ['conv', 'Convolution']], mode, v => { mode = v; tS.show(mode === 'tiled'); draw(); });
  el.append(segBox);
  const ctr = h('div', { class: 'controls' });
  const tS = UI.slider(ctr, { label: 'Tiled: number of kernels t', min: 2, max: 3, step: 1, value: t, fmt: v => String(v), onInput: v => { t = v; draw(); } });
  tS.show(false);
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'connections between an input layer and an output layer' });
  wrap.append(cv);
  const tbl = h('div', { class: 'grids' });
  const calc = h('div', { class: 'calcbox' });
  el.append(wrap, tbl, calc);
  const COLS = ['s1', 's2', 's3', 's4', 's5', 's7'];
  const setOf = j => mode === 'local' ? j : mode === 'tiled' ? j % t : 0;
  const names = 'abc';
  const sub = n => String(n).split('').map(d => '₀₁₂₃₄₅₆₇₈₉'[+d]).join('');
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = 170;
    const ctx = fitCanvas(cv, W, H);
    const Xi = i => 30 + (i + .5) * (W - 60) / N, Xo = j => Xi(j + 1);
    const Yi = H - 28, Yo = 30;
    for (let j = 0; j < O; j++) for (let i = 0; i < N; i++) {
      const linked = mode === 'full' || (i >= j && i < j + k);
      if (!linked) continue;
      ctx.strokeStyle = mode === 'full' ? alpha(c.muted, .45) : c[COLS[setOf(j) % COLS.length]];
      ctx.lineWidth = mode === 'full' ? 1 : 2;
      ctx.beginPath(); ctx.moveTo(Xi(i), Yi); ctx.lineTo(Xo(j), Yo); ctx.stroke();
    }
    for (let i = 0; i < N; i++) { dot(ctx, Xi(i), Yi, 9, c.surface2, c.border); label(ctx, 'x' + sub(i + 1), Xi(i), Yi + 17, { align: 'center', size: 10 }); }
    for (let j = 0; j < O; j++) { dot(ctx, Xo(j), Yo, 9, c.surface2, c.border); label(ctx, 's' + sub(j + 1), Xo(j), Yo - 17, { align: 'center', size: 10 }); }
    // weight table
    tbl.innerHTML = '';
    if (mode !== 'full') {
      const T = Array.from({ length: O }, (_, j) => [0, 1, 2].map(m => mode === 'local' ? `w${sub(j + 1)}${names[m]}` : mode === 'tiled' ? names[m] + sub(setOf(j) + 1) : names[m]));
      tbl.append(gwrap('weight used by output s_j (rows) for its 1st, 2nd, 3rd input', cellGrid(T, { cs: 38, fmt: v => v, color: (i) => alpha(c[COLS[setOf(i) % COLS.length]], .3) })));
    }
    const P = { full: N * O, local: O * k, tiled: t * k, conv: k }[mode];
    calc.innerHTML = `distinct weights:  fully connected ${N}×${O} = ${N * O}  ·  locally connected ${O}×${k} = ${O * k}  ·  tiled t×k = ${t}×${k} = ${t * k}  ·  convolution k = ${k}
current: <b>${P}</b>
${mode === 'local' ? 'locally connected (unshared convolution, 9.9): Z(i,j,k) = Σ V(l, j+m−1, k+n−1)·w(i,j,k,l,m,n) — a 6-D weight tensor, own weights at every location' : mode === 'tiled' ? `tiled (9.10): Z(i,j,k) = Σ V(l, j+m−1, k+n−1)·K(i,l,m,n, j%t+1, k%t+1) — neighbours use different kernels, cycling every t = ${t} positions` : mode === 'conv' ? 'convolution: one kernel reused at every position (t = 1)' : 'every output reads every input with its own weight'}
<span class="c">${mode === 'local' ? 'use when features are local but differ by location (mouth only in the bottom of a face crop)' : mode === 'tiled' ? 't = 1 → convolution, t = output width → locally connected' : mode === 'conv' ? 'biases: usually one per output channel, shared across positions' : 'no notion of “nearby”: the most flexible, the most parameters'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 12) Back-propagation through a convolution ---------- */
Widgets.convGrad = (el) => {
  UI.head(el, 'Gradients of a convolution layer (1-D, worked numbers)', 'Forward: z_t = Σⱼ kⱼ·x(t·s + j). Loss J = ½Σ(z − y)². Backward: G = ∂J/∂z flows back to the kernel and to the input (the input gradient = multiplication by the transposed matrix). <b>Click an input cell.</b>');
  let s = 1, seed = 3, selI = 2;
  const segBox = h('div', { class: 'toggles' });
  UI.seg(segBox, [['1', 'Stride 1'], ['2', 'Stride 2']], '1', v => { s = +v; draw(); });
  UI.button(segBox, 'New random numbers', () => { seed++; draw(); });
  el.append(segBox);
  const box = h('div', { class: 'stack', style: 'gap:6px' });
  const calc = h('div', { class: 'calcbox' });
  el.append(box, calc);
  const N = 7, K = 3;
  function draw() {
    const rng = mulberry32(seed * 31);
    const ri = (a, b) => a + Math.floor(rng() * (b - a + 1));
    const x = Array.from({ length: N }, () => ri(-2, 3)), k = Array.from({ length: K }, () => ri(-1, 2));
    const T = Math.floor((N - K) / s) + 1;
    const z = Array.from({ length: T }, (_, t) => k.reduce((a, v, j) => a + v * x[t * s + j], 0));
    const y = Array.from({ length: T }, () => ri(-2, 2));
    const G = z.map((v, t) => v - y[t]);
    const J = 0.5 * G.reduce((a, v) => a + v * v, 0);
    const dK = k.map((_, j) => G.reduce((a, g, t) => a + g * x[t * s + j], 0));
    const dX = x.map((_, i) => G.reduce((a, g, t) => { const j = i - t * s; return a + (j >= 0 && j < K ? g * k[j] : 0); }, 0));
    const dB = G.reduce((a, v) => a + v, 0);
    // numerical check of dK[0]
    const Jof = kk => 0.5 * Array.from({ length: T }, (_, t) => kk.reduce((a, v, j) => a + v * x[t * s + j], 0) - y[t]).reduce((a, v) => a + v * v, 0);
    const eps = 1e-5, num = (Jof([k[0] + eps, k[1], k[2]]) - Jof([k[0] - eps, k[1], k[2]])) / (2 * eps);
    const users = Array.from({ length: T }, (_, t) => t).filter(t => selI - t * s >= 0 && selI - t * s < K);
    const row = (lab, arr, o = {}) => h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, lab), cellGrid([arr], Object.assign({ cs: 30 }, o)));
    box.innerHTML = '';
    box.append(
      row('input x', x, { hl: (i, j) => j === selI, onClick: (i, j) => { selI = j; draw(); } }),
      row('kernel k', k),
      row('output z', z, { hl2: (i, j) => users.includes(j) }),
      row('target y', y),
      row('G = z − y', G, { hl2: (i, j) => users.includes(j) }),
      row('∂J/∂k', dK),
      row('∂J/∂x', dX, { hl: (i, j) => j === selI })
    );
    const M = Array.from({ length: T }, (_, t) => Array.from({ length: N }, (_, i) => { const j = i - t * s; return j >= 0 && j < K ? k[j] : 0; }));
    box.append(h('div', { class: 'grids' },
      gwrap(`forward matrix M (${T}×${N}): z = Mx`, cellGrid(M, { cs: 24, dim: (i, j) => M[i][j] === 0 })),
      gwrap(`backward: ∂J/∂x = Mᵀ G  (Mᵀ is ${N}×${T})`, cellGrid(M[0].map((_, i) => M.map(r => r[i])), { cs: 24, dim: (i, j) => M[j][i] === 0, hl: (i) => i === selI }))
    ));
    calc.innerHTML = `J = ½ Σ (z − y)² = ½ (${G.map(v => pn(v) + '²').join(' + ')}) = <b>${numStr(J, 1)}</b>
kernel gradient (9.11, 1-D):  ∂J/∂k₀ = Σₜ Gₜ · x(t·s + 0) = ${G.map((g, t) => `${pn(g)}×${pn(x[t * s])}`).join(' + ')} = <b>${numStr(dK[0])}</b>   (numerical check: ${num.toFixed(4)})
input gradient (9.13, 1-D):   ∂J/∂x${selI} = Σ over outputs that used x${selI} of  Gₜ · k(${selI} − t·s) = ${users.length ? users.map(t => `${pn(G[t])}×${pn(k[selI - t * s])}`).join(' + ') : '0 (no output uses it)'} = <b>${numStr(dX[selI])}</b>
shared bias b (one per channel): ∂J/∂b = Σₜ Gₜ = <b>${numStr(dB)}</b>
<span class="c">every kernel weight is used at every position, so its gradient SUMS the contributions from all positions — that is parameter sharing seen from the backward pass</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 13) Variable-size inputs ---------- */
Widgets.variableInput = (el) => {
  UI.head(el, 'One network, inputs of any length', 'The same kernel (1, 0, −1) is applied as many times as fits; then pooling over 4 regions (quarters of the feature map) always gives exactly 4 numbers for the classifier.');
  let n = 11;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Input length n', min: 6, max: 22, step: 1, value: n, fmt: v => String(v), onInput: v => { n = v; draw(); } });
  el.append(ctr);
  const box = h('div', { class: 'stack', style: 'gap:6px' });
  const calc = h('div', { class: 'calcbox' });
  el.append(box, calc);
  const rng = mulberry32(21);
  const base = Array.from({ length: 22 }, () => Math.floor(rng() * 6));
  const COLS = ['s1', 's3', 's4', 's7'];
  function draw() {
    const c = Colors.get();
    const x = base.slice(0, n), K = [1, 0, -1];
    const z = []; for (let t = 0; t + 3 <= n; t++) z.push(x[t] - x[t + 2]);
    const Lz = z.length;
    const bounds = [0, 1, 2, 3, 4].map(r => Math.floor(r * Lz / 4));
    const reg = t => { for (let r = 0; r < 4; r++) if (t >= bounds[r] && t < bounds[r + 1]) return r; return 3; };
    const pooled = [0, 1, 2, 3].map(r => { const seg = z.slice(bounds[r], bounds[r + 1]); return seg.length ? Math.max(...seg) : 0; });
    box.innerHTML = '';
    box.append(
      h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, `input (${n})`), cellGrid([x], { cs: 24, scale: 6 })),
      h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, `feature map (${Lz})`), cellGrid([z], { cs: 24, color: (i, j) => alpha(c[COLS[reg(j)]], .3) })),
      h('div', { class: 'unitrow' }, h('span', { class: 'rl' }, 'max per quarter'), cellGrid([pooled], { cs: 34, color: (i, j) => alpha(c[COLS[j]], .45) })),
      h('div', { class: 'io' }, h('span', { class: 'box' }, '4 numbers'), h('span', { class: 'arr' }, '→'), h('span', { class: 'box' }, 'fixed-size classifier (always 4 inputs)'))
    );
    calc.innerHTML = `feature map length = n − k + 1 = ${n} − 3 + 1 = <b>${Lz}</b>   <span class="c">(the kernel is simply applied more or fewer times)</span>
quarters: ${[0, 1, 2, 3].map(r => `[${bounds[r]}, ${bounds[r + 1]})`).join('  ')}   → region r covers ⌊r·${Lz}/4⌋ … ⌊(r+1)·${Lz}/4⌋ − 1
pooled = (${pooled.map(v => numStr(v)).join(', ')})  — always 4 values, whatever n is
as a matrix: the same 3 kernel numbers fill a ${Lz}×${n} Toeplitz matrix — its size changes with the input, its parameters do not
<span class="c">sensible only when the length varies because there is more or less of the SAME kind of observation (longer recording, wider image)</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 14) Build a small CNN: shapes, parameters, receptive field ---------- */
Widgets.cnnBuilder = (el) => {
  UI.head(el, 'A complete small CNN, layer by layer', 'Input: a 32×32 colour image (3 channels). conv → ReLU → pool, twice, then a fully connected layer to 10 classes. Change the design and watch shapes, parameters and cost.');
  let k = 5, c1 = 16, c2 = 32, pad = 'same', pool = true;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Kernel size k×k', min: 1, max: 7, step: 2, value: k, fmt: v => v + '×' + v, onInput: v => { k = v; draw(); } });
  UI.slider(ctr, { label: 'Channels in conv 1', min: 4, max: 64, step: 4, value: c1, fmt: v => String(v), onInput: v => { c1 = v; draw(); } });
  UI.slider(ctr, { label: 'Channels in conv 2', min: 8, max: 128, step: 8, value: c2, fmt: v => String(v), onInput: v => { c2 = v; draw(); } });
  el.append(ctr);
  const tg = h('div', { class: 'toggles' });
  UI.seg(tg, [['same', 'Same padding'], ['valid', 'Valid padding']], pad, v => { pad = v; draw(); });
  UI.chip(tg, '2×2 max pooling', '--s3', pool, v => { pool = v; draw(); });
  el.append(tg);
  const tbox = h('div');
  const calc = h('div', { class: 'calcbox' });
  el.append(tbox, calc);
  function draw() {
    const rows = [];
    let H = 32, C = 3, rf = 1, jump = 1, totalP = 0, totalM = 0;
    rows.push(['input', `${H}×${H}×${C}`, '—', '—', '1']);
    const conv = (name, Cout) => {
      const Ho = pad === 'same' ? H : H - k + 1;
      const P = k * k * C * Cout + Cout, M = Ho * Ho * k * k * C * Cout;
      rf += (k - 1) * jump; H = Ho; C = Cout; totalP += P; totalM += M;
      rows.push([name, `${H}×${H}×${C}`, P.toLocaleString('en-US'), fmt(M, 0), String(rf)]);
      rows.push(['ReLU', `${H}×${H}×${C}`, '0', '—', String(rf)]);
    };
    const pl = () => { if (!pool) return; rf += jump; jump *= 2; H = Math.floor(H / 2); rows.push(['max pool 2×2, stride 2', `${H}×${H}×${C}`, '0', '—', String(rf)]); };
    conv(`conv ${k}×${k}, ${c1} ch`, c1); pl();
    conv(`conv ${k}×${k}, ${c2} ch`, c2); pl();
    const flat = H * H * C, Pfc = flat * 10 + 10;
    totalP += Pfc; totalM += flat * 10;
    rows.push(['flatten', String(flat), '0', '—', '']);
    rows.push(['fully connected → 10', '10', Pfc.toLocaleString('en-US'), fmt(flat * 10, 0), 'all']);
    tbox.innerHTML = '';
    tbox.append(h('div', { class: 'scroll-x' }, h('table', { class: 'data-table' },
      h('thead', {}, h('tr', {}, ['Layer', 'Output shape', 'Parameters', 'Multiply-adds', 'Receptive field'].map(t => h('th', {}, t)))),
      h('tbody', {}, rows.map(r => h('tr', {}, r.map(v => h('td', {}, v)))))
    )));
    const H1 = pad === 'same' ? 32 : 33 - k;
    const fcInstead = 32 * 32 * 3 * H1 * H1 * c1;
    calc.innerHTML = `conv parameters = k·k·C_in·C_out + C_out (one shared bias per output channel): conv 1 = ${k}·${k}·3·${c1} + ${c1} = <b>${(k * k * 3 * c1 + c1).toLocaleString('en-US')}</b>
conv cost = H_out·W_out·k·k·C_in·C_out multiply-adds
total: <b>${totalP.toLocaleString('en-US')}</b> parameters, ${fmt(totalM, 0)} multiply-adds per image
a fully connected layer with the same output as conv 1 would need 3 072 × ${(H1 * H1 * c1).toLocaleString('en-US')} = <b>${fmt(fcInstead, 1)}</b> weights
<span class="c">receptive field grows by (k − 1)×jump per conv and doubles the jump after each pooling; the final layers “see” most of the image</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- quiz ---------- */
window.QUIZ = [
  { q: 'What does “convolutional network” mean, minimally?', opts: ['A network with only pooling layers', 'A neural network that uses convolution instead of general matrix multiplication in at least one layer', 'A network trained with a special optimizer', 'A network for text only'], a: 1, why: 'That is the definition: at least one layer replaces the full matrix multiply by a convolution.' },
  { q: 'Most deep-learning libraries compute S(i,j) = Σ I(i+m, j+n) K(m,n). What is this called?', opts: ['True convolution', 'Cross-correlation (convolution without flipping the kernel)', 'Pooling', 'A Toeplitz inverse'], a: 1, why: 'Cross-correlation skips the kernel flip. Libraries still call it convolution; for a learned kernel it makes no difference.' },
  { q: 'A layer with m = 1 000 inputs, n = 1 000 outputs and kernel width k = 3. How many weights does a dense layer need versus a convolution?', opts: ['1 000 vs 1 000', '1 000 000 vs 3', '3 000 vs 3', '1 000 000 vs 3 000'], a: 1, why: 'Dense: m×n = 10⁶. Sparse connectivity alone: k×n = 3 000 connections; with parameter sharing only k = 3 distinct weights.' },
  { q: 'What is the receptive field of a unit three convolutional layers above the input, with kernel width 3 (stride 1)?', opts: ['3', '5', '7', '9'], a: 2, why: '1 + depth × (k − 1) = 1 + 3×2 = 7 input positions.' },
  { q: 'Convolution is equivariant to translation. This means…', opts: ['Shifting the input leaves the output exactly unchanged', 'Shifting the input shifts the output by the same amount', 'The output is the same for every input', 'It is also equivariant to rotation and scaling'], a: 1, why: 'Equivariance: f(g(x)) = g(f(x)). Invariance (output unchanged) is what pooling approximately adds. Convolution is not naturally equivariant to scale or rotation.' },
  { q: 'Max pooling with width 3 and stride 1. The input shifts by one pixel. What typically happens?', opts: ['All pooled values change', 'Only some pooled values change, because max only reports the largest value, not its position', 'Pooling outputs become zero', 'The number of outputs doubles'], a: 1, why: 'Pooling gives approximate invariance to small translations: many windows still contain the same maximum.' },
  { q: 'Why can pooling be harmful for some tasks?', opts: ['It adds too many parameters', 'It is an infinitely strong prior of translation invariance; if precise location matters, it causes underfitting', 'It makes back-propagation impossible', 'It only works on colour images'], a: 1, why: 'When the task depends on exact positions (e.g. whether two edges meet), pooling everything throws needed information away.' },
  { q: 'Input width 32, kernel 5, stride 1. Output widths for valid, same and full convolution?', opts: ['28, 32, 36', '32, 32, 32', '27, 32, 37', '28, 30, 36'], a: 0, why: 'valid m − k + 1 = 28; same keeps 32; full m + k − 1 = 36.' },
  { q: 'Input width 9, kernel 3, stride 2, no padding. How many outputs?', opts: ['3', '4', '7', '9'], a: 1, why: '⌊(9 − 3)/2⌋ + 1 = 3 + 1 = 4.' },
  { q: 'A locally connected (unshared) layer is useful when…', opts: ['The same feature can appear anywhere in the image', 'Features are local but differ by location, e.g. a mouth only at the bottom of an aligned face', 'The input has variable size', 'We want fewer parameters than a convolution'], a: 1, why: 'Local connectivity is kept, but every location gets its own weights. It has more parameters than convolution and cannot handle variable sizes.' },
  { q: 'During back-propagation through a convolution, how is the gradient for the input computed?', opts: ['By the same forward convolution', 'By multiplying the output gradient by the transpose of the convolution matrix (a transposed convolution)', 'It is always zero', 'By numerical differences'], a: 1, why: 'Since z = Mx, ∂J/∂x = Mᵀ G. Its exact size depends on the stride and padding used in the forward pass.' },
  { q: 'Why can a CNN process images of different sizes while an MLP cannot?', opts: ['CNNs have no parameters', 'The kernel is just applied more or fewer times; pooling into a fixed number of regions gives a fixed-size output', 'CNNs resize every image', 'MLPs can only use one channel'], a: 1, why: 'The parameters (kernels) do not depend on the input size; a final pooling into e.g. quadrants produces a fixed number of features.' }
];
