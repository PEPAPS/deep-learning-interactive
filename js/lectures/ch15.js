/* =========================================================
   Lecture 13 · Ch 15.2 · Transfer learning and domain adaptation
   ========================================================= */
'use strict';

/* ---------- helpers ---------- */
const fx = (v, d = 2) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d).replace('-', '−');
const vec = (a, d = 2) => '(' + a.map(v => fx(v, d)).join(', ') + ')';
const sq = v => v * v;
const sum = a => a.reduce((s, v) => s + v, 0);
const calcBox = () => h('div', { class: 'calcbox' });
const stat = (k, v) => h('div', { class: 'stat' }, h('div', { class: 'k' }, k), h('div', { class: 'v', html: v }));
const pct = v => (100 * v).toFixed(0) + '%';
function cloud(rng, cx, cy, sd, n) { const out = []; for (let i = 0; i < n; i++) out.push([cx + sd * gaussian(rng), cy + sd * gaussian(rng)]); return out; }
function panel(ctx, W, H, ox, xr, yr, title, col) {
  const p = makePlot(W, H, xr, yr, { l: 26, r: 8, t: 22, b: 20 });
  ctx.save(); ctx.translate(ox, 0);
  drawAxes(ctx, p, { grid: false, nx: 4, ny: 4 });
  label(ctx, title, p.pad.l + 4, 9, { size: 11, bold: true, color: col });
  return { p, done: () => ctx.restore() };
}

/* ---------- 1) transfer, adaptation, drift: what stays, what changes ---------- */
Widgets.twoSettings = (el) => {
  UI.head(el, 'Setting 1 → setting 2: what stays the same, what changes?', 'Two distributions, P₁ (lots of data) and P₂ (little data). Pick a situation to see which part of the problem is shared — that shared part is what makes transfer possible.');
  let mode = 'transfer';
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, [['transfer', 'Transfer learning'], ['adapt', 'Domain adaptation'], ['drift', 'Concept drift'], ['multi', 'Multitask']], mode, v => { mode = v; draw(); });
  el.append(seg);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'two settings compared' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  const INFO = {
    transfer: { t1: 'P₁: cats vs. dogs (10 000 labeled photos)', t2: 'P₂: ants vs. wasps (20 labeled photos)', same: 'the kind of input (photos) and the low-level factors: edges, shapes, lighting', diff: 'the target: new categories', plan: 'reuse the representation learned on P₁; learn only the new output mapping from the few P₂ examples',
      txt: 'input = photos in both settings; target = different categories\nassumption: many factors that explain the variation in P₁ (edges, shapes, lighting…) are relevant to P₂\nso the representation learned on P₁ lets us generalize from only a few P₂ examples' },
    adapt: { t1: 'P₁: reviews of books & films (many labels)', t2: 'P₂: reviews of TVs & phones (few/no labels)', same: 'the task and the optimal input→output mapping: is the statement positive or negative?', diff: 'the input distribution: vocabulary and style shift', plan: 'keep the task; adapt the model to the shifted inputs',
      txt: 'same task (sentiment), same underlying function positive/neutral/negative\ninput distribution slightly different: “gripping plot” vs “crisp screen”\nthe model must cope with a shift of the inputs, not learn a new target' },
    drift: { t1: 'P₁: the data this year', t2: 'P₂: the data next year', same: 'the task', diff: 'the data distribution, gradually, over time', plan: 'treat it as transfer learning due to gradual change: keep adapting',
      txt: 'concept drift = the input (and sometimes the mapping) changes slowly over time\na form of transfer learning: P₂ is P₁ moved a little\nboth concept drift and transfer learning can be viewed as particular forms of multitask learning' },
    multi: { t1: 'task A (its own data)', t2: 'task B (its own data)', same: 'a representation used by both tasks', diff: 'the tasks and their outputs', plan: 'one shared representation benefits from the training data of both tasks',
      txt: 'multitask learning: use data from setting 1 to extract information useful in setting 2 (and vice versa)\nusing the same representation in both settings lets it benefit from the data available for both tasks\ntransfer learning and concept drift are special cases of this' }
  };
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 250, ctx = fitCanvas(cv, W, H);
    const half = (W - 20) / 2, rng = mulberry32(5);
    const A = panel(ctx, half, H, 0, [-3, 3], [-3, 3], INFO[mode].t1, c.s1);
    const draw2 = (p, cls, cols, shift) => {
      ctx.save(); p.clip(ctx);
      cls.forEach((cl, k) => cloud(rng, cl[0] + (shift ? shift[0] : 0), cl[1] + (shift ? shift[1] : 0), cl[2], 40).forEach(q => dot(ctx, p.X(q[0]), p.Y(q[1]), 3, alpha(cols[k], .7))));
      ctx.restore();
    };
    const cls1 = [[-1.2, 1, 0.55], [1.2, -0.8, 0.55]];
    draw2(A.p, cls1, [c.s1, c.s2]);
    label(ctx, mode === 'multi' ? 'class 1 · class 2' : mode === 'transfer' ? 'cat · dog' : 'positive · negative', A.p.pad.l + 4, H - 6, { size: 10, color: c.text2 });
    A.done();
    const B = panel(ctx, half, H, half + 20, [-3, 3], [-3, 3], INFO[mode].t2, c.s3);
    if (mode === 'transfer') { draw2(B.p, [[-0.6, -1.4, 0.5], [1.4, 1.2, 0.5]], [c.s7, c.s4]); label(ctx, 'ant · wasp (new classes, same feature axes)', B.p.pad.l + 4, H - 6, { size: 10, color: c.text2 }); }
    else if (mode === 'adapt') { draw2(B.p, cls1, [c.s1, c.s2], [1.1, 0.9]); arrow(ctx, B.p.X(-1.2), B.p.Y(1), B.p.X(-0.2), B.p.Y(1.8), c.text, 1.6, 7); label(ctx, 'same classes, inputs shifted', B.p.pad.l + 4, H - 6, { size: 10, color: c.text2 }); }
    else if (mode === 'drift') { [0.35, 0.7, 1.05].forEach((t, i) => { ctx.globalAlpha = 0.35 + 0.3 * i; draw2(B.p, cls1, [c.s1, c.s2], [t, t * 0.6]); }); ctx.globalAlpha = 1; label(ctx, 'the same data drifting month by month', B.p.pad.l + 4, H - 6, { size: 10, color: c.text2 }); }
    else { draw2(B.p, [[-1.5, -1, 0.5], [0.3, 1.5, 0.5], [1.8, -1.2, 0.5]], [c.s7, c.s4, c.s8]); label(ctx, 'a different task on related inputs', B.p.pad.l + 4, H - 6, { size: 10, color: c.text2 }); }
    B.done();
    arrow(ctx, half + 2, H / 2, half + 18, H / 2, c.muted, 2, 7);
    stats.innerHTML = '';
    stats.append(stat('shared', INFO[mode].same), stat('changes', INFO[mode].diff), stat('plan', INFO[mode].plan));
    calc.innerHTML = INFO[mode].txt;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) few-shot: raw features vs. a representation learned on P1 ---------- */
Widgets.fewShot = (el) => {
  UI.head(el, 'Why transfer helps: learn new classes from a handful of examples', 'Three new classes (P₂). Left: raw input features — the classes overlap. Right: features learned on the big P₁ dataset — the same points, cleanly separated. A nearest-centroid classifier is fit on n labeled examples per class and tested on 150 per class.');
  let n = 2, rel = 1, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Labeled examples per new class (n)', min: 1, max: 20, step: 1, value: n, fmt: v => String(v), onInput: v => { n = v; draw(); } });
  UI.slider(ctr, { label: 'How relevant P₁’s factors are to P₂', min: 0, max: 1, step: 0.05, value: rel, fmt: v => fx(v), onInput: v => { rel = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Resample', () => { seed++; draw(); });
  el.append(ctr, btns);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'raw features versus learned features' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  const C = [[-1.3, 1.1], [1.4, 0.9], [0.1, -1.4]];
  function experiment(sd, rng) {
    const train = C.map(cc => cloud(rng, cc[0], cc[1], sd, n));
    const test = C.map(cc => cloud(rng, cc[0], cc[1], sd, 150));
    const cent = train.map(pts => [mean(pts.map(q => q[0])), mean(pts.map(q => q[1]))]);
    let ok = 0;
    test.forEach((pts, k) => pts.forEach(q => { let b = 0, bd = 1e9; cent.forEach((m, j) => { const d = sq(q[0] - m[0]) + sq(q[1] - m[1]); if (d < bd) { bd = d; b = j; } }); if (b === k) ok++; }));
    return { train, test, cent, acc: ok / 450 };
  }
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 270, ctx = fitCanvas(cv, W, H);
    const half = (W - 20) / 2, cols = [c.s1, c.s2, c.s7];
    const sdRaw = 1.05, sdRep = 1.05 - 0.75 * rel;
    const r1 = experiment(sdRaw, mulberry32(seed * 11 + 1)), r2 = experiment(sdRep, mulberry32(seed * 11 + 1));
    const show = (res, ox, title) => {
      const P = panel(ctx, half, H, ox, [-3.5, 3.5], [-3.5, 3.5], title, c.text);
      ctx.save(); P.p.clip(ctx);
      res.test.forEach((pts, k) => pts.forEach(q => dot(ctx, P.p.X(q[0]), P.p.Y(q[1]), 2.2, alpha(cols[k], .35))));
      res.train.forEach((pts, k) => pts.forEach(q => dot(ctx, P.p.X(q[0]), P.p.Y(q[1]), 5, cols[k], c.text)));
      res.cent.forEach((m, k) => { ctx.strokeStyle = cols[k]; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(P.p.X(m[0]) - 7, P.p.Y(m[1]) - 7); ctx.lineTo(P.p.X(m[0]) + 7, P.p.Y(m[1]) + 7); ctx.moveTo(P.p.X(m[0]) + 7, P.p.Y(m[1]) - 7); ctx.lineTo(P.p.X(m[0]) - 7, P.p.Y(m[1]) + 7); ctx.stroke(); });
      ctx.restore();
      label(ctx, `test accuracy ${pct(res.acc)}`, P.p.pad.l + 4, H - 6, { size: 11, bold: true, color: res.acc > 0.9 ? c.good : c.critical });
      P.done();
    };
    show(r1, 0, 'raw input features (pixels)');
    show(r2, half + 20, 'features learned on P₁');
    stats.innerHTML = '';
    stats.append(stat('labeled per class', String(n)), stat('accuracy, raw', pct(r1.acc)), stat('accuracy, learned features', pct(r2.acc)), stat('cluster spread', `${fx(sdRaw)} → ${fx(sdRep)}`));
    calc.innerHTML = `nearest-centroid classifier: centroid_k = average of the n labeled points of class k;  predict the class whose centroid is closest\nlearned-feature centroids: ${r2.cent.map((m, k) => `c${k + 1} = ${vec(m)}`).join(',  ')}\nbig dots = the n labeled P₂ examples; small dots = 450 test points; × = estimated centroids\n<span class="c">${rel < 0.3 ? 'With P₁’s factors irrelevant to P₂ the learned features are no better than raw pixels: transfer needs shared factors of variation.' : n <= 2 ? 'With cleanly separated classes, one or two labeled examples per class are enough (this is one-shot learning). On raw features the same examples are hopeless.' : 'More labels help both, but the learned representation reaches high accuracy with far fewer — the data of P₁ did most of the work.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) which layers to share ---------- */
Widgets.whichLayers = (el) => {
  UI.head(el, 'Which layers to share: lower, upper, or your choice', 'A 6-layer network. Blue layers are reused from setting 1 (frozen); orange layers are trained on the small setting-2 dataset. Pick a scenario, or click layers to toggle them.');
  const SIZES = [256, 192, 128, 96, 64, 32, 10];
  const NAMES = ['input', 'layer 1', 'layer 2', 'layer 3', 'layer 4', 'layer 5', 'output'];
  let reuse = [true, true, true, false, false, false];
  let scen = 'vision';
  const SC = {
    vision: { reuse: [true, true, true, true, false, false], why: 'Vision: new categories (ants/wasps) share the <b>semantics of the input</b> with cats/dogs — edges, shapes, lighting. Reuse the lower layers; retrain the top.' },
    speech: { reuse: [false, false, true, true, true, true], why: 'Speech: every speaker needs <b>different early processing</b> (same phonemes sound different), but the output must be <b>valid sentences</b> for everyone. Share the upper layers; make the input layers task-specific.' },
    custom: { reuse: null, why: 'Your own split. Rule of thumb: share the layers whose job is the same in both settings.' }
  };
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, [['vision', 'Vision: new categories'], ['speech', 'Speech: new speaker'], ['custom', 'Custom (click layers)']], scen, v => { scen = v; if (SC[v].reuse) reuse = SC[v].reuse.slice(); draw(); });
  el.append(seg);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart clickable', role: 'img', 'aria-label': 'network layers, reused or retrained' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), why = h('div', { class: 'callout info' }), calc = calcBox();
  el.append(wrap, stats, why, calc);
  let geo = null;
  cv.addEventListener('click', e => {
    if (!geo) return;
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left;
    const k = geo.findIndex(([a, b]) => px >= a && px <= b);
    if (k >= 0) { reuse[k] = !reuse[k]; scen = 'custom'; draw(); }
  });
  function draw() {
    if (SC[scen].reuse) reuse = SC[scen].reuse.slice();
    const c = Colors.get(), W = boxWidth(wrap), H = 210, ctx = fitCanvas(cv, W, H);
    const n = SIZES.length, X = i => 24 + (i + .5) * (W - 48) / n;
    geo = [];
    let pRe = 0, pTr = 0; const parts = [];
    for (let i = 1; i < n; i++) {
      const pm = SIZES[i - 1] * SIZES[i] + SIZES[i];
      if (reuse[i - 1]) pRe += pm; else pTr += pm;
      parts.push(`${NAMES[i]}: ${SIZES[i - 1]}×${SIZES[i]} + ${SIZES[i]} = ${pm.toLocaleString('en-US')} ${reuse[i - 1] ? '(reused)' : '(trained on P₂)'}`);
      const col = reuse[i - 1] ? c.s1 : c.s2, x0 = (X(i - 1) + X(i)) / 2;
      ctx.fillStyle = alpha(col, .18); ctx.fillRect(X(i - 1) + 14, 30, X(i) - X(i - 1) - 28, 120);
      geo.push([X(i - 1) + 14, X(i) - 14]);
      label(ctx, reuse[i - 1] ? 'reused' : 'trained', x0, 162, { align: 'center', size: 10, bold: true, color: col });
      label(ctx, pm.toLocaleString('en-US') + ' w', x0, 176, { align: 'center', size: 9, color: c.muted });
    }
    for (let i = 0; i < n; i++) {
      const hgt = 30 + 90 * Math.sqrt(SIZES[i] / 256), x = X(i), y = 90 - hgt / 2;
      ctx.fillStyle = c.surface; ctx.strokeStyle = c.border; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x - 13, y, 26, hgt, 6) : ctx.rect(x - 13, y, 26, hgt); ctx.fill(); ctx.stroke();
      label(ctx, String(SIZES[i]), x, 90, { align: 'center', size: 10, bold: true, color: c.text });
      label(ctx, NAMES[i], x, 16, { align: 'center', size: 10, color: c.text2 });
    }
    label(ctx, 'click a weight block to toggle it', W - 10, H - 8, { align: 'right', size: 10, color: c.muted });
    stats.innerHTML = '';
    stats.append(stat('reused from setting 1', pRe.toLocaleString('en-US') + ' <small>weights</small>'), stat('trained on setting 2', pTr.toLocaleString('en-US') + ' <small>weights</small>'), stat('with 50 P₂ examples', pTr > 20000 ? '⚠ far too many weights to fit' : pTr > 5000 ? 'borderline' : 'fine: few weights per example'));
    why.innerHTML = `<span class="ico">i</span><div>${SC[scen].why}</div>`;
    calc.innerHTML = parts.join('\n') + `\nreused: <b>${pRe.toLocaleString('en-US')}</b>   trained from scratch on P₂: <b>${pTr.toLocaleString('en-US')}</b>   (${pct(pRe / (pRe + pTr))} of the network comes from setting 1)`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) domain adaptation: a shifted input distribution ---------- */
Widgets.domainShift = (el) => {
  UI.head(el, 'Domain adaptation: same task, shifted inputs', 'Sentiment classifier trained on book reviews (source). Feature 1 = positive-word score (what the task depends on); feature 2 = topic/style words. Electronics reviews (target) are the same task with shifted inputs. Slide the shift and adapt.');
  let dx = 0.3, dy = 1.4, adapt = 'none', seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Shift along feature 1 (vocabulary)', min: -2, max: 2, step: 0.1, value: dx, fmt: v => fx(v, 1), onInput: v => { dx = v; draw(); } });
  UI.slider(ctr, { label: 'Shift along feature 2 (style)', min: -2, max: 2, step: 0.1, value: dy, fmt: v => fx(v, 1), onInput: v => { dy = v; draw(); } });
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, [['none', 'No adaptation'], ['center', 'Re-center on target mean (no labels)'], ['few', 'Fine-tune with 6 target labels']], adapt, v => { adapt = v; draw(); });
  el.append(ctr, seg);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'source and target domains with the decision boundary' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  const rngS = mulberry32(3);
  const src = [cloud(rngS, 1, 0, 0.55, 80), cloud(rngS, -1, 0, 0.55, 80)];
  const centroid = pts => [mean(pts.map(q => q[0])), mean(pts.map(q => q[1]))];
  const fitLine = (pos, neg) => { const a = centroid(pos), b = centroid(neg); const w = [a[0] - b[0], a[1] - b[1]], m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; return { w, b: -(w[0] * m[0] + w[1] * m[1]) }; };
  const accOf = (L, pos, neg) => { let ok = 0; pos.forEach(q => { if (L.w[0] * q[0] + L.w[1] * q[1] + L.b > 0) ok++; }); neg.forEach(q => { if (L.w[0] * q[0] + L.w[1] * q[1] + L.b <= 0) ok++; }); return ok / (pos.length + neg.length); };
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 300, ctx = fitCanvas(cv, W, H);
    const rngT = mulberry32(seed * 7 + 2);
    const tgt = [cloud(rngT, 1 + dx, dy, 0.55, 80), cloud(rngT, -1 + dx, dy, 0.55, 80)];
    const L0 = fitLine(src[0], src[1]);
    let L = L0, note = '';
    if (adapt === 'center') {
      const ms = centroid([...src[0], ...src[1]]), mt = centroid([...tgt[0], ...tgt[1]]);
      const sh = [mt[0] - ms[0], mt[1] - ms[1]];
      L = { w: L0.w, b: L0.b - (L0.w[0] * sh[0] + L0.w[1] * sh[1]) };
      note = `estimated shift = target mean − source mean = ${vec(sh)} (no labels needed) → move the boundary by the same amount`;
    } else if (adapt === 'few') {
      L = fitLine(tgt[0].slice(0, 3), tgt[1].slice(0, 3));
      note = 'boundary re-fit from only 3 + 3 labeled target reviews (the big dots) — few labels go a long way when the task is unchanged';
    }
    const p = makePlot(W, H, [-4, 4], [-3, 4], { l: 30, r: 10, t: 12, b: 26 });
    drawAxes(ctx, p, { xlabel: 'feature 1: positive-word score', ylabel: 'feature 2: style' });
    ctx.save(); p.clip(ctx);
    src[0].forEach(q => dot(ctx, p.X(q[0]), p.Y(q[1]), 2.5, alpha(c.s1, .3)));
    src[1].forEach(q => dot(ctx, p.X(q[0]), p.Y(q[1]), 2.5, alpha(c.s2, .3)));
    tgt[0].forEach((q, i) => dot(ctx, p.X(q[0]), p.Y(q[1]), adapt === 'few' && i < 3 ? 6 : 3.5, c.s1, adapt === 'few' && i < 3 ? c.text : null));
    tgt[1].forEach((q, i) => dot(ctx, p.X(q[0]), p.Y(q[1]), adapt === 'few' && i < 3 ? 6 : 3.5, c.s2, adapt === 'few' && i < 3 ? c.text : null));
    const lineOf = (Ln, col, dash) => { ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.setLineDash(dash || []); ctx.beginPath(); if (Math.abs(Ln.w[1]) > 1e-6) { for (const xx of [-4, 4]) { const yy = -(Ln.w[0] * xx + Ln.b) / Ln.w[1]; xx === -4 ? ctx.moveTo(p.X(xx), p.Y(yy)) : ctx.lineTo(p.X(xx), p.Y(yy)); } } else { const xx = -Ln.b / Ln.w[0]; ctx.moveTo(p.X(xx), p.Y(-3)); ctx.lineTo(p.X(xx), p.Y(4)); } ctx.stroke(); ctx.setLineDash([]); };
    lineOf(L0, alpha(c.text, .4), [5, 4]);
    lineOf(L, c.text);
    ctx.restore();
    label(ctx, 'faint = source (books); solid = target (electronics)', p.pad.l + 6, p.pad.t + 10, { size: 10, color: c.text2 });
    label(ctx, 'dashed = boundary learned on the source; solid = boundary in use', p.pad.l + 6, p.pad.t + 24, { size: 10, color: c.text2 });
    const accS = accOf(L0, src[0], src[1]), accT0 = accOf(L0, tgt[0], tgt[1]), accT = accOf(L, tgt[0], tgt[1]);
    stats.innerHTML = '';
    stats.append(stat('source accuracy', pct(accS)), stat('target, source boundary', pct(accT0)), stat('target, after adaptation', pct(accT)));
    calc.innerHTML = `source boundary: ${fx(L0.w[0])}·f₁ + ${fx(L0.w[1])}·f₂ + ${fx(L0.b)} = 0   (direction ≈ feature 1 only: the task depends on positive words, not on style)\ntarget = source shifted by (${fx(dx, 1)}, ${fx(dy, 1)})   →   accuracy with the old boundary: <b>${pct(accT0)}</b>${note ? '\n' + note + `   →   <b>${pct(accT)}</b>` : ''}\n<span class="c">${Math.abs(dx) < 0.4 ? 'A shift along the style axis (feature 2) is harmless: the underlying function positive/negative does not depend on it.' : 'A shift along feature 1 (different vocabulary for “good”) moves the classes across the old boundary — the same function needs a re-adapted model.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) concept drift ---------- */
Widgets.driftDemo = (el) => {
  UI.head(el, 'Concept drift: the data move while the model stands still', 'The two classes slowly rotate and shift month by month. A model trained once (month 0) keeps its old boundary; another is retrained every k months on fresh data.');
  let t = 12, every = 6;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Month', min: 0, max: 36, step: 1, value: t, fmt: v => String(v), onInput: v => { t = v; draw(); } });
  UI.slider(ctr, { label: 'Retrain every k months', min: 1, max: 12, step: 1, value: every, fmt: v => String(v), onInput: v => { every = v; draw(); } });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'drifting classes and two boundaries' });
  wrap.append(cv);
  const chart = new LineChart(el, { xr: [0, 36], yr: [0.4, 1.02], aspect: .38, minH: 160, maxH: 200, xlabel: 'month', ylabel: 'accuracy', xname: 'month', vfmt: v => pct(v), snapX: v => Math.round(v), aria: 'accuracy over time' });
  const calc = calcBox();
  el.append(calc);
  const base = (() => { const r = mulberry32(9); return [cloud(r, 0, 0, 0.5, 120), cloud(r, 0, 0, 0.5, 120)]; })();
  const centers = m => { const a = m * 4 * Math.PI / 180, s = m * 0.03; return [[1.1 * Math.cos(a) + s, 1.1 * Math.sin(a)], [-1.1 * Math.cos(a) + s, -1.1 * Math.sin(a)]]; };
  const dataAt = m => { const cc = centers(m); return base.map((pts, k) => pts.map(q => [q[0] + cc[k][0], q[1] + cc[k][1]])); };
  const boundaryAt = m => { const cc = centers(m); const w = [cc[0][0] - cc[1][0], cc[0][1] - cc[1][1]], mid = [(cc[0][0] + cc[1][0]) / 2, (cc[0][1] + cc[1][1]) / 2]; return { w, b: -(w[0] * mid[0] + w[1] * mid[1]) }; };
  const accOf = (L, d) => { let ok = 0; d[0].forEach(q => { if (L.w[0] * q[0] + L.w[1] * q[1] + L.b > 0) ok++; }); d[1].forEach(q => { if (L.w[0] * q[0] + L.w[1] * q[1] + L.b <= 0) ok++; }); return ok / 240; };
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 240, ctx = fitCanvas(cv, W, H);
    const d = dataAt(t), L0 = boundaryAt(0), tr = Math.floor(t / every) * every, Lk = boundaryAt(tr);
    const p = makePlot(W, H, [-3.5, 3.5], [-2.6, 2.6], { l: 28, r: 10, t: 10, b: 22 });
    drawAxes(ctx, p, { grid: false });
    ctx.save(); p.clip(ctx);
    d[0].forEach(q => dot(ctx, p.X(q[0]), p.Y(q[1]), 3, alpha(c.s1, .6)));
    d[1].forEach(q => dot(ctx, p.X(q[0]), p.Y(q[1]), 3, alpha(c.s2, .6)));
    const lineOf = (Ln, col, dash) => { ctx.strokeStyle = col; ctx.lineWidth = 2.2; ctx.setLineDash(dash || []); ctx.beginPath(); let first = true; for (const xx of [-4, 4]) { const yy = Math.abs(Ln.w[1]) > 1e-6 ? -(Ln.w[0] * xx + Ln.b) / Ln.w[1] : null; if (yy == null) continue; first ? ctx.moveTo(p.X(xx), p.Y(yy)) : ctx.lineTo(p.X(xx), p.Y(yy)); first = false; } if (Math.abs(Ln.w[1]) <= 1e-6) { const xx = -Ln.b / Ln.w[0]; ctx.moveTo(p.X(xx), p.Y(-3)); ctx.lineTo(p.X(xx), p.Y(3)); } ctx.stroke(); ctx.setLineDash([]); };
    lineOf(L0, c.critical, [5, 4]);
    lineOf(Lk, c.good);
    ctx.restore();
    label(ctx, `month ${t}`, p.pad.l + 6, p.pad.t + 10, { size: 12, bold: true, color: c.text });
    label(ctx, 'dashed red: trained at month 0', p.pad.l + 6, p.pad.t + 26, { size: 10, color: c.critical });
    label(ctx, `green: retrained at month ${tr}`, p.pad.l + 6, p.pad.t + 40, { size: 10, color: c.good });
    const a0 = [], ak = [];
    for (let m = 0; m <= 36; m++) { const dm = dataAt(m); a0.push([m, accOf(L0, dm)]); ak.push([m, accOf(boundaryAt(Math.floor(m / every) * every), dm)]); }
    chart.set([{ name: 'trained once (month 0)', color: 'critical', data: a0 }, { name: `retrained every ${every} months`, color: 'good', data: ak }, { name: 'now', color: 'muted', data: [[t, 0.4], [t, 1.02]], dash: [4, 4], noHover: true }]);
    calc.innerHTML = `month ${t}: class centres rotated by ${t * 4}° and shifted by ${fx(t * 0.03)}\naccuracy of the month-0 model: <b>${pct(accOf(L0, d))}</b>     model retrained at month ${tr}: <b>${pct(accOf(Lk, d))}</b>\n<span class="c">Concept drift is transfer learning in slow motion: each month’s data are a slightly different P₂. Keeping the representation but refreshing it with recent data is the usual answer.</span>`;
  }
  draw();
  return { redraw: () => { chart.draw(); draw(); } };
};

/* ---------- 6) multitask: statistical strength of a shared representation ---------- */
Widgets.multitaskDemo = (el) => {
  UI.head(el, 'Multitask learning: the shared part learns from both datasets', 'Two tasks share a representation (the trunk) and have their own heads. Toy model: the shared part must estimate one quantity μ from noisy data. Alone, task B has only n_B examples; shared with task A it effectively has n_A + n_B.');
  let nA = 120, nB = 8, sd = 1, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Examples of task A (n_A)', min: 0, max: 300, step: 10, value: nA, fmt: v => String(v), onInput: v => { nA = v; draw(); } });
  UI.slider(ctr, { label: 'Examples of task B (n_B)', min: 1, max: 60, step: 1, value: nB, fmt: v => String(v), onInput: v => { nB = v; draw(); } });
  UI.slider(ctr, { label: 'Noise σ of each example', min: 0.3, max: 2, step: 0.1, value: sd, fmt: v => fx(v, 1), onInput: v => { sd = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Resample', () => { seed++; draw(); });
  el.append(ctr, btns);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'shared trunk with two heads and the estimate of the shared quantity' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 260, ctx = fitCanvas(cv, W, H);
    const rng = mulberry32(seed * 13 + 5), mu = 2;
    const A = Array.from({ length: nA }, () => mu + sd * gaussian(rng)), B = Array.from({ length: nB }, () => mu + sd * gaussian(rng));
    const mB = nB ? mean(B) : NaN, mAB = (sum(A) + sum(B)) / (nA + nB);
    const seB = sd / Math.sqrt(nB), seAB = sd / Math.sqrt(nA + nB);
    // left: architecture
    const lx = 20, bw = Math.min(250, W * 0.42);
    const box = (x, y, w, hh, txt, col) => { ctx.fillStyle = alpha(col, .18); ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, hh, 8) : ctx.rect(x, y, w, hh); ctx.fill(); ctx.stroke(); label(ctx, txt, x + w / 2, y + hh / 2, { align: 'center', size: 10.5, bold: true, color: c.text }); };
    box(lx, 20, bw * 0.42, 34, 'data A (n_A)', c.s1); box(lx + bw * 0.58, 20, bw * 0.42, 34, 'data B (n_B)', c.s2);
    box(lx + bw * 0.15, 90, bw * 0.7, 40, 'shared representation', c.s3);
    box(lx, 170, bw * 0.42, 34, 'head A → loss J_A', c.s1); box(lx + bw * 0.58, 170, bw * 0.42, 34, 'head B → loss J_B', c.s2);
    arrow(ctx, lx + bw * 0.21, 56, lx + bw * 0.4, 88, c.s1, 1.5, 6); arrow(ctx, lx + bw * 0.79, 56, lx + bw * 0.6, 88, c.s2, 1.5, 6);
    arrow(ctx, lx + bw * 0.4, 132, lx + bw * 0.21, 168, c.s1, 1.5, 6); arrow(ctx, lx + bw * 0.6, 132, lx + bw * 0.79, 168, c.s2, 1.5, 6);
    label(ctx, 'gradients of J_A + J_B both reach the shared part', lx + bw / 2, H - 14, { align: 'center', size: 10, color: c.text2 });
    // right: sampling distributions
    const px0 = lx + bw + 24, pw = W - px0 - 10;
    const p = makePlot(pw, H, [mu - 2.2, mu + 2.2], [0, 1], { l: 10, r: 6, t: 22, b: 24 });
    ctx.save(); ctx.translate(px0, 0);
    drawAxes(ctx, p, { grid: false, yticks: [], xlabel: 'estimate of μ' });
    label(ctx, 'how precisely the shared quantity is estimated', p.pad.l + 2, 9, { size: 10.5, bold: true, color: c.text });
    const bell = (m, s, col, fill) => { const top = normPdf(0, 0, Math.min(seB, seAB)); ctx.strokeStyle = col; ctx.fillStyle = alpha(col, .18); ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i <= 120; i++) { const xx = mu - 2.2 + 4.4 * i / 120, yy = normPdf(xx, m, s) / top; i ? ctx.lineTo(p.X(xx), p.Y(yy)) : ctx.moveTo(p.X(xx), p.Y(yy)); } ctx.stroke(); if (fill) { ctx.lineTo(p.X(mu + 2.2), p.Y(0)); ctx.lineTo(p.X(mu - 2.2), p.Y(0)); ctx.closePath(); ctx.fill(); } };
    bell(mu, seB, c.s2, true); bell(mu, seAB, c.s3, true);
    ctx.strokeStyle = c.text; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(p.X(mu), p.Y(0)); ctx.lineTo(p.X(mu), p.Y(1)); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, 'true μ', p.X(mu), p.pad.t + 4, { align: 'center', size: 10, color: c.text });
    dot(ctx, p.X(clamp(mB, mu - 2.2, mu + 2.2)), p.Y(0.04), 5, c.s2, c.surface); dot(ctx, p.X(clamp(mAB, mu - 2.2, mu + 2.2)), p.Y(0.1), 5, c.s3, c.surface);
    label(ctx, 'orange: B alone · green: shared with A', p.pad.l + 2, H - 8, { size: 10, color: c.text2 });
    ctx.restore();
    stats.innerHTML = '';
    stats.append(stat('B alone: std. error σ/√n_B', fx(seB, 3)), stat('shared: σ/√(n_A+n_B)', fx(seAB, 3)), stat('precision gain', (seB / seAB).toFixed(1) + '×'), stat('this sample: error B / shared', `${fx(Math.abs(mB - mu))} / ${fx(Math.abs(mAB - mu))}`));
    calc.innerHTML = `shared loss J = J_A + J_B — the shared representation is updated by the gradients of both tasks\nestimate from B only: mean of ${nB} noisy values = ${fx(mB)} (true μ = 2); its standard error σ/√n_B = ${fx(sd, 1)}/√${nB} = <b>${fx(seB, 3)}</b>\nestimate from A + B: mean of ${nA + nB} values = ${fx(mAB)}; standard error σ/√(n_A+n_B) = ${fx(sd, 1)}/√${nA + nB} = <b>${fx(seAB, 3)}</b>\n<span class="c">The shared part benefits from the training data available for both tasks (statistical strength). Transfer learning is this with the tasks learned one after the other; concept drift is this with the tasks spread over time.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) one-shot learning in a good representation ---------- */
Widgets.oneShotDemo = (el) => {
  UI.head(el, 'One-shot learning: one labeled example per class is enough — if the representation is good', 'Three new classes, unlabeled, in a representation learned during the first stage. Click a point to give it the next label (or let the demo pick one per class). Every other point is then classified by its nearest labeled example.');
  let quality = 'good', labeled = [], seed = 1;
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, [['good', 'Representation separates the classes'], ['poor', 'Poor representation (overlap)']], quality, v => { quality = v; labeled = []; draw(); });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Label one random example per class', () => { labeled = [0, 1, 2].map(k => ({ k, i: Math.floor(Math.random() * 60) })); draw(); }, 'btn primary');
  UI.button(btns, 'Clear labels', () => { labeled = []; draw(); });
  UI.button(btns, 'New points', () => { seed++; labeled = []; draw(); });
  el.append(seg, btns);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart clickable', role: 'img', 'aria-label': 'points in representation space with labeled examples' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  const C = [[-1.4, 1.1], [1.5, 0.9], [0, -1.5]];
  let pts = null, P = null;
  function gen() { const rng = mulberry32(seed * 3 + 1), sd = quality === 'good' ? 0.35 : 1.0; pts = C.map(cc => cloud(rng, cc[0], cc[1], sd, 60)); }
  cv.addEventListener('click', e => {
    if (!P || !pts) return;
    const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    let best = null, bd = 1e9;
    pts.forEach((cl, k) => cl.forEach((q, i) => { const d = sq(P.X(q[0]) - px) + sq(P.Y(q[1]) - py); if (d < bd) { bd = d; best = { k, i }; } }));
    if (best && bd < 400) { labeled.push(best); if (labeled.length > 3) labeled.shift(); draw(); }
  });
  function draw() {
    gen();
    const c = Colors.get(), W = boxWidth(wrap), H = 300, ctx = fitCanvas(cv, W, H);
    const cols = [c.s1, c.s2, c.s7];
    P = makePlot(W, H, [-3.5, 3.5], [-3.5, 3.5], { l: 28, r: 10, t: 10, b: 22 });
    drawAxes(ctx, P, { grid: false, xlabel: 'representation dim 1', ylabel: 'dim 2' });
    const labs = labeled.map((L, j) => ({ ...L, lab: j, pt: pts[L.k][L.i] }));
    let ok = 0, tot = 0;
    ctx.save(); P.clip(ctx);
    pts.forEach((cl, k) => cl.forEach(q => {
      let pred = -1;
      if (labs.length) { let bd = 1e9; labs.forEach(L => { const d = sq(q[0] - L.pt[0]) + sq(q[1] - L.pt[1]); if (d < bd) { bd = d; pred = L.lab; } }); }
      const col = pred < 0 ? c.muted : cols[pred];
      dot(ctx, P.X(q[0]), P.Y(q[1]), 3.2, alpha(col, .55));
      if (pred >= 0) { tot++; if (labs[pred].k === k) ok++; }
    }));
    labs.forEach(L => { dot(ctx, P.X(L.pt[0]), P.Y(L.pt[1]), 9, cols[L.lab], c.text); label(ctx, String.fromCharCode(65 + L.lab), P.X(L.pt[0]), P.Y(L.pt[1]), { align: 'center', size: 11, bold: true, color: '#fff' }); });
    ctx.restore();
    label(ctx, labs.length ? `labels given: ${labs.length} · points coloured by nearest labeled example` : 'no labels yet — click a point or press the button', P.pad.l + 6, P.pad.t + 10, { size: 10.5, color: c.text2 });
    const purity = tot ? ok / tot : 0;
    const consistent = labs.length === 3 && new Set(labs.map(L => L.k)).size === 3;
    stats.innerHTML = '';
    stats.append(stat('labels used', `${labs.length} <small>/ 180 points</small>`), stat('points agreeing with their label’s cluster', tot ? pct(purity) : '—'), stat('verdict', !labs.length ? 'label something' : consistent && purity > 0.9 ? 'one example per class was enough' : consistent ? 'clusters overlap: one example is not enough' : 'label one point in each cluster'));
    const q0 = pts[1][5];
    calc.innerHTML = labs.length ? `test point x = ${vec(q0)}:  distances to the labeled examples = ${labs.map(L => `${String.fromCharCode(65 + L.lab)}: ${fx(Math.hypot(q0[0] - L.pt[0], q0[1] - L.pt[1]))}`).join(',  ')}  → label ${String.fromCharCode(65 + labs.reduce((b, L, j) => Math.hypot(q0[0] - L.pt[0], q0[1] - L.pt[1]) < Math.hypot(q0[0] - labs[b].pt[0], q0[1] - labs[b].pt[1]) ? j : b, 0))}\n<span class="c">${quality === 'good' ? 'Because the first stage learned to separate the underlying classes cleanly, all test examples of a class cluster around one point — so a single labeled example tells us the label of the whole cluster.' : 'With a poor representation the clusters overlap and nearest-example classification fails: one-shot learning is only possible thanks to a good first-stage representation.'}</span>` : 'Click points (or press the button) to label them.';
  }
  draw();
  return { redraw: draw };
};

/* ---------- 8) zero-shot learning from descriptions ---------- */
Widgets.zeroShotDemo = (el) => {
  UI.head(el, 'Zero-shot learning: recognize a class you have never seen an image of', 'The model was trained to detect attributes from images (seen classes). Unseen classes are known only from text (“a zebra has four legs, hooves and stripes”). Pick an image, or toggle the detected attributes, and watch the matching.');
  const ATTR = ['four legs', 'two legs', 'pointy ears', 'stripes', 'wings', 'fur', 'feathers', 'swims', 'long neck', 'black & white', 'hooves', 'large'];
  const CLS = [
    { n: 'cat', seen: true, a: ['four legs', 'pointy ears', 'fur'] },
    { n: 'dog', seen: true, a: ['four legs', 'fur'] },
    { n: 'horse', seen: true, a: ['four legs', 'fur', 'hooves', 'large'] },
    { n: 'sparrow', seen: true, a: ['two legs', 'wings', 'feathers'] },
    { n: 'goldfish', seen: true, a: ['swims'] },
    { n: 'zebra', seen: false, a: ['four legs', 'fur', 'hooves', 'stripes', 'black & white', 'large'] },
    { n: 'penguin', seen: false, a: ['two legs', 'wings', 'feathers', 'swims', 'black & white'] },
    { n: 'giraffe', seen: false, a: ['four legs', 'fur', 'hooves', 'long neck', 'large'] }
  ];
  const IMG = { zebra: ['four legs', 'fur', 'hooves', 'stripes', 'black & white', 'large'], penguin: ['two legs', 'wings', 'feathers', 'swims', 'black & white'], cat: ['four legs', 'pointy ears', 'fur'], giraffe: ['four legs', 'fur', 'hooves', 'long neck', 'large'] };
  let pred = new Set(IMG.zebra);
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, Object.keys(IMG).map(k => [k, `photo of a ${k}`]), 'zebra', v => { pred = new Set(IMG[v]); chips.forEach(ch => ch.set(pred.has(ch.name))); draw(); });
  const tog = h('div', { class: 'toggles' });
  const chips = ATTR.map(a => { const ch = UI.chip(tog, a, '--s3', pred.has(a), v => { if (v) pred.add(a); else pred.delete(a); draw(); }); ch.name = a; return ch; });
  el.append(seg, h('div', { class: 'note' }, 'Attributes detected in the image (the trained part — toggle to simulate detection errors):'), tog);
  const bars = h('div', { class: 'hbars' }), calc = calcBox();
  el.append(bars, calc);
  function draw() {
    const scores = CLS.map(cl => { const set = new Set(cl.a); let agree = 0; ATTR.forEach(a => { if (pred.has(a) === set.has(a)) agree++; }); return { cl, agree, s: agree / ATTR.length }; });
    const best = scores.reduce((b, x) => (x.agree > b.agree ? x : b), scores[0]);
    bars.innerHTML = '';
    scores.sort((a, b) => b.agree - a.agree).forEach(x => {
      bars.append(h('div', { class: 'hbar' },
        h('div', { class: 'lab' }, x.cl.n, ' ', h('span', { class: 'pill', style: x.cl.seen ? '' : 'background:color-mix(in srgb,var(--s4) 30%,var(--surface))' }, x.cl.seen ? 'seen in training' : 'text description only')),
        h('div', { class: 'track' }, h('div', { class: 'fill', style: `width:${100 * x.s}%;--c:${x === best ? 'var(--good)' : 'var(--s1)'}` }), h('span', { class: 'val' }, `${x.agree} / ${ATTR.length} attributes agree`))));
    });
    const matched = ATTR.filter(a => pred.has(a) && best.cl.a.includes(a));
    calc.innerHTML = `detected attributes: {${[...pred].join(', ') || '—'}}\nscore(class) = number of the ${ATTR.length} attributes on which the detection and the class description agree\nbest match: <b>${best.cl.n}</b> (${best.agree}/${ATTR.length}; present in both: ${matched.join(', ') || 'none'})${best.cl.seen ? '' : '   ← recognized without ever seeing an image of it'}\n<span class="c">Zero-shot learning is possible only because additional information (the text descriptions) was exploited: the image→attribute detector was learned on seen classes, the attribute→class step comes from text.</span>`;
  }
  draw();
  return { redraw: () => {} };
};

/* ---------- 9) linking two representation spaces (machine translation) ---------- */
Widgets.translateMap = (el) => {
  UI.head(el, 'Linking two representation spaces: translate a word you never saw translated', 'Left: English word vectors. Right: Spanish word vectors (same structure, rotated and shifted). A linear map between the spaces is fit from a few matched pairs (anchors). Every other English word is mapped across and matched to its nearest Spanish neighbour.');
  const EN = [['cat', 1, 1], ['dog', 1.4, 0.7], ['horse', 0.7, 1.5], ['one', -1, 1], ['two', -1.4, 1.2], ['three', -0.8, 1.6], ['red', 1, -1], ['blue', 1.4, -1.3], ['green', 0.7, -1.5], ['king', -1, -1], ['queen', -1.4, -0.8], ['man', -0.8, -1.5], ['woman', -1.3, -1.4]];
  const ES = ['gato', 'perro', 'caballo', 'uno', 'dos', 'tres', 'rojo', 'azul', 'verde', 'rey', 'reina', 'hombre', 'mujer'];
  const ANCHOR_ORDER = [0, 9, 3, 6, 11, 2, 7, 4];
  let nA = 3, sel = 1;
  const rng = mulberry32(77), a = 35 * Math.PI / 180, sc = 1.1, sh = [0.4, -0.2];
  const es = EN.map(([, x, y]) => [sc * (x * Math.cos(a) - y * Math.sin(a)) + sh[0] + 0.12 * gaussian(rng), sc * (x * Math.sin(a) + y * Math.cos(a)) + sh[1] + 0.12 * gaussian(rng)]);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Matched pairs used to fit the map (anchors)', min: 2, max: 8, step: 1, value: nA, fmt: v => String(v), onInput: v => { nA = v; draw(); } });
  const sl = UI.select(ctr, 'Inspect word', EN.map((w, i) => [String(i), w[0]]), String(sel), v => { sel = +v; draw(); });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'English and Spanish word spaces linked by a learned map' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  function fit() {
    const idx = ANCHOR_ORDER.slice(0, nA);
    const X = idx.map(i => [1, EN[i][1], EN[i][2]]);
    const wx = ridgeFit(X, idx.map(i => es[i][0]), 1e-6), wy = ridgeFit(X, idx.map(i => es[i][1]), 1e-6);
    return { idx, map: q => [wx[0] + wx[1] * q[0] + wx[2] * q[1], wy[0] + wy[1] * q[0] + wy[2] * q[1]], wx, wy };
  }
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), H = 300, ctx = fitCanvas(cv, W, H);
    const half = (W - 20) / 2, F = fit();
    const col = i => [c.s1, c.s1, c.s1, c.s2, c.s2, c.s2, c.s7, c.s7, c.s7, c.s3, c.s3, c.s3, c.s3][i];
    const A = panel(ctx, half, H, 0, [-2.5, 2.5], [-2.5, 2.5], 'English word space', c.text);
    EN.forEach(([w, x, y], i) => { const anc = F.idx.includes(i); dot(ctx, A.p.X(x), A.p.Y(y), i === sel ? 7 : 4.5, col(i), anc ? c.text : c.surface); label(ctx, w + (anc ? ' ⚓' : ''), A.p.X(x) + 7, A.p.Y(y) - 8, { size: 10, bold: i === sel, color: c.text }); });
    A.done();
    const B = panel(ctx, half, H, half + 20, [-3, 3], [-3, 3], 'Spanish word space (+ mapped English words ○)', c.text);
    let correct = 0, tested = 0, selInfo = null;
    EN.forEach(([w, x, y], i) => {
      const m = F.map([x, y]);
      let nn = 0, bd = 1e9; es.forEach((q, j) => { const d = sq(q[0] - m[0]) + sq(q[1] - m[1]); if (d < bd) { bd = d; nn = j; } });
      const anc = F.idx.includes(i);
      if (!anc) { tested++; if (nn === i) correct++; }
      if (i === sel) selInfo = { m, nn, d: Math.sqrt(bd), anc };
      ctx.strokeStyle = alpha(col(i), .7); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(B.p.X(m[0]), B.p.Y(m[1]), i === sel ? 7 : 4.5, 0, 2 * Math.PI); ctx.stroke();
      if (i === sel) { ctx.strokeStyle = c.text; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(B.p.X(m[0]), B.p.Y(m[1])); ctx.lineTo(B.p.X(es[nn][0]), B.p.Y(es[nn][1])); ctx.stroke(); ctx.setLineDash([]); }
    });
    es.forEach((q, j) => { const anc = F.idx.includes(j); dot(ctx, B.p.X(q[0]), B.p.Y(q[1]), 4.5, col(j), anc ? c.text : c.surface); label(ctx, ES[j], B.p.X(q[0]) + 7, B.p.Y(q[1]) + 9, { size: 10, color: c.text2 }); });
    B.done();
    arrow(ctx, half + 2, H / 2, half + 18, H / 2, c.muted, 2, 7);
    stats.innerHTML = '';
    stats.append(stat('anchors (matched pairs)', String(nA)), stat('words never seen translated', String(tested)), stat('correct nearest neighbour', `${correct} / ${tested} <small>(${pct(correct / Math.max(1, tested))})</small>`));
    const w = EN[sel];
    calc.innerHTML = `linear map fit by least squares on the anchors:  x′ = ${fx(F.wx[1])}·x + ${fx(F.wx[2])}·y + ${fx(F.wx[0])},   y′ = ${fx(F.wy[1])}·x + ${fx(F.wy[2])}·y + ${fx(F.wy[0])}\n“${w[0]}” ${selInfo.anc ? '(an anchor)' : '(never seen in a translation pair)'}: (${fx(w[1])}, ${fx(w[2])}) → mapped to ${vec(selInfo.m)} → nearest Spanish vector: <b>${ES[selInfo.nn]}</b> (distance ${fx(selInfo.d)})${selInfo.nn === sel ? ' ✓' : ' ✗ (should be ' + ES[sel] + ')'}\n<span class="c">${nA <= 2 ? 'Two anchors are too few to pin down the rotation, scale and shift reliably — add more matched pairs.' : 'Because both languages have a distributed representation with the same structure, a link learned from a few matched pairs transfers to every other word. Learning the two representations and the link jointly works even better.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- quiz ---------- */
window.QUIZ = [
  { q: 'Transfer learning means…', opts: ['training two separate models on two datasets', 'exploiting what was learned in one setting (P₁) to improve generalization in another (P₂)', 'copying the input to the output', 'using a larger learning rate'], a: 1, why: 'What has been learned in one setting is exploited to improve generalization in another setting, assuming many factors of variation are shared.' },
  { q: 'The key assumption behind transfer learning is that…', opts: ['the two settings have exactly the same labels', 'many factors that explain the variations in P₁ are relevant to P₂', 'P₂ has more data than P₁', 'the network is linear'], a: 1, why: 'Shared underlying factors (edges, shapes, lighting…) are what make the representation learned on P₁ useful for P₂.' },
  { q: 'In domain adaptation…', opts: ['the task changes but the inputs stay the same', 'the task and the optimal mapping stay the same but the input distribution shifts', 'the network must be retrained from scratch', 'labels are never available'], a: 1, why: 'Example: sentiment analysis trained on book reviews applied to electronics reviews — the same positive/negative function, different vocabulary and style.' },
  { q: 'When the output semantics are shared but the inputs differ a lot (e.g. speech from different speakers), it makes sense to…', opts: ['share the upper layers and use task-specific preprocessing', 'share the lower layers only', 'share nothing', 'share only the input layer'], a: 0, why: 'The layers near the output (valid sentences) are shared; the layers near the input adapt to each speaker.' },
  { q: 'Concept drift is…', opts: ['a sudden change of the task', 'gradual change of the data distribution over time, a form of transfer learning', 'a bug in the optimizer', 'the same as overfitting'], a: 1, why: 'Concept drift can be viewed as transfer learning due to gradual changes in the data distribution over time.' },
  { q: 'Multitask learning helps because…', opts: ['each task gets its own representation', 'using the same representation in both settings lets it benefit from the training data of both tasks', 'it removes the need for labels', 'it doubles the learning rate'], a: 1, why: 'The shared representation receives gradients from all tasks, pooling their statistical strength.' },
  { q: 'One-shot learning is possible because…', opts: ['the representation learned in the first stage cleanly separates the underlying classes', 'the model memorizes the single example pixel by pixel', 'one example is always enough', 'the test set is small'], a: 0, why: 'Test examples of a class cluster around the same point in representation space, so one labeled example reveals the label of the whole cluster.' },
  { q: 'Zero-shot learning…', opts: ['needs one labeled example per class', 'uses no information at all', 'exploits additional information (e.g. text descriptions) exploited during training to recognize classes with no labeled examples', 'only works for images'], a: 2, why: 'Having read that a cat has four legs and pointy ears, the learner can guess an image is a cat without having seen one — because extra information was exploited.' },
  { q: 'How can a word with no translation example still be translated?', opts: ['by guessing randomly', 'by learning distributed representations of each language and a link between the two spaces from matched sentence pairs', 'by using a dictionary only', 'it cannot'], a: 1, why: 'The link (possibly two-way) relating the two representation spaces generalizes to words that never appeared in a matched pair.' },
  { q: 'Transfer across the two representation spaces works best when…', opts: ['the two representations and the relations between them are learned jointly', 'the spaces have different dimensions', 'only one language is used', 'no anchors are used'], a: 0, why: 'Learning the representations and their relation jointly aligns the spaces best.' }
];
