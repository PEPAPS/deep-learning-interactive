/* =========================================================
   Lecture 14 · Supplement · Self-attention
   ========================================================= */
'use strict';

/* ---------- helpers ---------- */
const fx = (v, d = 2) => (Math.abs(v) < 5e-13 ? 0 : v).toFixed(d).replace('-', '−');
const pn = (v, d = 2) => (v < -1e-12 ? '(' + fx(v, d) + ')' : fx(v, d));
const vec = (a, d = 2) => '[' + a.map(v => fx(v, d)).join(', ') + ']';
const sq = v => v * v;
const sum = a => a.reduce((s, v) => s + v, 0);
const calcBox = () => h('div', { class: 'calcbox' });
const stat = (k, v) => h('div', { class: 'stat' }, h('div', { class: 'k' }, k), h('div', { class: 'v', html: v }));
const gw = (cap, node) => h('div', { class: 'gwrap' }, h('div', { class: 'gcap' }, cap), node);
const gop = s => h('div', { class: 'gop' }, s);
const matmul = (A, B) => A.map(r => B[0].map((_, j) => sum(r.map((v, k) => v * B[k][j]))));
const transpose = A => A[0].map((_, j) => A.map(r => r[j]));
const softmax = o => { const m = Math.max(...o), e = o.map(v => Math.exp(v - m)), s = sum(e); return e.map(v => v / s); };

/* number grid with optional row/column highlight and click */
function numGrid(M, o = {}) {
  const c = Colors.get(), cs = o.cs || 40, cols = M[0].length;
  const flat = M.flat(), scale = o.scale ?? Math.max(1e-9, ...flat.map(Math.abs));
  const g = h('div', { class: 'cgrid' + (o.onClick ? ' click' : ''), style: `grid-template-columns:${o.rowLabels ? 'auto ' : ''}repeat(${cols}, ${cs}px);--cs:${cs}px` });
  if (o.colLabels) { if (o.rowLabels) g.append(h('div', { class: 'cell pad', style: 'width:auto;border:0' })); o.colLabels.forEach(t => g.append(h('div', { class: 'cell pad', style: 'border:0;font-family:var(--font);font-weight:600' }, t))); }
  M.forEach((row, i) => {
    if (o.rowLabels) g.append(h('div', { class: 'cell pad', style: 'width:auto;border:0;font-family:var(--font);font-weight:600;padding:0 6px;justify-content:flex-end' }, o.rowLabels[i]));
    row.forEach((v, j) => {
      const cls = ['cell'];
      if (o.hl && o.hl(i, j)) cls.push('hl');
      if (o.hl2 && o.hl2(i, j)) cls.push('hl2');
      if (o.dim && o.dim(i, j)) cls.push('dim');
      const cell = h('div', { class: cls.join(' '), title: o.title ? o.title(i, j, v) : null }, o.fmt ? o.fmt(v) : fx(v, o.d ?? 2));
      const t = clamp(Math.abs(v) / scale, 0, 1);
      if (o.heat) cell.style.background = alpha(c.s1, 0.06 + 0.8 * t);
      else if (Math.abs(v) > 1e-12) cell.style.background = alpha(v > 0 ? c.s1 : c.s2, 0.08 + 0.42 * t);
      if (o.onClick) cell.addEventListener('click', () => o.onClick(i, j));
      g.append(cell);
    });
  });
  return g;
}

/* ---------- the toy data: sentences with 4-feature embeddings ---------- */
/* features: [question-ness, verb-ness, noun/pronoun-ness, function-word-ness] */
const FEAT = ['question', 'verb', 'noun', 'function'];
const SENT = {
  how: { words: ['How', 'are', 'you', 'doing', '?'], X: [[1, 0.1, 0, 0], [0.1, 1, 0, 0.2], [0, 0.1, 1, 0], [0.2, 0.9, 0.1, 0], [0.6, 0, 0, 1]] },
  cat: { words: ['The', 'cat', 'sat', 'on', 'the', 'mat'], X: [[0, 0, 0.1, 1], [0, 0.1, 1, 0], [0, 1, 0.1, 0], [0, 0, 0, 0.9], [0, 0, 0.1, 1], [0, 0, 1, 0.1]] },
  she: { words: ['She', 'gave', 'him', 'a', 'book'], X: [[0, 0, 0.8, 0.3], [0, 1, 0, 0], [0, 0, 0.8, 0.3], [0, 0, 0, 1], [0, 0, 1, 0]] }
};
const SENT_OPTS = [['how', '“How are you doing ?”'], ['cat', '“The cat sat on the mat”'], ['she', '“She gave him a book”']];
const DQ = 2, DV = 3, D = 4;
/* hand-designed weights: verbs look for nouns, nouns look for verbs */
const W_DESIGN = {
  q: [[0.4, 0.6], [1.4, 0], [0, 1.2], [0.3, 0.3]],
  k: [[0, 0.5], [0, 1.4], [1.4, 0], [0.2, 0.2]],
  v: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, 0.3, 0.3]]
};
function randomW(seed) {
  const r = mulberry32(seed);
  const mk = (rows, cols) => Array.from({ length: rows }, () => Array.from({ length: cols }, () => Math.round(80 * gaussian(r)) / 100));
  return { q: mk(D, DQ), k: mk(D, DQ), v: mk(D, DV) };
}
function attention(X, Wt, o = {}) {
  const Q = matmul(X, Wt.q), K = matmul(X, Wt.k), V = matmul(X, Wt.v);
  const raw = matmul(Q, transpose(K));
  const scale = o.unscaled ? 1 : Math.sqrt(DQ);
  const S = raw.map(r => r.map(v => v / scale * (o.temp || 1)));
  const A = S.map(softmax);
  const Z = matmul(A, V);
  return { Q, K, V, raw, S, A, Z, scale };
}
function weightsSeg(parent, state, redraw) {
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, [['design', 'Hand-designed weights (what training could find)'], ['random', 'Random weights (untrained)']], state.wmode, v => { state.wmode = v; redraw(); });
  const b = UI.button(seg, 'Re-initialize', () => { state.seed++; state.wmode = 'random'; segCtl.set('random'); redraw(); });
  const segCtl = { set: v => { seg.querySelectorAll('button').forEach(bt => { if (bt !== b) bt.classList.toggle('on', bt.textContent.startsWith(v === 'random' ? 'Random' : 'Hand')); }); } };
  parent.append(seg);
}
const getW = state => (state.wmode === 'design' ? W_DESIGN : randomW(state.seed));
function sentenceSeg(parent, state, redraw) {
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, SENT_OPTS, state.sent, v => { state.sent = v; state.i = Math.min(state.i, SENT[v].words.length - 1); redraw(); });
  parent.append(seg);
}
function tokenPicker(parent, state, redraw, labelTxt) {
  const box = h('div', { class: 'gridbtns' });
  const render = () => {
    box.innerHTML = '';
    SENT[state.sent].words.forEach((w, i) => {
      const b = h('button', { type: 'button', class: i === state.i ? 'on' : '' }, `${i + 1}: ${w}`);
      b.addEventListener('click', () => { state.i = i; redraw(); });
      box.append(b);
    });
  };
  parent.append(h('div', { class: 'note' }, labelTxt || 'Pick the token i whose computation to show:'), box);
  return render;
}

/* ---------- 1) how far must information travel? ---------- */
Widgets.pathLength = (el) => {
  UI.head(el, 'How many steps from token i to token j?', 'Three ways to let position j use information from position i. RNN: through every state in between. CNN: through a stack of layers, each seeing k neighbours. Self-attention: one direct link.');
  let n = 10, k = 3, i = 1, j = 8;
  const ctr = h('div', { class: 'controls' });
  const sN = UI.slider(ctr, { label: 'Sequence length n', min: 4, max: 16, step: 1, value: n, fmt: v => String(v), onInput: v => { n = v; i = Math.min(i, n - 1); j = Math.min(j, n - 1); sI.set(i); sJ.set(j); draw(); } });
  const sI = UI.slider(ctr, { label: 'Token i', min: 0, max: 15, step: 1, value: i, fmt: v => String(+v + 1), onInput: v => { i = Math.min(v, n - 1); draw(); } });
  const sJ = UI.slider(ctr, { label: 'Token j', min: 0, max: 15, step: 1, value: j, fmt: v => String(+v + 1), onInput: v => { j = Math.min(v, n - 1); draw(); } });
  UI.slider(ctr, { label: 'CNN kernel size k', min: 3, max: 7, step: 2, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'paths between two tokens in an RNN, a CNN and self-attention' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(wrap, stats, calc);
  function draw() {
    const c = Colors.get(), W = boxWidth(wrap), dist = Math.abs(i - j), L = Math.max(1, Math.ceil(dist / (k - 1)));
    const rowH = 26, cnnH = 40 + (L + 1) * rowH, H = 120 + 100 + cnnH;
    const ctx = fitCanvas(cv, W, H);
    const X = t => 30 + (t + .5) * (W - 60) / n;
    const tokens = (y, hot) => { for (let t = 0; t < n; t++) { const on = t === i || t === j; dot(ctx, X(t), y, 9, on ? (t === i ? c.s2 : c.s1) : c.surface2, on ? c.text : c.border); label(ctx, String(t + 1), X(t), y, { align: 'center', size: 9, bold: on, color: on ? '#fff' : c.text2 }); } };
    // RNN
    let y = 36;
    label(ctx, `RNN: ${dist} sequential steps`, 8, y - 22, { size: 11, bold: true, color: c.text });
    tokens(y);
    const lo = Math.min(i, j), hi = Math.max(i, j);
    for (let t = 0; t < n - 1; t++) arrow(ctx, X(t) + 10, y, X(t + 1) - 10, y, t >= lo && t < hi ? c.s8 : alpha(c.muted, .4), t >= lo && t < hi ? 2.2 : 1.2, 6);
    // attention
    y = 130;
    label(ctx, 'Self-attention: 1 step, every pair directly', 8, y - 22, { size: 11, bold: true, color: c.text });
    for (let t = 0; t < n; t++) if (t !== j) { ctx.strokeStyle = alpha(c.muted, .3); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(j), y - 9); ctx.quadraticCurveTo((X(j) + X(t)) / 2, y - 14 - 2.2 * Math.abs(t - j) * 4, X(t), y - 9); ctx.stroke(); }
    ctx.strokeStyle = c.s8; ctx.lineWidth = 2.4; ctx.beginPath(); ctx.moveTo(X(j), y - 9); ctx.quadraticCurveTo((X(j) + X(i)) / 2, y - 14 - 2.2 * dist * 4, X(i), y - 9); ctx.stroke();
    tokens(y);
    // CNN
    const y0 = 220 + (L) * rowH + 10;
    label(ctx, `CNN (k = ${k}): ${L} stacked layer${L > 1 ? 's' : ''} (receptive field grows by ${k - 1} per layer)`, 8, 206, { size: 11, bold: true, color: c.text });
    const reach = l => Math.floor(l * (k - 1) / 2) + (l * (k - 1)) % 2 * 0; // radius after l layers (centered)
    for (let l = 0; l <= L; l++) {
      const yy = y0 - l * rowH;
      for (let t = 0; t < n; t++) {
        const inCone = l === 0 ? Math.abs(t - j) <= Math.ceil(L * (k - 1) / 2) : Math.abs(t - j) <= Math.ceil((L - l) * (k - 1) / 2);
        const isEnd = l === 0 && t === i, isTop = l === L && t === j;
        dot(ctx, X(t), yy, l === 0 ? 9 : 6, isEnd ? c.s2 : isTop ? c.s1 : inCone ? alpha(c.s8, .35) : c.surface2, isEnd || isTop ? c.text : c.border);
        if (l === 0) label(ctx, String(t + 1), X(t), yy, { align: 'center', size: 9, bold: isEnd, color: isEnd ? '#fff' : c.text2 });
        if (l > 0 && inCone) { for (let m = -Math.floor(k / 2); m <= Math.floor(k / 2); m++) { const s = t + m; if (s < 0 || s >= n) continue; const sIn = Math.abs(s - j) <= Math.ceil((L - l + 1) * (k - 1) / 2); ctx.strokeStyle = alpha(c.s8, sIn ? .5 : .1); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(X(t), yy + 6); ctx.lineTo(X(s), yy + rowH - (l === 1 ? 9 : 6)); ctx.stroke(); } }
      }
      label(ctx, l === 0 ? 'input' : `layer ${l}`, 4, yy, { size: 9, color: c.muted });
    }
    void reach;
    stats.innerHTML = '';
    stats.append(stat('distance |i − j|', String(dist)), stat('RNN path', `${dist} <small>steps, in order</small>`), stat('CNN path', `${L} <small>layer${L > 1 ? 's' : ''}</small>`), stat('self-attention path', '1 <small>step, all pairs in parallel</small>'));
    calc.innerHTML = `RNN: h(${j + 1}) depends on x(${i + 1}) only through h(${lo + 2})…h(${hi}) → ${dist} multiplications by the recurrent Jacobian (vanishing/exploding risk, L11)\nCNN: after l layers each unit sees 1 + l(k − 1) inputs → need 1 + l(k − 1) ≥ ${dist + 1} → l ≥ ${dist}/${k - 1} → <b>${L}</b> layer${L > 1 ? 's' : ''}\nself-attention: output ${j + 1} = Σⱼ a(${j + 1}, j)·v(j) includes j = ${i + 1} directly → path length <b>1</b>; all n² pairs are computed at once (no sequential bottleneck)`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) tokens → embedding matrix X ---------- */
Widgets.tokenEmbed = (el) => {
  UI.head(el, 'From words to a matrix of numbers', 'Each token is looked up in an embedding table and becomes a vector of d numbers (here d = 4, with made-up, readable features). Stacking the n vectors gives the input matrix X (n × d).');
  const st = { sent: 'how', i: 2 };
  sentenceSeg(el, st, draw);
  const pick = tokenPicker(el, st, draw, 'Click a token to highlight its row:');
  const out = h('div', { class: 'grids' }), stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(out, stats, calc);
  function draw() {
    pick();
    const S = SENT[st.sent], n = S.words.length;
    out.innerHTML = '';
    out.append(gw(`X = embeddings (n × d = ${n} × ${D})`, numGrid(S.X, { rowLabels: S.words, colLabels: FEAT, hl: (r) => r === st.i, cs: 54, d: 1 })));
    stats.innerHTML = '';
    stats.append(stat('tokens n', String(n)), stat('embedding size d', String(D)), stat('shape of X', `${n} × ${D}`), stat('table size', `V × d <small>(V = vocabulary)</small>`));
    calc.innerHTML = `token ${st.i + 1} = “${S.words[st.i]}” → row ${st.i + 1} of X = ${vec(S.X[st.i], 1)}   (${FEAT.map((f, k) => `${f} ${fx(S.X[st.i][k], 1)}`).join(', ')})\nthe embedding table has one row per vocabulary word; its entries are parameters learned by back-propagation like any weight\n<span class="c">Self-attention never sees the words themselves — only these vectors. Note that nothing in X says where a word is in the sentence (see the “position” slide).</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) the three projections ---------- */
Widgets.qkvProjection = (el) => {
  UI.head(el, 'Three projections of the same input: Q = XW<sub>q</sub>, K = XW<sub>k</sub>, V = XW<sub>v</sub>', 'The same embedding x<sub>i</sub> is multiplied by three different weight matrices to play three roles: a query (what am I looking for?), a key (what do I contain?) and a value (what do I pass on?).');
  const st = { sent: 'how', i: 1, wmode: 'design', seed: 1, which: 'q' };
  sentenceSeg(el, st, draw);
  weightsSeg(el, st, draw);
  const seg = h('div', { class: 'toggles' });
  UI.seg(seg, [['q', 'Query  Q = XW_q'], ['k', 'Key  K = XW_k'], ['v', 'Value  V = XW_v']], st.which, v => { st.which = v; draw(); });
  el.append(seg);
  const pick = tokenPicker(el, st, draw);
  const out = h('div', { class: 'grids' }), calc = calcBox();
  el.append(out, calc);
  function draw() {
    pick();
    const S = SENT[st.sent], Wt = getW(st), Wm = Wt[st.which], R = matmul(S.X, Wm), n = S.words.length, dout = Wm[0].length;
    const name = { q: 'Q', k: 'K', v: 'V' }[st.which], role = { q: 'query: what token i is looking for', k: 'key: what token j offers to be matched against', v: 'value: the information token j hands over' }[st.which];
    out.innerHTML = '';
    out.append(gw(`X (${n} × ${D})`, numGrid(S.X, { rowLabels: S.words, hl: r => r === st.i, cs: 40, d: 1 })), gop('×'), gw(`W_${st.which} (${D} × ${dout})`, numGrid(Wm, { rowLabels: FEAT, cs: 44 })), gop('='), gw(`${name} (${n} × ${dout})`, numGrid(R, { rowLabels: S.words, hl: r => r === st.i, cs: 50 })));
    const x = S.X[st.i];
    calc.innerHTML = `${role}\nrow i = ${st.i + 1} (“${S.words[st.i]}”):  ${name.toLowerCase()}_${st.i + 1} = x_${st.i + 1} · W_${st.which}\n` + R[st.i].map((v, j) => `  ${name.toLowerCase()}_${st.i + 1}[${j + 1}] = ${x.map((xv, k) => `${fx(xv, 1)}·${pn(Wm[k][j])}`).join(' + ')} = <b>${fx(v)}</b>`).join('\n') + `\n<span class="c">${st.wmode === 'design' ? 'In these hand-designed weights the “verb” feature feeds query slot 1 and the “noun” feature feeds key slot 1 — so verbs will look for nouns. Training discovers such weights by itself.' : 'Untrained random weights: the projections are arbitrary mixtures of the features. Training changes W_q, W_k, W_v so that useful tokens end up matching.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) scores = Q Kᵀ ---------- */
Widgets.scoreMatrix = (el) => {
  UI.head(el, 'Scores: every query against every key', 'Entry (i, j) of QKᵀ is the dot product q<sub>i</sub>·k<sub>j</sub> — how well what token i looks for matches what token j offers. Dividing by √d<sub>k</sub> keeps the numbers in a sensible range. Click a cell.');
  const st = { sent: 'how', i: 1, j: 2, wmode: 'design', seed: 1, scaled: true };
  sentenceSeg(el, st, draw);
  weightsSeg(el, st, draw);
  const tog = h('div', { class: 'toggles' });
  UI.chip(tog, 'divide by √d_k (scaled dot product)', '--s1', st.scaled, v => { st.scaled = v; draw(); });
  el.append(tog);
  const out = h('div', { class: 'grids' }), calc = calcBox();
  el.append(out, calc);
  function draw() {
    const S = SENT[st.sent], Wt = getW(st), n = S.words.length;
    const at = attention(S.X, Wt, { unscaled: !st.scaled });
    if (st.i >= n) st.i = 0; if (st.j >= n) st.j = 0;
    out.innerHTML = '';
    out.append(gw(`Q (${n} × ${DQ})`, numGrid(at.Q, { rowLabels: S.words, hl: r => r === st.i, cs: 44 })), gop('·'), gw(`Kᵀ (${DQ} × ${n})`, numGrid(transpose(at.K), { colLabels: S.words, hl: (r, cc) => cc === st.j, cs: 44 })), gop(st.scaled ? '/ √d_k =' : '='),
      gw(`scores S (${n} × ${n}): rows = queries, columns = keys`, numGrid(at.S, { rowLabels: S.words, colLabels: S.words, hl: (r, cc) => r === st.i && cc === st.j, hl2: (r, cc) => r === st.i && cc !== st.j, cs: 44, heat: true, onClick: (r, cc) => { st.i = r; st.j = cc; draw(); } })));
    const q = at.Q[st.i], k = at.K[st.j];
    calc.innerHTML = `score(i = ${st.i + 1} “${S.words[st.i]}”, j = ${st.j + 1} “${S.words[st.j]}”) = q_${st.i + 1} · k_${st.j + 1} = ${q.map((v, m) => `${pn(v)}·${pn(k[m])}`).join(' + ')} = ${fx(at.raw[st.i][st.j])}${st.scaled ? `\ndivide by √d_k = √${DQ} = ${fx(at.scale, 3)}  →  <b>${fx(at.S[st.i][st.j])}</b>` : `  →  <b>${fx(at.S[st.i][st.j])}</b> (unscaled)`}\nrow ${st.i + 1} (all keys): ${at.S[st.i].map((v, cc) => `${S.words[cc]} ${fx(v)}`).join(' · ')}\n<span class="c">A dot product is large when the two vectors point the same way: “${S.words[st.i]}” and “${S.words[st.j]}” ${at.S[st.i][st.j] > 0.5 ? 'match well' : 'do not match much'}. The whole n × n matrix is one matrix product — all pairs at once. ${st.scaled ? '' : 'Without the √d_k scaling, scores grow with the dimension and the softmax on the next slide becomes too peaky.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) softmax of a row ---------- */
Widgets.softmaxRow = (el) => {
  UI.head(el, 'Softmax turns one row of scores into attention weights', 'For query i, exponentiate every score and divide by their sum: the weights are positive and add up to 1. A multiplier on the scores shows why scaling matters: large scores → almost one-hot, small → almost uniform.');
  const st = { sent: 'how', i: 1, wmode: 'design', seed: 1, temp: 1 };
  sentenceSeg(el, st, draw);
  weightsSeg(el, st, draw);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Multiply the scores by', min: 0.25, max: 4, step: 0.25, value: st.temp, fmt: v => '×' + fx(v, 2), onInput: v => { st.temp = v; draw(); } });
  el.append(ctr);
  const pick = tokenPicker(el, st, draw, 'Pick the query token i (one row):');
  const wrap = h('div', { class: 'canvas-wrap' }), cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'scores, exponentials and normalized weights for one query' });
  wrap.append(cv);
  const calc = calcBox();
  el.append(wrap, calc);
  function draw() {
    pick();
    const S = SENT[st.sent], Wt = getW(st), n = S.words.length, at = attention(S.X, Wt, { temp: st.temp });
    const c = Colors.get(), W = boxWidth(wrap), H = 230, ctx = fitCanvas(cv, W, H);
    const s = at.S[st.i], e = s.map(v => Math.exp(v - Math.max(...s))), a = at.A[st.i];
    const third = W / 3;
    const bars = (ox, vals, title, col, ymax, fmtv) => {
      const p = makePlot(third - 8, H, [-0.5, n - 0.5], [Math.min(0, ...vals) * 1.1, ymax], { l: 30, r: 6, t: 24, b: 28 });
      ctx.save(); ctx.translate(ox, 0);
      drawAxes(ctx, p, { xticks: vals.map((_, k) => k), xfmt: t => S.words[t], ny: 4 });
      label(ctx, title, p.pad.l + 2, 10, { size: 11, bold: true, color: c.text });
      const bw = p.pw / n * 0.6;
      vals.forEach((v, k) => { ctx.fillStyle = alpha(col, .75); ctx.fillRect(p.X(k) - bw / 2, Math.min(p.Y(0), p.Y(v)), bw, Math.abs(p.Y(v) - p.Y(0))); label(ctx, fmtv(v), p.X(k), Math.min(p.Y(0), p.Y(v)) - 7, { align: 'center', size: 9, color: c.text2 }); });
      ctx.restore();
    };
    bars(0, s, 'scores s_j', c.s7, Math.max(1, ...s) * 1.25, v => fx(v, 1));
    bars(third, e, 'exp(s_j − max)', c.s2, Math.max(...e) * 1.25, v => fx(v, 2));
    bars(2 * third, a, 'weights a_ij = exp / Σexp', c.s1, 1.15, v => fx(v, 2));
    const Z = sum(e);
    const ent = -sum(a.map(v => (v > 0 ? v * Math.log(v) : 0)));
    calc.innerHTML = `query i = ${st.i + 1} “${S.words[st.i]}”;  scores (× ${fx(st.temp, 2)}): ${vec(s)}\nexp(s_j − max): ${vec(e, 3)}   sum Σ = ${fx(Z, 3)}   (subtracting the max changes nothing in the result and avoids huge numbers)\na_${st.i + 1}j = exp(s_j − max)/Σ = ${vec(a, 3)}   → sum = ${fx(sum(a), 3)} ✓\n<span class="c">${ent < 0.6 ? 'Peaky: almost all the weight is on one token — large scores act like a hard choice.' : ent > Math.log(n) - 0.15 ? 'Nearly uniform: tiny scores make every token count the same (attention cannot “choose”).' : 'A soft mixture: a few tokens get most of the weight.'} Entropy of the weights: ${fx(ent)} (max = ln ${n} = ${fx(Math.log(n))}).</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 6) weighted sum of values ---------- */
Widgets.weightedSum = (el) => {
  UI.head(el, 'The output: a weighted mix of the values', 'The new vector for token i is z<sub>i</sub> = Σ<sub>j</sub> a<sub>ij</sub> v<sub>j</sub>: every token’s value, weighted by how much token i attends to it. Bar widths are the attention weights.');
  const st = { sent: 'how', i: 1, wmode: 'design', seed: 1 };
  sentenceSeg(el, st, draw);
  weightsSeg(el, st, draw);
  const pick = tokenPicker(el, st, draw, 'Pick the token i whose output to compute:');
  const rows = h('div', { class: 'stack' }), out = h('div', { class: 'grids' }), calc = calcBox();
  el.append(rows, out, calc);
  function draw() {
    pick();
    const S = SENT[st.sent], Wt = getW(st), n = S.words.length, at = attention(S.X, Wt);
    const a = at.A[st.i];
    rows.innerHTML = '';
    const list = h('div', { class: 'hbars' });
    S.words.forEach((w, j) => list.append(h('div', { class: 'hbar' },
      h('div', { class: 'lab' }, `a(${st.i + 1}, ${j + 1}) · v_${j + 1}  — “${w}”`),
      h('div', { class: 'track', style: 'display:flex;align-items:center;gap:8px' },
        h('div', { class: 'fill', style: `width:${100 * a[j]}%;--c:var(--s1)` }),
        h('span', { class: 'val' }, `${fx(a[j], 2)} × ${vec(at.V[j])}`)))));
    rows.append(list);
    out.innerHTML = '';
    out.append(gw(`A (${n} × ${n}) — row ${st.i + 1} highlighted`, numGrid(at.A, { rowLabels: S.words, colLabels: S.words, hl: r => r === st.i, cs: 40, heat: true })), gop('×'), gw(`V (${n} × ${DV})`, numGrid(at.V, { rowLabels: S.words, cs: 44 })), gop('='), gw(`Z (${n} × ${DV})`, numGrid(at.Z, { rowLabels: S.words, hl: r => r === st.i, cs: 48 })));
    calc.innerHTML = `z_${st.i + 1} = Σ_j a(${st.i + 1}, j) · v_j\n` + at.Z[st.i].map((zv, m) => `  z_${st.i + 1}[${m + 1}] = ${a.map((av, j) => `${fx(av, 2)}·${pn(at.V[j][m])}`).join(' + ')} = <b>${fx(zv)}</b>`).join('\n') + `\n<span class="c">z_${st.i + 1} is the “hidden state” of “${S.words[st.i]}” after one self-attention layer: its own value blended with the values of the tokens it attends to. Since the weights sum to 1, z stays in the range of the values.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) the whole pipeline ---------- */
Widgets.attentionPlayground = (el) => {
  UI.head(el, 'The whole pipeline on one sentence', 'Click a token in the attention map to follow its computation from embedding to output. Compare hand-designed weights with random ones.');
  const st = { sent: 'how', i: 1, wmode: 'design', seed: 1 };
  sentenceSeg(el, st, draw);
  weightsSeg(el, st, draw);
  const out = h('div', { class: 'grids' }), stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(out, stats, calc);
  function draw() {
    const S = SENT[st.sent], Wt = getW(st), n = S.words.length, at = attention(S.X, Wt);
    if (st.i >= n) st.i = 0;
    out.innerHTML = '';
    out.append(gw('attention weights A — click a row (query) to inspect it', numGrid(at.A, { rowLabels: S.words, colLabels: S.words, hl: r => r === st.i, cs: 44, heat: true, onClick: r => { st.i = r; draw(); } })), gop('→'), gw(`outputs Z (${n} × ${DV})`, numGrid(at.Z, { rowLabels: S.words, hl: r => r === st.i, cs: 48 })));
    const top = at.A[st.i].map((v, j) => [v, j]).sort((x, y) => y[0] - x[0]);
    stats.innerHTML = '';
    stats.append(stat('shapes', `X ${n}×${D} · W_q,W_k ${D}×${DQ} · W_v ${D}×${DV}`), stat(`“${S.words[st.i]}” attends most to`, `“${S.words[top[0][1]]}” <small>(${fx(top[0][0], 2)})</small>`), stat('then', `“${S.words[top[1][1]]}” <small>(${fx(top[1][0], 2)})</small>`));
    const i = st.i;
    calc.innerHTML = `1  x_${i + 1} = ${vec(S.X[i], 1)}  (embedding of “${S.words[i]}”)\n2  q_${i + 1} = x_${i + 1}W_q = ${vec(at.Q[i])}     k_j = x_jW_k     v_j = x_jW_v\n3  scores s_${i + 1}j = q_${i + 1}·k_j / √${DQ}: ${S.words.map((w, j) => `${w} ${fx(at.S[i][j])}`).join(' · ')}\n4  softmax → a_${i + 1}j: ${S.words.map((w, j) => `${w} ${fx(at.A[i][j], 2)}`).join(' · ')}\n5  z_${i + 1} = Σ_j a_${i + 1}j v_j = ${vec(at.Z[i])}\n<span class="c">${st.wmode === 'design' ? 'With the designed weights, verbs attend to nouns/pronouns and nouns to verbs — a pattern a trained model might use to link “doing” with “you”. Nothing here depends on the order of the tokens.' : 'Random weights give an arbitrary attention pattern. Training adjusts W_q, W_k, W_v (and the embeddings) so that the weights become useful for the task.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 8) cost of a layer ---------- */
Widgets.costCompare = (el) => {
  UI.head(el, 'What a layer costs: self-attention vs. recurrent vs. convolutional', 'Per-layer complexity, how many operations must run one after the other, and the longest path between two positions (the standard comparison from the Transformer paper). Slide n, d and k.');
  let ln = 7, ld = 8, k = 3;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Sequence length n', min: 3, max: 14, step: 1, value: ln, fmt: v => String(2 ** v), onInput: v => { ln = v; draw(); } });
  UI.slider(ctr, { label: 'Representation size d', min: 5, max: 11, step: 1, value: ld, fmt: v => String(2 ** v), onInput: v => { ld = v; draw(); } });
  UI.slider(ctr, { label: 'Convolution kernel k', min: 3, max: 9, step: 2, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  el.append(ctr);
  const table = h('div', { class: 'scroll-x' }), bars = h('div', { class: 'hbars' }), calc = calcBox();
  el.append(table, bars, calc);
  const big = v => (v >= 1e12 ? (v / 1e12).toFixed(1) + ' T' : v >= 1e9 ? (v / 1e9).toFixed(1) + ' G' : v >= 1e6 ? (v / 1e6).toFixed(1) + ' M' : v >= 1e3 ? (v / 1e3).toFixed(1) + ' k' : String(v));
  function draw() {
    const n = 2 ** ln, d = 2 ** ld;
    const att = n * n * d, rnn = n * d * d, cnn = k * n * d * d;
    const rows = [
      ['Self-attention', 'O(n² · d)', big(att), 'O(1)', 'O(1)'],
      ['Recurrent', 'O(n · d²)', big(rnn), 'O(n) = ' + n, 'O(n) = ' + n],
      ['Convolutional', 'O(k · n · d²)', big(cnn), 'O(1)', `O(log_k n) ≈ ${Math.ceil(Math.log(n) / Math.log(k))}`]
    ];
    table.innerHTML = '';
    table.append(h('table', { class: 'data-table' }, h('thead', {}, h('tr', {}, ['Layer type', 'Complexity per layer', 'with these n, d, k', 'Sequential operations', 'Maximum path length'].map(t => h('th', {}, t)))), h('tbody', {}, rows.map(r => h('tr', {}, r.map(v => h('td', {}, v)))))));
    const mx = Math.max(att, rnn, cnn);
    bars.innerHTML = '';
    [['self-attention n²d', att, 's1'], ['recurrent nd²', rnn, 's8'], ['convolutional knd²', cnn, 's6']].forEach(([lab, v, col]) => bars.append(h('div', { class: 'hbar' }, h('div', { class: 'lab' }, lab), h('div', { class: 'track' }, h('div', { class: 'fill', style: `width:${100 * v / mx}%;--c:var(--${col})` }), h('span', { class: 'val' }, big(v) + ' mult.')))));
    calc.innerHTML = `n = ${n} tokens, d = ${d}:   n²d = ${n}² × ${d} = <b>${big(att)}</b>     nd² = ${n} × ${d}² = <b>${big(rnn)}</b>     knd² = ${k} × ${n} × ${d}² = <b>${big(cnn)}</b>\nthe attention matrix alone has n² = ${big(n * n)} entries per head and layer — this is the “computational & memory cost” of self-attention\n<span class="c">${n < d ? 'n < d: self-attention is the cheapest per layer AND fully parallel with path length 1 — this is the usual regime for sentences.' : 'n > d: the n² term dominates — long sequences (long documents, high-resolution images, audio) are where self-attention becomes expensive.'} The recurrent layer is cheap but its n sequential steps cannot run in parallel.</span>`;
  }
  draw();
  return { redraw: () => {} };
};

/* ---------- 9) the comparison table, interactive ---------- */
Widgets.archTable = (el) => {
  UI.head(el, 'CNN vs. RNN vs. self-attention: click any cell', 'The comparison table from the slide. Each cell links back to the lecture slide that explains the idea.');
  const COLS = ['CNN', 'RNN', 'Self-attention (Transformer)'];
  const ROWS = ['Characteristics', 'Advantages', 'Disadvantages'];
  const T = [
    [
      { t: 'Local sliding kernels over the spatial/temporal dimension; sparse interactions, parameter sharing and (translation) equivariant representations', e: 'A small kernel slides over the input; each output depends on a few neighbours (sparse), the same weights are reused everywhere (sharing), and shifting the input shifts the output (equivariance).', l: [['ch09.html#/s-sparse', 'Sparse interactions (L10)'], ['ch09.html#/s-sharing', 'Parameter sharing (L10)'], ['ch09.html#/s-equivariance', 'Equivariance (L10)']] },
      { t: 'Sequential step-by-step state updating; shared transition weights over steps', e: 'A state h(t) is updated one step at a time with the same weights W at every step.', l: [['ch10.html#/s-unfold', 'Unfolding the recurrence (L11)'], ['ch10.html#/s-forward', 'Forward equations (L11)']] },
      { t: 'Pairwise query–key–value interactions', e: 'Every token produces a query, a key and a value; each output mixes all values weighted by query–key matches.', l: [['#/s-qkv', 'Q, K, V (this lecture)'], ['#/s-playground', 'The full pipeline']] }
    ],
    [
      { t: 'High GPU/TPU parallelization; compact parameter set', e: 'All positions of a layer are computed independently (parallel), and the kernel has very few weights regardless of input size.', l: [['ch09.html#/s-sharing', 'Why CNN layers are tiny (L10)'], ['ch09.html#/s-data-types', 'Variable-size inputs (L10)']] },
      { t: 'Dynamic processing of arbitrary sequence lengths; capturing long-range global dependencies; reusing processing units', e: 'The same cell runs for any length; in principle the state can carry information across the whole sequence; one set of units is reused at every step.', l: [['ch10.html#/s-seq', 'Any length, shared weights (L11)'], ['ch10.html#/s-lstm', 'LSTM: carrying information far (L11)']] },
      { t: 'Direct interaction between any two inputs, capturing long-range dependencies; fully parallelizable', e: 'Path length 1 between any pair; the whole layer is a few matrix products with no sequential dependency.', l: [['#/s-why', 'Path lengths (this lecture)'], ['#/s-cost', 'Sequential operations (this lecture)']] }
    ],
    [
      { t: 'Hard to capture long-range global dependencies', e: 'The receptive field grows only k − 1 per layer, so distant positions interact only after many layers (or with pooling/dilation).', l: [['ch09.html#/s-sparse', 'Receptive field grows with depth (L10)'], ['#/s-why', 'Path length of a CNN (this lecture)']] },
      { t: 'Sequential training bottleneck; vanishing/exploding gradients', e: 'Step t needs step t − 1: no parallelism along the sequence. Gradients through many steps are products of Jacobians that vanish or explode.', l: [['ch10.html#/s-bptt', 'Back-propagation through time (L11)'], ['ch10.html#/s-longterm', 'Vanishing / exploding gradients (L11)'], ['ch10.html#/s-clip', 'Gradient clipping (L11)']] },
      { t: 'Computational & memory cost; requiring massive training datasets', e: 'The n × n attention matrix costs O(n²d) time and n² memory per head and layer. With little built-in structure (no locality prior), attention needs a lot of data to learn what to attend to.', l: [['#/s-cost', 'Cost per layer (this lecture)'], ['ch09.html#/s-prior', 'Convolution as a strong prior (L10)'], ['ch05-1.html#/s-nfl', 'No free lunch (L1)']] }
    ]
  ];
  let sel = [0, 2];
  const wrap = h('div', { class: 'scroll-x' }), detail = h('div', { class: 'card' });
  el.append(wrap, detail);
  function draw() {
    wrap.innerHTML = '';
    wrap.append(h('table', { class: 'data-table' },
      h('thead', {}, h('tr', {}, h('th', {}, ''), COLS.map(cn => h('th', {}, cn)))),
      h('tbody', {}, ROWS.map((rn, r) => h('tr', {}, h('td', {}, rn), T[r].map((cell, cc) => {
        const b = h('button', { type: 'button', class: 'btn', style: `text-align:left;white-space:normal;line-height:1.35;${sel[0] === r && sel[1] === cc ? 'border-color:var(--accent);background:var(--accent-soft)' : ''}` }, cell.t);
        b.addEventListener('click', () => { sel = [r, cc]; draw(); });
        return h('td', { style: 'font-weight:400' }, b);
      }))))));
    const cell = T[sel[0]][sel[1]];
    detail.innerHTML = '';
    detail.append(h('h3', {}, h('span', { class: 'dot' }), `${COLS[sel[1]]} · ${ROWS[sel[0]]}`), h('p', {}, cell.e), h('div', { class: 'topic-btns' }, cell.l.map(([href, txt]) => { const a = h('a', { class: 'topic-btn', href, style: 'text-decoration:none' }, txt + ' →'); return a; })));
  }
  draw();
  return { redraw: () => {} };
};

/* ---------- 10) order does not matter — unless you add position ---------- */
Widgets.permuteDemo = (el) => {
  UI.head(el, 'Self-attention ignores word order — unless you add position information', 'Shuffle the sentence. Without position vectors, each token gets exactly the same output as before (just moved with it): the layer is permutation-equivariant. Adding a position vector to each embedding breaks the tie.');
  const st = { sent: 'how', wmode: 'design', seed: 1, perm: null, pos: false };
  sentenceSeg(el, st, () => { st.perm = null; draw(); });
  weightsSeg(el, st, draw);
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Shuffle the tokens', () => { const n = SENT[st.sent].words.length, p = [...Array(n).keys()]; for (let a = n - 1; a > 0; a--) { const b = Math.floor(Math.random() * (a + 1)); [p[a], p[b]] = [p[b], p[a]]; } st.perm = p; draw(); }, 'btn primary');
  UI.button(btns, 'Original order', () => { st.perm = null; draw(); });
  const tog = h('div', { class: 'toggles' });
  UI.chip(tog, 'add a position vector to each embedding', '--s4', st.pos, v => { st.pos = v; draw(); });
  el.append(h('div', { class: 'controls' }, btns, tog));
  const out = h('div', { class: 'grids' }), stats = h('div', { class: 'stat-row' }), calc = calcBox();
  el.append(out, stats, calc);
  const posVec = (t, n) => [0.3 * Math.sin(t / n * Math.PI), 0.3 * Math.cos(t / n * Math.PI), 0.3 * (t / (n - 1)), 0.3 * (1 - t / (n - 1))];
  function draw() {
    const S = SENT[st.sent], n = S.words.length, Wt = getW(st), perm = st.perm || [...Array(n).keys()];
    const addPos = X => (st.pos ? X.map((x, t) => x.map((v, k) => v + posVec(t, n)[k])) : X);
    const X0 = addPos(S.X), words1 = perm.map(p => S.words[p]), X1 = addPos(perm.map(p => S.X[p]));
    const a0 = attention(X0, Wt), a1 = attention(X1, Wt);
    out.innerHTML = '';
    out.append(gw('original order → Z', numGrid(a0.Z, { rowLabels: S.words, cs: 48 })), gop('vs'), gw((st.perm ? 'shuffled' : 'same') + ' order → Z', numGrid(a1.Z, { rowLabels: words1, cs: 48 })));
    // compare: for each token, output in the shuffled run vs original run
    let maxDiff = 0;
    perm.forEach((p, t) => { const d = Math.max(...a1.Z[t].map((v, m) => Math.abs(v - a0.Z[p][m]))); maxDiff = Math.max(maxDiff, d); });
    stats.innerHTML = '';
    stats.append(stat('largest change in any token’s output', fx(maxDiff, 3)), stat('verdict', !st.perm ? 'shuffle to compare' : maxDiff < 1e-9 ? 'identical: order was invisible' : 'outputs changed: order now matters'));
    const w = S.words[perm[0]];
    calc.innerHTML = `${st.perm ? `shuffled order: ${words1.join(' ')}` : 'original order: ' + S.words.join(' ')}\n“${w}”: output in the original run ${vec(a0.Z[perm[0]])}   in this run ${vec(a1.Z[0])}\n<span class="c">${st.pos ? 'Each embedding now carries a position vector, so the same word at a different place has a different x, hence different q, k, v and output. (Transformers use sinusoidal or learned position encodings for this.)' : 'Scores depend only on the vectors q_i·k_j, never on i or j themselves — so permuting the rows of X permutes the rows of Z and changes nothing else. “How are you” and “you are How” are the same to a bare self-attention layer.'}</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- quiz ---------- */
window.QUIZ = [
  { q: 'In self-attention, the three vectors computed from each token embedding are…', opts: ['input, hidden, output', 'query, key, value', 'weight, bias, activation', 'mean, variance, scale'], a: 1, why: 'Q = XW_q, K = XW_k, V = XW_v: the same embedding plays three roles through three learned projections.' },
  { q: 'The attention score between token i and token j is…', opts: ['the sum of their embeddings', 'the dot product q_i · k_j (scaled by √d_k)', 'the distance between their positions', 'always 1'], a: 1, why: 'Entry (i, j) of QKᵀ measures how well what i looks for matches what j offers.' },
  { q: 'Why apply a softmax to each row of the score matrix?', opts: ['to make the scores negative', 'to turn them into positive weights that sum to 1', 'to remove the diagonal', 'to sort the tokens'], a: 1, why: 'Softmax gives a proper weighting; the output z_i is then a convex combination of the values.' },
  { q: 'The output z_i of self-attention is…', opts: ['the value of token i only', 'the weighted sum Σ_j a_ij v_j of all values', 'the largest key', 'the query q_i'], a: 1, why: 'Every token’s value contributes, weighted by the attention weight a_ij.' },
  { q: 'Compared with an RNN, the path length between two distant tokens in self-attention is…', opts: ['the same', 'longer', 'O(1): a direct interaction', 'infinite'], a: 2, why: 'Any two inputs interact directly in one layer — the key to capturing long-range dependencies.' },
  { q: 'The main cost of self-attention comes from…', opts: ['the n × n attention matrix: O(n² d) time and n² memory per layer', 'the softmax', 'the embedding table', 'the bias terms'], a: 0, why: 'Every pair of tokens is scored, so cost grows quadratically with sequence length.' },
  { q: 'Which statement matches the table on the slide?', opts: ['CNNs capture long-range dependencies easily', 'RNNs are fully parallelizable', 'Self-attention requires massive training datasets', 'RNNs have no vanishing-gradient problem'], a: 2, why: 'Self-attention’s disadvantages: computational & memory cost, and the need for massive datasets. CNNs struggle with long-range dependencies; RNNs have a sequential bottleneck and vanishing/exploding gradients.' },
  { q: 'What property of CNNs does the table list as a characteristic?', opts: ['pairwise query-key-value interactions', 'sparse interactions, parameter sharing and equivariant representations', 'sequential state updating', 'teacher forcing'], a: 1, why: 'Local sliding kernels give sparse interactions, shared parameters and translation equivariance (L10).' },
  { q: 'A bare self-attention layer applied to a shuffled sentence…', opts: ['produces the same outputs, just shuffled the same way', 'fails to run', 'produces all zeros', 'produces completely different outputs'], a: 0, why: 'Scores depend only on the vectors, not on positions: the layer is permutation-equivariant, so position information must be added explicitly.' },
  { q: 'Why divide the scores by √d_k?', opts: ['to make the matrix square', 'to keep the scores from growing with the dimension, which would make the softmax too peaky', 'to normalize the values', 'it is only a convention with no effect'], a: 1, why: 'Dot products of d_k-dimensional vectors have variance proportional to d_k; scaling keeps the softmax in a useful range.' }
];
