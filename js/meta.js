/* =========================================================
   meta.js — Part B demos (Chapter 8.7)
   layer interference · batch norm · Polyak averaging · pretraining ·
   model design · continuation · curriculum
   ========================================================= */
'use strict';

/* =========================================================
   1) Deep linear chain: first-order prediction vs. simultaneous update
   ========================================================= */
Widgets.chain = (el) => {
  UI.head(el, 'One gradient step, many layers: predicted vs. actual',
    'Network <b>ŷ = x·w₁·w₂···w<sub>l</sub></b> with x = 1. We choose ε so that the gradient <b>predicts</b> ŷ will drop by exactly Δ. Then every layer is updated at once — what <b>actually</b> happens?');
  let L = 8, w = 0.7, D = 0.1, uneven = false, mode = 'auto', lgEps = -3;
  const modeBox = h('div', { class: 'ctl', style: 'flex-basis:100%' }, h('span', { class: 'ctl-label' }, h('span', {}, 'How the learning rate ε is chosen')));
  UI.seg(modeBox, [['auto', 'ε = Δ / gᵀg  (lecture’s example)'], ['fixed', 'One fixed ε for every case']], mode, v => { mode = v; dS.show(v === 'auto'); eS.show(v === 'fixed'); draw(); });
  const ctr = h('div', { class: 'controls' });
  ctr.append(modeBox);
  UI.slider(ctr, { label: 'Number of layers l', min: 1, max: 20, step: 1, value: L, fmt: v => String(v), onInput: v => { L = v; draw(); } });
  UI.slider(ctr, { label: 'Typical weight size w', min: 0.3, max: 2, step: 0.05, value: w, fmt: v => v.toFixed(2), onInput: v => { w = v; draw(); } });
  const dS = UI.slider(ctr, { label: 'Wanted decrease Δ', min: 0.01, max: 0.5, step: 0.01, value: D, fmt: v => v.toFixed(2), onInput: v => { D = v; draw(); } });
  const eS = UI.slider(ctr, { label: 'Fixed learning rate ε', min: -5, max: -1, step: 0.05, value: lgEps, fmt: v => (10 ** v).toPrecision(2), onInput: v => { lgEps = v; draw(); } });
  eS.show(false);
  const tg = h('div', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', {}, 'Weights')));
  UI.chip(tg, 'Make them uneven (±40%)', '--s4', uneven, v => { uneven = v; draw(); });
  ctr.append(tg);
  const chainBox = h('div', { style: 'display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:.76rem' });
  const stats = h('div', { class: 'stat-row' });
  const bars = h('div', { class: 'hbars' });
  const verdict = h('div');
  el.append(ctr, chainBox, stats, bars, verdict);

  function draw() {
    const c = Colors.get();
    const rng = mulberry32(11);
    const ws = Array.from({ length: L }, () => w * (uneven ? Math.exp(0.4 * (2 * rng() - 1)) : 1));
    const y = ws.reduce((a, b) => a * b, 1);
    const g = ws.map(wi => y / wi);
    const gg = g.reduce((a, b) => a + b * b, 0);
    const eps = mode === 'auto' ? D / gg : 10 ** lgEps;
    const pred = eps * gg; // first-order predicted decrease of ŷ
    const solo = g.map(gi => eps * gi * gi);
    const yNew = ws.reduce((a, wi, i) => a * (wi - eps * g[i]), 1);
    const actual = y - yNew;
    const r = actual / pred;

    chainBox.innerHTML = '';
    chainBox.append(h('span', { class: 'pill' }, 'x = 1'));
    ws.forEach((wi, i) => {
      chainBox.append(h('span', { class: 'muted' }, '→'));
      chainBox.append(h('span', { class: 'pill', title: `If only layer ${i + 1} moved, ŷ would drop by ${fmt(solo[i], 4)}`, style: 'display:inline-flex;flex-direction:column;line-height:1.2;padding:3px 8px' },
        h('b', {}, `w${i + 1}=${wi.toFixed(2)}`), h('span', { style: 'font-size:.66rem' }, `alone: −${fmt(solo[i], 3)}`)));
    });
    chainBox.append(h('span', { class: 'muted' }, '→'), h('span', { class: 'pill' }, 'ŷ = ' + fmt(y, 4)));

    stats.innerHTML = '';
    stats.append(
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'ŷ before'), h('div', { class: 'v' }, fmt(y, 4))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, mode === 'auto' ? 'ε chosen = Δ / gᵀg' : 'fixed ε'), h('div', { class: 'v' }, fmt(eps, 4))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Gradient predicts ŷ →'), h('div', { class: 'v' }, fmt(y - pred, 4))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Actually ŷ →'), h('div', { class: 'v' }, fmt(yNew, 4)))
    );

    const scale = 3;
    const row = (lab, v, col) => {
      const mag = Math.abs(v) / pred;
      const wpc = clamp(mag / scale * 100, 0.8, 100);
      const txt = (v >= 0 ? 'ŷ drops by ' : 'ŷ RISES by ') + fmt(Math.abs(v), 4) + (mag > scale ? '  (off the chart ▶)' : '');
      return h('div', { class: 'hbar' }, h('div', { class: 'lab' }, lab),
        h('div', { class: 'track' }, h('div', { class: 'fill', style: `width:${wpc}%;--c:${v >= 0 ? col : c.critical}` }), h('span', { class: 'val' }, txt)));
    };
    bars.innerHTML = '';
    bars.append(
      row('Gradient’s prediction (first-order)', pred, c.s1),
      row('Sum of each layer moved alone', solo.reduce((a, b) => a + b, 0), c.s3),
      row('All layers moved together (reality)', actual, c.s2)
    );

    verdict.innerHTML = '';
    let cls, ico, msg;
    if (Math.abs(r - 1) < 0.05) {
      cls = 'good'; ico = '✓';
      msg = mode === 'auto'
        ? 'Here the first-order prediction is close, because ε came out small compared with the weights. Try w ≈ 0.5–0.7 with more layers, or switch to one fixed ε and raise w above 1.'
        : 'Here the first-order prediction is close, because the step is small compared with the weights. Keep this ε and change l or w: the same ε can suddenly behave very differently.';
    }
    else if (r < 0) { cls = 'bad'; ico = '✗'; msg = `Wrong direction! Every layer stepped "downhill", yet the output went UP by ${fmt(-actual, 3)}. The higher-order cross-terms (like ε²g₁g₂w₃···w_l) took over.`; }
    else if (r > 1) { cls = 'bad'; ico = '!'; msg = `Overshoot: the output dropped ${fmt(r, 2)}× more than planned. A learning rate that is "right" for each layer alone is too big when all of them move together.`; }
    else { cls = 'warn'; ico = '!'; msg = `Undershoot: only ${(r * 100).toFixed(0)}% of the planned change happened. The layers' updates interfere with each other.`; }
    verdict.append(h('div', { class: 'callout ' + cls }, h('span', { class: 'ico' }, ico), h('div', {}, msg)));
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   2) Batch norm step by step
   ========================================================= */
Widgets.bnSteps = (el) => {
  UI.head(el, 'Batch norm on a real minibatch',
    'm = 8 examples × 3 hidden units. The first two sliders imitate <b>the layers below changing</b> (rescaling and shifting this layer’s input — the shift is just like a bias b). Watch the middle row: Ĥ does not move.');
  const m = 8, n = 3, delta = 1e-8;
  const rng = mulberry32(42);
  const means = [1.5, -0.5, 3], sds = [1, 2.2, 0.6];
  const Z = Array.from({ length: m }, () => means.map((mu, j) => mu + sds[j] * gaussian(rng)));
  let s = 1, b = 0, gam = 1, bet = 0;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Lower layers rescale input by', min: 0.2, max: 4, step: 0.05, value: s, fmt: v => v.toFixed(2) + '×', onInput: v => { s = v; draw(); } });
  UI.slider(ctr, { label: 'Lower layers shift input by (bias b)', min: -5, max: 5, step: 0.1, value: b, fmt: v => (v >= 0 ? '+' : '') + v.toFixed(1), onInput: v => { b = v; draw(); } });
  UI.slider(ctr, { label: 'Learned scale γ', min: 0.2, max: 3, step: 0.05, value: gam, fmt: v => v.toFixed(2), onInput: v => { gam = v; draw(); } });
  UI.slider(ctr, { label: 'Learned shift β', min: -3, max: 3, step: 0.1, value: bet, fmt: v => v.toFixed(1), onInput: v => { bet = v; draw(); } });
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Values before and after batch normalization' });
  wrap.append(cv);
  el.append(ctr, wrap);
  const lg = UI.legend(el, []);
  const tables = h('div', { class: 'ntables scroll-x' });
  el.append(tables);

  function stats(M) {
    const mu = [0, 1, 2].map(j => M.reduce((a, r) => a + r[j], 0) / m);
    const sd = [0, 1, 2].map(j => Math.sqrt(delta + M.reduce((a, r) => a + (r[j] - mu[j]) ** 2, 0) / m));
    return { mu, sd };
  }
  function table(title, M, extraRows) {
    return h('table', { class: 'ntable' }, h('caption', {}, title),
      h('thead', {}, h('tr', {}, h('th', {}, ''), ...[1, 2, 3].map(j => h('th', {}, 'unit ' + j)))),
      h('tbody', {},
        ...M.map((r, i) => h('tr', {}, h('th', {}, 'ex ' + (i + 1)), ...r.map(v => h('td', {}, fmt(v, 2))))),
        ...(extraRows || []).map(([lab, vals]) => h('tr', { class: 'stat-r' }, h('th', {}, lab), ...vals.map(v => h('td', {}, fmt(v, 2)))))));
  }
  function draw() {
    const c = Colors.get();
    const cols = [c.s1, c.s2, c.s3];
    lg.set([{ label: 'unit 1', color: cols[0], kind: 'dot' }, { label: 'unit 2', color: cols[1], kind: 'dot' }, { label: 'unit 3', color: cols[2], kind: 'dot' }, { label: 'mean of each unit', color: c.text }]);
    const Hm = Z.map(r => r.map(v => s * v + b));
    const st = stats(Hm);
    const Hn = Hm.map(r => r.map((v, j) => (v - st.mu[j]) / st.sd[j]));
    const Out = Hn.map(r => r.map(v => gam * v + bet));
    const W = boxWidth(wrap), lane = 12, rowH = 3 * lane + 44, H = rowH * 3 + 6;
    const ctx = fitCanvas(cv, W, H);
    const rowsDef = [
      ['H — what this layer receives', Hm, null],
      ['Ĥ = (H − μ) / σ — always mean 0, std 1', Hn, [-8, 8]],
      ['γĤ + β — what the next layer sees', Out, [-8, 8]]
    ];
    rowsDef.forEach(([title, M, rng0], k) => {
      let r = rng0;
      if (!r) { const mx = Math.max(...M.flat().map(Math.abs)) * 1.1 + 0.5; r = [-mx, mx]; }
      const top = k * rowH;
      const p = makePlot(W, rowH, r, [0, 1], { l: 12, r: 12, t: 20, b: 18 });
      const base = top + rowH - 18;
      ctx.save();
      ctx.font = '600 11.5px Inter, system-ui, sans-serif'; ctx.fillStyle = c.text2; ctx.textBaseline = 'top'; ctx.textAlign = 'left';
      ctx.fillText(title, 12, top + 3);
      ctx.strokeStyle = c.axis; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(12, base + .5); ctx.lineTo(W - 12, base + .5); ctx.stroke();
      ctx.font = '10px Inter, system-ui, sans-serif'; ctx.fillStyle = c.muted; ctx.textAlign = 'center';
      for (const t of niceTicks(r[0], r[1], 8)) { const x = p.X(t); ctx.beginPath(); ctx.moveTo(x, base); ctx.lineTo(x, base + 3); ctx.stroke(); ctx.fillText(tickLabel(t), x, base + 4); }
      for (let j = 0; j < 3; j++) {
        const yy = base - 10 - (2 - j) * lane;
        const mu = M.reduce((a, row) => a + row[j], 0) / m;
        ctx.strokeStyle = c.text; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(p.X(mu), yy - 6); ctx.lineTo(p.X(mu), yy + 6); ctx.stroke();
        for (const row of M) dot(ctx, clamp(p.X(row[j]), 12, W - 12), yy, 4, cols[j], c.surface);
      }
      ctx.restore();
    });
    tables.innerHTML = '';
    tables.append(
      table('H (rows = examples, columns = units)', Hm, [['μ', st.mu], ['σ', st.sd]]),
      table('Ĥ = (H − μ) / σ', Hn),
      table(`γĤ + β  (γ=${gam.toFixed(2)}, β=${bet.toFixed(1)})`, Out)
    );
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   3) Three ways to keep activations normalized (illustration)
   ========================================================= */
Widgets.bnApproaches = (el) => {
  UI.head(el, 'Fighting the optimizer vs. removing the fight',
    'Each training step, the gradient “wants” to push a unit’s mean upward a little. Illustration of what the unit’s mean does under the three strategies (hover to read values).');
  const T = 30;
  const push = t => 0.35 * (0.5 + 0.25 * Math.sin(t / 3));
  const pen = [[0, 0]]; let mPen = 0;
  for (let t = 0; t < T; t++) { mPen = mPen + push(t) - 0.3 * mPen; pen.push([t + 1, mPen]); }
  const ren = [];
  let wasted = 0;
  for (let t = 0; t < T; t++) { ren.push([t, 0], [t + 0.97, push(t)]); wasted += push(t); }
  ren.push([T, 0]);
  const bn = Array.from({ length: T + 1 }, (_, t) => [t, 0]);
  const chart = new LineChart(el, { xr: [0, T], yr: [-0.05, 0.8], xlabel: 'training step', ylabel: 'mean of the unit', xname: 'step', aspect: .5, maxH: 300, refY: 0 });
  const lg = UI.legend(el, []);
  const note = h('div', { class: 'row3' });
  el.append(note);
  function draw() {
    const c = Colors.get();
    chart.set([
      { name: 'Penalty in the loss', color: 's4', data: pen },
      { name: 'Renormalize after each step', color: 's8', data: ren },
      { name: 'Batch normalization', color: 's3', data: bn, width: 3 }
    ]);
    lg.set([{ label: 'Penalty in the loss', color: c.s4 }, { label: 'Renormalize after each step', color: c.s8 }, { label: 'Batch normalization', color: c.s3 }]);
    note.innerHTML = '';
    note.append(
      h('div', { class: 'callout warn' }, h('span', { class: 'ico' }, '!'), h('div', {}, h('b', { class: 'tag' }, 'Penalty:'), `settles near ${fmt(mPen, 2)}, not 0 — the normalization is imperfect.`)),
      h('div', { class: 'callout warn' }, h('span', { class: 'ico' }, '!'), h('div', {}, h('b', { class: 'tag' }, 'Renormalize:'), `mean is reset every step, but ${fmt(wasted, 1)} units of update were proposed and thrown away.`)),
      h('div', { class: 'callout good' }, h('span', { class: 'ico' }, '✓'), h('div', {}, h('b', { class: 'tag' }, 'Batch norm:'), 'mean is 0 by definition, and the gradient never even proposes changing it.'))
    );
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   4) Training-time batch statistics vs. test-time running averages
   ========================================================= */
Widgets.runningStats = (el) => {
  UI.head(el, 'Collect running averages during training, use them at test time',
    'This unit’s pre-activations really have mean 3 and std 2. Every training minibatch gives a noisy estimate; the running average (momentum 0.1) smooths them.');
  const TRUE_MU = 3, TRUE_SD = 2, delta = 1e-5;
  let B = 8, rng, batches, runMu, runVar, testX = 5;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Minibatch size', min: 2, max: 64, step: 1, value: B, fmt: v => String(v), onInput: v => { B = v; } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Train on 1 minibatch', () => { train(1); }, 'btn primary');
  UI.button(btns, 'Train on 25', () => { train(25); });
  UI.button(btns, 'Reset', () => { reset(); });
  ctr.append(btns);
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 10], yr: [-1, 6], xlabel: 'minibatch #', ylabel: 'estimated mean', xname: 'batch', aspect: .45, maxH: 260, refY: TRUE_MU, snapX: x => Math.round(x) });
  const lg = UI.legend(el, []);
  const test = h('div', { class: 'card', style: 'box-shadow:none;background:var(--surface-2)' });
  el.append(test);
  function reset() { rng = mulberry32(5); batches = []; runMu = 0; runVar = 1; draw(); }
  function train(k) {
    for (let i = 0; i < k; i++) {
      const xs = Array.from({ length: B }, () => TRUE_MU + TRUE_SD * gaussian(rng));
      const mu = xs.reduce((a, b) => a + b, 0) / B;
      const v = xs.reduce((a, b) => a + (b - mu) ** 2, 0) / Math.max(1, B - 1);
      runMu = 0.9 * runMu + 0.1 * mu;
      runVar = 0.9 * runVar + 0.1 * v;
      batches.push({ mu, run: runMu });
    }
    draw();
  }
  function draw() {
    const c = Colors.get();
    const n = batches.length;
    chart.set([
      { name: 'batch mean μ_B', color: 's1', style: 'dots', data: batches.map((q, i) => [i + 1, q.mu]) },
      { name: 'running mean μ̂', color: 's2', data: [[0, 0], ...batches.map((q, i) => [i + 1, q.run])] }
    ], { xr: [0, Math.max(10, n)] });
    lg.set([{ label: 'batch mean μ_B (noisy)', color: c.s1, kind: 'dot' }, { label: 'running mean μ̂ (smooth)', color: c.s2 }, { label: 'true mean = 3', color: c.muted, kind: 'dash' }]);
    test.innerHTML = '';
    const sd = Math.sqrt(runVar + delta);
    const norm = (testX - runMu) / sd;
    const sl = h('div', { class: 'controls' });
    test.append(h('h3', {}, h('span', { class: 'dot' }), 'Test time: one single example arrives'));
    UI.slider(sl, { label: 'Test input value x', min: -4, max: 10, step: 0.1, value: testX, fmt: v => v.toFixed(1), onInput: v => { testX = v; draw(); } });
    test.append(sl,
      h('div', { class: 'stat-row', style: 'margin-top:8px' },
        h('div', { class: 'stat' }, h('div', { class: 'k' }, 'With running stats (μ̂, σ̂)'), h('div', { class: 'v' }, fmt(norm, 3), h('small', {}, ` = (${testX.toFixed(1)} − ${fmt(runMu, 2)}) / ${fmt(sd, 2)}`))),
        h('div', { class: 'stat' }, h('div', { class: 'k' }, 'With a “batch” of just this example'), h('div', { class: 'v' }, '0.000', h('small', {}, ' = (x − x) / √δ')))),
      h('div', { class: 'callout info', style: 'margin-top:8px' }, h('span', { class: 'ico' }, 'i'),
        h('div', {}, 'A batch of one has μ = x and σ = 0, so ', h('b', {}, 'every'), ' input would be normalized to 0 and the network would output the same thing for every image. Running averages fix this — and make the prediction for one example independent of whatever else is in the batch.')));
  }
  reset();
  return { redraw: draw };
};

/* =========================================================
   5) Batch norm in CNNs: which numbers share statistics?
   ========================================================= */
Widgets.cnnBN = (el) => {
  UI.head(el, 'BN in a convolutional layer: which numbers share one μ and σ?',
    'Feature maps for a minibatch of 2 images × 3 channels × 4×4 positions. <b>Click any cell</b> to light up every number used to compute its mean and standard deviation.');
  let mode = 'channel', sel = { n: 0, c: 1, i: 1, j: 2 };
  const segBox = h('div');
  UI.seg(segBox, [['channel', 'Per channel ✓'], ['position', 'Per position ✗']], mode, v => { mode = v; draw(); });
  const svgBox = h('div');
  const out = h('div');
  el.append(segBox, svgBox, out);
  function draw() {
    const c = Colors.get();
    const cols = [c.s1, c.s2, c.s3];
    const cs = 17, gap = 2, gw = 4 * cs + 3 * gap, cg = 16, lw = 64;
    const W = lw + 3 * gw + 2 * cg + 4, Hh = 22 + 2 * (gw + 26);
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`); svg.setAttribute('class', 'diagram'); svg.style.maxWidth = '520px';
    const mk = (tag, a) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); svg.append(e); return e; };
    for (let ch = 0; ch < 3; ch++) {
      const t = mk('text', { x: lw + ch * (gw + cg) + gw / 2, y: 13, 'text-anchor': 'middle', 'font-size': 11, fill: c.text2, 'font-weight': 600 });
      t.textContent = 'channel ' + (ch + 1);
    }
    let count = 0;
    for (let n = 0; n < 2; n++) {
      const y0 = 22 + n * (gw + 26);
      const t = mk('text', { x: 0, y: y0 + gw / 2 + 4, 'font-size': 11, fill: c.text2, 'font-weight': 600 });
      t.textContent = 'image ' + (n + 1);
      for (let ch = 0; ch < 3; ch++) {
        for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
          const on = mode === 'channel' ? ch === sel.c : (ch === sel.c && i === sel.i && j === sel.j);
          if (on) count++;
          const isSel = n === sel.n && ch === sel.c && i === sel.i && j === sel.j;
          const r = mk('rect', {
            x: lw + ch * (gw + cg) + j * (cs + gap), y: y0 + i * (cs + gap), width: cs, height: cs, rx: 3,
            fill: on ? cols[ch] : alpha(cols[ch], .14), stroke: isSel ? c.text : 'none', 'stroke-width': isSel ? 2.5 : 0, style: 'cursor:pointer'
          });
          r.addEventListener('click', () => { sel = { n, c: ch, i, j }; draw(); });
        }
      }
    }
    svgBox.innerHTML = ''; svgBox.append(svg);
    out.innerHTML = '';
    out.append(mode === 'channel'
      ? h('div', { class: 'callout good' }, h('span', { class: 'ico' }, '✓'), h('div', {}, `Statistics come from ${count} numbers: every image in the batch × every position in channel ${sel.c + 1}. The same filter produced all of them, so they should be treated the same way everywhere — a cat’s ear is normalized identically in the corner or the centre.`))
      : h('div', { class: 'callout bad' }, h('span', { class: 'ico' }, '✗'), h('div', {}, `Only ${count} numbers per statistic, and each position gets its own μ and σ. The feature map’s statistics would now depend on location, breaking the “same detector everywhere” idea of convolution (and the estimates are very noisy).`)));
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   6) Polyak averaging
   ========================================================= */
Widgets.polyak = (el) => {
  UI.head(el, 'Averaging the path lands nearer the bottom',
    'Gradient descent with a large learning rate bounces from wall to wall of a narrow valley. Compare the raw iterate with the plain average (Polyak) and the exponential moving average.');
  const a = 0.5, b = 10;
  const f = (x, y) => 0.5 * (a * x * x + b * y * y);
  const dom = [-5, 5, -2.5, 2.5];
  const start = [-4.6, 1.9];
  let lr = 0.19, noise = 0, al = 0.9, N = 70;
  let pts, avg, ema, k, timer = null, bg = null, bgKey = '';
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Learning rate ε', min: 0.05, max: 0.199, step: 0.001, value: lr, fmt: v => v.toFixed(3), onInput: v => { lr = v; reset(); } });
  UI.slider(ctr, { label: 'Gradient noise (SGD)', min: 0, max: 3, step: 0.05, value: noise, fmt: v => v.toFixed(2), onInput: v => { noise = v; reset(); } });
  UI.slider(ctr, { label: 'EMA decay α', min: 0.5, max: 0.99, step: 0.01, value: al, fmt: v => v.toFixed(2), onInput: v => { al = v; reset(); } });
  const btns = h('div', { class: 'btns' });
  const runBtn = UI.button(btns, 'Run ▶', () => (timer ? pause() : play()), 'btn primary');
  UI.button(btns, 'Reset', () => { pause(); reset(); });
  ctr.append(btns);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Valley with optimizer path and averaged points' });
  wrap.append(cv);
  el.append(ctr, wrap);
  const lg = UI.legend(el, []);
  el.append(h('div', { class: 'note' }, 'Distance to the true minimum (log scale) — lower is better:'));
  const chart = new LineChart(el, { xr: [0, N], yr: [1e-3, 10], logY: true, xlabel: 'iteration', xname: 'iteration', aspect: .38, maxH: 220, minH: 160, snapX: x => Math.round(x) });
  function reset() {
    const rng = mulberry32(3);
    pts = [start.slice()]; avg = [start.slice()]; ema = [start.slice()]; k = 0;
    let th = start.slice(), sum = start.slice(), e = start.slice();
    for (let t = 1; t <= N; t++) {
      const g = [a * th[0] + noise * gaussian(rng), b * th[1] + noise * gaussian(rng)];
      th = [th[0] - lr * g[0], th[1] - lr * g[1]];
      pts.push(th);
      sum = [sum[0] + th[0], sum[1] + th[1]];
      avg.push([sum[0] / (t + 1), sum[1] / (t + 1)]);
      e = [al * e[0] + (1 - al) * th[0], al * e[1] + (1 - al) * th[1]];
      ema.push(e);
    }
    k = timer ? 0 : N;
    draw();
  }
  function play() { pause(); k = 0; runBtn.textContent = 'Pause ❚❚'; timer = setInterval(() => { k++; draw(); if (k >= N) pause(); }, 90); }
  function pause() { if (timer) clearInterval(timer); timer = null; runBtn.textContent = 'Run ▶'; }
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = Math.round(W / 2);
    const ctx = fitCanvas(cv, W, H);
    const p = makePlot(W, H, [dom[0], dom[1]], [dom[2], dom[3]], { l: 0, r: 0, t: 0, b: 0 });
    const key = W + '|' + Theme.current();
    if (key !== bgKey) { bg = contourImage(f, dom, Math.round(W / 2), Math.round(H / 2), 14); bgKey = key; }
    ctx.drawImage(bg, 0, 0, W, H);
    const line = (arr, col, w) => {
      ctx.strokeStyle = c.surface; ctx.lineWidth = w + 2.5; ctx.lineJoin = 'round';
      ctx.beginPath(); arr.slice(0, k + 1).forEach((q, i) => i ? ctx.lineTo(p.X(q[0]), p.Y(q[1])) : ctx.moveTo(p.X(q[0]), p.Y(q[1]))); ctx.stroke();
      ctx.strokeStyle = col; ctx.lineWidth = w;
      ctx.beginPath(); arr.slice(0, k + 1).forEach((q, i) => i ? ctx.lineTo(p.X(q[0]), p.Y(q[1])) : ctx.moveTo(p.X(q[0]), p.Y(q[1]))); ctx.stroke();
    };
    line(pts, alpha(c.s1, .75), 1.4);
    pts.slice(1, k + 1).forEach(q => dot(ctx, p.X(q[0]), p.Y(q[1]), 2.4, c.s1));
    line(avg, c.s2, 2.4);
    line(ema, c.s3, 2.4);
    const last = arr => arr[Math.min(k, arr.length - 1)];
    dot(ctx, p.X(last(pts)[0]), p.Y(last(pts)[1]), 6, c.s1, c.surface);
    dot(ctx, p.X(last(avg)[0]), p.Y(last(avg)[1]), 6, c.s2, c.surface);
    dot(ctx, p.X(last(ema)[0]), p.Y(last(ema)[1]), 6, c.s3, c.surface);
    ctx.save(); ctx.fillStyle = c.text; ctx.font = '600 14px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', p.X(0), p.Y(0)); ctx.restore();
    lg.set([{ label: 'raw iterate θ(t)', color: c.s1 }, { label: 'Polyak: plain average of all θ', color: c.s2 }, { label: 'EMA: θ̂ = αθ̂ + (1−α)θ', color: c.s3 }, { label: 'minimum ★', color: c.text, kind: 'dot' }]);
    const dist = arr => arr.slice(0, k + 1).map((q, i) => [i, Math.max(1e-3, Math.hypot(q[0], q[1]))]);
    chart.set([
      { name: 'raw', color: 's1', data: dist(pts), width: 1.5 },
      { name: 'Polyak avg', color: 's2', data: dist(avg) },
      { name: 'EMA', color: 's3', data: dist(ema) }
    ]);
  }
  reset();
  return { redraw: () => { bgKey = ''; draw(); } };
};

/* =========================================================
   7) Greedy layer-wise supervised pretraining (stepper)
   ========================================================= */
Widgets.pretrain = (el) => {
  UI.head(el, 'Greedy supervised pretraining, step by step', 'Use the buttons to walk through how a deep network is built one hidden layer at a time.');
  const steps = [
    { t: 'Train a shallow network', d: 'Start with something easy to optimize: input → one hidden layer h₁ → output. Train it on the real task (same labels y).', cols: ['x', 'h₁', 'ŷ'], w: ['train', 'train'] },
    { t: 'Keep the hidden layer, throw away the output layer', d: 'W₁ has learned useful features. The output layer U₁ was only scaffolding, so it is discarded.', cols: ['x', 'h₁', 'ŷ'], w: ['keep', 'discard'] },
    { t: 'Stack a new hidden layer + new output', d: 'Feed h₁’s outputs into a fresh one-hidden-layer network (h₂ → ŷ) and train it with the same objective. Only the new weights are trained in this stage.', cols: ['x', 'h₁', 'h₂', 'ŷ'], w: ['keep', 'train', 'train'] },
    { t: 'Repeat for as many layers as you want', d: 'Discard the output again, stack h₃, train. Each stage is a small, easy supervised problem.', cols: ['x', 'h₁', 'h₂', 'h₃', 'ŷ'], w: ['keep', 'keep', 'train', 'train'] },
    { t: 'Fine-tune everything together', d: 'The greedy solution is a great starting point. Now a joint optimization trains all layers at once to fix what the greedy stages could not see.', cols: ['x', 'h₁', 'h₂', 'h₃', 'ŷ'], w: ['tune', 'tune', 'tune', 'tune'] }
  ];
  let i = 0;
  const nav = h('div', { class: 'btns' });
  const prev = UI.button(nav, '← Previous', () => { i = Math.max(0, i - 1); draw(); });
  const next = UI.button(nav, 'Next step →', () => { i = Math.min(steps.length - 1, i + 1); draw(); }, 'btn primary');
  const title = h('div', { style: 'font-weight:650' });
  const desc = h('div', { class: 'note' });
  const svgBox = h('div');
  el.append(nav, title, desc, svgBox);
  const lg = UI.legend(el, []);
  function draw() {
    const c = Colors.get();
    const st = steps[i];
    prev.disabled = i === 0; next.disabled = i === steps.length - 1;
    title.textContent = `Stage ${i + 1} of ${steps.length}: ${st.t}`;
    desc.textContent = st.d;
    const colOf = { train: c.s2, keep: c.s1, discard: c.muted, tune: c.s3 };
    const ns = 'http://www.w3.org/2000/svg';
    const W = 560, Hh = 210;
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`); svg.setAttribute('class', 'diagram'); svg.style.maxWidth = '620px';
    const mk = (tag, a, txt) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); if (txt) e.textContent = txt; svg.append(e); return e; };
    const nCols = st.cols.length;
    const xs = st.cols.map((_, k) => 50 + k * (W - 100) / (5 - 1));
    const sizes = st.cols.map((nm, k) => k === 0 ? 3 : k === nCols - 1 ? 2 : 4);
    const ys = sizes.map(sz => Array.from({ length: sz }, (_, q) => 95 + (q - (sz - 1) / 2) * 36));
    for (let k = 0; k < nCols - 1; k++) {
      const s = st.w[k];
      for (const y1 of ys[k]) for (const y2 of ys[k + 1]) {
        mk('line', { x1: xs[k], y1, x2: xs[k + 1], y2, stroke: colOf[s], 'stroke-width': s === 'discard' ? 1.2 : 1.8, 'stroke-dasharray': s === 'discard' ? '4 4' : '', opacity: s === 'discard' ? .6 : .85 });
      }
      const lbl = k === nCols - 2 ? `U${i <= 1 ? 1 : i === 2 ? 2 : 3}` : `W${k + 1}`;
      mk('text', { x: (xs[k] + xs[k + 1]) / 2, y: 190, 'text-anchor': 'middle', 'font-size': 12, fill: colOf[s], 'font-weight': 700 }, lbl + (s === 'discard' ? ' ✗' : ''));
    }
    st.cols.forEach((nm, k) => {
      const isOutDiscard = k === nCols - 1 && st.w[nCols - 2] === 'discard';
      ys[k].forEach(y => mk('circle', { cx: xs[k], cy: y, r: 10, fill: Colors.get().surface, stroke: isOutDiscard ? c.muted : c.text, 'stroke-width': 2, 'stroke-dasharray': isOutDiscard ? '3 3' : '' }));
      mk('text', { x: xs[k], y: 22, 'text-anchor': 'middle', 'font-size': 13, fill: c.text, 'font-weight': 700 }, nm);
    });
    svgBox.innerHTML = ''; svgBox.append(svg);
    lg.set([{ label: 'being trained now', color: c.s2 }, { label: 'pretrained, kept', color: c.s1 }, { label: 'discarded', color: c.muted, kind: 'dash' }, { label: 'fine-tuned jointly', color: c.s3 }]);
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   8) Pretraining variants: deeper init, transfer, teacher-student hints
   ========================================================= */
Widgets.pretrainVariants = (el) => {
  UI.head(el, 'Three relatives of supervised pretraining', 'Pick a tab.');
  let tab = 'deep', k = 3, hints = true;
  const tabs = h('div', { class: 'tabs' });
  const body = h('div', { class: 'stack', style: 'gap:10px' });
  el.append(tabs, body);
  const defs = [['deep', 'Initialize a deeper net'], ['transfer', 'Transfer learning'], ['fitnets', 'Teacher → student hints']];
  const tbtns = defs.map(([id, l]) => { const b = h('button', { type: 'button', class: 'tab' }, l); b.addEventListener('click', () => { tab = id; draw(); }); tabs.append(b); return [id, b]; });
  const ns = 'http://www.w3.org/2000/svg';
  function svgEl(W, Hh) {
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${Hh}`); svg.setAttribute('class', 'diagram');
    const mk = (tag, a, txt) => { const e = document.createElementNS(ns, tag); for (const q in a) e.setAttribute(q, a[q]); if (txt != null) e.textContent = txt; svg.append(e); return e; };
    return [svg, mk];
  }
  function draw() {
    const c = Colors.get();
    tbtns.forEach(([id, b]) => b.classList.toggle('on', id === tab));
    body.innerHTML = '';
    if (tab === 'deep') {
      const [svg, mk] = svgEl(600, 175);
      const bw = 26, gp = 4, x0 = 20;
      mk('text', { x: x0, y: 16, 'font-size': 12, fill: c.text2, 'font-weight': 600 }, 'Pretrained network (11 weight layers)');
      for (let q = 0; q < 11; q++) mk('rect', { x: x0 + q * (bw + gp), y: 24, width: bw, height: 34, rx: 5, fill: c.s1 });
      mk('text', { x: x0, y: 106, 'font-size': 12, fill: c.text2, 'font-weight': 600 }, 'New, deeper network (19 weight layers), then trained jointly');
      for (let q = 0; q < 19; q++) {
        const copied = q < 4 || q >= 16;
        mk('rect', { x: x0 + q * (bw + gp), y: 114, width: bw, height: 34, rx: 5, fill: copied ? c.s1 : alpha(c.s7, .85) });
      }
      for (let q = 0; q < 4; q++) mk('line', { x1: x0 + q * (bw + gp) + bw / 2, y1: 60, x2: x0 + q * (bw + gp) + bw / 2, y2: 112, stroke: c.s1, 'stroke-width': 1.5, 'stroke-dasharray': '3 3' });
      for (let q = 0; q < 3; q++) mk('line', { x1: x0 + (8 + q) * (bw + gp) + bw / 2, y1: 60, x2: x0 + (16 + q) * (bw + gp) + bw / 2, y2: 112, stroke: c.s1, 'stroke-width': 1.5, 'stroke-dasharray': '3 3' });
      mk('text', { x: x0 + 10 * (bw + gp), y: 168, 'text-anchor': 'middle', 'font-size': 11, fill: c.text2 }, '12 middle layers start random');
      body.append(svg);
      body.append(UI.legend(h('div'), [{ label: 'copied from the pretrained net', color: c.s1, kind: 'dot' }, { label: 'random initialization', color: c.s7, kind: 'dot' }]).el);
      body.append(h('p', { class: 'note' }, 'Instead of growing one layer at a time, pretrain a whole (shallower) deep conv net, then copy its first four and last three layers into a much deeper net. Only the middle is random, so the deep net starts close to something that already works.'));
    } else if (tab === 'transfer') {
      const ctr = h('div', { class: 'controls' });
      UI.slider(ctr, { label: 'Layers copied from network A (k)', min: 0, max: 8, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
      body.append(ctr);
      const [svg, mk] = svgEl(600, 170);
      const bw = 52, gp = 8, x0 = 20;
      mk('text', { x: x0, y: 16, 'font-size': 12, fill: c.text2, 'font-weight': 600 }, 'Network A — trained on task A (half of the ImageNet classes, lots of data)');
      for (let q = 0; q < 8; q++) mk('rect', { x: x0 + q * (bw + gp), y: 24, width: bw, height: 34, rx: 6, fill: c.s1 });
      mk('text', { x: x0, y: 100, 'font-size': 12, fill: c.text2, 'font-weight': 600 }, 'Network B — same size, trained on task B (other classes, fewer examples)');
      for (let q = 0; q < 8; q++) {
        mk('rect', { x: x0 + q * (bw + gp), y: 108, width: bw, height: 34, rx: 6, fill: q < k ? c.s1 : alpha(c.s7, .85) });
        if (q < k) mk('line', { x1: x0 + q * (bw + gp) + bw / 2, y1: 60, x2: x0 + q * (bw + gp) + bw / 2, y2: 106, stroke: c.s1, 'stroke-width': 1.5, 'stroke-dasharray': '3 3' });
      }
      mk('text', { x: x0, y: 162, 'font-size': 11, fill: c.text2 }, 'input side ←');
      mk('text', { x: 580, y: 162, 'text-anchor': 'end', 'font-size': 11, fill: c.text2 }, '→ output side');
      body.append(svg);
      body.append(UI.legend(h('div'), [{ label: 'copied from A', color: c.s1, kind: 'dot' }, { label: 'random, then all trained jointly on B', color: c.s7, kind: 'dot' }]).el);
      body.append(h('p', { class: 'note' }, k === 0 ? 'k = 0: network B starts from scratch and must learn everything from its small dataset.'
        : k <= 3 ? `k = ${k}: the low layers (edges, colours, textures) are general and transfer well to the new classes.`
          : k <= 6 ? `k = ${k}: middle layers are more specialized; they still help, because B can fine-tune them jointly.`
            : `k = ${k}: almost everything is reused; the top layers were very specific to task A, but joint training on B can still adapt them.`));
    } else {
      const ctr = h('div', { class: 'controls' });
      const box = h('div', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', {}, 'Extra objective')));
      UI.chip(box, 'Use hints from the teacher', '--s4', hints, v => { hints = v; draw(); });
      ctr.append(box); body.append(ctr);
      const [svg, mk] = svgEl(600, 230);
      mk('text', { x: 20, y: 16, 'font-size': 12, fill: c.text2, 'font-weight': 600 }, 'Teacher: wide & shallow (5 layers) — easy to train');
      for (let q = 0; q < 5; q++) mk('rect', { x: 20 + q * 70, y: 26, width: 56, height: 54, rx: 7, fill: q === 2 ? c.s4 : c.s1 });
      mk('text', { x: 20, y: 136, 'font-size': 12, fill: c.text2, 'font-weight': 600 }, 'Student: thin & deep (11–19 layers) — hard to train alone');
      for (let q = 0; q < 11; q++) mk('rect', { x: 20 + q * 34, y: 150, width: 24, height: 30, rx: 5, fill: q === 5 ? c.s4 : alpha(c.s7, .85) });
      mk('text', { x: 400, y: 170, 'font-size': 12, fill: c.text, 'font-weight': 600 }, '→ ŷ (task loss)');
      if (hints) {
        mk('rect', { x: 430, y: 92, width: 120, height: 30, rx: 7, fill: 'none', stroke: c.s4, 'stroke-width': 2 });
        mk('text', { x: 490, y: 111, 'text-anchor': 'middle', 'font-size': 11, fill: c.text }, 'regressor r(·)');
        mk('path', { d: `M ${20 + 5 * 34 + 12} 150 C ${20 + 5 * 34 + 12} 120, 380 107, 430 107`, fill: 'none', stroke: c.s4, 'stroke-width': 2 });
        mk('path', { d: `M 550 107 C 600 107, 600 53, ${20 + 2 * 70 + 56} 53`, fill: 'none', stroke: c.s4, 'stroke-width': 2, 'stroke-dasharray': '5 4' });
        mk('text', { x: 495, y: 140, 'text-anchor': 'middle', 'font-size': 11, fill: c.text2 }, 'match the teacher’s middle layer');
      }
      body.append(svg);
      body.append(hints
        ? h('div', { class: 'callout good' }, h('span', { class: 'ico' }, '✓'), h('div', {}, h('b', { class: 'tag' }, 'With hints:'), 'the student’s lower layers get two jobs — help the final prediction AND reproduce the teacher’s middle layer through r(·). These hints tell the hidden layers how to be used, which makes optimization much easier. A thin, deep student can then generalize well with far fewer parameters.'))
        : h('div', { class: 'callout bad' }, h('span', { class: 'ico' }, '✗'), h('div', {}, h('b', { class: 'tag' }, 'Without hints:'), 'in the experiments the deep, thin student trained poorly — bad on the training set and on the test set. The signal from the output alone was not enough to guide its many middle layers.')));
    }
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   9) Activation shapes: saturating vs. (piecewise) linear
   ========================================================= */
Widgets.actShapes = (el) => {
  UI.head(el, 'Sigmoid vs. ReLU and their slopes', 'Hover to read values. The dashed lines are the derivatives — that is the factor the gradient gets multiplied by when it passes back through the unit.');
  const xs = Array.from({ length: 241 }, (_, i) => -6 + i * 0.05);
  const sig = x => 1 / (1 + Math.exp(-x));
  const chart = new LineChart(el, { xr: [-6, 6], yr: [-0.1, 1.6], xlabel: 'pre-activation a', xname: 'a', aspect: .45, maxH: 250, vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  function draw() {
    const c = Colors.get();
    chart.set([
      { name: 'sigmoid(a)', color: 's1', data: xs.map(x => [x, sig(x)]) },
      { name: "sigmoid′(a)", color: 's1', data: xs.map(x => [x, sig(x) * (1 - sig(x))]), dash: [5, 4] },
      { name: 'ReLU(a)', color: 's2', data: xs.map(x => [x, Math.max(0, x)]) },
      { name: "ReLU′(a)", color: 's2', data: xs.map(x => [x, x > 0 ? 1 : 0]), dash: [5, 4] }
    ]);
    lg.set([{ label: 'sigmoid', color: c.s1 }, { label: 'sigmoid slope (max 0.25)', color: c.s1, kind: 'dash' }, { label: 'ReLU', color: c.s2 }, { label: 'ReLU slope (exactly 1 when active)', color: c.s2, kind: 'dash' }]);
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   10) Gradient flow through depth — a real backward pass in the browser
   ========================================================= */
Widgets.gradFlow = (el) => {
  UI.head(el, 'How much gradient reaches each layer?',
    'A real randomly-initialized network (width 24, batch 16) is run forward and backward right here. Bars show ‖∂J/∂W‖ for each layer, on a log scale (each gridline = 10× ).');
  let L = 20, act = 'sigmoid', skip = false, aux = false, seed = 1;
  const n = 24, B = 16;
  const ctr = h('div', { class: 'controls' });
  UI.select(ctr, 'Activation', [['sigmoid', 'Sigmoid'], ['tanh', 'Tanh'], ['relu', 'ReLU']], act, v => { act = v; draw(); });
  UI.slider(ctr, { label: 'Depth (layers)', min: 4, max: 40, step: 1, value: L, fmt: v => String(v), onInput: v => { L = v; draw(); } });
  el.append(ctr);
  const tg = h('div', { class: 'toggles' });
  UI.chip(tg, 'Skip connections (h + f(h))', '--s3', skip, v => { skip = v; draw(); });
  UI.chip(tg, 'Auxiliary head in the middle', '--s4', aux, v => { aux = v; draw(); });
  UI.button(tg, 'New random weights', () => { seed++; draw(); });
  el.append(tg);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Gradient norm per layer' });
  const tip = h('div', { class: 'tip', hidden: true });
  wrap.append(cv, tip);
  el.append(wrap);
  const out = h('div');
  el.append(out);
  let norms = [], plot = null, hover = -1;

  const phi = { sigmoid: a => 1 / (1 + Math.exp(-a)), tanh: Math.tanh, relu: a => (a > 0 ? a : 0) };
  const dphi = {
    sigmoid: a => { const s = 1 / (1 + Math.exp(-a)); return s * (1 - s); },
    tanh: a => { const t = Math.tanh(a); return 1 - t * t; },
    relu: a => (a > 0 ? 1 : 0)
  };
  function compute() {
    const rng = mulberry32(seed * 7919 + 13);
    const rnd = (len, sd) => Float64Array.from({ length: len }, () => gaussian(rng) * sd);
    const sd = act === 'relu' ? Math.sqrt(2 / n) : Math.sqrt(1 / n);
    const Ws = Array.from({ length: L }, () => rnd(n * n, sd));
    const V = rnd(n, Math.sqrt(1 / n)), Va = rnd(n, Math.sqrt(1 / n));
    const tgt = rnd(B, 1);
    const s = skip ? 1 / Math.sqrt(L) : 1;
    const f = phi[act], df = dphi[act];
    const hs = [rnd(B * n, 1)], as = [];
    for (let l = 0; l < L; l++) {
      const hIn = hs[l], W = Ws[l], a = new Float64Array(B * n), hOut = new Float64Array(B * n);
      for (let b = 0; b < B; b++) for (let j = 0; j < n; j++) {
        let acc = 0;
        for (let i = 0; i < n; i++) acc += hIn[b * n + i] * W[i * n + j];
        a[b * n + j] = acc;
        hOut[b * n + j] = skip ? hIn[b * n + j] + s * f(acc) : f(acc);
      }
      as.push(a); hs.push(hOut);
    }
    const out = (hh, v) => Array.from({ length: B }, (_, b) => { let acc = 0; for (let j = 0; j < n; j++) acc += hh[b * n + j] * v[j]; return acc; });
    const y = out(hs[L], V);
    let dh = new Float64Array(B * n);
    for (let b = 0; b < B; b++) for (let j = 0; j < n; j++) dh[b * n + j] = (y[b] - tgt[b]) / B * V[j];
    const mid = Math.floor(L / 2);
    const ya = out(hs[mid], Va);
    const res = new Array(L);
    for (let l = L - 1; l >= 0; l--) {
      if (aux && l + 1 === mid) for (let b = 0; b < B; b++) for (let j = 0; j < n; j++) dh[b * n + j] += (ya[b] - tgt[b]) / B * Va[j];
      const a = as[l], hIn = hs[l], W = Ws[l];
      const da = new Float64Array(B * n);
      for (let q = 0; q < B * n; q++) da[q] = (skip ? s : 1) * dh[q] * df(a[q]);
      let gn = 0;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        let acc = 0;
        for (let b = 0; b < B; b++) acc += hIn[b * n + i] * da[b * n + j];
        gn += acc * acc;
      }
      res[l] = Math.sqrt(gn);
      const dPrev = new Float64Array(B * n);
      for (let b = 0; b < B; b++) for (let i = 0; i < n; i++) {
        let acc = skip ? dh[b * n + i] : 0;
        for (let j = 0; j < n; j++) acc += da[b * n + j] * W[i * n + j];
        dPrev[b * n + i] = acc;
      }
      dh = dPrev;
    }
    return { res, mid };
  }
  let mid = 0;
  function draw() {
    const r = compute();
    norms = r.res; mid = r.mid;
    render();
  }
  function render() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = clamp(Math.round(W * .45), 200, 300);
    const ctx = fitCanvas(cv, W, H);
    const logs = norms.map(v => Math.log10(Math.max(v, 1e-300)));
    let lo = Math.floor(Math.min(...logs)) - 1, hi = Math.ceil(Math.max(...logs)) + 0.3;
    if (hi - lo < 3) lo = hi - 3;
    const p = makePlot(W, H, [0.4, L + 0.6], [lo, hi], { l: 50, b: 30 });
    plot = p;
    const yt = niceTicks(lo, hi, 6).filter(Number.isInteger);
    drawAxes(ctx, p, { yticks: yt, yfmt: t => '10' + sup(t), xticks: niceTicks(1, L, 8).filter(Number.isInteger), xlabel: 'layer (1 = closest to the input)' });
    const bw = Math.max(2, p.pw / L - 2);
    norms.forEach((v, l) => {
      const x = p.X(l + 1) - bw / 2, y = p.Y(logs[l]), yb = p.Y(lo);
      ctx.fillStyle = hover === l ? c.s2 : c.s1;
      ctx.beginPath();
      const rr = Math.min(4, bw / 2);
      ctx.moveTo(x, yb); ctx.lineTo(x, y + rr); ctx.quadraticCurveTo(x, y, x + rr, y); ctx.lineTo(x + bw - rr, y); ctx.quadraticCurveTo(x + bw, y, x + bw, y + rr); ctx.lineTo(x + bw, yb); ctx.closePath();
      ctx.fill();
    });
    if (aux) {
      const X = p.X(mid + 0.5);
      ctx.strokeStyle = c.s4; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(X, p.pad.t); ctx.lineTo(X, H - p.pad.b); ctx.stroke(); ctx.setLineDash([]);
      label(ctx, 'aux head', X + 4, p.pad.t + 8, { color: c.text, bold: true });
    }
    const ratio = norms[L - 1] / norms[0];
    out.innerHTML = '';
    const bad = ratio > 1e3;
    out.append(h('div', { class: 'callout ' + (bad ? 'bad' : ratio > 10 ? 'warn' : 'good') }, h('span', { class: 'ico' }, bad ? '✗' : ratio > 10 ? '!' : '✓'),
      h('div', {}, `Layer 1 receives ${ratio >= 1 ? fmt(ratio, 1) + '× less' : fmt(1 / ratio, 1) + '× more'} gradient than layer ${L}. `,
        bad ? 'The early layers barely learn — this is the vanishing-gradient problem. Try ReLU, skip connections, or the auxiliary head.'
          : 'Gradient reaches the early layers, so every layer can learn at a similar pace.',
        aux ? ' The auxiliary head injects a fresh, large gradient at the dashed line: compare the bars just below it with the run without the head.' : '')));
  }
  cv.addEventListener('mousemove', e => {
    if (!plot) return;
    const r = cv.getBoundingClientRect();
    const l = Math.round(plot.invX(e.clientX - r.left)) - 1;
    if (l < 0 || l >= L) { hover = -1; tip.hidden = true; render(); return; }
    hover = l; render();
    tip.innerHTML = `<div class="tk">layer ${l + 1}</div><div>‖∂J/∂W‖ = <b>${fmt(norms[l], 3)}</b></div>`;
    tip.hidden = false;
    const X = plot.X(l + 1);
    tip.style.left = clamp(X + 10, 0, plot.W - tip.offsetWidth) + 'px';
    tip.style.top = '8px';
  });
  cv.addEventListener('mouseleave', () => { hover = -1; tip.hidden = true; render(); });
  draw();
  return { redraw: render };
};

/* =========================================================
   11) Continuation methods: blur the loss, then sharpen it
   ========================================================= */
const CPRESETS = {
  wiggly: { label: 'Bumpy bowl — continuation works', dom: [-4, 4], yr: [0, 3.6], f: t => 0.08 * (t - 1) ** 2 + 0.6 * Math.sin(3 * t) + 1, start: -3 },
  hole: { label: 'Narrow deep hole — tracks to the wrong minimum', dom: [-4.5, 4.5], yr: [-0.9, 1.9], f: t => 0.03 * t * t - Math.exp(-((t - 2) ** 2) / 2) - 1.8 * Math.exp(-((t + 2.5) ** 2) / (2 * 0.12 * 0.12)) + 1, start: 0.5 },
  negsq: { label: 'J(θ) = −θ² — never becomes convex', dom: [-4, 4], yr: [-17, 1], f: t => -t * t, start: 0.4 }
};
function blurred(f, t, s) {
  if (s < 1e-3) return f(t);
  let sum = 0, ws = 0;
  const K = 60;
  for (let k = -K; k <= K; k++) { const u = k / K * 4; const w = Math.exp(-u * u / 2); sum += w * f(t + u * s); ws += w; }
  return sum / ws;
}
function descend1d(fn, t0, dom) {
  let t = t0, lr = 0.5;
  for (let it = 0; it < 500; it++) {
    const g = (fn(t + 1e-4) - fn(t - 1e-4)) / 2e-4;
    if (Math.abs(g) < 1e-6) break;
    let step = -lr * g, nt = t + step, tries = 0;
    const ft = fn(t);
    while (fn(nt) > ft && tries++ < 40) { step /= 2; nt = t + step; }
    if (Math.abs(nt - t) < 1e-10) break;
    t = clamp(nt, dom[0], dom[1]);
    if (t === dom[0] || t === dom[1]) break;
  }
  return t;
}
Widgets.continuation = (el, ds) => {
  const keys = (ds.presets || 'wiggly,hole,negsq').split(',');
  let key = keys[0], sigma = 0, cont = null, plain = null, timer = null;
  let start = CPRESETS[key].start;
  UI.head(el, ds.title || 'Blur the landscape, then slowly sharpen it',
    'The blue curve is J blurred with a Gaussian of width σ. <b>Run continuation</b> solves a sequence of less and less blurred problems, each starting from the previous answer.');
  const ctr = h('div', { class: 'controls' });
  if (keys.length > 1) UI.select(ctr, 'Loss function', keys.map(k => [k, CPRESETS[k].label]), key, v => { key = v; start = CPRESETS[v].start; stS.set(start); stop(); cont = plain = null; sigma = 0; sgS.set(0); draw(); });
  const sgS = UI.slider(ctr, { label: 'Blur width σ', min: 0, max: 2, step: 0.01, value: sigma, fmt: v => v.toFixed(2), onInput: v => { sigma = v; draw(); } });
  const stS = UI.slider(ctr, { label: 'Starting point θ₀', min: -4, max: 4, step: 0.05, value: start, fmt: v => v.toFixed(2), onInput: v => { start = v; stop(); cont = plain = null; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Run continuation ▶', () => run(), 'btn primary');
  UI.button(btns, 'Plain gradient descent', () => { plain = descend1d(CPRESETS[key].f, start, CPRESETS[key].dom); draw(); });
  UI.button(btns, 'Reset', () => { stop(); cont = plain = null; sigma = 0; sgS.set(0); draw(); });
  ctr.append(btns);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Loss curve and blurred loss curve' });
  wrap.append(cv);
  el.append(ctr, wrap);
  const lg = UI.legend(el, []);
  const out = h('div', { class: 'stack', style: 'gap:8px' });
  el.append(out);
  function stop() { if (timer) clearInterval(timer); timer = null; }
  function run() {
    stop();
    const P = CPRESETS[key];
    const sched = [2, 1.5, 1.1, 0.8, 0.55, 0.35, 0.2, 0.1, 0];
    let i = 0;
    cont = { th: start, trail: [], stage: 0 };
    const tick = () => {
      if (i >= sched.length) { stop(); draw(); return; }
      sigma = sched[i]; sgS.set(sigma);
      cont.th = descend1d(t => blurred(P.f, t, sigma), cont.th, P.dom);
      cont.trail.push([cont.th, sigma]);
      cont.stage = i + 1;
      i++;
      draw();
    };
    tick();
    timer = setInterval(tick, 650);
  }
  function draw() {
    const c = Colors.get();
    const P = CPRESETS[key];
    const W = boxWidth(wrap), H = clamp(Math.round(W * .55), 230, 360);
    const ctx = fitCanvas(cv, W, H);
    const p = makePlot(W, H, P.dom, P.yr, { l: 40 });
    drawAxes(ctx, p, { xlabel: 'θ', ylabel: 'J(θ)' });
    const N = 360;
    const curve = fn => { const pts = []; for (let k = 0; k <= N; k++) { const t = P.dom[0] + (P.dom[1] - P.dom[0]) * k / N; pts.push([t, fn(t)]); } return pts; };
    const orig = curve(P.f);
    const bl = sigma > 0 ? curve(t => blurred(P.f, t, sigma)) : null;
    ctx.save(); p.clip(ctx);
    const stroke = (pts, col, w, dash) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash || []); ctx.beginPath(); pts.forEach(([x, y], i) => i ? ctx.lineTo(p.X(x), p.Y(y)) : ctx.moveTo(p.X(x), p.Y(y))); ctx.stroke(); ctx.setLineDash([]); };
    stroke(orig, bl ? alpha(c.text, .45) : c.text, bl ? 1.6 : 2.4);
    if (bl) stroke(bl, c.s1, 2.6);
    // global min of original
    let gm = orig[0]; for (const q of orig) if (q[1] < gm[1]) gm = q;
    const hasMin = key !== 'negsq';
    if (hasMin) { ctx.fillStyle = c.text; ctx.font = '600 15px Inter, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('★', p.X(gm[0]), p.Y(gm[1]) + 12); }
    // blurred min
    if (bl && hasMin) { let bm = bl[0]; for (const q of bl) if (q[1] < bm[1]) bm = q; dot(ctx, p.X(bm[0]), p.Y(bm[1]), 5, c.surface, c.s1); }
    // start marker
    ctx.strokeStyle = c.text; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(p.X(start), p.Y(P.f(start)), 6, 0, Math.PI * 2); ctx.stroke();
    if (plain != null) dot(ctx, p.X(plain), p.Y(P.f(plain)), 6.5, c.s7, c.surface);
    if (cont) {
      const lastI = cont.trail.length - 1;
      cont.trail.forEach(([t], i) => {
        if (i === lastI) return;
        dot(ctx, p.X(t), p.Y(P.yr[0]) - 7, 3.2, c.s2, c.surface); // earlier stages, along the bottom
      });
      const t = cont.trail[lastI][0];
      const y = sigma > 0 ? blurred(P.f, t, sigma) : P.f(t);
      dot(ctx, p.X(t), p.Y(y), 7, c.s2, c.surface);
    }
    ctx.restore();
    label(ctx, 'start', p.X(start), p.Y(P.f(start)) - 14, { color: c.text, align: 'center', bold: true });
    lg.set([
      { label: 'original J(θ)', color: c.text },
      { label: `blurred J (σ = ${sigma.toFixed(2)})`, color: c.s1 },
      { label: 'continuation (orange dots along the bottom = its answer at each stage)', color: c.s2, kind: 'dot' },
      { label: 'plain gradient descent', color: c.s7, kind: 'dot' },
      { label: 'global minimum ★', color: c.text, kind: 'dot' }
    ]);
    out.innerHTML = '';
    const rows = [];
    if (cont) rows.push(['Continuation' + (cont.stage < 9 ? ` (stage ${cont.stage}/9)` : ''), cont.th]);
    if (plain != null) rows.push(['Plain gradient descent', plain]);
    if (rows.length) {
      out.append(h('table', { class: 'readout' }, h('thead', {}, h('tr', {}, h('th', {}, 'Method'), h('th', {}, 'ends at θ'), h('th', {}, 'J(θ)'), h('th', {}, 'Global minimum?'))),
        h('tbody', {}, rows.map(([n, t]) => {
          const atEdge = Math.abs(t - P.dom[0]) < 1e-6 || Math.abs(t - P.dom[1]) < 1e-6;
          const ok = hasMin && Math.abs(t - gm[0]) < 0.15;
          return h('tr', {}, h('td', {}, n), h('td', { class: 'num' }, fmt(t, 3)), h('td', { class: 'num' }, fmt(P.f(t), 3)),
            h('td', {}, !hasMin ? (atEdge ? '✗ ran off the edge — there is no minimum' : '✗ no minimum exists') : ok ? '✓ yes' : '✗ no — a local minimum'));
        }))));
    }
    const msg = {
      wiggly: 'With heavy blur the bumps vanish and only the big bowl remains, so gradient descent finds its bottom easily. Each sharper stage starts inside the right basin, so continuation tracks all the way to the global minimum — while plain gradient descent from θ₀ = −3 gets stuck in the first bump.',
      hole: 'Blurring washes out the narrow hole (it is too thin to survive the averaging), so the blurred minimum sits in the wide basin on the right. Continuation then tracks that basin and ends in a local minimum. Blurring became convex, but pointed the wrong way.',
      negsq: 'Blurring a hill gives the same hill, just lower: E[−θ′²] = −θ² − σ². No amount of blur creates a minimum, so continuation has nothing to track.'
    }[key];
    out.append(h('div', { class: 'callout info' }, h('span', { class: 'ico' }, 'i'), h('div', {}, msg)));
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   12) Curriculum learning: deterministic vs stochastic
   ========================================================= */
Widgets.curriculum = (el) => {
  UI.head(el, 'What goes into each minibatch as training progresses?',
    'Each dot is one training example in a minibatch of 24. Slide through training (or press Play).');
  let t = 0, timer = null;
  const ctr = h('div', { class: 'controls' });
  const sl = UI.slider(ctr, { label: 'Training progress', min: 0, max: 100, step: 1, value: t, fmt: v => v + '%', onInput: v => { t = v; stop(); draw(); } });
  const btns = h('div', { class: 'btns' });
  const playBtn = UI.button(btns, 'Play ▶', () => (timer ? stop() : play()), 'btn primary');
  ctr.append(btns);
  const rows = h('div', { class: 'row2' });
  const detBox = h('div', { class: 'stack', style: 'gap:6px' }), stoBox = h('div', { class: 'stack', style: 'gap:6px' });
  rows.append(detBox, stoBox);
  el.append(ctr, rows);
  const lg = UI.legend(el, []);
  el.append(h('div', { class: 'note' }, 'Stochastic curriculum: chance of drawing each difficulty as training progresses (hover to read):'));
  const chart = new LineChart(el, { xr: [0, 100], yr: [0, 1], xlabel: 'training progress %', ylabel: 'probability', xname: 'progress %', aspect: .36, maxH: 210, minH: 160, vfmt: v => (v * 100).toFixed(0) + '%', snapX: x => Math.round(x) });
  const probs = tau => ({ easy: 0.6 * (1 - tau), med: 0.3, hard: 0.1 + 0.6 * tau });
  function play() { if (t >= 100) t = 0; playBtn.textContent = 'Pause ❚❚'; timer = setInterval(() => { t = Math.min(100, t + 1); sl.set(t); draw(); if (t >= 100) stop(); }, 70); }
  function stop() { if (timer) clearInterval(timer); timer = null; playBtn.textContent = 'Play ▶'; }
  function draw() {
    const c = Colors.get();
    const colOf = [c.ord1, c.ord2, c.ord3];
    const tau = t / 100;
    const detLevel = tau < 1 / 3 ? 0 : tau < 2 / 3 ? 1 : 2;
    const det = Array(24).fill(detLevel);
    const rng = mulberry32(1000 + t);
    const pr = probs(tau);
    const sto = Array.from({ length: 24 }, () => { const u = rng(); return u < pr.easy ? 0 : u < pr.easy + pr.med ? 1 : 2; }).sort();
    const grid = arr => h('div', { class: 'dotgrid' }, arr.map(l => h('span', { style: `--c:${colOf[l]}`, title: ['easy', 'medium', 'hard'][l] })));
    const share = arr => Math.round(arr.filter(x => x === 2).length / arr.length * 100);
    detBox.innerHTML = ''; stoBox.innerHTML = '';
    detBox.append(h('b', {}, 'Deterministic curriculum'), h('span', { class: 'note' }, 'Only one difficulty at a time, switched in stages.'), grid(det), h('span', { class: 'note' }, `hard examples in this batch: ${share(det)}%`));
    stoBox.append(h('b', {}, 'Stochastic curriculum'), h('span', { class: 'note' }, 'Always a random mix; the share of hard ones slowly grows.'), grid(sto), h('span', { class: 'note' }, `hard examples in this batch: ${share(sto)}%`));
    lg.set([{ label: 'easy', color: c.ord1, kind: 'dot' }, { label: 'medium', color: c.ord2, kind: 'dot' }, { label: 'hard (long-term dependencies)', color: c.ord3, kind: 'dot' }]);
    const xs = Array.from({ length: 101 }, (_, i) => i);
    chart.set([
      { name: 'easy', color: 'ord1', data: xs.map(x => [x, probs(x / 100).easy]) },
      { name: 'medium', color: 'ord2', data: xs.map(x => [x, probs(x / 100).med]) },
      { name: 'hard', color: 'ord3', data: xs.map(x => [x, probs(x / 100).hard]) },
      { name: 'now', color: 'text', style: 'dots', r: 4, noHover: true, data: [[t, pr.hard]] }
    ]);
  }
  draw();
  return { redraw: draw };
};
