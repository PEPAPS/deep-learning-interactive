/* =========================================================
   Lecture 1 · Ch 5.1–5.3 · Machine Learning Basics I
   ========================================================= */
'use strict';

/* ---------- 1) T, P, E builder ---------- */
Widgets.tpe = (el) => {
  UI.head(el, 'Turn any application into a learning problem', 'Pick an application. Every learning problem needs a <b>Task</b>, a <b>Performance measure</b> and <b>Experience</b> (data).');
  const apps = [
    ['Spam filter', 'Classification: e-mail → {spam, not spam}', 'Accuracy (or error rate) on new e-mails. Maybe weighted: losing a real e-mail is worse than letting one spam through.', 'Supervised: thousands of e-mails already labelled “spam / not spam” by users.'],
    ['House prices', 'Regression: features (size, rooms, location) → price, a real number', 'Mean squared error on houses that were not used for training.', 'Supervised: records of past sales with their final prices.'],
    ['Speech recognition', 'Transcription: audio waveform → sequence of words', 'Word error rate: partial credit for every correct word instead of all-or-nothing.', 'Supervised: hours of recordings with their written transcripts.'],
    ['English → French', 'Machine translation: word sequence → word sequence', 'Overlap with human translations, or human ratings.', 'Supervised: pairs of sentences that mean the same thing.'],
    ['Pedestrian detection', 'Structured output: camera image → boxes around every pedestrian', 'Missed pedestrians must be ~0; false alarms per hour should be small.', 'Supervised: street images where people drew boxes around pedestrians.'],
    ['Credit-card fraud', 'Anomaly detection: flag purchases that do not look like you', 'Fraud caught vs. false alarms (blocked honest purchases).', 'Mostly unsupervised: your normal purchase history (fraud is rare and unlabelled).'],
    ['Customer groups', 'Clustering (unsupervised): split customers into similar groups', 'Hard to define! Usefulness of the groups for marketing.', 'Unsupervised: purchase records with no labels at all.'],
    ['Game-playing agent', 'Choose actions that win the game', 'Win rate or total reward.', 'Reinforcement learning: the agent plays, the environment answers, and the data depend on its own actions (a feedback loop).'],
    ['Image generator', 'Synthesis & sampling: create new images like the training ones', 'Hard to measure: human judgement, or likelihood (often intractable).', 'Unsupervised: a large collection of example images.']
  ];
  let k = 0;
  const grid = h('div', { class: 'gridbtns' });
  const out = h('div', { class: 'stack', style: 'gap:10px' });
  el.append(grid, out);
  const btns = apps.map((a, i) => { const b = h('button', { type: 'button' }, a[0]); b.addEventListener('click', () => { k = i; draw(); }); grid.append(b); return b; });
  function draw() {
    const [name, T, P, E] = apps[k];
    btns.forEach((b, i) => b.classList.toggle('on', i === k));
    out.innerHTML = '';
    out.append(
      h('div', { class: 'compare' },
        h('div', { style: 'border-top:3px solid var(--s1)' }, h('b', {}, 'T · Task'), T),
        h('div', { style: 'border-top:3px solid var(--s2)' }, h('b', {}, 'P · Performance measure'), P),
        h('div', { style: 'border-top:3px solid var(--s3)' }, h('b', {}, 'E · Experience'), E)),
      h('div', { class: 'callout info' }, h('span', { class: 'ico' }, 'i'),
        h('div', {}, h('b', {}, `“The ${name.toLowerCase()} program learns`), ` if its performance (${P.split(/[.:]/)[0].toLowerCase()}) at the task improves as it gets more experience.”`)));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) Task explorer ---------- */
Widgets.taskExplorer = (el) => {
  UI.head(el, 'The main kinds of tasks', 'Click a task to see what goes in, what comes out, a real example, and where it appears later in this course.');
  const tasks = [
    ['Classification', 'x ∈ ℝⁿ (e.g. pixels)', 'one of k classes {1,…,k} — or a probability for each class', 'Handwritten digits, face recognition, object recognition, spam filter, music genre.', ['ch06-1.html#/s-softmax', 'Softmax output units (L4)']],
    ['Classification with missing inputs', 'x with some features missing', 'a class', 'Medical diagnosis: some tests are expensive or invasive, so they are often not done. Instead of one function per missing-subset, learn p(x) and “marginalize out” what is missing.', ['ch05-4.html#/s-gen-disc', 'Generative models (L2)']],
    ['Regression', 'x ∈ ℝⁿ', 'a real number y ∈ ℝ', 'Insurance claim amounts, stock prices, weather forecasts.', ['ch06-1.html#/s-output-linear', 'Linear output units (L4)']],
    ['Transcription', 'unstructured data (image, audio)', 'discrete text', 'Reading house numbers in Google Street View photos; speech recognition.', ['ch10.html#/s-rnn-designs', 'RNN sequence outputs (L11)']],
    ['Machine translation', 'sequence of symbols in language A', 'sequence of symbols in language B', 'English → French.', ['ch10.html#/s-rnn-designs', 'Sequence-to-sequence RNNs (L11)']],
    ['Structured output', 'anything', 'a vector / tree / image whose parts are tightly related', 'Parsing a sentence into a grammar tree, labelling every pixel (segmentation), writing a caption for an image.', ['ch09.html#/s-cnn-arch', 'CNNs (L10) + RNNs (L11)']],
    ['Anomaly detection', 'events or objects', 'flag “unusual” ones', 'Credit-card fraud: a thief’s purchases come from a different distribution than yours.', null],
    ['Synthesis & sampling', '(maybe a condition, e.g. text)', 'new examples like the training data', 'Video-game textures, art, speech synthesis — many outputs are correct, and we want variety.', null],
    ['Imputation of missing values', 'x with some entries missing', 'predicted values of the missing entries', 'Filling in gaps in sensor logs or surveys.', null],
    ['Denoising', 'corrupted example x̃', 'clean example x (or p(x | x̃))', 'Removing noise from photos or audio.', ['ch07.html#/s-noise', 'Noise robustness (L6)']],
    ['Density estimation', 'x', 'a probability p_model(x) for every x', 'Learning where data cluster. In principle it solves the other tasks (e.g. imputation via p(xᵢ | x₋ᵢ)), but those computations are often intractable.', ['ch05-4.html#/s-mle', 'Maximum likelihood (L2)']]
  ];
  let k = 0;
  const grid = h('div', { class: 'gridbtns' });
  const out = h('div', { class: 'stack', style: 'gap:10px' });
  el.append(grid, out);
  const btns = tasks.map((t, i) => { const b = h('button', { type: 'button' }, t[0]); b.addEventListener('click', () => { k = i; draw(); }); grid.append(b); return b; });
  function draw() {
    const [name, inp, outp, ex, link] = tasks[k];
    btns.forEach((b, i) => b.classList.toggle('on', i === k));
    out.innerHTML = '';
    out.append(
      h('div', { class: 'io' }, h('span', { class: 'box' }, inp), h('span', { class: 'arr' }, '→'), h('span', { class: 'box', style: 'border-color:var(--accent)' }, 'model f'), h('span', { class: 'arr' }, '→'), h('span', { class: 'box' }, outp)),
      h('p', {}, h('b', {}, name + ': '), ex),
      link ? h('p', { class: 'note' }, 'Later in the course: ', h('a', { href: link[0] }, link[1])) : null);
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) Which mistakes should a metric punish? ---------- */
Widgets.metricChoice = (el) => {
  UI.head(el, 'Choosing P: many medium mistakes, or one huge mistake?', 'Two regression models make errors on the same 10 test houses. Move the sliders and see which model each metric prefers.');
  let med = 2, big = 15;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Model A: every error is', min: 0.5, max: 5, step: 0.1, value: med, fmt: v => v.toFixed(1), onInput: v => { med = v; draw(); } });
  UI.slider(ctr, { label: 'Model B: 9 perfect, one error of', min: 1, max: 40, step: 0.5, value: big, fmt: v => v.toFixed(1), onInput: v => { big = v; draw(); } });
  const bars = h('div', { class: 'hbars' });
  const calc = h('div', { class: 'calcbox' });
  const verdict = h('div');
  el.append(ctr, bars, calc, verdict);
  function draw() {
    const c = Colors.get();
    const A = Array(10).fill(med), B = [...Array(9).fill(0), big];
    const mae = a => mean(a.map(Math.abs)), msq = a => mean(a.map(v => v * v));
    bars.innerHTML = '';
    const maxv = Math.max(med, big);
    [['Model A errors', A, c.s1], ['Model B errors', B, c.s2]].forEach(([lab, arr, col]) => {
      const row = h('div', { style: 'display:grid;grid-template-columns:repeat(10,1fr);gap:3px;align-items:end;height:60px' },
        arr.map(v => h('div', { title: 'error ' + v, style: `height:${Math.max(2, v / maxv * 60)}px;background:${col};border-radius:3px 3px 0 0` })));
      bars.append(h('div', { class: 'hbar' }, h('div', { class: 'lab' }, lab), row));
    });
    calc.innerHTML = `<span class="c">MAE = (1/m)·Σ|error|          MSE = (1/m)·Σ error²</span>
A: MAE = (10 × ${med.toFixed(1)}) / 10 = <b>${mae(A).toFixed(2)}</b>      MSE = (10 × ${med.toFixed(1)}²) / 10 = <b>${msq(A).toFixed(2)}</b>
B: MAE = (9 × 0 + ${big.toFixed(1)}) / 10 = <b>${mae(B).toFixed(2)}</b>      MSE = (9 × 0 + ${big.toFixed(1)}²) / 10 = <b>${msq(B).toFixed(2)}</b>`;
    const maeP = mae(A) < mae(B) ? 'A' : 'B', mseP = msq(A) < msq(B) ? 'A' : 'B';
    verdict.innerHTML = '';
    verdict.append(h('div', { class: 'callout ' + (maeP !== mseP ? 'warn' : 'info') }, h('span', { class: 'ico' }, maeP !== mseP ? '!' : 'i'),
      h('div', {}, `Mean absolute error prefers model ${maeP}; mean squared error prefers model ${mseP}. `,
        maeP !== mseP ? 'Same models, opposite verdicts! Squaring punishes big mistakes much more, so the choice of P is a design decision about what “good” means for your task.' : 'Here both agree. Make model B’s single mistake bigger to see them disagree.')));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) Design matrix & kinds of experience ---------- */
Widgets.designMatrix = (el) => {
  UI.head(el, 'A dataset as a design matrix', 'Rows are examples (iris flowers), columns are features (measurements in cm). Switch the kind of experience the algorithm gets.');
  const rows = [
    [5.1, 3.5, 1.4, 0.2, 0], [4.9, 3.0, 1.4, 0.2, 0], [4.7, 3.2, 1.3, 0.2, 0],
    [7.0, 3.2, 4.7, 1.4, 1], [6.4, 3.2, 4.5, 1.5, 1], [5.5, 2.3, 4.0, 1.3, 1],
    [6.3, 3.3, 6.0, 2.5, 2], [5.8, 2.7, 5.1, 1.9, 2], [7.1, 3.0, 5.9, 2.1, 2]
  ];
  const names = ['setosa', 'versicolor', 'virginica'];
  const rng = mulberry32(7);
  const stats = [[1.46, 0.17, 0.25, 0.1], [4.26, 0.47, 1.33, 0.2], [5.55, 0.55, 2.03, 0.27]];
  const cloud = stats.map(([ml, sl, mw, sw]) => Array.from({ length: 50 }, () => [ml + sl * gaussian(rng), Math.max(0.05, mw + sw * gaussian(rng))]));
  let mode = 'sup';
  const segBox = h('div');
  UI.seg(segBox, [['sup', 'Supervised'], ['unsup', 'Unsupervised'], ['semi', 'Semi-supervised']], mode, v => { mode = v; draw(); });
  const tbl = h('div', { class: 'scroll-x' });
  el.append(segBox, tbl);
  const chart = new LineChart(el, { xr: [0.5, 7.5], yr: [0, 3], xlabel: 'petal length (cm)', ylabel: 'petal width (cm)', xname: 'petal length', aspect: .5, maxH: 260 });
  const lg = UI.legend(el, []);
  const note = h('div');
  el.append(note);
  function draw() {
    const c = Colors.get();
    const cols = [c.s1, c.s2, c.s3];
    tbl.innerHTML = '';
    const head = h('tr', {}, h('th', {}, 'i'), ...['sepal length', 'sepal width', 'petal length', 'petal width'].map(t => h('th', {}, t)), mode === 'unsup' ? null : h('th', {}, 'label y'));
    const body = rows.map((r, i) => {
      const hidden = mode === 'semi' && i % 3 !== 0;
      return h('tr', {}, h('th', {}, i + 1), ...r.slice(0, 4).map(v => h('td', {}, v.toFixed(1))),
        mode === 'unsup' ? null : h('td', { style: hidden ? 'color:var(--muted)' : `color:${cols[r[4]]};font-weight:700` }, hidden ? '?' : names[r[4]]));
    });
    tbl.append(h('table', { class: 'ntable' }, h('caption', {}, 'Design matrix X (first 9 of 150 rows) — X ∈ ℝ¹⁵⁰ˣ⁴, X₍ᵢ,₁₎ = sepal length of plant i'), h('thead', {}, head), h('tbody', {}, body)));
    if (mode === 'unsup') {
      chart.set([{ name: 'flower', color: 'muted', style: 'dots', data: cloud.flat(), r: 3 }]);
      lg.set([{ label: 'flowers (no labels — the algorithm must find the groups itself, e.g. by clustering)', color: c.muted, kind: 'dot' }]);
    } else {
      chart.set(cloud.map((pts, k) => ({ name: names[k], color: 's' + (k + 1), style: 'dots', r: 3, data: mode === 'semi' ? pts.filter((_, i) => i % 6 === 0) : pts }))
        .concat(mode === 'semi' ? [{ name: 'unlabelled', color: 'muted', style: 'dots', r: 2.5, data: cloud.flat().filter((_, i) => i % 6 !== 0), noHover: true }] : []));
      lg.set(names.map((n, k) => ({ label: n, color: cols[k], kind: 'dot' })).concat(mode === 'semi' ? [{ label: 'unlabelled', color: c.muted, kind: 'dot' }] : []));
    }
    note.innerHTML = '';
    note.append(h('div', { class: 'callout info' }, h('span', { class: 'ico' }, 'i'), h('div', {}, {
      sup: 'Supervised: every example comes with a label y given by a “teacher”. The algorithm learns p(y | x): predict the species from the measurements.',
      unsup: 'Unsupervised: only the features. The algorithm learns properties of p(x) itself — e.g. that the flowers form groups (clusters), or the full density.',
      semi: 'Semi-supervised: only some examples have labels (“?” = unknown). The many unlabelled points still show where the groups are.'
    }[mode])));
    note.append(h('p', { class: 'note', style: 'margin-top:6px' }, 'Plot: 50 flowers per species simulated to match the real iris averages; the table shows real iris rows.'));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Linear regression: drag the line, see the squares, solve exactly ---------- */
Widgets.linreg = (el) => {
  UI.head(el, 'Linear regression: ŷ = w·x + b',
    'Each shaded square has side = the error of one point, so its <b>area is the squared error</b>. MSE is the average area. Move w and b, or solve the normal equations. Click the plot to add a point.');
  const rng = mulberry32(21);
  const base = Array.from({ length: 7 }, (_, i) => { const x = 1 + i * 1.3; return [x, 1.5 + 0.55 * x + 0.9 * gaussian(rng)]; });
  let pts = base.map(p => p.slice()), w = 0.2, b = 3;
  const ctr = h('div', { class: 'controls' });
  const wS = UI.slider(ctr, { label: 'weight w (slope)', min: -1, max: 2, step: 0.01, value: w, fmt: v => v.toFixed(2), onInput: v => { w = v; draw(); } });
  const bS = UI.slider(ctr, { label: 'bias b (intercept)', min: -3, max: 8, step: 0.05, value: b, fmt: v => v.toFixed(2), onInput: v => { b = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Solve the normal equations', () => { const s = solve(); w = s[1]; b = s[0]; wS.set(w); bS.set(b); draw(); }, 'btn primary');
  UI.button(btns, 'Reset points', () => { pts = base.map(p => p.slice()); draw(); });
  ctr.append(btns);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart clickable', role: 'img', 'aria-label': 'Data points, regression line and squared errors' });
  wrap.append(cv);
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, wrap, calc);
  let p = null;
  cv.addEventListener('click', e => {
    if (!p) return;
    const r = cv.getBoundingClientRect();
    const x = p.invX(e.clientX - r.left), y = p.invY(e.clientY - r.top);
    if (x < 0 || x > 10 || y < -1 || y > 10) return;
    pts.push([x, y]); if (pts.length > 14) pts.shift(); draw();
  });
  function solve() { return ridgeFit(pts.map(([x]) => [1, x]), pts.map(q => q[1]), 0); }
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = clamp(Math.round(W * .62), 240, 380);
    const ctx = fitCanvas(cv, W, H);
    p = makePlot(W, H, [0, 10], [-1, 10], { l: 34 });
    drawAxes(ctx, p, { xlabel: 'x (feature)', ylabel: 'y (target)' });
    ctx.save(); p.clip(ctx);
    pts.forEach(([x, y]) => {
      const yh = w * x + b, X = p.X(x), Y = p.Y(y), Yh = p.Y(yh), side = Math.abs(Y - Yh);
      ctx.fillStyle = alpha(c.s2, .16); ctx.strokeStyle = alpha(c.s2, .7); ctx.lineWidth = 1;
      ctx.fillRect(X, Math.min(Y, Yh), side, side); ctx.strokeRect(X, Math.min(Y, Yh), side, side);
      ctx.strokeStyle = c.s2; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X, Y); ctx.lineTo(X, Yh); ctx.stroke();
    });
    ctx.strokeStyle = c.text; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(p.X(0), p.Y(b)); ctx.lineTo(p.X(10), p.Y(10 * w + b)); ctx.stroke();
    pts.forEach(([x, y]) => dot(ctx, p.X(x), p.Y(y), 5, c.s1, c.surface));
    ctx.restore();
    const m = pts.length;
    const errs = pts.map(([x, y]) => w * x + b - y);
    const mseV = mean(errs.map(e => e * e));
    const s = solve();
    const best = mean(pts.map(([x, y]) => (s[1] * x + s[0] - y) ** 2));
    const sx = pts.reduce((a, q) => a + q[0], 0), sxx = pts.reduce((a, q) => a + q[0] * q[0], 0), sy = pts.reduce((a, q) => a + q[1], 0), sxy = pts.reduce((a, q) => a + q[0] * q[1], 0);
    calc.innerHTML = `<span class="c">1) errors  eᵢ = ŷᵢ − yᵢ = (w·xᵢ + b) − yᵢ</span>
   ${errs.slice(0, 5).map(e => e.toFixed(2)).join(', ')}${m > 5 ? ', …' : ''}
<span class="c">2) MSE = (1/m)·Σ eᵢ²  with m = ${m}</span>
   = (${errs.slice(0, 4).map(e => e.toFixed(2) + '²').join(' + ')}${m > 4 ? ' + …' : ''}) / ${m} = <b>${mseV.toFixed(3)}</b>
<span class="c">3) Normal equations  (XᵀX)·[b, w]ᵀ = Xᵀy   (X has a column of 1s for the bias)</span>
   [ ${m}      ${sx.toFixed(1)} ] [b]   [ ${sy.toFixed(1)} ]
   [ ${sx.toFixed(1)}  ${sxx.toFixed(1)} ] [w] = [ ${sxy.toFixed(1)} ]
   → b = <b>${s[0].toFixed(3)}</b>, w = <b>${s[1].toFixed(3)}</b>,  smallest possible MSE = <b>${best.toFixed(3)}</b>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 6) Capacity: polynomial degree ---------- */
Widgets.polyfit = (el) => {
  UI.head(el, 'Capacity in action: fitting polynomials',
    'True function: sin(2πx) plus noise. The model is a polynomial of degree d: ŷ = w₀ + w₁x + … + w_d x^d — still linear in the weights, so the normal equations solve it.');
  let deg = 3, n = 10, noise = 0.25, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Degree d (capacity)', min: 0, max: 9, step: 1, value: deg, fmt: v => String(v), onInput: v => { deg = v; draw(); } });
  UI.slider(ctr, { label: 'Training points m', min: 4, max: 60, step: 1, value: n, fmt: v => String(v), onInput: v => { n = v; draw(); } });
  UI.slider(ctr, { label: 'Noise σ', min: 0, max: 0.6, step: 0.01, value: noise, fmt: v => v.toFixed(2), onInput: v => { noise = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'New training sample', () => { seed++; draw(); });
  ctr.append(btns);
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 1], yr: [-1.8, 1.8], xlabel: 'x', ylabel: 'y', xname: 'x', aspect: .55, maxH: 320 });
  const lg = UI.legend(el, []);
  const stat = h('div', { class: 'stat-row' });
  const verdict = h('div');
  el.append(stat, verdict);
  function draw() {
    const c = Colors.get();
    const train = makeData(n, noise, seed);
    const wv = fitPoly(train, deg);
    const xs = Array.from({ length: 201 }, (_, i) => i / 200);
    const testSet = makeData(300, noise, 999);
    chart.set([
      { name: 'true sin(2πx)', color: 'muted', data: xs.map(x => [x, TRUE_F(x)]), dash: [5, 4], noHover: true },
      { name: 'model ŷ', color: 's2', data: xs.map(x => [x, evalPoly(wv, x)]) },
      { name: 'training point', color: 's1', style: 'dots', r: 4.5, data: train, noHover: true }
    ]);
    lg.set([{ label: 'true function', color: c.muted, kind: 'dash' }, { label: `fitted polynomial, degree ${deg}`, color: c.s2 }, { label: 'training points', color: c.s1, kind: 'dot' }]);
    const tr = polyMSE(wv, train), te = polyMSE(wv, testSet);
    stat.innerHTML = '';
    stat.append(
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Parameters (d + 1)'), h('div', { class: 'v' }, String(deg + 1))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Training MSE'), h('div', { class: 'v' }, fmt(tr, 3))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Test MSE (300 new points)'), h('div', { class: 'v' }, fmt(te, 3))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Gap (test − train)'), h('div', { class: 'v' }, fmt(te - tr, 3))));
    const nv = noise * noise;
    let cls, ico, msg;
    if (tr > nv + 0.12) { cls = 'warn'; ico = '!'; msg = 'Underfitting: even the training error is large. The model family is too simple (capacity too low) to follow the curve.'; }
    else if (te > 2.5 * (nv + 0.01) + 0.05) { cls = 'bad'; ico = '✗'; msg = 'Overfitting: the curve passes (almost) through every training point, but wiggles wildly between them, so the test error is large. Capacity is too high for this amount of data.'; }
    else { cls = 'good'; ico = '✓'; msg = 'Good fit: training error is about the noise level and the test error is close to it — the capacity matches the task and the data.'; }
    verdict.innerHTML = '';
    verdict.append(h('div', { class: 'callout ' + cls }, h('span', { class: 'ico' }, ico), h('div', {}, msg, deg >= n - 1 ? ` With d + 1 = ${deg + 1} ≥ m = ${n} parameters, the polynomial can pass through every training point exactly.` : '')));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) The U-shaped curve & the effect of more data ---------- */
Widgets.capacityCurve = (el) => {
  UI.head(el, 'Training vs. test error as capacity grows',
    'Averaged over 40 random training sets (noise σ = 0.25, so the best possible MSE — the <b>Bayes error</b> — is σ² = 0.0625, dashed line).');
  let n = 12, deg2 = 3;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Training points m', min: 6, max: 80, step: 1, value: n, fmt: v => String(v), onInput: v => { n = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 9], yr: [0.01, 10], logY: true, xlabel: 'degree d (capacity)', xname: 'degree', aspect: .45, maxH: 260, refY: 0.0625, snapX: x => Math.round(x) });
  const lg = UI.legend(el, []);
  const note = h('div');
  el.append(note);
  el.append(h('div', { class: 'note', style: 'margin-top:6px' }, 'Second view — fix the degree and add data:'));
  const ctr2 = h('div', { class: 'controls' });
  UI.slider(ctr2, { label: 'Degree d', min: 1, max: 9, step: 1, value: deg2, fmt: v => String(v), onInput: v => { deg2 = v; draw2(); } });
  el.append(ctr2);
  const chart2 = new LineChart(el, { xr: [5, 100], yr: [0.01, 10], logY: true, xlabel: 'training points m', xname: 'm', aspect: .4, maxH: 230, refY: 0.0625, snapX: x => Math.round(x / 5) * 5 });
  const test = makeData(300, 0.25, 4242);
  function avgErr(deg, m) {
    let tr = 0, te = 0, R = 40;
    for (let r = 0; r < R; r++) {
      const d = makeData(m, 0.25, 100 + r * 7 + m * 1000);
      const wv = fitPoly(d, deg, 1e-9);
      tr += Math.max(1e-6, polyMSE(wv, d)); te += Math.min(1e3, polyMSE(wv, test));
    }
    return [tr / R, te / R];
  }
  function draw() {
    const c = Colors.get();
    const rows = Array.from({ length: 10 }, (_, d) => [d, ...avgErr(d, n)]);
    const best = rows.reduce((a, r) => (r[2] < a[2] ? r : a));
    chart.set([
      { name: 'training error', color: 's1', data: rows.map(r => [r[0], Math.max(0.0101, r[1])]) },
      { name: 'test (generalization) error', color: 's2', data: rows.map(r => [r[0], Math.min(9.9, r[2])]) },
      { name: 'best capacity', color: 'text', style: 'dots', r: 6, noHover: true, data: [[best[0], best[2]]] }
    ]);
    lg.set([{ label: 'training error', color: c.s1 }, { label: 'test error', color: c.s2 }, { label: 'optimal capacity (lowest test error)', color: c.text, kind: 'dot' }, { label: 'Bayes error σ²', color: c.muted, kind: 'dash' }]);
    note.innerHTML = '';
    note.append(h('div', { class: 'callout info' }, h('span', { class: 'ico' }, 'i'), h('div', {},
      `With m = ${n} points the best degree is ${best[0]}. Left of it: underfitting zone (both errors high). Right of it: overfitting zone (training error keeps falling, test error rises — the gap grows).`)));
    draw2();
  }
  function draw2() {
    const c = Colors.get();
    const ms = Array.from({ length: 20 }, (_, i) => 5 + i * 5).filter(m => m > deg2);
    const rows = ms.map(m => [m, ...avgErr(deg2, m)]);
    chart2.set([
      { name: 'training error', color: 's1', data: rows.map(r => [r[0], Math.max(0.0101, r[1])]) },
      { name: 'test error', color: 's2', data: rows.map(r => [r[0], Math.min(9.9, r[2])]) }
    ]);
    void c;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 8) No free lunch ---------- */
Widgets.nfl = (el) => {
  UI.head(el, 'No free lunch, by brute force',
    'We have seen some training points (all labelled 1). Three new points A, B, C are unseen. Choose your algorithm’s predictions, then check every possible “true world”.');
  let pred = [1, 1, 1], prior = 'uniform';
  const ctr = h('div', { class: 'controls' });
  const pbox = h('div', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', {}, 'Your algorithm predicts for A, B, C')));
  const pbtns = h('div', { class: 'btns' });
  const pb = [0, 1, 2].map(i => UI.button(pbtns, '', () => { pred[i] = 1 - pred[i]; draw(); }));
  pbox.append(pbtns); ctr.append(pbox);
  const segBox = h('div', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', {}, 'How likely is each world?')));
  UI.seg(segBox, [['uniform', 'All worlds equally likely'], ['smooth', 'Real-world assumption']], prior, v => { prior = v; draw(); });
  ctr.append(segBox);
  const tbl = h('div', { class: 'scroll-x' });
  const out = h('div');
  el.append(ctr, tbl, out);
  function draw() {
    pb.forEach((b, i) => { b.textContent = 'ABC'[i] + ' = ' + pred[i]; b.className = 'btn' + (pred[i] ? ' primary' : ''); });
    const worlds = Array.from({ length: 8 }, (_, k) => [(k >> 2) & 1, (k >> 1) & 1, k & 1]);
    const wt = wd => prior === 'uniform' ? 1 : 3 ** wd.reduce((a, v) => a + v, 0);
    const tot = worlds.reduce((a, wd) => a + wt(wd), 0);
    let accMine = 0, accZero = 0;
    const body = worlds.map(wd => {
      const a1 = wd.filter((v, i) => v === pred[i]).length / 3, a0 = wd.filter(v => v === 0).length / 3;
      accMine += a1 * wt(wd) / tot; accZero += a0 * wt(wd) / tot;
      return h('tr', {}, h('td', {}, wd.join(' ')), h('td', {}, (wt(wd) / tot * 100).toFixed(1) + '%'), h('td', {}, (a1 * 100).toFixed(0) + '%'), h('td', {}, (a0 * 100).toFixed(0) + '%'));
    });
    tbl.innerHTML = '';
    tbl.append(h('table', { class: 'ntable' }, h('caption', {}, 'All 2³ = 8 possible true labelings of A, B, C'),
      h('thead', {}, h('tr', {}, h('th', {}, 'true A B C'), h('th', {}, 'probability'), h('th', {}, 'your accuracy'), h('th', {}, '“always 0” accuracy'))),
      h('tbody', {}, body, h('tr', { class: 'stat-r' }, h('td', {}, 'weighted average'), h('td', {}, '100%'), h('td', {}, (accMine * 100).toFixed(1) + '%'), h('td', {}, (accZero * 100).toFixed(1) + '%')))));
    out.innerHTML = '';
    out.append(prior === 'uniform'
      ? h('div', { class: 'callout warn' }, h('span', { class: 'ico' }, '!'), h('div', {}, 'Averaged over all worlds equally, every algorithm scores exactly 50% — even the silly “always predict 0”. That is the no free lunch theorem.'))
      : h('div', { class: 'callout good' }, h('span', { class: 'ico' }, '✓'), h('div', {}, 'Assume the real world is “smooth” (new points tend to share the label of the training points, weight 3 per label 1). Now predicting 1s wins. Good learning = good assumptions about the real world.')));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 9) Weight decay ---------- */
Widgets.weightDecay = (el) => {
  UI.head(el, 'Weight decay on a degree-9 polynomial', 'J(w) = MSE_train + λ·wᵀw. Slide λ from tiny to huge and watch the curve go from wild to flat.');
  let lg10 = -3;
  const train = makeData(10, 0.25, 3);
  const test = makeData(300, 0.25, 77);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'λ (strength of preference for small weights)', min: -9, max: 1, step: 0.1, value: lg10, fmt: v => (10 ** v).toExponential(1), onInput: v => { lg10 = v; draw(); } });
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 1], yr: [-1.8, 1.8], xlabel: 'x', ylabel: 'y', xname: 'x', aspect: .55, maxH: 300 });
  const lg = UI.legend(el, []);
  const stat = h('div', { class: 'stat-row' });
  const calc = h('div', { class: 'calcbox' });
  const verdict = h('div');
  el.append(stat, calc, verdict);
  function draw() {
    const c = Colors.get();
    const lam = 10 ** lg10;
    const wv = fitPoly(train, 9, lam);
    const xs = Array.from({ length: 201 }, (_, i) => i / 200);
    chart.set([
      { name: 'true', color: 'muted', data: xs.map(x => [x, TRUE_F(x)]), dash: [5, 4], noHover: true },
      { name: 'model ŷ', color: 's2', data: xs.map(x => [x, evalPoly(wv, x)]) },
      { name: 'train', color: 's1', style: 'dots', r: 4.5, data: train, noHover: true }
    ]);
    lg.set([{ label: 'true function', color: c.muted, kind: 'dash' }, { label: 'degree-9 fit with weight decay', color: c.s2 }, { label: 'training points', color: c.s1, kind: 'dot' }]);
    const tr = polyMSE(wv, train), te = polyMSE(wv, test), ww = dotv(wv.slice(1), wv.slice(1));
    stat.innerHTML = '';
    stat.append(
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Training MSE'), h('div', { class: 'v' }, fmt(tr, 3))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Test MSE'), h('div', { class: 'v' }, fmt(te, 3))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'wᵀw (size of weights)'), h('div', { class: 'v' }, fmt(ww, 2))));
    calc.innerHTML = `<span class="c">J(w) = MSE_train + λ · wᵀw</span>
     = ${tr.toFixed(4)} + ${lam.toExponential(1)} × ${fmt(ww, 3)} = <b>${fmt(tr + lam * ww, 4)}</b>
<span class="c">weights w₁…w₉ (on t = 2x−1):</span> ${wv.slice(1).map(v => fmt(v, 1)).join(', ')}`;
    verdict.innerHTML = '';
    const cls = lg10 < -6 ? ['bad', '✗', 'λ ≈ 0: no preference, so the 10 weights fit the 10 points exactly and the weights become huge → overfitting.']
      : lg10 > -0.5 ? ['warn', '!', 'λ very large: small weights matter more than fitting the data, so the curve flattens toward a constant → underfitting.']
        : ['good', '✓', 'Medium λ: the model still has degree 9 (same hypothesis space) but prefers small weights, giving a smooth curve that generalizes.'];
    verdict.append(h('div', { class: 'callout ' + cls[0] }, h('span', { class: 'ico' }, cls[1]), h('div', {}, cls[2])));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 10) Choosing a hyper-parameter with a validation set ---------- */
Widgets.hyperSearch = (el) => {
  UI.head(el, 'Pick the degree with a validation set', '40 training examples are split 80% / 20% into a training part and a validation part. The test set is kept locked away until the very end.');
  let seed = 5;
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'New dataset', () => { seed++; draw(); }, 'btn primary');
  el.append(btns);
  const chart = new LineChart(el, { xr: [0, 9], yr: [0.01, 10], logY: true, xlabel: 'degree d (hyper-parameter)', xname: 'degree', aspect: .45, maxH: 250, snapX: x => Math.round(x) });
  const lg = UI.legend(el, []);
  const out = h('div', { class: 'stack', style: 'gap:8px' });
  el.append(out);
  const test = makeData(300, 0.25, 31337);
  function draw() {
    const c = Colors.get();
    const all = makeData(40, 0.25, 500 + seed);
    const tr = all.slice(0, 32), va = all.slice(32);
    const rows = Array.from({ length: 10 }, (_, d) => { const wv = fitPoly(tr, d, 1e-9); return [d, polyMSE(wv, tr), polyMSE(wv, va), wv]; });
    const bestV = rows.reduce((a, r) => (r[2] < a[2] ? r : a));
    const bestT = rows.reduce((a, r) => (r[1] < a[1] ? r : a));
    chart.set([
      { name: 'training error', color: 's1', data: rows.map(r => [r[0], Math.max(0.0101, r[1])]) },
      { name: 'validation error', color: 's2', data: rows.map(r => [r[0], Math.min(9.9, r[2])]) },
      { name: 'chosen', color: 'text', style: 'dots', r: 6, noHover: true, data: [[bestV[0], bestV[2]]] }
    ]);
    lg.set([{ label: 'error on the 32 training examples', color: c.s1 }, { label: 'error on the 8 validation examples', color: c.s2 }, { label: 'chosen degree', color: c.text, kind: 'dot' }]);
    out.innerHTML = '';
    out.append(
      h('div', { class: 'stat-row' },
        h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Chosen by validation'), h('div', { class: 'v' }, 'd = ' + bestV[0])),
        h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Would be chosen by training error'), h('div', { class: 'v' }, 'd = ' + bestT[0])),
        h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Test MSE of chosen model'), h('div', { class: 'v' }, fmt(polyMSE(bestV[3], test), 3)))),
      h('div', { class: 'callout info' }, h('span', { class: 'ico' }, 'i'), h('div', {}, 'Training error always prefers the biggest capacity (it can only go down as d grows). That is why hyper-parameters that control capacity must be chosen on data the training algorithm never saw — the validation set.')));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 11) k-fold cross-validation ---------- */
Widgets.kfold = (el) => {
  UI.head(el, 'k-fold cross-validation', 'A small dataset of 20 examples. Each trial holds out one fold as the test set and trains on the rest; the estimate is the average over the k trials.');
  let k = 5, deg = 3;
  const data = makeData(20, 0.25, 8080);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Number of folds k', min: 2, max: 10, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  UI.slider(ctr, { label: 'Polynomial degree being evaluated', min: 1, max: 7, step: 1, value: deg, fmt: v => String(v), onInput: v => { deg = v; draw(); } });
  const grid = h('div', { style: 'display:grid;gap:4px' });
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, grid, calc);
  function draw() {
    const c = Colors.get();
    const fold = i => Math.floor(i * k / data.length);
    grid.innerHTML = '';
    const errs = [];
    for (let t = 0; t < k; t++) {
      const tr = data.filter((_, i) => fold(i) !== t), te = data.filter((_, i) => fold(i) === t);
      const wv = fitPoly(tr, deg, 1e-9);
      const e = polyMSE(wv, te);
      errs.push(e);
      grid.append(h('div', { style: 'display:grid;grid-template-columns:70px 1fr 90px;gap:8px;align-items:center;font-size:.78rem' },
        h('span', {}, 'trial ' + (t + 1)),
        h('div', { style: `display:grid;grid-template-columns:repeat(${data.length},1fr);gap:2px` }, data.map((_, i) => h('span', { style: `height:16px;border-radius:3px;background:${fold(i) === t ? c.s2 : alpha(c.s1, .55)}` }))),
        h('span', { style: 'font-family:var(--mono)' }, 'MSE ' + fmt(e, 3))));
    }
    grid.append(UI.legend(h('div'), [{ label: 'used for training in this trial', color: alpha(c.s1, .55), kind: 'dot' }, { label: 'held-out test fold', color: c.s2, kind: 'dot' }]).el);
    const avg = mean(errs), sd = Math.sqrt(mean(errs.map(e => (e - avg) ** 2)));
    calc.innerHTML = `<span class="c">CV estimate = (1/k)·Σ_trial MSE_trial</span>
  = (${errs.map(e => fmt(e, 3)).join(' + ')}) / ${k} = <b>${fmt(avg, 3)}</b>
<span class="c">spread between trials (std): ${fmt(sd, 3)} — every example is used for testing exactly once</span>`;
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   Quiz
   ========================================================= */
window.QUIZ = [
  { q: 'In Mitchell’s definition of learning, what are T, P and E?', opts: ['Training, Prediction, Error', 'Task, Performance measure, Experience', 'Test set, Parameters, Epochs', 'Theory, Practice, Evaluation'], a: 1, why: 'A program learns from experience E with respect to tasks T and performance measure P if its performance at T, measured by P, improves with E.' },
  { q: 'Predicting tomorrow’s temperature (a real number) is which kind of task?', opts: ['Classification', 'Regression', 'Density estimation', 'Transcription'], a: 1, why: 'Regression outputs a numerical value, f: ℝⁿ → ℝ. Classification outputs one of k categories.' },
  { q: 'What does the design matrix X ∈ ℝ^{m×n} contain?', opts: ['One example per column', 'One example per row and one feature per column', 'The model weights', 'Only the labels'], a: 1, why: 'Each row is an example (e.g. one iris flower), each column a feature (e.g. petal length). Labels go in a separate vector y.' },
  { q: 'The normal equations w = (XᵀX)⁻¹Xᵀy come from…', opts: ['setting the gradient of the training MSE to zero', 'maximizing the test error', 'random search', 'counting misclassifications'], a: 0, why: '∇_w MSE_train = 0 gives XᵀXw = Xᵀy. Solving that linear system is the whole learning algorithm for linear regression.' },
  { q: 'A degree-9 polynomial fits 10 training points perfectly but does badly on new points. This is…', opts: ['underfitting', 'overfitting', 'the Bayes error', 'regularization'], a: 1, why: 'Training error ≈ 0 but a large gap to the test error = overfitting: capacity too high for the amount of data.' },
  { q: 'Which statement about the Bayes error is true?', opts: ['It is 0 for a perfect model', 'It is the error of an oracle that knows the true distribution — caused by noise in the data', 'It always equals the training error', 'It grows with model capacity'], a: 1, why: 'Even the true distribution p(x, y) cannot predict noise, so its error (here σ² = 0.0625) is the floor no model can beat.' },
  { q: 'The no free lunch theorem says…', opts: ['deep learning always wins', 'averaged over all possible data-generating distributions, all classifiers have the same error on unseen points', 'more data never helps', 'regularization is useless'], a: 1, why: 'Only when we make assumptions about real-world distributions can one algorithm beat another — which is why we design preferences (regularization) for the tasks we care about.' },
  { q: 'Regularization is defined as any modification that aims to reduce…', opts: ['training error but not generalization error', 'generalization error but not training error', 'the number of parameters only', 'computation time'], a: 1, why: 'For example weight decay J(w) = MSE + λwᵀw prefers smaller weights, trading a bit of training fit for better generalization.' },
  { q: 'Why can’t we choose the polynomial degree using the training set?', opts: ['It is too slow', 'The training error always prefers the largest capacity, which overfits', 'The degree is not a number', 'Because of the Bayes error'], a: 1, why: 'Hyper-parameters controlling capacity must be chosen on a validation set (typically 20% of the training data) that the training algorithm does not see.' },
  { q: 'In k-fold cross-validation, how many times is each example used as test data?', opts: ['Never', 'Exactly once', 'k times', 'k − 1 times'], a: 1, why: 'The data are split into k disjoint folds; trial i tests on fold i and trains on the others, so each example is tested exactly once and the k errors are averaged.' }
];
