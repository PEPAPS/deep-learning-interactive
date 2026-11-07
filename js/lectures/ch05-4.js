/* =========================================================
   Lecture 2 · Ch 5.4–5.7 · Machine Learning Basics II
   ========================================================= */
'use strict';

/* small helper: HTML histogram */
function histBars(values, lo, hi, bins, color, markX) {
  const counts = new Array(bins).fill(0);
  values.forEach(v => { const k = clamp(Math.floor((v - lo) / (hi - lo) * bins), 0, bins - 1); counts[k]++; });
  const mx = Math.max(1, ...counts);
  const wrap = h('div', { style: 'position:relative' });
  wrap.append(h('div', { style: `display:grid;grid-template-columns:repeat(${bins},1fr);gap:2px;align-items:end;height:110px;border-bottom:1px solid var(--axis)` },
    counts.map((n, i) => h('div', { title: `${(lo + i * (hi - lo) / bins).toFixed(2)}–${(lo + (i + 1) * (hi - lo) / bins).toFixed(2)}: ${n}`, style: `height:${n / mx * 108}px;background:${color};border-radius:3px 3px 0 0;min-height:${n ? 2 : 0}px` }))));
  if (markX != null) wrap.append(h('div', { title: 'true value', style: `position:absolute;top:0;bottom:0;left:${(markX - lo) / (hi - lo) * 100}%;border-left:2px dashed var(--text)` }));
  wrap.append(h('div', { style: 'display:flex;justify-content:space-between;font-size:.7rem;color:var(--muted)' }, h('span', {}, lo), h('span', {}, ((lo + hi) / 2).toFixed(2)), h('span', {}, hi)));
  return wrap;
}

/* ---------- 1) An estimator is a random variable ---------- */
Widgets.coinSampler = (el) => {
  UI.head(el, 'An estimate is a random variable', 'A coin lands heads (1) with true probability θ. Each dataset = m flips; the estimator is θ̂ = (number of heads) / m. Draw many datasets and look at how θ̂ is spread.');
  let theta = 0.3, m = 10, est = [], last = null, rng = mulberry32(1);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'True θ (unknown in real life)', min: 0.05, max: 0.95, step: 0.05, value: theta, fmt: v => v.toFixed(2), onInput: v => { theta = v; est = []; last = null; draw(); } });
  UI.slider(ctr, { label: 'Flips per dataset m', min: 1, max: 100, step: 1, value: m, fmt: v => String(v), onInput: v => { m = v; est = []; last = null; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Draw 1 dataset', () => { draws(1); }, 'btn primary');
  UI.button(btns, 'Draw 500 datasets', () => { draws(500); });
  UI.button(btns, 'Clear', () => { est = []; last = null; draw(); });
  ctr.append(btns);
  const flips = h('div', { style: 'display:flex;flex-wrap:wrap;gap:3px;min-height:16px' });
  const hist = h('div');
  const stats = h('div', { class: 'stat-row' });
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, flips, hist, stats, calc);
  function draws(n) {
    for (let r = 0; r < n; r++) { const f = Array.from({ length: m }, () => (rng() < theta ? 1 : 0)); last = f; est.push(mean(f)); }
    draw();
  }
  function draw() {
    const c = Colors.get();
    flips.innerHTML = '';
    if (last) last.slice(0, 100).forEach(v => flips.append(h('span', { title: v ? 'heads (1)' : 'tails (0)', style: `width:12px;height:12px;border-radius:50%;border:2px solid ${c.s1};background:${v ? c.s1 : 'transparent'}` })));
    hist.innerHTML = '';
    hist.append(h('div', { class: 'note' }, `Histogram of θ̂ over ${est.length} datasets (dashed line = true θ):`), histBars(est, 0, 1, 20, c.s2, theta));
    const avg = est.length ? mean(est) : NaN, sd = est.length > 1 ? Math.sqrt(mean(est.map(v => (v - avg) ** 2))) : NaN;
    stats.innerHTML = '';
    stats.append(
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Average of θ̂ (≈ E[θ̂])'), h('div', { class: 'v' }, isFinite(avg) ? avg.toFixed(3) : '—')),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Bias = E[θ̂] − θ'), h('div', { class: 'v' }, isFinite(avg) ? (avg - theta).toFixed(3) : '—')),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Std of θ̂ (measured)'), h('div', { class: 'v' }, isFinite(sd) ? sd.toFixed(3) : '—')),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Theory √(θ(1−θ)/m)'), h('div', { class: 'v' }, Math.sqrt(theta * (1 - theta) / m).toFixed(3))));
    const k = last ? last.reduce((a, b) => a + b, 0) : 0;
    calc.innerHTML = last ? `<span class="c">last dataset: m = ${m} flips, ${k} heads</span>
θ̂ = (1/m)·Σ x⁽ⁱ⁾ = ${k} / ${m} = <b>${(k / m).toFixed(3)}</b>   (true θ = ${theta.toFixed(2)}; a new dataset gives a different θ̂)` : '<span class="c">Press “Draw 1 dataset”.</span>';
  }
  draw();
  return { redraw: draw };
};

/* ---------- 2) Biased vs unbiased variance estimator ---------- */
Widgets.varianceBias = (el) => {
  UI.head(el, 'Two estimators of the variance', 'Data ~ Gaussian with true variance σ² = 4. For each of 4 000 datasets of size m we compute both estimators and average them.');
  let m = 4;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Dataset size m', min: 2, max: 40, step: 1, value: m, fmt: v => String(v), onInput: v => { m = v; draw(); } });
  const bars = h('div', { class: 'hbars' });
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, bars, calc);
  function draw() {
    const c = Colors.get();
    const rng = mulberry32(9 + m), s2 = 4, R = 4000;
    let b = 0, u = 0, first = null;
    for (let r = 0; r < R; r++) {
      const x = Array.from({ length: m }, () => 2 * gaussian(rng));
      const mu = mean(x), ss = x.reduce((a, v) => a + (v - mu) ** 2, 0);
      b += ss / m; u += ss / (m - 1);
      if (!first) first = { x, mu, ss };
    }
    b /= R; u /= R;
    const row = (lab, v, col) => h('div', { class: 'hbar' }, h('div', { class: 'lab' }, lab), h('div', { class: 'track' }, h('div', { class: 'fill', style: `width:${v / 5 * 100}%;--c:${col}` }), h('div', { class: 'ref', style: `left:${4 / 5 * 100}%;border-color:var(--text)` }), h('span', { class: 'val' }, v.toFixed(3))));
    bars.innerHTML = '';
    bars.append(row('Average of σ̂²ₘ = (1/m)Σ(x−x̄)²', b, c.s2), row('Average of σ̃²ₘ = (1/(m−1))Σ(x−x̄)²', u, c.s3), row('Theory for σ̂²ₘ: (m−1)/m · σ²', (m - 1) / m * s2, c.muted));
    bars.append(h('div', { class: 'legend' }, h('span', {}, h('i', { class: 'dash', style: `--c:${c.text}` }), 'true σ² = 4')));
    const f = first;
    calc.innerHTML = `<span class="c">one dataset (m = ${m}):</span> x = ${f.x.slice(0, 6).map(v => v.toFixed(2)).join(', ')}${m > 6 ? ', …' : ''}
x̄ = ${f.mu.toFixed(3)},  Σ(xᵢ − x̄)² = ${f.ss.toFixed(3)}
biased:   σ̂² = ${f.ss.toFixed(3)} / ${m} = <b>${(f.ss / m).toFixed(3)}</b>
unbiased: σ̃² = ${f.ss.toFixed(3)} / ${m - 1} = <b>${(f.ss / (m - 1)).toFixed(3)}</b>
<span class="c">bias of σ̂² = E[σ̂²] − σ² = −σ²/m = −4/${m} = ${(-4 / m).toFixed(3)}  (measured: ${(b - 4).toFixed(3)})</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 3) Standard error & 95% confidence intervals ---------- */
Widgets.ciDemo = (el) => {
  UI.head(el, 'How sure is a test-error estimate?', 'A classifier’s true error rate is p. We test it on 25 independent test sets of m examples each and draw a 95% confidence interval for each one.');
  let p = 0.2, lm = 2, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'True error rate p', min: 0.02, max: 0.5, step: 0.01, value: p, fmt: v => v.toFixed(2), onInput: v => { p = v; draw(); } });
  UI.slider(ctr, { label: 'Test-set size m', min: 1.3, max: 3.5, step: 0.05, value: lm, fmt: v => String(Math.round(10 ** v)), onInput: v => { lm = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'New test sets', () => { seed++; draw(); }, 'btn primary');
  ctr.append(btns);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Confidence intervals' });
  wrap.append(cv);
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, wrap);
  const lg = UI.legend(el, []);
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const m = Math.round(10 ** lm), rng = mulberry32(seed * 31 + m);
    const sets = Array.from({ length: 25 }, () => {
      let e = 0; for (let i = 0; i < m; i++) if (rng() < p) e++;
      const ph = e / m, se = Math.sqrt(ph * (1 - ph) / m);
      return { e, ph, se, lo: ph - 1.96 * se, hi: ph + 1.96 * se };
    });
    const W = boxWidth(wrap), H = 260;
    const ctx = fitCanvas(cv, W, H);
    const P = makePlot(W, H, [0, Math.min(1, p * 2.2 + 0.1)], [0, 26], { l: 20, b: 26 });
    drawAxes(ctx, P, { xlabel: 'error rate', yticks: [], grid: false });
    ctx.strokeStyle = c.text; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(P.X(p), P.pad.t); ctx.lineTo(P.X(p), H - P.pad.b); ctx.stroke(); ctx.setLineDash([]);
    let miss = 0;
    sets.forEach((s, i) => {
      const ok = s.lo <= p && p <= s.hi; if (!ok) miss++;
      const y = P.Y(i + 1), col = ok ? c.s1 : c.critical;
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(P.X(Math.max(0, s.lo)), y); ctx.lineTo(P.X(s.hi), y); ctx.stroke();
      dot(ctx, P.X(s.ph), y, 3.5, col, c.surface);
    });
    lg.set([{ label: `interval contains the true p (${25 - miss} of 25)`, color: c.s1 }, { label: `misses (${miss}) — expected about 5%`, color: c.critical }, { label: 'true p', color: c.text, kind: 'dash' }]);
    const s = sets[0];
    calc.innerHTML = `<span class="c">test set #1: ${s.e} errors out of m = ${m}</span>
p̂ = ${s.e}/${m} = ${s.ph.toFixed(4)}
SE = √(p̂(1 − p̂)/m) = √(${s.ph.toFixed(3)} × ${(1 - s.ph).toFixed(3)} / ${m}) = <b>${s.se.toFixed(4)}</b>
95% CI = p̂ ± 1.96·SE = [${s.lo.toFixed(4)}, ${s.hi.toFixed(4)}]   <span class="c">← 4× more test data halves the width</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 4) Bias–variance decomposition with polynomials ---------- */
Widgets.biasVariance = (el) => {
  UI.head(el, 'Bias and variance of a fitted curve', '25 different training sets (m = 15 points each, noise σ = 0.25). Thin lines = the 25 fitted curves; thick line = their average.');
  let deg = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Polynomial degree (capacity)', min: 0, max: 9, step: 1, value: deg, fmt: v => String(v), onInput: v => { deg = v; draw(); } });
  el.append(ctr);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Many fitted curves' });
  wrap.append(cv); el.append(wrap);
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  el.append(h('div', { class: 'note' }, 'Decomposition for every degree (log scale, averaged over 60 training sets):'));
  const chart = new LineChart(el, { xr: [0, 9], yr: [1e-3, 30], logY: true, xlabel: 'degree', xname: 'degree', aspect: .42, maxH: 240, snapX: x => Math.round(x) });
  const lg2 = UI.legend(el, []);
  const xs = Array.from({ length: 41 }, (_, i) => 0.02 + i * 0.024);
  function stats(d, R) {
    const fits = Array.from({ length: R }, (_, r) => fitPoly(makeData(15, 0.25, 700 + r), d, 1e-9));
    const preds = fits.map(w => xs.map(x => evalPoly(w, x)));
    const avg = xs.map((_, j) => mean(preds.map(p => p[j])));
    const bias2 = mean(xs.map((x, j) => (avg[j] - TRUE_F(x)) ** 2));
    const vari = mean(xs.map((_, j) => mean(preds.map(p => (p[j] - avg[j]) ** 2))));
    return { fits, avg, bias2, vari };
  }
  let table = null;
  function draw() {
    const c = Colors.get();
    const S = stats(deg, 25);
    const W = boxWidth(wrap), H = clamp(Math.round(W * .55), 220, 330);
    const ctx = fitCanvas(cv, W, H);
    const P = makePlot(W, H, [0, 1], [-2, 2], { l: 30 });
    drawAxes(ctx, P, { xlabel: 'x' });
    ctx.save(); P.clip(ctx);
    S.fits.forEach(w => { ctx.strokeStyle = alpha(c.s1, .35); ctx.lineWidth = 1; ctx.beginPath(); for (let i = 0; i <= 100; i++) { const x = i / 100; const y = evalPoly(w, x); i ? ctx.lineTo(P.X(x), P.Y(y)) : ctx.moveTo(P.X(x), P.Y(y)); } ctx.stroke(); });
    ctx.strokeStyle = c.muted; ctx.setLineDash([5, 4]); ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i <= 100; i++) { const x = i / 100; i ? ctx.lineTo(P.X(x), P.Y(TRUE_F(x))) : ctx.moveTo(P.X(x), P.Y(TRUE_F(x))); } ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = c.s2; ctx.lineWidth = 3; ctx.beginPath(); xs.forEach((x, j) => j ? ctx.lineTo(P.X(x), P.Y(S.avg[j])) : ctx.moveTo(P.X(x), P.Y(S.avg[j]))); ctx.stroke();
    ctx.restore();
    lg.set([{ label: 'one fit per training set', color: alpha(c.s1, .6) }, { label: 'average fit E[f̂(x)]', color: c.s2 }, { label: 'true f(x)', color: c.muted, kind: 'dash' }]);
    calc.innerHTML = `<span class="c">bias²  = avg over x of (E[f̂(x)] − f(x))²       = gap between thick orange and dashed</span>
bias²  = <b>${S.bias2.toFixed(4)}</b>
<span class="c">variance = avg over x of E[(f̂(x) − E[f̂(x)])²]  = spread of the thin lines</span>
variance = <b>${S.vari.toFixed(4)}</b>
expected test MSE ≈ bias² + variance + noise σ² = ${S.bias2.toFixed(4)} + ${S.vari.toFixed(4)} + 0.0625 = <b>${(S.bias2 + S.vari + 0.0625).toFixed(4)}</b>`;
    if (!table) table = Array.from({ length: 10 }, (_, d) => { const s = stats(d, 60); return [d, s.bias2, s.vari]; });
    chart.set([
      { name: 'bias²', color: 's2', data: table.map(r => [r[0], Math.max(1e-3, r[1])]) },
      { name: 'variance', color: 's1', data: table.map(r => [r[0], Math.max(1e-3, Math.min(29, r[2]))]) },
      { name: 'bias² + var + σ²', color: 'text', data: table.map(r => [r[0], Math.min(29, r[1] + r[2] + 0.0625)]) },
      { name: 'current', color: 's4', style: 'dots', r: 6, noHover: true, data: [[deg, Math.min(29, table[deg][1] + table[deg][2] + 0.0625)]] }
    ]);
    lg2.set([{ label: 'bias² (falls with capacity)', color: c.s2 }, { label: 'variance (rises with capacity)', color: c.s1 }, { label: 'total expected error (U-shape)', color: c.text }]);
  }
  draw();
  return { redraw: draw };
};

/* ---------- 5) Maximum likelihood for a coin ---------- */
Widgets.mleCoin = (el) => {
  UI.head(el, 'Maximum likelihood for a coin', 'You observed k heads in m flips. For every candidate θ, how probable is exactly this data? The best θ is the one that makes the data most probable.');
  let m = 10, k = 7;
  const ctr = h('div', { class: 'controls' });
  const kS = UI.slider(ctr, { label: 'Heads observed k', min: 0, max: m, step: 1, value: k, fmt: v => String(v), onInput: v => { k = v; draw(); } });
  UI.slider(ctr, { label: 'Flips m', min: 1, max: 100, step: 1, value: m, fmt: v => String(v), onInput: v => { m = v; kS.input.max = v; if (k > m) { k = m; } kS.set(k); draw(); } });
  el.append(ctr);
  const ch1 = new LineChart(el, { xr: [0, 1], yr: [0, 1.05], xlabel: 'θ', ylabel: 'likelihood (scaled to max 1)', xname: 'θ', aspect: .36, maxH: 200, minH: 150 });
  const ch2 = new LineChart(el, { xr: [0, 1], yr: [-40, 0], xlabel: 'θ', ylabel: 'log-likelihood', xname: 'θ', aspect: .36, maxH: 200, minH: 150 });
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const ts = Array.from({ length: 199 }, (_, i) => (i + 1) / 200);
    const ll = t => k * Math.log(t) + (m - k) * Math.log(1 - t);
    const th = k / m, best = (k === 0 || k === m) ? 0 : ll(th);
    ch1.set([{ name: 'L(θ)', color: 's1', data: ts.map(t => [t, Math.exp(ll(t) - best)]) }, { name: 'θ_ML', color: 'text', style: 'dots', r: 5, noHover: true, data: [[th, 1]] }]);
    const lo = Math.min(-5, ll(0.05), ll(0.95));
    ch2.set([{ name: 'log L(θ)', color: 's2', data: ts.map(t => [t, ll(t)]) }, { name: 'θ_ML', color: 'text', style: 'dots', r: 5, noHover: true, data: (k > 0 && k < m) ? [[th, best]] : [] }], { yr: [Math.max(lo, -200), Math.max(1, best + 1)] });
    const pr = Math.pow(0.6, 2000);
    calc.innerHTML = `<span class="c">likelihood of these flips:</span> L(θ) = θ^k (1−θ)^(m−k) = θ^${k} (1−θ)^${m - k}
<span class="c">log-likelihood:</span>  log L(θ) = k·log θ + (m−k)·log(1−θ)
<span class="c">set the derivative to 0:</span>  k/θ − (m−k)/(1−θ) = 0  ⇒  θ_ML = k/m = ${k}/${m} = <b>${th.toFixed(3)}</b>
<span class="c">why use logs?</span> 0.6^2000 in floating point = ${pr} (underflow!) but 2000·log 0.6 = ${(2000 * Math.log(0.6)).toFixed(1)} is fine`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 6) Gaussian likelihood = MSE ---------- */
Widgets.gaussLik = (el) => {
  UI.head(el, 'Linear regression as maximum likelihood', 'The model predicts a whole Gaussian p(y | x) = N(y; ŷ(x), σ²) at every x (the bells). The likelihood of a point is the height of its bell where the point lies.');
  const rng = mulberry32(12);
  const pts = Array.from({ length: 7 }, (_, i) => { const x = 1 + i * 1.3; return [x, 1 + 0.6 * x + 0.8 * gaussian(rng)]; });
  let w = 0.3, b = 2.5, sg = 1;
  const ctr = h('div', { class: 'controls' });
  const wS = UI.slider(ctr, { label: 'w', min: -0.5, max: 1.5, step: 0.01, value: w, fmt: v => v.toFixed(2), onInput: v => { w = v; draw(); } });
  const bS = UI.slider(ctr, { label: 'b', min: -2, max: 6, step: 0.05, value: b, fmt: v => v.toFixed(2), onInput: v => { b = v; draw(); } });
  UI.slider(ctr, { label: 'noise σ (fixed by the user)', min: 0.4, max: 2.5, step: 0.05, value: sg, fmt: v => v.toFixed(2), onInput: v => { sg = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Maximize the likelihood', () => { const s = ridgeFit(pts.map(([x]) => [1, x]), pts.map(q => q[1]), 0); b = s[0]; w = s[1]; wS.set(w); bS.set(b); draw(); }, 'btn primary');
  ctr.append(btns);
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Regression line with Gaussian noise model' });
  wrap.append(cv);
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, wrap, calc);
  function draw() {
    const c = Colors.get();
    const W = boxWidth(wrap), H = clamp(Math.round(W * .6), 240, 360);
    const ctx = fitCanvas(cv, W, H);
    const P = makePlot(W, H, [0, 10.5], [-1, 10], { l: 30 });
    drawAxes(ctx, P, { xlabel: 'x', ylabel: 'y' });
    ctx.save(); P.clip(ctx);
    ctx.strokeStyle = c.text; ctx.lineWidth = 2.2; ctx.beginPath(); ctx.moveTo(P.X(0), P.Y(b)); ctx.lineTo(P.X(10.5), P.Y(10.5 * w + b)); ctx.stroke();
    const scale = 1.3 / normPdf(0, 0, 0.4);
    pts.forEach(([x, y]) => {
      const mu = w * x + b;
      ctx.fillStyle = alpha(c.s1, .15); ctx.strokeStyle = alpha(c.s1, .7); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(P.X(x), P.Y(mu - 3.5 * sg));
      for (let t = -3.5; t <= 3.5; t += 0.1) ctx.lineTo(P.X(x + normPdf(t * sg, 0, sg) * scale), P.Y(mu + t * sg));
      ctx.lineTo(P.X(x), P.Y(mu + 3.5 * sg)); ctx.closePath(); ctx.fill(); ctx.stroke();
      const dens = normPdf(y, mu, sg);
      ctx.strokeStyle = c.s2; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(P.X(x), P.Y(y)); ctx.lineTo(P.X(x + dens * scale), P.Y(y)); ctx.stroke();
      dot(ctx, P.X(x), P.Y(y), 4.5, c.s2, c.surface);
    });
    ctx.restore();
    const m = pts.length, errs = pts.map(([x, y]) => w * x + b - y), msev = mean(errs.map(e => e * e));
    const dens = pts.map(([x, y]) => normPdf(y, w * x + b, sg));
    const logL = dens.reduce((a, d) => a + Math.log(d), 0);
    calc.innerHTML = `<span class="c">p(y⁽ⁱ⁾|x⁽ⁱ⁾) = (1/(σ√2π))·exp(−(ŷ⁽ⁱ⁾ − y⁽ⁱ⁾)²/(2σ²))   (orange bar length)</span>
   ${dens.slice(0, 4).map(d => d.toFixed(3)).join(', ')}, …
log L = Σ log p = −m·log σ − (m/2)·log 2π − Σ(ŷ − y)²/(2σ²)
      = −${m}·log ${sg.toFixed(2)} − ${m / 2}·log 2π − ${(m * msev).toFixed(3)}/(2·${(sg * sg).toFixed(3)}) = <b>${logL.toFixed(3)}</b>
MSE = ${msev.toFixed(4)}   <span class="c">→ log L = constant − (m/(2σ²))·MSE, so max log L ⇔ min MSE</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 7) Consistency ---------- */
Widgets.consistency = (el) => {
  UI.head(el, 'Consistent vs. merely unbiased', 'Data ~ N(μ = 2, 1). As the dataset grows, the sample mean homes in on μ. The estimator “just use the first sample x⁽¹⁾” is also unbiased — but never improves.');
  let seed = 1;
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'New random runs', () => { seed++; draw(); }, 'btn primary');
  el.append(btns);
  const chart = new LineChart(el, { xr: [1, 300], yr: [-0.5, 4.5], xlabel: 'm (examples seen)', ylabel: 'estimate of μ', xname: 'm', aspect: .5, maxH: 280, refY: 2, snapX: x => Math.round(x) });
  const lg = UI.legend(el, []);
  function draw() {
    const c = Colors.get();
    const series = [];
    for (let r = 0; r < 4; r++) {
      const rng = mulberry32(seed * 100 + r);
      const xs = Array.from({ length: 300 }, () => 2 + gaussian(rng));
      let s = 0; const avg = xs.map((v, i) => { s += v; return [i + 1, s / (i + 1)]; });
      series.push({ name: 'sample mean, run ' + (r + 1), color: 's1', data: avg, width: 1.6 });
      series.push({ name: 'first sample, run ' + (r + 1), color: 's2', data: [[1, xs[0]], [300, xs[0]]], dash: [5, 4], width: 1.6 });
    }
    chart.set(series);
    lg.set([{ label: 'sample mean μ̂ₘ — consistent: converges to μ', color: c.s1 }, { label: 'x⁽¹⁾ — unbiased but not consistent', color: c.s2, kind: 'dash' }, { label: 'true μ = 2', color: c.muted, kind: 'dash' }]);
  }
  draw();
  return { redraw: draw };
};

/* ---------- 8) Bayesian updating for a coin ---------- */
Widgets.betaCoin = (el) => {
  UI.head(el, 'Bayesian learning: prior × likelihood → posterior', 'Belief about a coin’s θ starts as a prior Beta(a, b). Each flip updates it with Bayes’ rule. Watch the posterior narrow (lose entropy) as data arrives.');
  let a = 2, b = 2, H_ = 0, T_ = 0, rng = mulberry32(4);
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Prior a (pseudo-heads)', min: 0.5, max: 20, step: 0.5, value: a, fmt: v => v.toFixed(1), onInput: v => { a = v; draw(); } });
  UI.slider(ctr, { label: 'Prior b (pseudo-tails)', min: 0.5, max: 20, step: 0.5, value: b, fmt: v => v.toFixed(1), onInput: v => { b = v; draw(); } });
  const btns = h('div', { class: 'btns' });
  UI.button(btns, '+ Heads', () => { H_++; draw(); });
  UI.button(btns, '+ Tails', () => { T_++; draw(); });
  UI.button(btns, 'Flip 10 (true θ = 0.7)', () => { for (let i = 0; i < 10; i++) rng() < 0.7 ? H_++ : T_++; draw(); }, 'btn primary');
  UI.button(btns, 'Reset data', () => { H_ = T_ = 0; draw(); });
  ctr.append(btns);
  el.append(ctr);
  const chart = new LineChart(el, { xr: [0, 1], yr: [0, 5], xlabel: 'θ', ylabel: 'density', xname: 'θ', aspect: .5, maxH: 280, vfmt: v => v.toFixed(2) });
  const lg = UI.legend(el, []);
  const stats = h('div', { class: 'stat-row' });
  const calc = h('div', { class: 'calcbox' });
  el.append(stats, calc);
  function draw() {
    const c = Colors.get();
    const m = H_ + T_, pa = a + H_, pb = b + T_;
    const ts = Array.from({ length: 199 }, (_, i) => (i + 1) / 200);
    const post = ts.map(t => betaPdf(t, pa, pb)), prior = ts.map(t => betaPdf(t, a, b));
    const lik = m ? ts.map(t => betaPdf(t, H_ + 1, T_ + 1)) : null;
    const ymax = Math.min(25, Math.max(2, ...post, ...prior, ...(lik || [0])) * 1.1);
    const S = [{ name: 'prior p(θ)', color: 'muted', data: ts.map((t, i) => [t, prior[i]]), dash: [5, 4] }];
    if (lik) S.push({ name: 'likelihood (normalized)', color: 's1', data: ts.map((t, i) => [t, lik[i]]), width: 1.5 });
    S.push({ name: 'posterior p(θ|data)', color: 's2', data: ts.map((t, i) => [t, post[i]]), width: 3 });
    chart.set(S, { yr: [0, ymax] });
    lg.set([{ label: 'prior', color: c.muted, kind: 'dash' }, { label: 'likelihood of the flips', color: c.s1 }, { label: 'posterior', color: c.s2 }]);
    const mle = m ? H_ / m : NaN, map = (pa > 1 && pb > 1) ? (pa - 1) / (pa + pb - 2) : NaN, pm = pa / (pa + pb), psd = Math.sqrt(pa * pb / ((pa + pb) ** 2 * (pa + pb + 1)));
    stats.innerHTML = '';
    stats.append(
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Data'), h('div', { class: 'v' }, `${H_} H, ${T_} T`)),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'MLE k/m'), h('div', { class: 'v' }, isFinite(mle) ? mle.toFixed(3) : '—')),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'MAP (posterior peak)'), h('div', { class: 'v' }, isFinite(map) ? map.toFixed(3) : '—')),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'P(next = heads)'), h('div', { class: 'v' }, pm.toFixed(3), h('small', {}, ` ± ${psd.toFixed(3)}`))));
    calc.innerHTML = `<span class="c">Bayes’ rule:  p(θ | data) = p(data | θ)·p(θ) / p(data)</span>
likelihood  p(data|θ) = θ^${H_}·(1−θ)^${T_}
prior       p(θ) ∝ θ^(a−1)·(1−θ)^(b−1) = θ^${(a - 1).toFixed(1)}·(1−θ)^${(b - 1).toFixed(1)}
posterior   ∝ θ^${(pa - 1).toFixed(1)}·(1−θ)^${(pb - 1).toFixed(1)}  =  <b>Beta(${pa.toFixed(1)}, ${pb.toFixed(1)})</b>
MAP = (a+k−1)/(a+b+m−2) = ${isFinite(map) ? map.toFixed(3) : '—'}   predictive p(x⁽ᵐ⁺¹⁾=1 | data) = ∫θ p(θ|data)dθ = (a+k)/(a+b+m) = <b>${pm.toFixed(3)}</b>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 9) MAP = weight decay: less variance, more bias ---------- */
Widgets.mapRidge = (el) => {
  UI.head(el, 'MAP with a Gaussian prior: trading variance for bias', 'True model y = 1.5·x + noise (σ = 1), only 4 training points per dataset. We estimate the slope w on 40 different datasets, with and without the prior w ~ N(0, 1/α).');
  let al = 2;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Prior precision α (weight-decay strength)', min: 0, max: 12, step: 0.1, value: al, fmt: v => v.toFixed(1), onInput: v => { al = v; draw(); } });
  const wrap = h('div', { class: 'canvas-wrap' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Slope estimates' });
  wrap.append(cv);
  const tbl = h('div', { class: 'scroll-x' });
  const calc = h('div', { class: 'calcbox' });
  el.append(ctr, wrap);
  const lg = UI.legend(el, []);
  el.append(tbl, calc);
  const rng = mulberry32(55);
  const sets = Array.from({ length: 40 }, () => Array.from({ length: 4 }, () => { const x = 2 * rng() - 1; return [x, 1.5 * x + gaussian(rng)]; }));
  function draw() {
    const c = Colors.get();
    const est = sets.map(d => { const sxy = d.reduce((s, [x, y]) => s + x * y, 0), sxx = d.reduce((s, [x]) => s + x * x, 0); return [sxy / sxx, sxy / (sxx + al)]; });
    const W = boxWidth(wrap), H = 130;
    const ctx = fitCanvas(cv, W, H);
    const P = makePlot(W, H, [-4, 7], [0, 2], { l: 70, b: 24, t: 6 });
    drawAxes(ctx, P, { yticks: [], grid: false, xlabel: 'estimated slope w' });
    ctx.strokeStyle = c.text; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(P.X(1.5), P.pad.t); ctx.lineTo(P.X(1.5), H - P.pad.b); ctx.stroke(); ctx.setLineDash([]);
    [['MLE', 0, c.s1, 1.5], ['MAP', 1, c.s2, 0.5]].forEach(([lab, k, col, yy]) => {
      label(ctx, lab, 8, P.Y(yy), { color: c.text, bold: true });
      est.forEach(e => dot(ctx, P.X(clamp(e[k], -4, 7)), P.Y(yy), 4, alpha(col, .75)));
    });
    lg.set([{ label: 'maximum likelihood slopes', color: c.s1, kind: 'dot' }, { label: 'MAP slopes (prior pulls toward 0)', color: c.s2, kind: 'dot' }, { label: 'true w = 1.5', color: c.text, kind: 'dash' }]);
    const st = k => { const v = est.map(e => e[k]); const mu = mean(v); return { mu, bias: mu - 1.5, vr: mean(v.map(x => (x - mu) ** 2)) }; };
    const A = st(0), B = st(1);
    tbl.innerHTML = '';
    tbl.append(h('table', { class: 'ntable' }, h('thead', {}, h('tr', {}, h('th', {}, ''), h('th', {}, 'mean'), h('th', {}, 'bias'), h('th', {}, 'variance'), h('th', {}, 'MSE = bias² + var'))),
      h('tbody', {}, [['MLE', A], ['MAP', B]].map(([n, s]) => h('tr', {}, h('th', {}, n), h('td', {}, s.mu.toFixed(3)), h('td', {}, s.bias.toFixed(3)), h('td', {}, s.vr.toFixed(3)), h('td', {}, (s.bias ** 2 + s.vr).toFixed(3)))))));
    const d = sets[0], sxy = d.reduce((s, [x, y]) => s + x * y, 0), sxx = d.reduce((s, [x]) => s + x * x, 0);
    calc.innerHTML = `<span class="c">−log posterior = ½·Σ(y − w·x)² + (α/2)·w² + const      (log-prior of N(0,1/α) = −(α/2)w² + const)</span>
dataset 1: Σxy = ${sxy.toFixed(3)}, Σx² = ${sxx.toFixed(3)}
w_MLE = Σxy / Σx²       = ${sxy.toFixed(3)} / ${sxx.toFixed(3)} = <b>${(sxy / sxx).toFixed(3)}</b>
w_MAP = Σxy / (Σx² + α) = ${sxy.toFixed(3)} / ${(sxx + al).toFixed(3)} = <b>${(sxy / (sxx + al)).toFixed(3)}</b>   <span class="c">← exactly weight decay</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 10) Curse of dimensionality ---------- */
Widgets.curse = (el) => {
  UI.head(el, 'The curse of dimensionality: empty cells everywhere', 'Cut each input axis into bins. The number of cells is binsᵈ — it explodes with the number of dimensions d, while the number of examples stays the same.');
  let bins = 10, lm = 3;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Bins per axis', min: 2, max: 20, step: 1, value: bins, fmt: v => String(v), onInput: v => { bins = v; draw(); } });
  UI.slider(ctr, { label: 'Training examples m', min: 1, max: 6, step: 0.1, value: lm, fmt: v => fmt(Math.round(10 ** v), 0), onInput: v => { lm = v; draw(); } });
  const wrap = h('div', { class: 'canvas-wrap', style: 'max-width:300px' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': '2-D grid with examples' });
  wrap.append(cv);
  const bars = h('div', { class: 'hbars' });
  el.append(ctr, h('div', { class: 'row2' }, h('div', {}, h('div', { class: 'note' }, 'd = 2 picture (up to 600 points drawn):'), wrap), h('div', {}, h('div', { class: 'note' }, 'Fraction of cells containing at least one example:'), bars)));
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const m = Math.round(10 ** lm);
    const W = Math.min(300, boxWidth(wrap)), H = W;
    const ctx = fitCanvas(cv, W, H);
    const rng = mulberry32(3);
    const occ = new Set(); const shown = Math.min(m, 600);
    const P = Array.from({ length: shown }, () => [rng(), rng()]);
    P.forEach(([x, y]) => occ.add(Math.floor(x * bins) + ',' + Math.floor(y * bins)));
    const cs = W / bins;
    for (let i = 0; i < bins; i++) for (let j = 0; j < bins; j++) {
      ctx.fillStyle = occ.has(i + ',' + j) ? alpha(c.s1, .25) : c.surface2; ctx.fillRect(i * cs + 1, j * cs + 1, cs - 2, cs - 2);
    }
    P.forEach(([x, y]) => dot(ctx, x * W, y * H, 1.8, c.s1));
    bars.innerHTML = '';
    const frac = d => { const N = bins ** d; return 1 - Math.exp(m * Math.log1p(-1 / N)); };
    for (let d = 1; d <= 10; d++) {
      const f = frac(d);
      bars.append(h('div', { class: 'hbar', style: 'grid-template-columns:60px 1fr' }, h('div', { class: 'lab' }, 'd = ' + d),
        h('div', { class: 'track', style: 'height:16px' }, h('div', { class: 'fill', style: `width:${Math.max(0.5, f * 100)}%;--c:${c.s1}` }), h('span', { class: 'val' }, (f * 100 < 0.01 ? f.toExponential(1) : (f * 100).toFixed(f * 100 < 1 ? 3 : 1) + '%')))));
    }
    const d = 10, N = bins ** d;
    calc.innerHTML = `<span class="c">cells = bins^d.  With ${bins} bins:  d=1 → ${bins},  d=2 → ${bins ** 2},  d=3 → ${fmt(bins ** 3, 0)},  d=10 → ${fmt(N, 0)}</span>
P(a given cell is empty) = (1 − 1/cells)^m        → for d=10: (1 − 1/${fmt(N, 0)})^${fmt(m, 0)} ≈ <b>${Math.exp(m * Math.log1p(-1 / N)).toFixed(6)}</b>
<span class="c">so a new point in 10-D almost surely lands in a cell where we have never seen any example.</span>`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 11) Local constancy (nearest neighbour) on a checkerboard ---------- */
Widgets.knnSmooth = (el) => {
  UI.head(el, 'The smoothness prior needs an example in every region', 'Target: a checkerboard. A 1-nearest-neighbour rule copies the label of the closest training point — it assumes the function is locally constant.');
  let tiles = 4, lm = 1.7, seed = 1;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Tiles per side (regions = tiles²)', min: 2, max: 12, step: 1, value: tiles, fmt: v => `${v} → ${v * v} regions`, onInput: v => { tiles = v; draw(); } });
  UI.slider(ctr, { label: 'Training examples m', min: 1, max: 3, step: 0.05, value: lm, fmt: v => String(Math.round(10 ** v)), onInput: v => { lm = v; draw(); } });
  const btns = h('div', { class: 'btns' }); UI.button(btns, 'Resample', () => { seed++; draw(); }); ctr.append(btns);
  const wrap = h('div', { class: 'canvas-wrap', style: 'max-width:340px' });
  const cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': 'Checkerboard prediction map' });
  wrap.append(cv);
  const stats = h('div', { class: 'stat-row' });
  el.append(ctr, wrap, stats);
  function draw() {
    const c = Colors.get();
    const m = Math.round(10 ** lm), rng = mulberry32(seed * 13 + m);
    const lab = (x, y) => (Math.floor(x * tiles) + Math.floor(y * tiles)) % 2;
    const P = Array.from({ length: m }, () => { const x = rng(), y = rng(); return [x, y, lab(x, y)]; });
    const nn = (x, y) => { let best = 1e9, l = 0; for (const p of P) { const d = (p[0] - x) ** 2 + (p[1] - y) ** 2; if (d < best) { best = d; l = p[2]; } } return l; };
    const W = Math.min(340, boxWidth(wrap)), G = 70, cs = W / G;
    const ctx = fitCanvas(cv, W, W);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const x = (i + .5) / G, y = (j + .5) / G; ctx.fillStyle = alpha(nn(x, y) ? c.s2 : c.s1, .35); ctx.fillRect(i * cs, j * cs, cs + .5, cs + .5); }
    ctx.strokeStyle = c.text; ctx.lineWidth = 1; ctx.globalAlpha = .35;
    for (let k = 1; k < tiles; k++) { ctx.beginPath(); ctx.moveTo(k * W / tiles, 0); ctx.lineTo(k * W / tiles, W); ctx.moveTo(0, k * W / tiles); ctx.lineTo(W, k * W / tiles); ctx.stroke(); }
    ctx.globalAlpha = 1;
    P.forEach(([x, y, l]) => dot(ctx, x * W, y * W, 3, l ? c.s2 : c.s1, c.surface));
    let ok = 0; const T = 1500, r2 = mulberry32(99);
    for (let t = 0; t < T; t++) { const x = r2(), y = r2(); if (nn(x, y) === lab(x, y)) ok++; }
    const occ = new Set(P.map(([x, y]) => Math.floor(x * tiles) + ',' + Math.floor(y * tiles))).size;
    stats.innerHTML = '';
    stats.append(
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Regions'), h('div', { class: 'v' }, String(tiles * tiles))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Regions with ≥1 example'), h('div', { class: 'v' }, String(occ))),
      h('div', { class: 'stat' }, h('div', { class: 'k' }, 'Test accuracy'), h('div', { class: 'v' }, (ok / T * 100).toFixed(1) + '%')));
  }
  draw();
  return { redraw: draw };
};

/* ---------- 12) Logistic regression by gradient descent ---------- */
Widgets.logreg = (el) => {
  UI.head(el, 'Logistic regression: p(y = 1 | x) = σ(w·x + b)', 'No closed-form solution — we minimize the negative log-likelihood (NLL) by gradient descent. Press Step and watch the S-curve fit the two classes.');
  const rng = mulberry32(17);
  const data = [...Array.from({ length: 12 }, () => [-1 + 1.1 * gaussian(rng), 0]), ...Array.from({ length: 12 }, () => [1.5 + 1.1 * gaussian(rng), 1])];
  let w = 0, b = 0, steps = 0, timer = null;
  const btns = h('div', { class: 'btns' });
  UI.button(btns, 'Step', () => { step(); draw(); }, 'btn primary');
  UI.button(btns, 'Run 200 steps', () => { if (timer) return; let n = 0; timer = setInterval(() => { for (let i = 0; i < 4; i++) step(); n += 4; draw(); if (n >= 200) { clearInterval(timer); timer = null; } }, 30); });
  UI.button(btns, 'Reset', () => { w = 0; b = 0; steps = 0; draw(); });
  el.append(btns);
  const chart = new LineChart(el, { xr: [-4, 5], yr: [-0.1, 1.1], xlabel: 'x', ylabel: 'p(y=1|x)', xname: 'x', aspect: .45, maxH: 250, vfmt: v => v.toFixed(3) });
  const lg = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  const nll = () => mean(data.map(([x, y]) => { const p = sigmoid(w * x + b); return -(y * Math.log(p + 1e-12) + (1 - y) * Math.log(1 - p + 1e-12)); }));
  function step() {
    let gw = 0, gb = 0;
    data.forEach(([x, y]) => { const e = sigmoid(w * x + b) - y; gw += e * x; gb += e; });
    w -= 0.5 * gw / data.length; b -= 0.5 * gb / data.length; steps++;
  }
  function draw() {
    const c = Colors.get();
    const xs = Array.from({ length: 181 }, (_, i) => -4 + i * 0.05);
    chart.set([
      { name: 'σ(w·x + b)', color: 's3', data: xs.map(x => [x, sigmoid(w * x + b)]), width: 2.5 },
      { name: 'class 0', color: 's1', style: 'dots', r: 4, noHover: true, data: data.filter(d => !d[1]).map(([x], i) => [x, -0.02 + (i % 3) * 0.02]) },
      { name: 'class 1', color: 's2', style: 'dots', r: 4, noHover: true, data: data.filter(d => d[1]).map(([x], i) => [x, 0.98 + (i % 3) * 0.02]) }
    ]);
    lg.set([{ label: 'model probability of class 1', color: c.s3 }, { label: 'class 0 examples (y = 0)', color: c.s1, kind: 'dot' }, { label: 'class 1 examples (y = 1)', color: c.s2, kind: 'dot' }]);
    const [x0, y0] = data[12], z = w * x0 + b, p = sigmoid(z);
    let gw = 0; data.forEach(([x, y]) => { gw += (sigmoid(w * x + b) - y) * x; }); gw /= data.length;
    calc.innerHTML = `<span class="c">step ${steps}:  w = ${w.toFixed(3)}, b = ${b.toFixed(3)},  NLL = (1/m)·Σ −[y log p + (1−y) log(1−p)] = <b>${nll().toFixed(4)}</b></span>
one example: x = ${x0.toFixed(2)}, y = ${y0}:  z = w·x + b = ${z.toFixed(3)},  p = σ(z) = 1/(1+e^(−z)) = ${p.toFixed(3)},  loss = −log p = ${(-Math.log(p + 1e-12)).toFixed(3)}
gradient: ∂NLL/∂w = (1/m)·Σ(σ(zᵢ) − yᵢ)·xᵢ = ${gw.toFixed(4)}   update: w ← w − 0.5 × ${gw.toFixed(4)}`;
  }
  draw();
  return { redraw: draw };
};

/* ---------- 13) Generative vs. discriminative ---------- */
Widgets.genDisc = (el) => {
  UI.head(el, 'Generative vs. discriminative on the same data', 'Generative: model p(x | Cₖ) and p(Cₖ), then Bayes. Discriminative: model p(C₂ | x) directly (logistic regression). Move the test point.');
  const rng = mulberry32(23);
  const A = Array.from({ length: 40 }, () => -1 + 0.8 * gaussian(rng)), B = Array.from({ length: 40 }, () => 1.2 + 1.0 * gaussian(rng));
  const st = v => { const mu = mean(v); return [mu, Math.sqrt(mean(v.map(x => (x - mu) ** 2)))]; };
  const [ma, sa] = st(A), [mb, sb] = st(B);
  let w = 0, b0 = 0;
  const all = [...A.map(x => [x, 0]), ...B.map(x => [x, 1])];
  for (let it = 0; it < 3000; it++) { let gw = 0, gb = 0; all.forEach(([x, y]) => { const e = sigmoid(w * x + b0) - y; gw += e * x; gb += e; }); w -= 0.3 * gw / all.length; b0 -= 0.3 * gb / all.length; }
  let xt = 0.2;
  const ctr = h('div', { class: 'controls' });
  UI.slider(ctr, { label: 'Test point x', min: -5, max: 6, step: 0.05, value: xt, fmt: v => v.toFixed(2), onInput: v => { xt = v; draw(); } });
  el.append(ctr);
  el.append(h('div', { class: 'note' }, 'Generative view: p(x|Cₖ)·p(Cₖ) for each class, and their sum p(x):'));
  const ch1 = new LineChart(el, { xr: [-5, 6], yr: [0, 0.32], xlabel: 'x', xname: 'x', aspect: .36, maxH: 200, minH: 150, vfmt: v => v.toFixed(3) });
  const lg1 = UI.legend(el, []);
  el.append(h('div', { class: 'note' }, 'Posterior p(C₂ | x) from both approaches:'));
  const ch2 = new LineChart(el, { xr: [-5, 6], yr: [0, 1], xlabel: 'x', xname: 'x', aspect: .32, maxH: 180, minH: 140, vfmt: v => v.toFixed(3) });
  const lg2 = UI.legend(el, []);
  const calc = h('div', { class: 'calcbox' });
  el.append(calc);
  function draw() {
    const c = Colors.get();
    const xs = Array.from({ length: 221 }, (_, i) => -5 + i * 0.05);
    const ga = x => 0.5 * normPdf(x, ma, sa), gb = x => 0.5 * normPdf(x, mb, sb);
    ch1.set([
      { name: 'p(x|C₁)p(C₁)', color: 's1', data: xs.map(x => [x, ga(x)]) },
      { name: 'p(x|C₂)p(C₂)', color: 's2', data: xs.map(x => [x, gb(x)]) },
      { name: 'p(x)', color: 'muted', data: xs.map(x => [x, ga(x) + gb(x)]), dash: [5, 4] },
      { name: 'test point', color: 'text', style: 'dots', r: 5, noHover: true, data: [[xt, ga(xt) + gb(xt)]] }
    ]);
    lg1.set([{ label: 'class 1 (generative)', color: c.s1 }, { label: 'class 2 (generative)', color: c.s2 }, { label: 'p(x) = sum (only generative models give this)', color: c.muted, kind: 'dash' }]);
    ch2.set([
      { name: 'generative', color: 's2', data: xs.map(x => [x, gb(x) / (ga(x) + gb(x))]) },
      { name: 'discriminative', color: 's3', data: xs.map(x => [x, sigmoid(w * x + b0)]), dash: [6, 4] },
      { name: 'test point', color: 'text', style: 'dots', r: 5, noHover: true, data: [[xt, gb(xt) / (ga(xt) + gb(xt))]] }
    ]);
    lg2.set([{ label: 'generative: Bayes’ rule', color: c.s2 }, { label: 'discriminative: σ(w·x + b)', color: c.s3, kind: 'dash' }]);
    const px = ga(xt) + gb(xt), post = gb(xt) / px;
    calc.innerHTML = `<span class="c">fitted: class 1 ~ N(${ma.toFixed(2)}, ${sa.toFixed(2)}²), class 2 ~ N(${mb.toFixed(2)}, ${sb.toFixed(2)}²), priors 0.5 / 0.5</span>
p(x|C₁)p(C₁) = ${ga(xt).toFixed(4)},  p(x|C₂)p(C₂) = ${gb(xt).toFixed(4)}
p(x) = sum = ${px.toFixed(4)}   p(C₂|x) = ${gb(xt).toFixed(4)} / ${px.toFixed(4)} = <b>${post.toFixed(3)}</b>
discriminative: σ(${w.toFixed(2)}·${xt.toFixed(2)} + ${b0.toFixed(2)}) = <b>${sigmoid(w * xt + b0).toFixed(3)}</b>
${px < 0.005 ? '<b>⚠ p(x) is tiny → outlier / novelty: the generative model can warn us; the discriminative one confidently answers anyway.</b>' : '<span class="c">move x far to the left or right to see outlier detection</span>'}`;
  }
  draw();
  return { redraw: draw };
};

/* =========================================================
   Quiz
   ========================================================= */
window.QUIZ = [
  { q: 'Why is a point estimate θ̂ considered a random variable in the frequentist view?', opts: ['Because θ is random', 'Because it is a function of the randomly drawn dataset', 'Because of floating-point errors', 'It is not random'], a: 1, why: 'The true θ is fixed but unknown; the data are random, so any function of the data (θ̂ = g(x⁽¹⁾,…,x⁽ᵐ⁾)) is random too.' },
  { q: 'The sample variance σ̂² = (1/m)Σ(x − x̄)² has bias…', opts: ['0', '−σ²/m', '+σ²/m', 'σ²'], a: 1, why: 'E[σ̂²] = (m−1)/m·σ², so the bias is −σ²/m. Dividing by m − 1 instead gives the unbiased estimator.' },
  { q: 'The standard error of the sample mean is…', opts: ['σ/m', 'σ/√m', 'σ²', '√m·σ'], a: 1, why: 'SE(μ̂ₘ) = √Var(μ̂ₘ) = σ/√m. Quadrupling the test set halves the error bars; the 95% CI is μ̂ ± 1.96·SE.' },
  { q: 'MSE of an estimator decomposes as…', opts: ['bias + variance', 'bias² + variance', 'bias × variance', 'variance − bias'], a: 1, why: 'MSE = E[(θ̂ − θ)²] = Bias(θ̂)² + Var(θ̂). Increasing capacity usually lowers bias and raises variance.' },
  { q: 'Why do we maximize the log-likelihood instead of the likelihood?', opts: ['It gives a different answer', 'Same argmax, but sums are numerically stable and easier to differentiate than products', 'Logs are faster to type', 'Only logs can be maximized'], a: 1, why: 'log is monotonic, so the maximizer is unchanged; and a product of many probabilities underflows to 0 while the sum of logs does not.' },
  { q: 'Minimizing the MSE of linear regression is equivalent to maximum likelihood when…', opts: ['p(y|x) is Bernoulli', 'p(y|x) is Gaussian with mean ŷ(x; w) and fixed variance', 'there is no noise', 'w is Gaussian'], a: 1, why: 'log N(y; ŷ, σ²) = const − (y − ŷ)²/(2σ²), so maximizing the log-likelihood = minimizing the squared error.' },
  { q: 'Maximum likelihood is equivalent to minimizing which quantity between p̂_data and p_model?', opts: ['Euclidean distance', 'KL divergence (i.e. cross-entropy)', 'Bias', 'Accuracy'], a: 1, why: 'D_KL(p̂_data‖p_model) = E[log p̂_data] − E[log p_model]; only the second term depends on the model, so minimizing KL = minimizing cross-entropy = maximizing likelihood.' },
  { q: 'MAP estimation with a Gaussian prior N(w; 0, (1/λ)I) on the weights corresponds to…', opts: ['dropout', 'weight decay (L2 penalty)', 'early stopping', 'no regularization'], a: 1, why: 'log p(w) = −(λ/2)wᵀw + const, which adds a weight-decay penalty to the log-likelihood: less variance, more bias.' },
  { q: 'The curse of dimensionality says that…', opts: ['high-dimensional data is always linear', 'the number of possible configurations grows exponentially with dimension, so most regions have no training data', 'more features always help', 'computation is free in high dimensions'], a: 1, why: 'With 10 bins per axis there are 10ᵈ cells; for d = 10 that is 10¹⁰ cells, far more than any training set can fill.' },
  { q: 'Logistic regression for binary classification…', opts: ['has a closed-form solution like linear regression', 'squashes w·x + b with a sigmoid and is trained by minimizing the negative log-likelihood with gradient descent', 'predicts real-valued targets', 'needs no training'], a: 1, why: 'p(y=1|x) = σ(θᵀx). There is no closed form for the optimal weights, so we use gradient descent on the NLL.' },
  { q: 'An advantage of generative models over discriminative ones is that they…', opts: ['always need less data', 'also give p(x), which can detect outliers / novel inputs', 'never use Bayes’ rule', 'are cheaper to train'], a: 1, why: 'Modelling p(x, Cₖ) yields p(x) = Σₖ p(x|Cₖ)p(Cₖ). But if you only need decisions, modelling p(Cₖ|x) directly is less demanding.' }
];
