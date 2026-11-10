/* =========================================================
   core.js — shared helpers, slide navigation, theme, charts
   ========================================================= */
'use strict';

/* ---------- tiny DOM helpers ---------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return e;
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;

/* deterministic RNG so demos look the same on every visit */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/* ---------- number formatting ---------- */
function fmt(v, d = 2) {
  if (!isFinite(v)) return v > 0 ? '∞' : v < 0 ? '−∞' : '—';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-3)) {
    const e = Math.floor(Math.log10(a));
    const m = v / 10 ** e;
    return `${m.toFixed(1)}×10${sup(e)}`.replace('-', '−');
  }
  return v.toFixed(d).replace('-', '−');
}
function sup(n) {
  const map = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  return String(n).split('').map(c => map[c] || c).join('');
}

/* ---------- theme ---------- */
const Theme = {
  current() {
    const t = document.documentElement.dataset.theme;
    if (t) return t;
    return window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  },
  set(t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem('dlx-theme', t); } catch (e) { /* storage unavailable */ }
    Colors.invalidate();
    window.dispatchEvent(new Event('themechange'));
  },
  init() {
    try {
      const t = localStorage.getItem('dlx-theme');
      if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
    } catch (e) { /* ignore */ }
    if (window.matchMedia) {
      matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', () => {
        Colors.invalidate();
        window.dispatchEvent(new Event('themechange'));
      });
    }
  }
};

/* ---------- colors read from CSS tokens ---------- */
const Colors = {
  _c: null,
  invalidate() { this._c = null; },
  get() {
    if (this._c) return this._c;
    const cs = getComputedStyle(document.documentElement);
    const v = n => cs.getPropertyValue(n).trim();
    this._c = {
      bg: v('--bg'), surface: v('--surface'), surface2: v('--surface-2'), border: v('--border'),
      text: v('--text'), text2: v('--text-2'), muted: v('--muted'), grid: v('--grid'), axis: v('--axis'),
      accent: v('--accent'),
      s1: v('--s1'), s2: v('--s2'), s3: v('--s3'), s4: v('--s4'), s5: v('--s5'), s6: v('--s6'), s7: v('--s7'), s8: v('--s8'),
      ord1: v('--ord-1'), ord2: v('--ord-2'), ord3: v('--ord-3'),
      lo: v('--contour-lo'), hi: v('--contour-hi'), line: v('--contour-line'),
      good: v('--good'), warn: v('--warn'), serious: v('--serious'), critical: v('--critical'),
      dark: Theme.current() === 'dark'
    };
    return this._c;
  }
};
function hexToRgb(hex) {
  const m = hex.replace('#', '');
  const n = parseInt(m.length === 3 ? m.split('').map(c => c + c).join('') : m, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function alpha(hex, a) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
}

/* ---------- canvas helpers ---------- */
function fitCanvas(canvas, cssW, cssH) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.height = cssH + 'px';
  const W = Math.round(cssW * dpr), H = Math.round(cssH * dpr);
  if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  return ctx;
}
function boxWidth(el) {
  return Math.max(260, Math.floor(el.getBoundingClientRect().width || el.clientWidth || 320));
}
function makePlot(W, H, xr, yr, pad) {
  pad = Object.assign({ l: 44, r: 14, t: 14, b: 30 }, pad || {});
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  return {
    W, H, xr, yr, pad, pw, ph,
    X: x => pad.l + (x - xr[0]) / (xr[1] - xr[0]) * pw,
    Y: y => pad.t + ph - (y - yr[0]) / (yr[1] - yr[0]) * ph,
    invX: px => xr[0] + (px - pad.l) / pw * (xr[1] - xr[0]),
    invY: py => yr[0] + (pad.t + ph - py) / ph * (yr[1] - yr[0]),
    clip(ctx) { ctx.beginPath(); ctx.rect(pad.l, pad.t, pw, ph); ctx.clip(); }
  };
}
function niceTicks(a, b, n = 5) {
  const span = b - a;
  if (!(span > 0)) return [a];
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let t = Math.ceil(a / step - 1e-9) * step; t <= b + step * 1e-9; t += step) out.push(+t.toPrecision(12));
  return out;
}
function tickLabel(t) {
  if (Math.abs(t) < 1e-12) return '0';
  const a = Math.abs(t);
  if (a >= 1e4 || a < 1e-2) return fmt(t, 0);
  return String(+t.toPrecision(4)).replace('-', '−');
}
function drawAxes(ctx, p, o = {}) {
  const c = Colors.get();
  ctx.save();
  ctx.font = '11px Inter, system-ui, sans-serif';
  ctx.lineWidth = 1;
  const xt = o.xticks || niceTicks(p.xr[0], p.xr[1], o.nx || 6);
  const yt = o.yticks || niceTicks(p.yr[0], p.yr[1], o.ny || 5);
  // grid
  if (o.grid !== false) {
    ctx.strokeStyle = c.grid;
    for (const t of yt) { const y = Math.round(p.Y(t)) + .5; ctx.beginPath(); ctx.moveTo(p.pad.l, y); ctx.lineTo(p.W - p.pad.r, y); ctx.stroke(); }
  }
  // axes
  ctx.strokeStyle = c.axis;
  ctx.beginPath();
  ctx.moveTo(p.pad.l + .5, p.pad.t); ctx.lineTo(p.pad.l + .5, p.H - p.pad.b + .5); ctx.lineTo(p.W - p.pad.r, p.H - p.pad.b + .5);
  ctx.stroke();
  ctx.fillStyle = c.muted;
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const t of xt) {
    const x = p.X(t);
    if (x < p.pad.l - 1 || x > p.W - p.pad.r + 1) continue;
    ctx.fillText(o.xfmt ? o.xfmt(t) : tickLabel(t), x, p.H - p.pad.b + 5);
  }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const t of yt) {
    const y = p.Y(t);
    if (y < p.pad.t - 1 || y > p.H - p.pad.b + 1) continue;
    ctx.fillText(o.yfmt ? o.yfmt(t) : tickLabel(t), p.pad.l - 6, y);
  }
  if (o.xlabel) { ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(o.xlabel, p.W - p.pad.r, p.H - p.pad.b - 4); }
  if (o.ylabel) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(o.ylabel, p.pad.l + 6, p.pad.t + 2); }
  ctx.restore();
}
function dot(ctx, x, y, r, fill, ring) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill; ctx.fill();
  if (ring) { ctx.lineWidth = 2; ctx.strokeStyle = ring; ctx.stroke(); }
}
function label(ctx, text, x, y, o = {}) {
  const c = Colors.get();
  ctx.save();
  ctx.font = (o.bold ? '600 ' : '') + (o.size || 11) + 'px Inter, system-ui, sans-serif';
  ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'middle';
  ctx.lineWidth = 3; ctx.strokeStyle = c.surface; ctx.lineJoin = 'round';
  ctx.strokeText(text, x, y);
  ctx.fillStyle = o.color || c.text2;
  ctx.fillText(text, x, y);
  ctx.restore();
}
function arrow(ctx, x1, y1, x2, y2, color, w = 2, head = 7) {
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.save();
  ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2 - Math.cos(a) * head * .6, y2 - Math.sin(a) * head * .6); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(a - .42), y2 - head * Math.sin(a - .42));
  ctx.lineTo(x2 - head * Math.cos(a + .42), y2 - head * Math.sin(a + .42));
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

/* Loss-landscape image: shaded bands + contour lines. Cached per function/theme/size. */
function contourImage(f, dom, w, h, levels = 16) {
  const c = Colors.get();
  const lo = hexToRgb(c.lo), hi = hexToRgb(c.hi), ln = hexToRgb(c.line);
  const vals = new Float64Array(w * h);
  let mn = Infinity, mx = -Infinity;
  for (let j = 0; j < h; j++) {
    const y = dom[3] - (j + .5) / h * (dom[3] - dom[2]);
    for (let i = 0; i < w; i++) {
      const x = dom[0] + (i + .5) / w * (dom[1] - dom[0]);
      const v = f(x, y);
      vals[j * w + i] = v;
      if (v < mn) mn = v; if (v > mx) mx = v;
    }
  }
  const s = (mx - mn) * 0.012 + 1e-12;
  const den = Math.log1p((mx - mn) / s);
  const tval = new Float64Array(w * h), band = new Int16Array(w * h);
  for (let k = 0; k < vals.length; k++) {
    const t = Math.log1p((vals[k] - mn) / s) / den;
    tval[k] = t; band[k] = Math.floor(t * levels);
  }
  const cv = document.createElement('canvas'); cv.width = w; cv.height = h;
  const cx = cv.getContext('2d');
  const img = cx.createImageData(w, h);
  for (let j = 0; j < h; j++) {
    for (let i = 0; i < w; i++) {
      const k = j * w + i;
      const edge = (i < w - 1 && band[k] !== band[k + 1]) || (j < h - 1 && band[k] !== band[k + w]);
      let r, g, b;
      if (edge) { [r, g, b] = ln; }
      else {
        const t = Math.pow(tval[k], .8);
        r = lerp(lo[0], hi[0], t); g = lerp(lo[1], hi[1], t); b = lerp(lo[2], hi[2], t);
      }
      img.data[k * 4] = r; img.data[k * 4 + 1] = g; img.data[k * 4 + 2] = b; img.data[k * 4 + 3] = 255;
    }
  }
  cx.putImageData(img, 0, 0);
  return cv;
}

/* ---------- form controls ---------- */
const UI = {
  slider(parent, o) {
    const f = o.fmt || (v => v);
    const out = h('output', {}, f(o.value));
    const input = h('input', { type: 'range', min: o.min, max: o.max, step: o.step, value: o.value, 'aria-label': o.label });
    const wrap = h('label', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', {}, o.label), out), input);
    input.addEventListener('input', () => { const v = +input.value; out.textContent = f(v); o.onInput && o.onInput(v); });
    parent.append(wrap);
    return {
      el: wrap, input,
      get: () => +input.value,
      set(v, fire) { input.value = v; out.textContent = f(+input.value); if (fire && o.onInput) o.onInput(+input.value); },
      show(b) { wrap.hidden = !b; }
    };
  },
  select(parent, labelText, options, value, onChange) {
    const sel = h('select', { 'aria-label': labelText }, options.map(([v, l]) => h('option', { value: v, selected: v === value }, l)));
    sel.addEventListener('change', () => onChange(sel.value));
    const wrap = h('label', { class: 'ctl' }, h('span', { class: 'ctl-label' }, h('span', {}, labelText)), sel);
    parent.append(wrap);
    return { el: wrap, sel, get: () => sel.value, set(v) { sel.value = v; }, show(b) { wrap.hidden = !b; } };
  },
  button(parent, text, onClick, cls = 'btn') {
    const b = h('button', { class: cls, type: 'button' }, text);
    b.addEventListener('click', onClick);
    parent.append(b);
    return b;
  },
  chip(parent, text, color, on, onChange) {
    const input = h('input', { type: 'checkbox' });
    input.checked = !!on;
    const lab = h('label', { class: 'tchip' + (on ? ' on' : ''), style: `--c:var(${color})` }, input, h('span', { class: 'sw' }), text);
    input.addEventListener('change', () => { lab.classList.toggle('on', input.checked); onChange(input.checked); });
    parent.append(lab);
    return { el: lab, input, get: () => input.checked, set(v) { input.checked = v; lab.classList.toggle('on', v); } };
  },
  seg(parent, options, value, onChange) {
    const wrap = h('div', { class: 'seg', role: 'group' });
    const btns = options.map(([v, l]) => {
      const b = h('button', { type: 'button', class: v === value ? 'on' : '', 'aria-pressed': v === value ? 'true' : 'false' }, l);
      b.addEventListener('click', () => { set(v); onChange(v); });
      wrap.append(b);
      return [v, b];
    });
    function set(v) { btns.forEach(([k, b]) => { b.classList.toggle('on', k === v); b.setAttribute('aria-pressed', k === v ? 'true' : 'false'); }); }
    parent.append(wrap);
    return { el: wrap, set };
  },
  legend(parent, items) {
    const lg = h('div', { class: 'legend' });
    const set = its => {
      lg.innerHTML = '';
      its.forEach(it => lg.append(h('span', {}, h('i', { class: it.kind || '', style: `--c:${it.color}` }), it.label)));
    };
    set(items);
    parent.append(lg);
    return { el: lg, set };
  },
  head(parent, title, sub) {
    parent.append(h('div', { class: 'w-head' }, h('div', { class: 'w-title' }, title), sub ? h('div', { class: 'w-sub', html: sub }) : null));
  }
};

/* ---------- a reusable line/dot chart with hover crosshair ---------- */
class LineChart {
  constructor(parent, opts) {
    this.o = Object.assign({ aspect: .55, maxH: 320, minH: 190, logY: false }, opts);
    this.wrap = h('div', { class: 'canvas-wrap' });
    this.cv = h('canvas', { class: 'chart', role: 'img', 'aria-label': this.o.aria || 'chart' });
    this.tip = h('div', { class: 'tip', hidden: true });
    this.wrap.append(this.cv, this.tip);
    parent.append(this.wrap);
    this.series = [];
    this.hx = null;
    this.cv.addEventListener('mousemove', e => this.onMove(e));
    this.cv.addEventListener('mouseleave', () => { this.hx = null; this.tip.hidden = true; this.draw(); });
  }
  set(series, o) { this.series = series; if (o) Object.assign(this.o, o); this.draw(); }
  ty(y) { return this.o.logY ? Math.log10(Math.max(y, 1e-300)) : y; }
  draw() {
    const c = Colors.get();
    const W = boxWidth(this.wrap);
    const H = clamp(Math.round(W * this.o.aspect), this.o.minH, this.o.maxH);
    const ctx = fitCanvas(this.cv, W, H);
    const xr = this.o.xr, yr = this.o.logY ? [Math.log10(this.o.yr[0]), Math.log10(this.o.yr[1])] : this.o.yr;
    const p = makePlot(W, H, xr, yr, this.o.pad);
    this.p = p;
    const yticks = this.o.logY ? niceTicks(yr[0], yr[1], 5).filter(t => Number.isInteger(t)) : undefined;
    drawAxes(ctx, p, {
      xlabel: this.o.xlabel, ylabel: this.o.ylabel, yticks,
      yfmt: this.o.logY ? (t => '10' + sup(t)) : this.o.yfmt, xfmt: this.o.xfmt
    });
    ctx.save(); p.clip(ctx);
    if (this.o.refY != null) {
      ctx.strokeStyle = c.muted; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
      const y = p.Y(this.ty(this.o.refY));
      ctx.beginPath(); ctx.moveTo(p.pad.l, y); ctx.lineTo(W - p.pad.r, y); ctx.stroke(); ctx.setLineDash([]);
    }
    for (const s of this.series) {
      const col = s.color.startsWith('#') ? s.color : c[s.color];
      if (s.style === 'dots') {
        for (const [x, y] of s.data) dot(ctx, p.X(x), p.Y(this.ty(y)), s.r || 3.5, col, c.surface);
        continue;
      }
      ctx.strokeStyle = col; ctx.lineWidth = s.width || 2; ctx.lineJoin = 'round';
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let started = false;
      for (const [x, y] of s.data) {
        if (!isFinite(y)) { started = false; continue; }
        const X = p.X(x), Y = p.Y(this.ty(y));
        if (!started) { ctx.moveTo(X, Y); started = true; } else ctx.lineTo(X, Y);
      }
      ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.restore();
    // hover layer
    if (this.hx != null && this.series.length) {
      const X = p.X(this.hx);
      ctx.strokeStyle = c.axis; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(X + .5, p.pad.t); ctx.lineTo(X + .5, H - p.pad.b); ctx.stroke();
      const rows = [];
      for (const s of this.series) {
        if (s.noHover || !s.data.length) continue;
        let best = s.data[0];
        for (const d of s.data) if (Math.abs(d[0] - this.hx) < Math.abs(best[0] - this.hx)) best = d;
        const col = s.color.startsWith('#') ? s.color : c[s.color];
        if (isFinite(best[1])) dot(ctx, p.X(best[0]), p.Y(this.ty(best[1])), 4.5, col, c.surface);
        rows.push(`<div><span class="ts" style="background:${col}"></span>${s.name}: <b>${(this.o.vfmt || fmt)(best[1])}</b></div>`);
      }
      this.tip.innerHTML = `<div class="tk">${this.o.xname || 'x'} = ${(this.o.hxfmt || (v => fmt(v, 2)))(this.hx)}</div>` + rows.join('');
      this.tip.hidden = false;
      const tw = this.tip.offsetWidth;
      let left = X + 12; if (left + tw > W) left = X - tw - 12;
      this.tip.style.left = Math.max(0, left) + 'px';
      this.tip.style.top = (p.pad.t + 4) + 'px';
    }
  }
  onMove(e) {
    if (!this.p) return;
    const r = this.cv.getBoundingClientRect();
    const px = e.clientX - r.left;
    if (px < this.p.pad.l || px > this.p.W - this.p.pad.r) { this.hx = null; this.tip.hidden = true; this.draw(); return; }
    let x = this.p.invX(px);
    if (this.o.snapX) x = this.o.snapX(x);
    this.hx = x; this.draw();
  }
}

/* ---------- KaTeX ---------- */
function renderMath(root) {
  if (window.renderMathInElement) {
    try {
      renderMathInElement(root, {
        delimiters: [{ left: '\\[', right: '\\]', display: true }, { left: '\\(', right: '\\)', display: false }],
        throwOnError: false
      });
    } catch (e) { console.warn('KaTeX render failed', e); }
  }
}

/* ---------- training-loop stages (used by chips + overview) ----------
   Each page defines window.TOPICS = { stageId: [[link, label], ...] }.
   A link is either an in-page slide id ("s-dropout") or a URL ("ch07.html#/s-dropout"). */
const STAGES = [
  { id: 'data', short: 'Data & task', name: 'Define the task and collect data',
    what: 'Decide what the model should do (classify, regress, translate…), how success is measured, and which examples it will learn from. Split the data into training, validation and test sets.' },
  { id: 'init', short: 'Init θ', name: 'Build the model & initialize the weights',
    what: 'Choose the architecture (layers, units, activations) and give every weight a starting value. The start decides which valley the training can reach.' },
  { id: 'batch', short: 'Minibatch', name: 'Sample a minibatch',
    what: 'Pick a small random set of training examples. Which examples you show, and how often, shapes what the network learns.' },
  { id: 'forward', short: 'Forward', name: 'Forward pass through the model',
    what: 'Push the minibatch through the layers to get predictions. The architecture decides what the network can represent and how easy it is to optimize.' },
  { id: 'loss', short: 'Loss J(θ)', name: 'Compute the loss J(θ)',
    what: 'Compare predictions with targets and average the errors, plus any regularization penalty. This number is the “height” of the landscape at the current weights.' },
  { id: 'backward', short: 'Backward ∇J', name: 'Backward pass: compute the gradient',
    what: 'Back-propagation applies the chain rule backwards through the network to get ∂J/∂θ for every weight.' },
  { id: 'update', short: 'Update θ', name: 'Update the weights',
    what: 'Use the gradient (and maybe curvature or past gradients) to choose a direction and step size, then move the weights. This is the optimizer.' },
  { id: 'final', short: 'Evaluate', name: 'Validate, stop, and test',
    what: 'Check performance on held-out data, decide when to stop, pick hyper-parameters, and finally measure the test error on data never used for training.' }
];
function goLink(link) {
  if (!link) return;
  if (/^s-/.test(link)) Deck.goId(link);
  else window.location.href = link;
}

/* ---------- widget registry ---------- */
const Widgets = {};
function initWidgetsIn(root) {
  $$('[data-widget]', root).forEach(el => {
    if (el._w) { el._w.redraw && el._w.redraw(); return; }
    const make = Widgets[el.dataset.widget];
    if (!make) { console.warn('Unknown widget', el.dataset.widget); return; }
    el.classList.add('widget');
    try { el._w = make(el, el.dataset) || {}; }
    catch (err) {
      console.error(err);
      el.append(h('p', { class: 'err' }, 'This demo failed to load: ' + err.message));
      el._w = {};
    }
  });
}
function redrawWidgetsIn(root) {
  $$('[data-widget]', root).forEach(el => { if (el._w && el._w.redraw) el._w.redraw(); });
}

/* ---------- course registry helpers (data lives in js/course.js) ---------- */
const Course = {
  list() { return (window.COURSE && COURSE.lectures) || []; },
  isHub() { return document.body.dataset.page === 'hub'; },
  id() { return document.body.dataset.lecture; },
  idx() { return this.list().findIndex(l => l.id === this.id()); },
  current() { return this.list()[this.idx()] || null; },
  prev() { const k = this.idx(); return k > 0 ? this.list()[k - 1] : null; },
  next() { const k = this.idx(); return k >= 0 && k < this.list().length - 1 ? this.list()[k + 1] : null; },
  href(l, slide) { return (this.isHub() ? 'lectures/' : '') + l.id + '.html' + (slide ? '#/' + slide : ''); },
  home() { return this.isHub() ? 'index.html' : '../index.html'; }
};

const ICONS = {
  menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M4 7h16M4 12h16M4 17h16"/></svg>',
  home: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l9-7 9 7"/><path d="M5 10v10h14V10"/></svg>',
  page: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>',
  theme: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 0 0 16z" fill="currentColor"/></svg>'
};

/* Builds the top bar, sidebar and bottom navigation so every lecture page stays small and consistent. */
function buildChrome() {
  const lec = Course.current();
  const hub = Course.isHub();
  const iconBtn = (id, icon, txt, title) => h('button', { id, class: 'icon-btn', type: 'button', title, html: icon + (txt ? `<span class="txt">${txt}</span>` : '') });
  const header = h('header', { class: 'topbar' },
    hub ? null : iconBtn('menuBtn', ICONS.menu, '', 'Open slide list'),
    h('div', { class: 'brand' },
      h('b', {}, hub ? 'Deep Learning — Interactive Course Notes' : (lec ? lec.title : document.title)),
      h('span', {}, hub ? 'Every lecture explained, illustrated and connected' : (lec ? `Lecture ${lec.n} · ${lec.chLabel || 'Chapter ' + lec.ch}` : ''))),
    hub ? null : h('a', { class: 'icon-btn', href: Course.home(), title: 'Course home', html: ICONS.home + '<span class="txt">Course</span>' }),
    hub ? null : iconBtn('modeBtn', ICONS.page, 'One page', 'Switch between slides and one long page'),
    iconBtn('themeBtn', ICONS.theme, 'Dark', 'Toggle light / dark theme'),
    h('div', { class: 'progress' }));
  document.body.prepend(header);
  if (hub) return;
  const main = $('main.deck');
  const layout = h('div', { class: 'layout' }, h('nav', { id: 'toc', class: 'sidebar', 'aria-label': 'Slides' }), h('div', { class: 'scrim' }));
  main.replaceWith(layout);
  layout.append(main);
  document.body.append(h('nav', { class: 'navbar', 'aria-label': 'Slide navigation' },
    h('button', { id: 'prevBtn', class: 'nav-btn', type: 'button', html: '← <span class="txt">Previous</span>' }),
    h('div', { id: 'where', class: 'where' }),
    h('button', { id: 'nextBtn', class: 'nav-btn primary', type: 'button', html: '<span class="txt">Next</span> →' })));
}

/* ---------- deck navigation ---------- */
const Deck = {
  slides: [], i: 0, page: false,
  init() {
    $('#themeBtn').addEventListener('click', () => Theme.set(Theme.current() === 'dark' ? 'light' : 'dark'));
    this.updateThemeIcon();
    window.addEventListener('themechange', () => this.updateThemeIcon());
    $$('[data-goto]').forEach(b => b.addEventListener('click', () => goLink(b.dataset.goto)));
    this.slides = $$('.slide');
    if (!this.slides.length) { // hub page: just draw widgets
      initWidgetsIn(document);
      let rt;
      window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => redrawWidgetsIn(document), 120); });
      window.addEventListener('themechange', () => redrawWidgetsIn(document));
      return;
    }
    const toc = $('#toc');
    const lec = Course.current();
    toc.append(h('a', { class: 'toc-home', href: Course.home() }, '← All lectures (course home)'));
    if (lec) toc.append(h('div', { class: 'toc-lec' }, h('span', { class: 'kicker' }, `Lecture ${lec.n} · ${lec.chLabel || 'Ch ' + lec.ch}`), h('b', {}, lec.title)));
    let part = null;
    this.slides.forEach((s, i) => {
      if (s.dataset.part !== part) {
        part = s.dataset.part;
        toc.append(h('div', { class: 'toc-part' }, part));
      }
      const b = h('button', { class: 'toc-item', type: 'button' }, h('span', { class: 'n' }, i + 1), h('span', {}, s.dataset.title));
      b.addEventListener('click', () => { this.go(i); document.body.classList.remove('toc-open'); });
      toc.append(b);
      s._toc = b;
      const holder = $('.stages', s);
      if (holder) {
        const on = (s.dataset.stages || '').split(',').filter(Boolean);
        holder.append(h('span', { class: 'lbl' }, 'Training step:'));
        STAGES.forEach(st => {
          const chip = h('button', { type: 'button', class: 'stage-chip' + (on.includes(st.id) ? ' on' : ''), title: st.name + ' — click to see the whole loop' }, st.short);
          chip.addEventListener('click', () => this.goStage(st.id));
          holder.append(chip);
        });
      }
    });
    if (Course.list().length) {
      toc.append(h('div', { class: 'toc-part' }, 'All lectures'));
      Course.list().forEach(l => toc.append(h('a', { class: 'toc-item' + (l.id === Course.id() ? ' active-lec' : ''), href: Course.href(l) },
        h('span', { class: 'n' }, 'L' + l.n), h('span', {}, `${l.title} `, h('span', { class: 'muted' }, `(${l.ch})`)))));
    }
    $('#prevBtn').addEventListener('click', () => {
      if (this.i === 0 && Course.prev()) location.href = Course.href(Course.prev());
      else this.go(this.i - 1);
    });
    $('#nextBtn').addEventListener('click', () => {
      if (this.i === this.slides.length - 1 && Course.next()) location.href = Course.href(Course.next());
      else this.go(this.i + 1);
    });
    $('#menuBtn').addEventListener('click', () => document.body.classList.toggle('toc-open'));
    $('.scrim').addEventListener('click', () => document.body.classList.remove('toc-open'));
    $('#modeBtn').addEventListener('click', () => this.setPage(!this.page));
    document.addEventListener('keydown', e => {
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
      if (e.altKey || e.ctrlKey || e.metaKey || this.page) return;
      if (e.key === 'ArrowRight' || e.key === 'PageDown') { this.go(this.i + 1); e.preventDefault(); }
      if (e.key === 'ArrowLeft' || e.key === 'PageUp') { this.go(this.i - 1); e.preventDefault(); }
    });
    window.addEventListener('hashchange', () => this.fromHash());
    let rt;
    window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => this.redrawVisible(), 120); });
    window.addEventListener('themechange', () => this.redrawVisible());
    try { if (localStorage.getItem('dlx-page') === '1') this.setPage(true, true); } catch (e) { /* ignore */ }
    this.fromHash(true);
  },
  updateThemeIcon() {
    const t = $('#themeBtn .txt');
    if (t) t.textContent = Theme.current() === 'dark' ? 'Light' : 'Dark';
  },
  fromHash(first) {
    const id = decodeURIComponent((location.hash || '').replace(/^#\/?/, ''));
    const idx = this.slides.findIndex(s => s.id === id);
    if (idx >= 0) this.go(idx, false);
    else if (first) this.go(0, false);
  },
  goId(id) { const i = this.slides.findIndex(s => s.id === id); if (i >= 0) this.go(i); },
  goStage(stageId) {
    this.goId('s-loop');
    const w = $('#s-loop [data-widget="loop"]');
    if (w && w._w && w._w.select) w._w.select(stageId);
  },
  go(i, push = true) {
    i = clamp(i, 0, this.slides.length - 1);
    this.i = i;
    const s = this.slides[i];
    if (this.page) s.scrollIntoView({ behavior: 'smooth', block: 'start' });
    else { this.slides.forEach((x, k) => x.classList.toggle('active', k === i)); window.scrollTo({ top: 0 }); }
    this.slides.forEach(x => x._toc.classList.toggle('active', x === s));
    s._toc.scrollIntoView({ block: 'nearest' });
    $('#where').innerHTML = `<b>${i + 1}</b> / ${this.slides.length} · ${s.dataset.title}`;
    const first = i === 0, last = i === this.slides.length - 1;
    $('#prevBtn').disabled = first && !Course.prev();
    $('#nextBtn').disabled = last && !Course.next();
    $('#prevBtn .txt').textContent = first && Course.prev() ? 'Previous lecture' : 'Previous';
    $('#nextBtn .txt').textContent = last && Course.next() ? 'Next lecture' : 'Next';
    $('.progress').style.width = ((i + 1) / this.slides.length * 100) + '%';
    if (push && ('#/' + s.id) !== location.hash) history.replaceState(null, '', '#/' + s.id);
    const lec = Course.current();
    document.title = s.dataset.title + ' · ' + (lec ? `L${lec.n} ${lec.title}` : 'Deep Learning');
    initWidgetsIn(s); // the slide is visible now, so widgets can measure their size
  },
  setPage(on, silent) {
    this.page = on;
    document.body.classList.toggle('page-mode', on);
    $('#modeBtn .txt').textContent = on ? 'Slides' : 'One page';
    try { localStorage.setItem('dlx-page', on ? '1' : '0'); } catch (e) { /* ignore */ }
    if (on) {
      this.slides.forEach(s => s.classList.add('active'));
      if (!this.io) {
        this.io = new IntersectionObserver(entries => {
          entries.forEach(en => {
            if (!en.isIntersecting) return;
            initWidgetsIn(en.target);
            if (this.page) {
              const k = this.slides.indexOf(en.target);
              if (k >= 0) {
                this.i = k;
                this.slides.forEach(x => x._toc.classList.toggle('active', x === en.target));
                $('.progress').style.width = ((k + 1) / this.slides.length * 100) + '%';
              }
            }
          });
        }, { rootMargin: '0px 0px -40% 0px' });
      }
      this.slides.forEach(s => this.io.observe(s));
      if (!silent) setTimeout(() => this.slides[this.i].scrollIntoView({ block: 'start' }), 30);
    } else {
      if (this.io) this.slides.forEach(s => this.io.unobserve(s));
      this.go(this.i, true);
    }
  },
  redrawVisible() {
    if (this.page) this.slides.forEach(s => redrawWidgetsIn(s));
    else redrawWidgetsIn(this.slides[this.i]);
  }
};

/* ---------- the training loop map (per lecture, or whole course on the hub) ---------- */
Widgets.loop = (el, ds) => {
  const topics = window.TOPICS || {};
  const course = ds.scope === 'course';
  UI.head(el, 'The training loop — click any step',
    course ? 'Every lecture improves one or more of these steps. Pick a step to see which lectures and ideas plug into it.'
      : 'Highlighted steps are the ones this lecture changes. Pick a step to see which ideas from this lecture live there.');
  const flow = h('div', { class: 'loop' });
  const detail = h('div', { class: 'loop-detail' });
  const cards = STAGES.map((st, k) => {
    const n = (topics[st.id] || []).length;
    const b = h('button', { type: 'button', class: 'loop-step' + (n ? ' has' : '') },
      h('span', { class: 'num' }, 'STEP ' + (k + 1)),
      h('span', { class: 'nm' }, st.name),
      h('span', { class: 'cnt' }, n ? `${n} topic${n > 1 ? 's' : ''} here` : 'not in this lecture'));
    b.addEventListener('click', () => select(st.id));
    flow.append(b);
    return b;
  });
  el.append(flow, h('div', { class: 'loop-back' }, '↺ steps 3 → 7 repeat thousands of times (one round per minibatch); step 8 checks progress along the way'), detail);
  function select(id) {
    const k = STAGES.findIndex(s => s.id === id);
    if (k < 0) return;
    const st = STAGES[k];
    cards.forEach((c, j) => c.classList.toggle('on', j === k));
    detail.innerHTML = '';
    const list = topics[st.id] || [];
    const btns = h('div', { class: 'topic-btns' });
    list.forEach(([link, name]) => {
      const b = h('button', { type: 'button', class: 'topic-btn' }, name + ' →');
      b.addEventListener('click', () => goLink(link));
      btns.append(b);
    });
    detail.append(h('div', { class: 'card' },
      h('h3', {}, h('span', { class: 'dot' }), 'Step ' + (k + 1) + ' · ' + st.name),
      h('p', {}, st.what),
      list.length ? h('div', { class: 'note', style: 'margin:8px 0 6px' }, course ? 'Where the course covers this step:' : 'Ideas from this lecture that live here:')
        : h('div', { class: 'note' }, 'This lecture does not change this step — the course home shows which lectures do.'),
      btns));
  }
  const firstWith = STAGES.find(s => (topics[s.id] || []).length);
  select(ds.start || (firstWith ? firstWith.id : 'update'));
  return { select };
};

/* ---------- boot ---------- */
Theme.init();
document.addEventListener('DOMContentLoaded', () => {
  buildChrome();
  renderMath(document.body);
  Deck.init();
});

/* ---------- small numeric helpers shared by the lectures ---------- */
/* Solve A x = b (Gaussian elimination with partial pivoting). A: n×n array of arrays. */
function solveLinear(A, b) {
  const n = b.length;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c];
    if (Math.abs(piv) < 1e-300) continue;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / piv;
      if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[n] / (Math.abs(r[i]) > 1e-300 ? r[i] : 1e-300));
}
/* Least squares with optional weight decay: minimize (1/m)·Σ(x·w − y)² + λ·Σ_{j≥1} w_j² (the first weight = bias is not penalized). */
function ridgeFit(X, y, lambda = 0) {
  const d = X[0].length, m = X.length;
  const A = Array.from({ length: d }, () => new Array(d).fill(0));
  const b = new Array(d).fill(0);
  X.forEach((row, i) => {
    for (let p = 0; p < d; p++) {
      b[p] += row[p] * y[i] / m;
      for (let q = 0; q < d; q++) A[p][q] += row[p] * row[q] / m;
    }
  });
  for (let p = 0; p < d; p++) A[p][p] += (p > 0 ? lambda : 0) + 1e-12;
  return solveLinear(A, b);
}
/* polynomial features of x ∈ [0,1], computed on t = 2x − 1 for numerical stability */
function polyRow(x, deg) { const t = 2 * x - 1; const r = [1]; for (let k = 1; k <= deg; k++) r.push(r[k - 1] * t); return r; }
function dotv(a, b) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function mean(a) { return a.reduce((s, v) => s + v, 0) / a.length; }
function mse(pred, y) { let s = 0; for (let i = 0; i < y.length; i++) s += (pred[i] - y[i]) ** 2; return s / y.length; }
function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

/* ---------- polynomial-regression toy problem (used by several lectures) ---------- */
function TRUE_F(x) { return Math.sin(2 * Math.PI * x); }
function makeData(n, noise, seed) {
  const rng = mulberry32(seed);
  return Array.from({ length: n }, () => { const x = rng(); return [x, TRUE_F(x) + noise * gaussian(rng)]; });
}
function fitPoly(data, deg, lambda = 0) { return ridgeFit(data.map(([x]) => polyRow(x, deg)), data.map(d => d[1]), lambda); }
function evalPoly(wv, x) { return dotv(wv, polyRow(x, wv.length - 1)); }
function polyMSE(wv, data) { return mean(data.map(([x, y]) => (evalPoly(wv, x) - y) ** 2)); }
/* log-gamma (Lanczos) — for Beta / Gaussian densities */
function lgamma(z) {
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1; let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
function betaPdf(t, a, b) { if (t <= 0 || t >= 1) return 0; return Math.exp((a - 1) * Math.log(t) + (b - 1) * Math.log(1 - t) - (lgamma(a) + lgamma(b) - lgamma(a + b))); }
function normPdf(x, mu, sd) { return Math.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI)); }
