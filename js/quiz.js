/* =========================================================
   quiz.js — self-check quiz widget (each lecture sets window.QUIZ)
   ========================================================= */
'use strict';



Widgets.quiz = (el) => {
  el.classList.remove('widget');
  el.classList.add('quiz');
  const answered = new Map();
  const score = h('div', { class: 'score' });
  const reset = h('button', { class: 'btn', type: 'button' }, 'Reset answers');
  reset.addEventListener('click', () => { answered.clear(); build(); });
  const list = h('div', { class: 'quiz' });
  el.append(h('div', { style: 'display:flex;gap:12px;align-items:center;flex-wrap:wrap' }, score, reset), list);
  function updateScore() {
    const right = [...answered.values()].filter(Boolean).length;
    score.textContent = `Score: ${right} / ${(window.QUIZ || []).length}` + (answered.size === (window.QUIZ || []).length ? (right === (window.QUIZ || []).length ? ' — perfect!' : ' — review the ones in red') : ` (${answered.size} answered)`);
  }
  function build() {
    list.innerHTML = '';
    (window.QUIZ || []).forEach((item, qi) => {
      const box = h('div', { class: 'q' });
      box.append(h('div', { class: 'qt' }, `${qi + 1}. `, h('span', { html: item.q })));
      const opts = h('div', { class: 'opts' });
      const why = h('div', { class: 'why', hidden: true, html: '<b>Why:</b> ' + item.why });
      const btns = item.opts.map((o, oi) => {
        const b = h('button', { type: 'button', class: 'opt', html: String.fromCharCode(65 + oi) + '. ' + o });
        b.addEventListener('click', () => {
          btns.forEach((x, k) => { x.disabled = true; if (k === item.a) x.classList.add('right'); });
          if (oi !== item.a) b.classList.add('wrong');
          answered.set(qi, oi === item.a);
          why.hidden = false;
          updateScore();
        });
        opts.append(b);
        return b;
      });
      box.append(opts, why);
      list.append(box);
    });
    renderMath(list);
    updateScore();
  }
  build();
  return {};
};
