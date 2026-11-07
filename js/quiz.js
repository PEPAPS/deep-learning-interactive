/* =========================================================
   quiz.js — self-check questions with explanations
   ========================================================= */
'use strict';

const QUIZ = [
  {
    q: 'Newton’s method is applied to a quadratic loss whose Hessian is positive definite. How many updates does it need to reach the minimum?',
    opts: ['Exactly one', 'About k, the number of parameters', 'It depends on the learning rate', 'It never reaches it exactly'],
    a: 0,
    why: 'The second-order Taylor expansion of a quadratic is the function itself, so jumping to the minimum of the approximation \\(\\theta^* = \\theta_0 - H^{-1}\\nabla J(\\theta_0)\\) lands exactly on the true minimum. There is no learning rate to tune.'
  },
  {
    q: 'Why does steepest descent with an exact line search zig-zag in a narrow valley?',
    opts: ['The learning rate is too large', 'At the end of each line search the new gradient is orthogonal to the previous direction', 'The Hessian is not invertible', 'Minibatch noise'],
    a: 1,
    why: 'At the lowest point along a line, the slope along that line is zero: \\(\\nabla J(\\theta)^\\top d_{t-1} = 0\\). So the next direction (the gradient) is at 90° to the previous one, and partly undoes earlier progress.'
  },
  {
    q: 'Two search directions are called conjugate when…',
    opts: ['\\(d_t^\\top d_{t-1} = 0\\)', '\\(d_t^\\top H\\, d_{t-1} = 0\\)', '\\(d_t = -d_{t-1}\\)', '\\(\\|d_t\\| = \\|d_{t-1}\\|\\)'],
    a: 1,
    why: 'Conjugate means “orthogonal as measured through the Hessian”. Moving along \\(d_t\\) then does not spoil the minimization already done along \\(d_{t-1}\\). On a k-dimensional quadratic, CG needs at most k line searches.'
  },
  {
    q: 'Near a saddle point, what can the plain (unregularized) Newton update do?',
    opts: ['Always escape faster than gradient descent', 'Move toward the saddle point, i.e. in the wrong direction', 'Stop because the gradient is zero', 'Nothing special happens'],
    a: 1,
    why: 'Newton jumps to wherever the local quadratic model has zero slope. With a negative eigenvalue that point is a saddle or maximum, so it can be attracted to it. Adding \\(\\alpha I\\) to the Hessian (Levenberg–Marquardt style) makes the step point downhill again.'
  },
  {
    q: 'What is the main reason BFGS is impractical for modern deep networks?',
    opts: ['It needs an exact line search', 'It stores a k × k inverse-Hessian approximation: O(k²) memory', 'It cannot use gradients', 'It only works for convex functions'],
    a: 1,
    why: 'With millions of parameters, \\(k^2\\) numbers is far beyond any memory. L-BFGS keeps only the last m pairs \\((s, y)\\) — O(mk) memory — and rebuilds the effect of M on the fly.'
  },
  {
    q: 'In batch normalization, why is it crucial to back-propagate through the computation of μ and σ?',
    opts: ['To make training faster on GPUs', 'So the gradient can never propose simply changing a unit’s mean or std — that component is zeroed out', 'Because μ and σ are learned parameters', 'It is not crucial; they can be treated as constants'],
    a: 1,
    why: 'Since the normalization would undo any change to the mean or standard deviation, back-propagating through it removes that useless part of the gradient. Earlier approaches (penalties, or renormalizing after each step) either normalized imperfectly or wasted effort.'
  },
  {
    q: 'Why do we add learned parameters γ and β after normalizing (γĤ + β)?',
    opts: ['To undo batch norm at test time', 'To keep the network’s expressive power while giving it easier learning dynamics', 'To reduce memory use', 'To replace the activation function'],
    a: 1,
    why: 'Forcing mean 0 / std 1 could limit what the layer can represent. γ and β give that freedom back, but now the mean is controlled by a single parameter β instead of a complicated interaction of all lower layers — much easier for gradient descent.'
  },
  {
    q: 'At test time, a batch-norm layer usually normalizes with…',
    opts: ['The statistics of the test minibatch', 'Running averages of μ and σ collected during training', 'μ = 0 and σ = 1', 'The statistics of the single test example'],
    a: 1,
    why: 'Running averages let the model evaluate one example at a time. A “batch” of one example would have σ = 0 and normalize everything to 0. (In PyTorch this is what model.eval() switches on.)'
  },
  {
    q: 'For non-convex problems, Polyak averaging is usually done with…',
    opts: ['A plain average of every point ever visited', 'An exponentially decaying running average', 'Only the last point', 'The point with the lowest training loss'],
    a: 1,
    why: 'Points from the distant past may be in a completely different region, separated by barriers. \\(\\hat\\theta^{(t)} = \\alpha\\hat\\theta^{(t-1)} + (1-\\alpha)\\theta^{(t)}\\) forgets them gradually.'
  },
  {
    q: 'In teacher–student training with hints (FitNets), what is the thin deep student trained to do?',
    opts: ['Only copy the teacher’s outputs', 'Predict the task output and also predict the teacher’s middle hidden layer', 'Train one layer at a time, greedily', 'Only predict the teacher’s middle layer'],
    a: 1,
    why: 'Its lower layers get two objectives: help the final prediction, and (through an extra regressor) reproduce the teacher’s middle layer. Without these hints the student performed poorly on both training and test sets.'
  },
  {
    q: 'Which statement matches the lecture’s view on “designing models to aid optimization”?',
    opts: ['Better optimizers caused most of the progress', 'Choosing an easy-to-optimize model family matters more than a powerful optimizer', 'Sigmoid units are easier to optimize than ReLUs', 'Skip connections make gradients vanish faster'],
    a: 1,
    why: 'Most progress came from changing the model (ReLU, LSTM, maxout, skip connections, auxiliary heads), while SGD with momentum from the 1980s is still used. Mostly-linear pieces keep gradients flowing and pointing toward far-away solutions.'
  },
  {
    q: 'A continuation method that blurs \\(J(\\theta) = -\\theta^2\\) fails because…',
    opts: ['It needs too many stages', 'The blurred function never becomes convex', 'It tracks to a local minimum', 'Blurring is too expensive'],
    a: 1,
    why: 'Blurring a downward parabola gives \\(-\\theta^2 - \\sigma^2\\): the same hill, only lower. The other failure modes are “too many stages” and “the blurred minimum tracks to a local, not global, minimum”.'
  },
  {
    q: 'Which curriculum gave clearly better results in the experiment described in the lecture?',
    opts: ['Deterministic: easy examples first, then only hard ones', 'Stochastic: always a random mix, with the share of hard examples growing over time', 'Hard examples first, then easy ones', 'No curriculum at all'],
    a: 1,
    why: 'The deterministic curriculum showed no improvement over ordinary training; the stochastic one, which always mixes easy and hard examples while slowly raising the share of hard ones, worked much better.'
  }
];

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
    score.textContent = `Score: ${right} / ${QUIZ.length}` + (answered.size === QUIZ.length ? (right === QUIZ.length ? ' — perfect!' : ' — review the ones in red') : ` (${answered.size} answered)`);
  }
  function build() {
    list.innerHTML = '';
    QUIZ.forEach((item, qi) => {
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
