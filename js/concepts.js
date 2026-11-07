/* =========================================================
   concepts.js — the course home page: story, concept map, lecture cards
   ========================================================= */
'use strict';

/* Course-wide training-loop map: every stage → the lecture slides that change it. */
window.TOPICS = {
  data: [['lectures/ch05-1.html#/s-experience', 'L1 · Experience: the dataset'], ['lectures/ch05-1.html#/s-hyper', 'L1 · Train / validation / test split'], ['lectures/ch07.html#/s-augment', 'L6 · Data augmentation'], ['lectures/ch09.html#/s-data-types', 'L10 · Grid data and channels'], ['lectures/ch10.html#/s-seq', 'L11 · Sequences']],
  init: [['lectures/ch06-1.html#/s-arch', 'L4 · Choose the architecture'], ['lectures/ch08-3.html#/s-init-scale', 'L8 · Initialization scale'], ['lectures/ch08-3.html#/s-init-bias', 'L8 · Bias initialization'], ['lectures/ch08-6.html#/s-pretrain', 'L9 · Pretraining'], ['lectures/ch09.html#/s-cnn-arch', 'L10 · CNN layers'], ['lectures/ch10.html#/s-esn', 'L11 · Echo state networks']],
  batch: [['lectures/ch04-3.html#/s-sgd', 'L3 · Stochastic gradient descent'], ['lectures/ch08-1.html#/s-minibatch', 'L7 · Minibatch size and noise'], ['lectures/ch08-6.html#/s-curriculum', 'L9 · Curriculum learning']],
  forward: [['lectures/ch06-1.html#/s-ffn', 'L4 · Feedforward networks'], ['lectures/ch06-1.html#/s-hidden', 'L4 · Hidden units'], ['lectures/ch07.html#/s-dropout', 'L6 · Dropout'], ['lectures/ch08-6.html#/s-bn-how', 'L9 · Batch normalization'], ['lectures/ch09.html#/s-conv2d', 'L10 · Convolution'], ['lectures/ch09.html#/s-pooling', 'L10 · Pooling'], ['lectures/ch10.html#/s-forward', 'L11 · RNN forward pass'], ['lectures/ch10.html#/s-lstm', 'L11 · LSTM']],
  loss: [['lectures/ch05-1.html#/s-performance', 'L1 · Performance measure (MSE)'], ['lectures/ch05-4.html#/s-mle', 'L2 · Maximum likelihood'], ['lectures/ch06-1.html#/s-cost-mle', 'L4 · Cross-entropy cost'], ['lectures/ch07.html#/s-l2', 'L6 · Weight-decay penalty'], ['lectures/ch08-1.html#/s-surrogate', 'L7 · Surrogate loss'], ['lectures/ch10.html#/s-teacher', 'L11 · Teacher forcing']],
  backward: [['lectures/ch06-5.html#/s-mlp-backprop', 'L5 · Back-propagation'], ['lectures/ch04-3.html#/s-jacobian-hessian', 'L3 · Jacobian and Hessian'], ['lectures/ch09.html#/s-conv-grad', 'L10 · Gradients of convolution'], ['lectures/ch10.html#/s-bptt', 'L11 · Back-propagation through time'], ['lectures/ch10.html#/s-longterm', 'L11 · Vanishing / exploding gradients']],
  update: [['lectures/ch04-3.html#/s-gd', 'L3 · Gradient descent'], ['lectures/ch08-1.html#/s-illcond', 'L7 · Ill-conditioning'], ['lectures/ch08-3.html#/s-momentum', 'L8 · Momentum'], ['lectures/ch08-3.html#/s-adam', 'L8 · Adam'], ['lectures/ch08-6.html#/s-newton', 'L9 · Newton, CG, BFGS'], ['lectures/ch10.html#/s-clip', 'L11 · Gradient clipping']],
  final: [['lectures/ch05-1.html#/s-generalization', 'L1 · Generalization'], ['lectures/ch05-1.html#/s-cv', 'L1 · Cross-validation'], ['lectures/ch05-4.html#/s-bias-variance', 'L2 · Bias–variance trade-off'], ['lectures/ch07.html#/s-early-stop', 'L6 · Early stopping'], ['lectures/ch08-6.html#/s-polyak', 'L9 · Polyak averaging']]
};

/* How each lecture hands over to the next one. */
const STORY = {
  'ch05-1': 'We now know what “learning” means and that the real goal is a low <b>test</b> error. But how do we pick good estimators in a principled way?',
  'ch05-4': 'Maximum likelihood gives us a loss to minimize for any model. Next: the tools to actually <b>minimize</b> it.',
  'ch04-3': 'We can minimize a function with gradients. Now we need a <b>flexible model</b> worth minimizing over: the neural network.',
  'ch06-1': 'A network is a big composition of functions. Training it needs <b>its gradient</b>, efficiently.',
  'ch06-5': 'Now we can train any network. But a big network can memorize the training set — next: how to make it <b>generalize</b>.',
  'ch07': 'With regularization the model generalizes. But <b>training</b> deep nets is still hard — why?',
  'ch08-1': 'We know the obstacles (noise, ill-conditioning, saddles, cliffs). Next: the <b>update rules</b> and starting points that get past them.',
  'ch08-3': 'First-order methods are the workhorse. Can <b>curvature</b> and other strategies help even more?',
  'ch08-6': 'We can train deep MLPs well. Now we <b>specialize</b> the architecture for images and grids.',
  'ch09': 'Convolution shares weights across space. For <b>sequences</b>, we share them across time.',
  'ch10': 'You have the whole toolbox: data → model → loss → gradients → updates → evaluation. Use the concept explorer below to review how it fits together.'
};

/* Concept map: every concept points to the slide that explains it and lists what it builds on. */
const CONCEPTS = [
  // Foundations
  { id: 'task-pe', g: 'Foundations', name: 'Task, performance, experience', lec: 'ch05-1', s: 's-what-learning', def: 'A program learns if its performance P at task T improves with experience E.', needs: [] },
  { id: 'mse', g: 'Foundations', name: 'Mean squared error', lec: 'ch05-1', s: 's-performance', def: 'Average of squared prediction errors — a common performance measure for regression.', needs: ['task-pe'] },
  { id: 'linreg', g: 'Foundations', name: 'Linear regression', lec: 'ch05-1', s: 's-linreg', def: 'Predict y = wᵀx; the best w solves the normal equations.', needs: ['task-pe', 'mse'] },
  { id: 'generalization', g: 'Foundations', name: 'Generalization & test error', lec: 'ch05-1', s: 's-generalization', def: 'What matters is the error on new data, not on the training set.', needs: ['task-pe'] },
  { id: 'capacity', g: 'Foundations', name: 'Capacity, over- & underfitting', lec: 'ch05-1', s: 's-capacity', def: 'Too little capacity underfits, too much overfits; the best is in between.', needs: ['generalization'] },
  { id: 'nfl', g: 'Foundations', name: 'No free lunch', lec: 'ch05-1', s: 's-nfl', def: 'No learner is best on all possible tasks; assumptions (priors) matter.', needs: ['generalization'] },
  { id: 'regularization', g: 'Foundations', name: 'Regularization (the idea)', lec: 'ch05-1', s: 's-regularization', def: 'Any change meant to reduce test error rather than training error.', needs: ['capacity'] },
  { id: 'hyper', g: 'Foundations', name: 'Hyper-parameters & validation set', lec: 'ch05-1', s: 's-hyper', def: 'Settings not learned by training; chosen on held-out validation data.', needs: ['generalization'] },
  { id: 'cv', g: 'Foundations', name: 'Cross-validation', lec: 'ch05-1', s: 's-cv', def: 'Rotate the validation fold to estimate error reliably on small datasets.', needs: ['hyper'] },
  // Estimation
  { id: 'bias-var', g: 'Probability & estimation', name: 'Bias & variance', lec: 'ch05-4', s: 's-bias-variance', def: 'Expected error = bias² + variance (+ noise); capacity trades one for the other.', needs: ['capacity'] },
  { id: 'mle', g: 'Probability & estimation', name: 'Maximum likelihood', lec: 'ch05-4', s: 's-mle', def: 'Choose parameters that make the observed data most probable.', needs: ['task-pe'] },
  { id: 'cond-ll', g: 'Probability & estimation', name: 'Conditional log-likelihood = MSE', lec: 'ch05-4', s: 's-cond-ll', def: 'With Gaussian noise, maximizing likelihood is the same as minimizing MSE.', needs: ['mle', 'mse'] },
  { id: 'map', g: 'Probability & estimation', name: 'Bayesian & MAP estimation', lec: 'ch05-4', s: 's-map', def: 'Combine likelihood with a prior; a Gaussian prior = weight decay.', needs: ['mle'] },
  { id: 'curse', g: 'Probability & estimation', name: 'Curse of dimensionality', lec: 'ch05-4', s: 's-curse', def: 'The number of regions explodes with dimension; local methods need too much data.', needs: ['generalization'] },
  { id: 'logreg', g: 'Probability & estimation', name: 'Logistic regression', lec: 'ch05-4', s: 's-prob-sl', def: 'A linear model squashed by a sigmoid gives class probabilities.', needs: ['mle', 'linreg'] },
  { id: 'gen-disc', g: 'Probability & estimation', name: 'Generative vs. discriminative', lec: 'ch05-4', s: 's-gen-disc', def: 'Model p(x, y) or directly p(y | x).', needs: ['logreg'] },
  // Building blocks
  { id: 'mlp', g: 'Network building blocks', name: 'Feedforward network (MLP)', lec: 'ch06-1', s: 's-ffn', def: 'Layers of linear maps + non-linearities, composed into one function.', needs: ['linreg'] },
  { id: 'relu', g: 'Network building blocks', name: 'Hidden units (ReLU & friends)', lec: 'ch06-1', s: 's-hidden', def: 'Non-linear activations; ReLU = max(0, z) is the default.', needs: ['mlp'] },
  { id: 'cost', g: 'Network building blocks', name: 'Cross-entropy cost', lec: 'ch06-1', s: 's-cost-mle', def: 'The loss that maximum likelihood gives for a network’s output distribution.', needs: ['mle', 'mlp'] },
  { id: 'output-units', g: 'Network building blocks', name: 'Output units (linear, sigmoid, softmax)', lec: 'ch06-1', s: 's-softmax', def: 'Turn the last layer into the parameters of an output distribution.', needs: ['cost', 'logreg'] },
  { id: 'uat', g: 'Network building blocks', name: 'Universal approximation & depth', lec: 'ch06-1', s: 's-uat', def: 'One wide hidden layer can approximate anything; depth does it far more efficiently.', needs: ['mlp'] },
  { id: 'comp-graph', g: 'Network building blocks', name: 'Computational graph', lec: 'ch06-5', s: 's-graph', def: 'Nodes = variables, edges = operations; any model is such a graph.', needs: ['mlp'] },
  { id: 'chain', g: 'Network building blocks', name: 'Chain rule', lec: 'ch06-5', s: 's-chain', def: 'Derivative of a composition = product of local derivatives.', needs: ['comp-graph'] },
  { id: 'backprop', g: 'Network building blocks', name: 'Back-propagation', lec: 'ch06-5', s: 's-mlp-backprop', def: 'Apply the chain rule backwards, reusing sub-results: all gradients for ~one forward pass of cost.', needs: ['chain', 'gd'] },
  // Optimization
  { id: 'gd', g: 'Optimization', name: 'Gradient descent', lec: 'ch04-3', s: 's-gd', def: 'Step against the gradient: θ ← θ − ε∇J(θ).', needs: [] },
  { id: 'hessian', g: 'Optimization', name: 'Jacobian & Hessian (curvature)', lec: 'ch04-3', s: 's-jacobian-hessian', def: 'Second derivatives tell how the gradient changes.', needs: ['gd'] },
  { id: 'condition', g: 'Optimization', name: 'Condition number', lec: 'ch04-3', s: 's-condition', def: 'Ratio of largest to smallest curvature; large → zig-zagging gradient descent.', needs: ['hessian'] },
  { id: 'kkt', g: 'Optimization', name: 'Constrained optimization (KKT)', lec: 'ch04-3', s: 's-constrained', def: 'Optimize subject to constraints using generalized Lagrange multipliers.', needs: ['gd'] },
  { id: 'sgd', g: 'Optimization', name: 'Stochastic gradient descent', lec: 'ch04-3', s: 's-sgd', def: 'Estimate the gradient from a small random minibatch.', needs: ['gd'] },
  { id: 'surrogate', g: 'Optimization', name: 'Empirical risk & surrogate loss', lec: 'ch08-1', s: 's-surrogate', def: 'We minimize a convenient proxy for the true objective, and stop early on validation.', needs: ['mle', 'generalization'] },
  { id: 'minibatch', g: 'Optimization', name: 'Minibatch noise', lec: 'ch08-1', s: 's-minibatch', def: 'Gradient noise shrinks like 1/√m — bigger batches give diminishing returns.', needs: ['sgd'] },
  { id: 'illcond', g: 'Optimization', name: 'Ill-conditioning', lec: 'ch08-1', s: 's-illcond', def: 'Steps that look downhill can increase the cost when curvature is large.', needs: ['condition'] },
  { id: 'saddle', g: 'Optimization', name: 'Local minima, saddles & plateaus', lec: 'ch08-1', s: 's-saddle', def: 'In high dimensions most critical points are saddles, not bad minima.', needs: ['hessian'] },
  { id: 'cliffs', g: 'Optimization', name: 'Cliffs & exploding gradients', lec: 'ch08-1', s: 's-cliffs', def: 'Very steep regions can catapult the parameters far away.', needs: ['gd'] },
  { id: 'lr-schedule', g: 'Optimization', name: 'Learning-rate schedule', lec: 'ch08-3', s: 's-sgd', def: 'Decay ε so that minibatch noise dies out near the minimum.', needs: ['sgd', 'minibatch'] },
  { id: 'momentum', g: 'Optimization', name: 'Momentum & Nesterov', lec: 'ch08-3', s: 's-momentum', def: 'Accumulate a velocity to move faster along consistent directions.', needs: ['sgd', 'illcond'] },
  { id: 'adaptive', g: 'Optimization', name: 'AdaGrad, RMSProp, Adam', lec: 'ch08-3', s: 's-adam', def: 'Per-parameter step sizes from running averages of squared gradients.', needs: ['sgd', 'momentum'] },
  { id: 'init', g: 'Optimization', name: 'Parameter initialization', lec: 'ch08-3', s: 's-init-scale', def: 'Random weights break symmetry; their scale controls vanishing/exploding signals.', needs: ['mlp'] },
  { id: 'newton', g: 'Optimization', name: 'Newton’s method', lec: 'ch08-6', s: 's-newton', def: 'Jump to the minimum of the local quadratic approximation: θ − H⁻¹g.', needs: ['hessian'] },
  { id: 'cg', g: 'Optimization', name: 'Conjugate gradients', lec: 'ch08-6', s: 's-cg', def: 'Search directions that do not undo each other’s progress.', needs: ['newton'] },
  { id: 'bfgs', g: 'Optimization', name: 'BFGS & L-BFGS', lec: 'ch08-6', s: 's-bfgs', def: 'Build up an approximation of H⁻¹ from gradients alone.', needs: ['newton'] },
  { id: 'batchnorm', g: 'Optimization', name: 'Batch normalization', lec: 'ch08-6', s: 's-bn-how', def: 'Normalize each unit over the minibatch, then rescale with learned γ, β.', needs: ['mlp', 'sgd'] },
  { id: 'polyak', g: 'Optimization', name: 'Polyak averaging', lec: 'ch08-6', s: 's-polyak', def: 'Average the parameters visited by the optimizer.', needs: ['sgd'] },
  { id: 'pretrain', g: 'Optimization', name: 'Pretraining & curriculum', lec: 'ch08-6', s: 's-pretrain', def: 'Start from a model trained on a simpler problem or simpler examples.', needs: ['init'] },
  // Regularization
  { id: 'weight-decay', g: 'Regularization', name: 'L² weight decay', lec: 'ch07', s: 's-l2', def: 'Penalize ½α‖w‖²: shrink weights, especially in low-curvature directions.', needs: ['regularization', 'map'] },
  { id: 'l1', g: 'Regularization', name: 'L¹ penalty & sparsity', lec: 'ch07', s: 's-l1', def: 'Penalize α‖w‖₁: many weights become exactly zero.', needs: ['regularization'] },
  { id: 'augment', g: 'Regularization', name: 'Data augmentation & noise', lec: 'ch07', s: 's-augment', def: 'Create new training examples by label-preserving transformations.', needs: ['generalization'] },
  { id: 'early-stop', g: 'Regularization', name: 'Early stopping', lec: 'ch07', s: 's-early-stop', def: 'Stop when validation error is lowest; acts like L² regularization.', needs: ['hyper', 'weight-decay'] },
  { id: 'ensemble', g: 'Regularization', name: 'Bagging & ensembles', lec: 'ch07', s: 's-bagging', def: 'Average several models: errors partially cancel.', needs: ['bias-var'] },
  { id: 'dropout', g: 'Regularization', name: 'Dropout', lec: 'ch07', s: 's-dropout', def: 'Randomly drop units during training — a cheap ensemble of sub-networks.', needs: ['ensemble', 'mlp'] },
  // Architectures
  { id: 'conv', g: 'Convolutional networks', name: 'Convolution', lec: 'ch09', s: 's-conv2d', def: 'Slide a small kernel over the input; each output = patch · kernel.', needs: ['mlp'] },
  { id: 'sharing', g: 'Convolutional networks', name: 'Sparse connections & parameter sharing', lec: 'ch09', s: 's-sharing', def: 'Few local connections, same weights everywhere → tiny, efficient layers.', needs: ['conv'] },
  { id: 'pooling', g: 'Convolutional networks', name: 'Pooling & invariance', lec: 'ch09', s: 's-pooling', def: 'Summarize neighbourhoods (max/average): robust to small shifts.', needs: ['conv'] },
  { id: 'strong-prior', g: 'Convolutional networks', name: 'Convolution as a strong prior', lec: 'ch09', s: 's-prior', def: 'Hard constraints (local, tied weights): great when true, underfits when false.', needs: ['sharing', 'map'] },
  { id: 'cnn', g: 'Convolutional networks', name: 'A complete CNN', lec: 'ch09', s: 's-cnn-arch', def: 'conv → ReLU → pool blocks, then a softmax classifier.', needs: ['conv', 'pooling', 'backprop', 'output-units'] },
  // Sequences
  { id: 'rnn', g: 'Sequence models', name: 'Recurrent network & unfolding', lec: 'ch10', s: 's-unfold', def: 'h(t) = f(h(t−1), x(t); θ) — the same θ at every time step.', needs: ['mlp', 'sharing'] },
  { id: 'bptt', g: 'Sequence models', name: 'Back-propagation through time', lec: 'ch10', s: 's-bptt', def: 'Back-prop on the unrolled graph; shared weights sum their gradients.', needs: ['rnn', 'backprop'] },
  { id: 'teacher', g: 'Sequence models', name: 'Teacher forcing', lec: 'ch10', s: 's-teacher', def: 'Feed the true previous output during training.', needs: ['rnn', 'mle'] },
  { id: 'longterm', g: 'Sequence models', name: 'Long-term dependencies', lec: 'ch10', s: 's-longterm', def: 'Products of many Jacobians vanish or explode.', needs: ['bptt', 'init'] },
  { id: 'lstm', g: 'Sequence models', name: 'LSTM & GRU', lec: 'ch10', s: 's-lstm', def: 'Gated self-loops let information and gradients survive many steps.', needs: ['longterm'] },
  { id: 'clip', g: 'Sequence models', name: 'Gradient clipping', lec: 'ch10', s: 's-clip', def: 'Rescale gradients whose norm exceeds a threshold.', needs: ['cliffs', 'bptt'] }
];

const lecById = id => Course.list().find(l => l.id === id);

/* ---------- the lecture cards ---------- */
Widgets.lectureCards = (el) => {
  el.classList.remove('widget');
  const grid = h('div', { class: 'lec-grid' });
  Course.list().forEach(l => grid.append(h('a', { class: 'lec-card', href: Course.href(l), style: `--c:var(${l.color})` },
    h('span', { class: 'num' }, `Lecture ${l.n} · Chapter ${l.ch}`),
    h('h3', {}, l.title),
    h('p', {}, h('i', {}, l.question)),
    h('p', {}, l.summary),
    h('div', { class: 'pills' }, l.topics.map(t => h('span', { class: 'pill' }, t))),
    h('span', { class: 'open' }, 'Open the interactive slides →'))));
  el.append(grid);
  return {};
};

/* ---------- the story of the course ---------- */
Widgets.story = (el) => {
  el.classList.remove('widget');
  const wrap = h('div', { class: 'story' });
  Course.list().forEach(l => wrap.append(h('div', { class: 'story-step', style: `--c:var(${l.color})` },
    h('div', { class: 'dotn' }, 'L' + l.n),
    h('div', { class: 'body' },
      h('div', { class: 'q' }, l.question),
      h('h4', {}, l.title),
      h('p', { html: STORY[l.id] || '' }),
      h('a', { href: Course.href(l) }, 'Open lecture ' + l.n + ' →')))));
  el.append(wrap);
  return {};
};

/* ---------- concept explorer ---------- */
Widgets.concepts = (el) => {
  UI.head(el, 'Concept explorer: what builds on what', 'Pick a concept. Left: what you need to know first. Right: what uses it later. Click any box to walk through the course.');
  const byId = Object.fromEntries(CONCEPTS.map(c => [c.id, c]));
  const usedBy = id => CONCEPTS.filter(c => c.needs.includes(id));
  const box = h('div', { class: 'cx' });
  const list = h('div', { class: 'cx-list' });
  const search = h('input', { type: 'search', placeholder: 'Search concepts…', 'aria-label': 'Search concepts' });
  const items = h('div');
  list.append(search, items);
  const main = h('div', { class: 'cx-main' });
  box.append(list, main);
  el.append(box);
  let cur = 'backprop';
  function renderList() {
    const q = search.value.trim().toLowerCase();
    items.innerHTML = '';
    let g = null;
    CONCEPTS.filter(c => !q || c.name.toLowerCase().includes(q) || c.def.toLowerCase().includes(q)).forEach(c => {
      if (c.g !== g) { g = c.g; items.append(h('div', { class: 'cx-group' }, g)); }
      const b = h('button', { type: 'button', class: 'cx-item' + (c.id === cur ? ' on' : '') }, c.name);
      b.addEventListener('click', () => { cur = c.id; render(); });
      items.append(b);
    });
  }
  const nodeBtn = c => {
    const l = lecById(c.lec);
    const b = h('button', { type: 'button', class: 'cx-node' }, c.name, h('small', {}, `L${l ? l.n : '?'} · ${l ? l.title : ''}`));
    b.addEventListener('click', () => { cur = c.id; render(); });
    return b;
  };
  function render() {
    renderList();
    const c = byId[cur], l = lecById(c.lec);
    const pre = c.needs.map(id => byId[id]).filter(Boolean), post = usedBy(c.id);
    main.innerHTML = '';
    main.append(h('div', { class: 'cx-flow' },
      h('div', { class: 'cx-col' }, h('span', { class: 'lab' }, 'Builds on'), pre.length ? pre.map(nodeBtn) : h('span', { class: 'cx-empty' }, 'A starting point — no prerequisites.')),
      h('div', { class: 'cx-arrow' }, '→'),
      h('div', { class: 'cx-center' },
        h('span', { class: 'kicker' }, `${c.g} · Lecture ${l.n}`),
        h('h4', {}, c.name),
        h('p', {}, c.def),
        h('a', { class: 'btn primary', href: Course.href(l, c.s) }, 'Open the slide →')),
      h('div', { class: 'cx-arrow' }, '→'),
      h('div', { class: 'cx-col' }, h('span', { class: 'lab' }, 'Used by'), post.length ? post.map(nodeBtn) : h('span', { class: 'cx-empty' }, 'Nothing later builds on this directly.'))));
  }
  search.addEventListener('input', renderList);
  render();
  return { redraw: () => {} };
};
