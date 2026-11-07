/* =========================================================
   course.js — the list of lectures (shared by every page)
   The concept map used on the home page lives in js/concepts.js.
   ========================================================= */
'use strict';

window.COURSE = {
  lectures: [
    {
      id: 'ch05-1', n: 1, ch: '5.1–5.3', color: '--s1',
      title: 'Machine Learning Basics I',
      question: 'What does it mean for a computer to “learn”?',
      summary: 'Task, performance measure and experience; linear regression as the first learning algorithm; capacity, overfitting and underfitting; regularization; validation sets and cross-validation.',
      topics: ['Task / Performance / Experience', 'Linear regression', 'Capacity', 'Overfitting', 'No free lunch', 'Validation']
    },
    {
      id: 'ch05-4', n: 2, ch: '5.4–5.7', color: '--s1',
      title: 'Machine Learning Basics II',
      question: 'How do we judge and build estimators from data?',
      summary: 'Estimators, bias and variance; maximum likelihood (and why MSE is MLE); Bayesian statistics and MAP; the curse of dimensionality; probabilistic supervised learning; generative vs. discriminative models.',
      topics: ['Bias & variance', 'Maximum likelihood', 'Bayesian / MAP', 'Curse of dimensionality', 'Generative vs. discriminative']
    },
    {
      id: 'ch04-3', n: 3, ch: '4.3–4.5 & 5.9', color: '--s7',
      title: 'Numerical Optimization & SGD',
      question: 'How do we actually find the best parameters?',
      summary: 'Gradient descent, critical points, Jacobian and Hessian, condition number, constrained optimization (KKT), linear least squares, and stochastic gradient descent.',
      topics: ['Gradient descent', 'Hessian & curvature', 'Condition number', 'KKT', 'Least squares', 'SGD']
    },
    {
      id: 'ch06-1', n: 4, ch: '6.1–6.4', color: '--s2',
      title: 'Deep Feedforward Networks',
      question: 'What is a neural network and how should we design one?',
      summary: 'Layers, hidden units and why we need non-linearity (XOR); cost functions from maximum likelihood; output units (linear, sigmoid, softmax); hidden units (ReLU and friends); depth and universal approximation.',
      topics: ['XOR', 'Cross-entropy cost', 'Sigmoid & softmax outputs', 'ReLU', 'Universal approximation', 'Depth']
    },
    {
      id: 'ch06-5', n: 5, ch: '6.5', color: '--s2',
      title: 'Back-Propagation',
      question: 'How do we compute the gradient of a huge network efficiently?',
      summary: 'Computational graphs, the chain rule, applying it recursively, back-propagation in an MLP, the general algorithm and its cost.',
      topics: ['Computational graph', 'Chain rule', 'Backprop for MLP', 'Reusing sub-results']
    },
    {
      id: 'ch07', n: 6, ch: '7', color: '--s3',
      title: 'Regularization',
      question: 'How do we make the network work on data it has never seen?',
      summary: 'L2 and L1 penalties, data augmentation, noise robustness and label smoothing, early stopping, bagging and ensembles, and dropout.',
      topics: ['L2 / weight decay', 'L1 / sparsity', 'Data augmentation', 'Early stopping', 'Bagging', 'Dropout']
    },
    {
      id: 'ch08-1', n: 7, ch: '8.1–8.2', color: '--s5',
      title: 'Optimization I: Why Training Is Hard',
      question: 'How is training different from pure optimization, and what goes wrong?',
      summary: 'Empirical risk, surrogate losses, minibatches; ill-conditioning, local minima, saddle points and plateaus, cliffs and exploding gradients, long-term dependencies, inexact gradients.',
      topics: ['Surrogate loss', 'Minibatches', 'Ill-conditioning', 'Saddle points', 'Cliffs', 'Local vs. global']
    },
    {
      id: 'ch08-3', n: 8, ch: '8.3–8.5', color: '--s5',
      title: 'Optimization II: SGD, Momentum, Adam & Initialization',
      question: 'Which update rule should we use, and where should we start?',
      summary: 'SGD and learning-rate schedules, momentum and Nesterov momentum, AdaGrad, RMSProp and Adam, and how to initialize the weights.',
      topics: ['Learning-rate decay', 'Momentum', 'Nesterov', 'AdaGrad / RMSProp', 'Adam', 'Initialization']
    },
    {
      id: 'ch08-6', n: 9, ch: '8.6–8.7', color: '--s5',
      title: 'Optimization III: Second-Order Methods & Strategies',
      question: 'Can curvature help, and what other tricks make training easier?',
      summary: 'Newton’s method, conjugate gradients, BFGS and L-BFGS; batch normalization, Polyak averaging, pretraining, model design, continuation and curriculum learning.',
      topics: ['Newton', 'Conjugate gradients', 'BFGS', 'Batch norm', 'Polyak averaging', 'Curriculum']
    },
    {
      id: 'ch09', n: 10, ch: '9', color: '--s6',
      title: 'Convolutional Networks',
      question: 'How do we build networks for images and other grids?',
      summary: 'The convolution operation; sparse interactions, parameter sharing and equivariance; pooling; convolution as a strong prior; stride, padding, local and tiled connections; gradients, bias and data types.',
      topics: ['Convolution', 'Parameter sharing', 'Pooling', 'Stride & padding', 'Receptive field']
    },
    {
      id: 'ch10', n: 11, ch: '10', color: '--s8',
      title: 'Recurrent Neural Networks',
      question: 'How do we build networks for sequences?',
      summary: 'Unfolding computational graphs, RNN designs and teacher forcing, back-propagation through time, bidirectional and deep RNNs, long-term dependencies, echo state networks, leaky units, LSTM and GRU, gradient clipping.',
      topics: ['Unfolding', 'BPTT', 'Teacher forcing', 'Vanishing gradients', 'LSTM / GRU', 'Gradient clipping']
    }
  ]
};
