# Deep Learning — interactive course notes

Interactive study notes for a Deep Learning course: **11 lectures, 191 slides, 130+ live demos**, from machine-learning basics to convolutional and recurrent networks.

Every slide has the same parts:

- **Key idea**: the one sentence to remember.
- **In plain words**: the idea explained without jargon.
- **Formulas**: rendered with KaTeX, with the textbook equation numbers.
- **Terms on this slide**: every symbol and word defined.
- **Calculation, step by step**: real numbers plugged into each formula.
- **A live demo**: sliders, step buttons and charts, with a panel that shows the current numbers.
- **Connections**: links back to what the idea builds on and forward to where it is used.

Each lecture starts with the **training loop** (data → init → minibatch → forward → loss → backward → update → evaluate) and ends with a summary table and a quiz.

The home page (`index.html`) connects the whole course:

- a course-wide training-loop map;
- the "story" of how each lecture leads to the next;
- a concept explorer with 61 concepts, showing what each builds on and what uses it;
- cards for every lecture.

## Lectures

| # | Chapter | Topic |
|---|---------|-------|
| 1 | 5.1–5.3 | Machine learning basics I: task/performance/experience, linear regression, capacity, regularization, validation |
| 2 | 5.4–5.7 | Machine learning basics II: bias & variance, maximum likelihood, Bayesian/MAP, curse of dimensionality |
| 3 | 4.3–4.5, 5.9 | Numerical optimization & SGD: gradient descent, Hessian, condition number, KKT, least squares |
| 4 | 6.1–6.4 | Deep feedforward networks: XOR, cost functions, output and hidden units, universal approximation |
| 5 | 6.5 | Back-propagation: computational graphs, chain rule, backprop in an MLP |
| 6 | 7 | Regularization: L2/L1, augmentation, early stopping, bagging, dropout |
| 7 | 8.1–8.2 | Optimization I: why training is hard (surrogate loss, minibatches, ill-conditioning, saddles, cliffs) |
| 8 | 8.3–8.5 | Optimization II: SGD schedules, momentum, Nesterov, AdaGrad, RMSProp, Adam, initialization |
| 9 | 8.6–8.7 | Optimization III: Newton, CG, BFGS, L-BFGS, batch norm, Polyak averaging, pretraining, continuation |
| 10 | 9 | Convolutional networks: convolution, sharing, equivariance, pooling, stride/padding, gradients |
| 11 | 10 | Recurrent networks: unfolding, BPTT, teacher forcing, bidirectional/deep/recursive nets, LSTM, GRU, clipping |

## Files

```
index.html               course home (loop map, story, concept explorer, lecture cards)
lectures/<id>.html       one slide deck per lecture
css/style.css            layout, components, light/dark theme
js/core.js               navigation, theme, charts, training-loop widget, shared maths helpers
js/course.js             the list of lectures
js/concepts.js           home-page data: course-wide loop map, story, concept map
js/quiz.js               quiz widget (each lecture defines its own questions)
js/lectures/<id>.js      the demos for each lecture
```

It is a static site (HTML + CSS + JavaScript) with no build step. Math is rendered with KaTeX from a CDN.

## View it locally

Run a small local server in this folder:

```bash
python -m http.server 8000
```

Then open http://localhost:8000.

## Publish on GitHub Pages

1. Push this folder to a GitHub repository (the `main` branch).
2. In the repository, open **Settings → Pages**, choose **Deploy from a branch**, then branch **main** and folder **/ (root)**, and **Save**.
3. After a minute or two the site is live at `https://<your-username>.github.io/<repo-name>/`.

Links to single slides also work, e.g. `lectures/ch08-3.html#/s-adam`.

## Notes

- After editing CSS or JS, bump the `?v=` number on the `<link>`/`<script>` tags so browsers fetch the new files instead of cached copies.
- These are personal study notes. All explanations, diagrams and demos are original. Equation numbers follow Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016). The course slides and textbook figures are not included.
