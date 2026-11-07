# Optimization for Training — interactive study slides

Interactive study slides for **Deep Learning, chapter 8.6–8.7**:

- **Part A — approximate second-order methods:** Newton's method, saddle points and Hessian regularization, line search and the zig-zag problem, conjugate gradients (Fletcher–Reeves / Polak–Ribière), BFGS and L-BFGS.
- **Part B — optimization strategies and meta-algorithms:** batch normalization, Polyak averaging, supervised pretraining (greedy, transfer learning, teacher hints), designing models for optimization, continuation methods and curriculum learning.

Every slide explains the idea in plain words, shows the formulas, says which step of the training loop it changes, and comes with a hands-on demo: an optimizer race on loss landscapes, batch norm worked through number by number, a real forward/backward pass that measures vanishing gradients, and more. A 13-question quiz is at the end.

It is a static site (HTML + CSS + JavaScript) with no build step. Math is rendered with KaTeX from a CDN.

## Files

```
index.html          all slides (text + formulas)
css/style.css       layout, light/dark theme
js/core.js          navigation, theme, chart helpers, training-loop map
js/second-order.js  Part A demos (playground, Taylor, Newton cost, ...)
js/meta.js          Part B demos (batch norm, Polyak, pretraining, ...)
js/quiz.js          quiz questions
```

## Put it on GitHub Pages

1. Create a new repository on GitHub, for example `dl-optimization-notes`.
2. Upload the **contents of this folder** (`index.html`, `css/`, `js/`, `README.md`) to the root of the repository. You can drag and drop them on the repository page via **Add file → Upload files**, or use git:

   ```bash
   git init
   git add .
   git commit -m "Interactive study slides for DL chapter 8.6-8.7"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

3. In the repository go to **Settings → Pages**. Under *Build and deployment* choose **Deploy from a branch**, pick branch **main** and folder **/ (root)**, then **Save**.
4. After a minute or two the site is live at `https://<your-username>.github.io/<repo-name>/`.

Links to single slides work too, e.g. `.../#/s-bfgs` or `.../#/s-bn-how`.

## View it locally

Opening `index.html` directly in a browser works. You can also run a tiny local server from this folder:

```bash
python -m http.server 8000
```

Then open http://localhost:8000.

## Notes

- If you edit the CSS or JS later, bump the `?v=` number on the `<link>`/`<script>` tags in `index.html` so browsers fetch the new files instead of cached ones.
- Explanations, diagrams and demos are original study notes. Equation numbers follow Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press, 2016). The course slides and textbook figures are not included.
