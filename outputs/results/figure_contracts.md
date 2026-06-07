# Figure Contracts

All figures are generated in Python with matplotlib and exported as editable
SVG, PDF, high-DPI PNG, and TIFF. The report is a source-backed analysis of
MADELON, not a decorative classification demo.

Core claims:

1. MADELON is balanced and contains many weak/noisy marginal features.
2. PCA, t-SNE, and UMAP reveal complementary linear and nonlinear structure.
3. Supervised feature selection is central because MADELON contains many probe
   and irrelevant dimensions.
4. Nonlinear models, especially RBF-SVM and boosting/tree ensembles, should be
   compared with linear and regularized baselines.
5. Bootstrap and sample-size experiments quantify stability rather than relying
   on a single accuracy number.

Leakage controls:

- The official train split is used for fitting and cross-validation.
- The official validation split is used only as the final test set.
- Scaling, PCA, feature selection, RFE, and model fitting are fit on training
  data before transforming/evaluating the final test set.
