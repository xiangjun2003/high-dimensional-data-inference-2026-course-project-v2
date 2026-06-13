# Advanced extension experiment summary

These experiments are isolated from the main project pipeline. They read the same
official MADELON train/validation split, but write only to `outputs/extensions`.

## Extension A: SCAD / nonconvex sparse logistic regression

Best sparse-logistic-family result by validation accuracy:
**SCAD Logistic via LLA**, lambda = 100.0, accuracy =
0.632, AUC = 0.642, selected
features = 2, BIC =
2711.5.

Best SCAD-only result: lambda = 100.0, accuracy =
0.632, AUC = 0.642, selected features =
2, BIC = 2711.5.

BIC-selected sparse logistic result: **SCAD Logistic via LLA**, lambda =
100.0, accuracy = 0.632, selected
features = 2, BIC =
2711.5.

## Extension B: Random Fourier Features / Nystroem kernel approximation

Best kernel-family result: **Exact RBF-SVM**, components =
NA, accuracy = 0.593, AUC =
0.631, train time = 3.524s.

## Extension C: Bayesian logistic regression with Laplace approximation

Best uncertainty result by AUC: **L2 Logistic MAP**, accuracy =
0.593, AUC = 0.623, Brier =
0.239, ECE = 0.031.

## Extension D: EM-GMM classifier and BIC model selection

Best GMM by validation accuracy: PCA-5,
K=8, covariance =
full, accuracy = 0.835,
BIC = 41138.6.

BIC-selected GMM: PCA-5,
K=8, covariance =
full, accuracy = 0.835,
BIC = 41138.6.

```json
{
  "config": {
    "scad_lambdas": [
      0.5,
      1.0,
      2.0,
      5.0,
      10.0,
      20.0,
      50.0,
      100.0,
      200.0
    ],
    "kernel_components": [
      50,
      100,
      200,
      500,
      1000
    ],
    "bayes_top_k": 20,
    "bayes_draws": 600,
    "gmm_pca_dims": [
      5,
      10,
      20
    ],
    "gmm_components": [
      1,
      2,
      3,
      5,
      8
    ],
    "gmm_covariance_types": [
      "diag",
      "tied",
      "full"
    ]
  },
  "best_sparse_family": {
    "method": "SCAD Logistic via LLA",
    "lambda": 100.0,
    "nonzero_features": 2,
    "bic": 2711.4847710733766,
    "train_time_sec": 0.18888254199555377,
    "accuracy": 0.6316666666666667,
    "auc": 0.6419222222222222,
    "f1": 0.6310517529215359
  },
  "best_scad_only": {
    "method": "SCAD Logistic via LLA",
    "lambda": 100.0,
    "nonzero_features": 2,
    "bic": 2711.4847710733766,
    "train_time_sec": 0.18888254199555377,
    "accuracy": 0.6316666666666667,
    "auc": 0.6419222222222222,
    "f1": 0.6310517529215359
  },
  "bic_selected_sparse_logistic": {
    "method": "SCAD Logistic via LLA",
    "lambda": 100.0,
    "nonzero_features": 2,
    "bic": 2711.4847710733766,
    "train_time_sec": 0.18888254199555377,
    "accuracy": 0.6316666666666667,
    "auc": 0.6419222222222222,
    "f1": 0.6310517529215359
  },
  "best_kernel_family": {
    "method": "Exact RBF-SVM",
    "feature_map": "implicit RBF",
    "classifier": "SVC",
    "components": null,
    "train_time_sec": 3.5244126670004334,
    "accuracy": 0.5933333333333334,
    "auc": 0.6307555555555555,
    "f1": 0.5892255892255892
  },
  "best_bayesian_family": {
    "method": "L2 Logistic MAP",
    "features": 20,
    "posterior_draws": 0,
    "accuracy": 0.5933333333333334,
    "auc": 0.6232444444444445,
    "f1": 0.5933333333333334,
    "brier": 0.238664984342356,
    "ece_10bin": 0.03088846703221336
  },
  "best_gmm_by_accuracy": {
    "method": "Class-conditional EM-GMM",
    "pca_components": 5,
    "gmm_components": 8,
    "covariance_type": "full",
    "bic": 41138.62451777332,
    "mean_em_lower_bound": -9.708603840480805,
    "converged": true,
    "train_time_sec": 0.15150849999918137,
    "accuracy": 0.835,
    "auc": 0.9191777777777779,
    "f1": 0.8379705400981997
  },
  "best_gmm_by_bic": {
    "method": "Class-conditional EM-GMM",
    "pca_components": 5,
    "gmm_components": 8,
    "covariance_type": "full",
    "bic": 41138.62451777332,
    "mean_em_lower_bound": -9.708603840480805,
    "converged": true,
    "train_time_sec": 0.15150849999918137,
    "accuracy": 0.835,
    "auc": 0.9191777777777779,
    "f1": 0.8379705400981997
  }
}
```
