# Analysis Summary

- Best base model: **Gradient Boosting** (accuracy = 0.755, AUC = 0.841).
- Best feature-selection result: **Random Forest + Random Forest**, k = 20
  (accuracy = 0.892, AUC = 0.964).
- Best PCA result: **RBF-SVM**, components = 5
  (accuracy = 0.835, AUC = 0.921).
- Best regularized logistic path result: **L1 Logistic**, C = 0.01
  (accuracy = 0.632, nonzero features = 2).
- Bootstrap best model used for error analysis: **Best FS: Random Forest + Random Forest (k=20)**.
- The official MADELON validation split is treated as the final test set.
