# High-dimensional feature selection and nonlinear classification on MADELON

[中文报告](#中文报告) | [English Summary](#english-summary)

Course project for **High-Dimensional Data Inference 2026**. This revised
project uses the UCI **MADELON** dataset instead of the previous RNA-Seq dataset,
because MADELON has stronger method differentiation and is designed for feature
selection under many irrelevant probe features.

Repository: `high-dimensional-data-inference-2026-course-project-v2`

---

## 中文报告

### 项目题目

**基于 MADELON 数据集的高维特征选择、降维可视化与非线性分类方法比较**

本项目完全使用 Python 和 `llm-26-cpu` conda 环境完成。绘图遵循 Nature-style scientific figure workflow：每张图先服务一个统计结论，使用 Python/matplotlib 生成白底、低饱和配色、可编辑 SVG/PDF、高 DPI PNG/TIFF，并保留源数据表。

### 1. Introduction

MADELON 是一个二分类高维人工数据集，包含 500 个连续特征，其中只有少数特征与类别结构相关，大量特征是 probe / irrelevant noise features。相比之前“几乎所有模型都达到 1.0”的数据，MADELON 更适合检验高维噪声特征环境下不同统计学习方法的差异。

本项目回答以下研究问题：

1. 原始 500 维空间中是否存在可视化可见的类别结构？
2. PCA、t-SNE、UMAP 能否揭示线性或非线性分离结构？
3. 面对大量无关特征，哪些特征选择方法更有效？
4. 线性模型与非线性模型在 MADELON 上差距有多大？
5. PCA 无监督降维与监督式特征选择哪个更有利于分类？
6. L1/L2/Elastic Net 正则化如何影响准确率和稀疏性？
7. 模型在样本量变化和 bootstrap 重采样下是否稳定？
8. 最佳模型的错误样本主要位于哪些低维区域？

### 2. Dataset

数据来源：UCI Machine Learning Repository, MADELON dataset。
项目下载官方 archive：`https://archive.ics.uci.edu/static/public/171/madelon.zip`

本项目使用官方 labeled splits：

| Split | Samples | Features | Class -1 | Class +1 | 用途 |
|---|---:|---:|---:|---:|---|
| Official train | 2,000 | 500 | 1,000 | 1,000 | 模型训练与交叉验证 |
| Official validation | 600 | 500 | 300 | 300 | 最终 held-out test |

官方 unlabeled test split 不用于本地评估。虽然 MADELON 不需要刻意强调 `p >> n`，但它仍是一个高维噪声特征问题：特征数 500、有效样本量中等、干扰特征很多，适合做特征选择、正则化、降维与非线性分类比较。

![Class distribution](outputs/figures/01_class_distribution.png)

### 3. Exploratory Data Analysis and Visualization

#### 3.1 Feature Distribution

![Feature mean distribution](outputs/figures/02_feature_mean_distribution.png)

![Feature standard deviation distribution](outputs/figures/03_feature_std_distribution.png)

![Class mean difference ranking](outputs/figures/04_class_mean_difference_ranking.png)

单变量边际差异排序显示，大量特征对类别区分贡献较弱。这正是 MADELON 的核心难点：只看单个特征通常不足以恢复类别结构。

#### 3.2 Dimensionality Reduction

![PCA explained variance](outputs/figures/05_pca_explained_variance.png)

![PCA 2D scatter](outputs/figures/06_pca_2d_scatter.png)

![PCA 3D scatter](outputs/figures/07_pca_3d_scatter.png)

![t-SNE 2D scatter](outputs/figures/08_tsne_2d_scatter.png)

![UMAP 2D scatter](outputs/figures/09_umap_2d_scatter.png)

PCA 二维/三维图中类别不能被简单线性投影完全分开；t-SNE 与 UMAP 显示一定局部结构，但仍存在混杂区域。这支持后续比较非线性模型、监督特征选择和 PCA 降维。

#### 3.3 Feature Redundancy and Ranking

![Top-50 feature correlation heatmap](outputs/figures/10_top50_feature_correlation_heatmap.png)

![Feature importance rankings](outputs/figures/11_feature_importance_rankings.png)

![Feature selection overlap](outputs/figures/12_feature_selection_overlap.png)

特征 ranking 覆盖计划中的所有方法：ANOVA F-test、Mutual Information、L1 Logistic、Random Forest importance、Permutation Importance、RFE。Top-feature overlap 图用于比较不同选择器是否选到相同特征集合。

### 4. Methods

#### 4.1 Dimensionality Reduction

- PCA：线性无监督降维，用于 explained variance、可视化与 PCA+k 分类实验。
- t-SNE：在 PCA-50 后做二维局部邻域可视化。
- UMAP：在 PCA-50 后做二维非线性结构可视化。

#### 4.2 Feature Selection

所有特征选择都只在官方 training split 上 fit，然后 transform official validation/test split，避免数据泄漏。

特征选择方法：

- ANOVA F-test
- Mutual Information
- L1 Logistic sparse coefficients
- Random Forest impurity-based importance
- Permutation Importance
- Recursive Feature Elimination, RFE

Top-k 设置完整覆盖计划：

$$k \in \{5,10,20,50,100,200,500\}$$

#### 4.3 Classification Models

基础模型完整覆盖必做和可选项：

- Majority Class baseline
- LDA
- Shrinkage LDA
- Regularized QDA
- Logistic Regression
- L1 Logistic
- L2 Logistic
- Elastic Net Logistic
- Linear SVM
- RBF-SVM
- KNN
- Random Forest
- Gradient Boosting
- Extra Trees
- MLP

#### 4.4 Evaluation

报告指标包括：

- Accuracy
- Balanced Accuracy
- AUC
- F1-score
- Training time
- Prediction time
- Selected feature count
- Bootstrap 95% CI
- Paired bootstrap accuracy difference

### 5. Experiments

#### Experiment 1: Base Model Comparison

使用全部 500 个特征训练基础分类器，并在官方 validation split 上评估。

![Base model accuracy](outputs/figures/13_model_accuracy_bar.png)

![Base model AUC](outputs/figures/14_model_auc_bar.png)

![Training time](outputs/figures/15_training_time_bar.png)

![Accuracy vs training time](outputs/figures/16_accuracy_vs_training_time.png)

| Model | Accuracy | Balanced Acc. | AUC | F1 | Train Time (s) |
|---|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.755 | 0.755 | 0.841 | 0.757 | 18.257 |
| Random Forest | 0.728 | 0.728 | 0.821 | 0.717 | 0.766 |
| Extra Trees | 0.682 | 0.682 | 0.754 | 0.676 | 0.308 |
| L1 Logistic | 0.632 | 0.632 | 0.642 | 0.631 | 6.863 |
| Elastic Net Logistic | 0.613 | 0.613 | 0.644 | 0.607 | 15.049 |
| RBF-SVM | 0.598 | 0.598 | 0.455 | 0.600 | 30.824 |
| MLP | 0.585 | 0.585 | 0.608 | 0.586 | 1.304 |
| Logistic Regression | 0.580 | 0.580 | 0.602 | 0.579 | 2.121 |
| Linear SVM | 0.580 | 0.580 | 0.598 | 0.580 | 1.710 |
| LDA | 0.577 | 0.577 | 0.604 | 0.569 | 1.300 |
| Shrinkage LDA | 0.577 | 0.577 | 0.618 | 0.567 | 0.294 |
| Regularized QDA | 0.567 | 0.567 | 0.589 | 0.575 | 13.172 |
| KNN | 0.557 | 0.557 | 0.567 | 0.515 | 0.547 |
| Majority Class | 0.500 | 0.500 | 0.500 | 0.000 | 0.000 |

基础模型结论：非线性集成模型显著优于简单线性模型，但直接使用全部 500 个特征时性能仍有限，说明无关特征对泛化有明显影响。

#### Experiment 2: Feature Selection + Classifier

完整大表见 [feature_selection_accuracy_table.csv](outputs/results/feature_selection_accuracy_table.csv) 和 [feature_selection_results.csv](outputs/results/feature_selection_results.csv)。

![Feature selection top-k accuracy](outputs/figures/18_feature_selection_topk_accuracy.png)

最佳结果：

| Feature Selection | Classifier | k | Accuracy | AUC | F1 |
|---|---|---:|---:|---:|---:|
| Random Forest importance | Random Forest | 20 | 0.892 | 0.964 | 0.892 |

核心发现：合适的监督特征选择显著提升性能。Random Forest / Permutation Importance 在 top 10-20 特征附近表现最好；当 k 增大到 500 时，噪声特征重新进入模型，性能下降。

#### Experiment 3: PCA + Classifier

完整表见 [pca_classification_results.csv](outputs/results/pca_classification_results.csv)。

![PCA dimension accuracy and AUC](outputs/figures/19_pca_dimension_accuracy_auc.png)

最佳 PCA 结果：

| Model | Components | Accuracy | AUC | F1 |
|---|---:|---:|---:|---:|
| PCA + RBF-SVM | 5 | 0.835 | 0.921 | 0.837 |

PCA 能降低维度和计算量，但它是无监督方法，不一定保留最有判别力的方向。结果上，PCA+RBF-SVM 明显强于全部特征上的 RBF-SVM，但仍低于最佳监督特征选择组合。

#### Experiment 4: Regularization Path

正则化路径完整表见 [regularization_path_results.csv](outputs/results/regularization_path_results.csv)。

![Regularization path](outputs/figures/20_regularization_path.png)

L1 Logistic 在 `C=0.01` 时只保留 2 个非零特征，accuracy 为 0.632。它展示了稀疏性与性能之间的取舍：强 L1 正则能产生极简模型，但 MADELON 的多变量结构意味着过少特征会限制分类性能。

#### Experiment 5: Sample-size Sensitivity

样本量敏感性实验完整表见 [sample_size_sensitivity.csv](outputs/results/sample_size_sensitivity.csv)。

$$n \in \{200,500,1000,1500,2000\}$$

每个样本量重复 5 次分层抽样，比较 Logistic Regression、L1 Logistic、Linear SVM、RBF-SVM、Random Forest。

![Sample-size learning curve](outputs/figures/21_sample_size_learning_curve.png)

该实验用于展示随着训练样本增加，非线性模型和树模型的稳定性变化，以及小样本下高维噪声特征对泛化的影响。

#### Experiment 6: Bootstrap Stability and Error Analysis

![Bootstrap accuracy CI](outputs/figures/22_bootstrap_accuracy_ci.png)

![Paired bootstrap accuracy difference](outputs/figures/23_paired_bootstrap_accuracy_diff.png)

Bootstrap 结果：

| Model | Accuracy | 95% Bootstrap CI |
|---|---:|---:|
| Best FS: Random Forest + Random Forest (k=20) | 0.892 | [0.865, 0.915] |
| Best PCA: RBF-SVM (PCs=5) | 0.835 | [0.805, 0.863] |
| Gradient Boosting | 0.755 | [0.720, 0.787] |
| Random Forest | 0.728 | [0.692, 0.762] |
| Extra Trees | 0.682 | [0.648, 0.717] |
| Majority Class | 0.500 | [0.460, 0.538] |

最佳模型混淆矩阵和错误样本投影：

![Best model confusion matrix](outputs/figures/17_best_model_confusion_matrix.png)

![Error samples in embeddings](outputs/figures/24_error_samples_in_embeddings.png)

错误样本在 PCA/t-SNE/UMAP 中主要位于类别混杂区域，说明模型错误不是随机分布，而与低维结构中的边界/重叠区域有关。

### 6. Reproducibility

本项目已用本机 `llm-26-cpu` conda 环境验证。运行命令：

```bash
conda activate llm-26-cpu
python scripts/validate_environment.py
python scripts/run_project.py
```

也可以直接使用本机已验证解释器：

```bash
/Users/xiangjun/opt/anaconda3/envs/llm-26-cpu/bin/python scripts/validate_environment.py
/Users/xiangjun/opt/anaconda3/envs/llm-26-cpu/bin/python scripts/run_project.py
```

输出目录：

- `outputs/results/*.csv`：全部实验结果表
- `outputs/results/*.md`：analysis summary 和 figure contract
- `outputs/figures/*.png`：GitHub 预览图
- `outputs/figures/*.svg`：可编辑矢量图
- `outputs/figures/*.pdf`：PDF 输出
- `outputs/figures/*.tiff`：Nature-style high-DPI raster export

### 7. Repository Structure

```text
src/hddi26_madelon/      Python package for data, analysis, plotting
scripts/                 One-click run and environment validation scripts
data/raw/                Downloaded UCI files, ignored by git
data/processed/          Processed CSV cache, ignored by git
outputs/results/         Source-backed tables and summaries
outputs/figures/         SVG/PDF/PNG/TIFF figures
notebooks/               Optional notebook workspace
reports/                 Optional written report drafts
```

### 8. Conclusion

MADELON 更适合作为本课程项目数据集：它不会让所有模型都接近 1.0，且能清楚展示高维噪声特征下不同方法的差异。基础模型中 Gradient Boosting 最强，但监督特征选择后 Random Forest 在 top 20 features 上达到最佳性能，accuracy 0.892、AUC 0.964。PCA+RBF-SVM 说明无监督降维也能显著提升核方法表现，但仍弱于监督式特征选择。L1/Elastic Net 提供了可解释的稀疏模型，但过强稀疏会牺牲预测性能。Bootstrap 与错误样本投影进一步表明最佳模型优势稳定，错误主要集中在低维混杂区域。

---

## English Summary

This revised course project uses the UCI MADELON dataset to study high-dimensional feature selection, dimensionality reduction, nonlinear classification, and model stability under many irrelevant probe features.

The project implements all required and optional components from the plan: PCA/t-SNE/UMAP visualization, ANOVA/MI/L1/RF/permutation/RFE feature selection, LDA/Shrinkage LDA/regularized QDA/logistic/SVM/KNN/Random Forest/Gradient Boosting/Extra Trees/MLP classifiers, feature-selection top-k experiments, PCA+k experiments, regularization paths, sample-size sensitivity, bootstrap confidence intervals, paired bootstrap comparisons, and embedding-based error analysis.

Key result: the best base model is Gradient Boosting with accuracy 0.755, while the best overall model is Random Forest after Random-Forest feature selection with k=20, reaching accuracy 0.892 and AUC 0.964 on the official validation split used as the final test set.
