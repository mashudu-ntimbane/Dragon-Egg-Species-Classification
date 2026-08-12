# Dragon Egg Species Classification

Logistic regression project for **COMS4048A/COMS7063A – Statistical Foundations of Data Science**, University of the Witwatersrand.

## Overview

This project builds a multi-class classification model to predict a dragon's **species (Dragon, Wyvern, or Hydra)** from physical, environmental, and incubation characteristics of its egg. The work covers the full ML pipeline: data cleaning, feature engineering and selection, model tuning, and detailed evaluation — with a strong emphasis on interpretability.

## Objectives

- Explore and clean an egg-attribute dataset with known data quality issues (duplicate columns, invalid values, class imbalance).
- Engineer features to capture physical, transformed, and interaction effects.
- Select the most informative features using ANOVA F-tests and mutual information, while checking for multicollinearity (VIF).
- Train and tune a multinomial logistic regression classifier.
- Evaluate performance using confusion matrices, classification reports, ROC/AUC, and learning curves.
- Interpret model coefficients and odds ratios to explain species predictions.

## Dataset

500 observations, 8 features: `MASS`, `VOL`, `AGE`, `SPOT` (shell spot count), `NEST` (nesting environment), `TEMP` (incubation temperature), a duplicate target column (removed), and target `SPEC` (species). Classes are imbalanced: Hydra (52%), Wyvern (34.6%), Dragon (13.4%). A physically impossible negative mass value was removed during cleaning.

## Methodology

1. **Exploratory Data Analysis** – class distribution, statistical summaries, boxplots, histograms, scatter plots, correlation matrix, and nest-type analysis by species.
2. **Feature Engineering** – `DENSITY` (mass/volume), log transforms (`LOG_MASS`, `LOG_VOL`), temperature categories, polynomial terms (`MASS_SQUARED`, `VOL_SQUARED`), a `TEMP_VOLCANIC` interaction term, and egg size categories.
3. **Feature Selection** – ANOVA F-test and mutual information scoring, cross-checked against VIF for multicollinearity.
4. **Model Construction** – stratified 60/20/20 train/validation/test split; preprocessing pipeline with `StandardScaler` for numeric features and `OneHotEncoder` for `NEST`; multinomial logistic regression tuned over the regularisation parameter C.
5. **Evaluation** – confusion matrices (raw and normalised), per-class precision/recall/F1, one-vs-rest ROC/AUC, misclassification analysis, and learning curves.
6. **Interpretation** – per-class model coefficients, overall feature importance, and odds ratios.

## Key Results

| Metric | Value |
|---|---|
| Optimal regularisation | C = 0.1 |
| Training accuracy | 74.92% |
| Validation accuracy | 80.00% |
| Test accuracy | 72.00% |

**Per-species performance (test set):**

| Species | Precision | Recall | F1-Score |
|---|---|---|---|
| Dragon | 0.50 | 0.38 | 0.43 |
| Wyvern | 0.73 | 0.88 | 0.80 |
| Hydra | 0.78 | 0.60 | 0.68 |

- **SPOT** (shell spot count) was the single most important predictor overall, strongly distinguishing Hydra (positive association) from Dragon (negative association) eggs.
- **NEST_Swamp** and **LOG_MASS** were also highly influential, reflecting the importance of both nesting environment and egg size.
- The model performed best on the majority Wyvern and Hydra classes; the minority Dragon class was frequently confused with Hydra, largely due to class imbalance.
- Final feature set: `LOG_MASS`, `LOG_VOL`, `TEMP`, `SPOT`, `NEST`, `TEMP_VOLCANIC`.

## Tools & Libraries

Python (Google Colab) · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Statsmodels (VIF)

## Future Work

- Address class imbalance via SMOTE, resampling, or class-weighted training.
- Explore non-linear models (Random Forest, XGBoost, neural networks).
- Expand the dataset, particularly for the underrepresented Dragon class.

## Plots

Save each figure as an image (e.g. PNG) in an `images/` folder in the repo, then update the paths below to match your filenames.

| Figure | Description | Suggested filename |
|---|---|---|
| Fig. 1 | Species distribution by nest type (%), showing class balance and environmental clustering. | <img width="2970" height="1765" alt="nest_distribution" src="https://github.com/user-attachments/assets/910f4196-3227-4da5-a797-331dd2d45556" /> |
| Fig. 2 | Boxplots of MASS, VOL, AGE, TEMP, and SPOT by species. |<img width="5370" height="3543" alt="boxplots_by_species" src="https://github.com/user-attachments/assets/f54fec72-c6f1-49b5-8ec3-89af2fc95417" />|
| Fig. 3 | Histograms of MASS, VOL, AGE, TEMP, and SPOT distributions, coloured by species. | <img width="5370" height="2965" alt="distributions" src="https://github.com/user-attachments/assets/0f56202f-ea0a-4d6b-b82f-54e3f45a78e7" />|
| Fig. 4 | Pairwise scatter plots (MASS vs. VOL, TEMP distribution, shell spots, nest type) coloured by species. | <img width="4170" height="3542" alt="relationships" src="https://github.com/user-attachments/assets/b341da79-c827-4127-b16c-9f69bb4f1250" />|
| Fig. 5 | Pearson correlation matrix of numerical predictors and encoded target. | <img width="2331" height="2043" alt="correlation_matrix" src="https://github.com/user-attachments/assets/1010ffcd-1d5a-49c1-a0c6-cd715033ee2d" />|
| Fig. 6 | Nesting environment frequency by species. | <img width="2970" height="1765" alt="nest_distribution" src="https://github.com/user-attachments/assets/f5d42b58-29d8-405c-9d46-9a01dea42acd" />|
| Fig. 7 | Feature importance: ANOVA F-scores vs. Mutual Information scores by feature. |<img width="4770" height="1766" alt="feature_importance" src="https://github.com/user-attachments/assets/f820b7bf-db1e-4bd8-8862-2489f577ef59" /> |
| Fig. 8 | Validation accuracy vs. regularisation parameter C (model tuning curve). | <img width="855" height="552" alt="Model tuning" src="https://github.com/user-attachments/assets/caf7f7f1-6a96-4047-a18e-8ae740588ec8" /> |
| Fig. 9 | Confusion matrix (raw counts) — predicted vs. actual species. | <img width="2444" height="2107" alt="confusion_matrix" src="https://github.com/user-attachments/assets/28ab6419-01b0-416b-ae50-9540a498ab25" />|
| Fig. 10 | Normalised confusion matrix (% per true class). | <img width="2458" height="2107" alt="confusion_matrix_normalized" src="https://github.com/user-attachments/assets/edcc9917-adfb-440d-8b23-570ec83fc21b" />|
| Fig. 11 | Precision, recall, and F1-score by species. |<img width="3004" height="1638" alt="metrics_by_species" src="https://github.com/user-attachments/assets/695d145f-2d53-41dc-9a25-285aeb21ae7e" /> |
| Fig. 12 | One-vs-rest ROC curves for Dragon, Wyvern, and Hydra. | <img width="2578" height="2113" alt="roc_curves" src="https://github.com/user-attachments/assets/54ed5536-4f19-4d12-8e1e-1dcdcced7875" />|
| Fig. 13 | Learning curves: training vs. cross-validation accuracy by training set size. | <img width="2565" height="1638" alt="learning_curves" src="https://github.com/user-attachments/assets/6054bb72-60da-448d-9e45-596b25bd8bad" />|
| Fig. 14 | Model coefficients per feature, split by species class. |<img width="1790" height="590" alt="Feature coefficients" src="https://github.com/user-attachments/assets/0fcaa5cf-10c3-4b25-a4a1-77496a3e4adb" />|
| Fig. 15 | Overall feature importance ranked by mean absolute coefficient value. | <img width="2970" height="1765" alt="overall_feature_importance" src="https://github.com/user-attachments/assets/924f441b-36ec-4c3b-9e29-b9e8b983c354" />|


## Repository Contents

- `Egg_Project_Report.pdf` – Full written report with methodology, figures, and results.
- `images/` – Plot images referenced above.
- 'egg_report.ipynb'

## Authors

Ntimbane Permision · Nemangwela Unarine · Shabane Wamashudu · Namogane Mechem Matsepe

Supervised coursework — University of the Witwatersrand, School of Computer Science and Applied Mathematics.
