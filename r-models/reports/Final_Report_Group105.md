# Predicting Remittance Transfer Costs from the United States to Latin America and the Caribbean

*Team Project: Number 105  |  ISyE 7406 – Fall 2026*

*Oscar Borge Donado – GT ID: 903965024, Yonny Leyva – GT ID: 903942855*


# Problem Statement

Every year, millions of Latino families in the United States send money back to relatives in Latin America and the Caribbean (LAC). These transfers are a primary source of income for receiving households. In 2023, the United States sent over $150 billion in remittances to LAC countries, making it the largest single source of remittance inflows into the region. Despite their scale and importance, the cost of sending that money remains stubbornly high. The United Nations set a target of 3% average cost per transaction under SDG 10.c, yet the corridors in this study average 5.44% as of 2025 Q1, with some destinations such as Cuba (12.0%) and Guyana (8.7%) far exceeding that benchmark.

The goal of this project is to understand why costs are where they are, and to build tools that help senders find lower-cost options. The World Bank Remittance Prices Worldwide (RPW) database tracks transaction-level pricing across hundreds of global corridors, capturing provider type, payment instrument, transfer speed, fees, exchange rate margins, and destination country for each offering. This report completes the three-part plan laid out in the midterm: (1) exploratory data analysis and feature selection (midterm work, condensed below); (2) fitting and comparing regression models, a classification task at the 3% threshold, and destination clustering (this report's new work); and (3) a deployed web application that lets a money sender get a predicted cost for their own transfer. All analysis was conducted in Python using pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, and statsmodels; the web application uses FastAPI and React/TypeScript. Figures are described in the Appendix.


# Proposed Dataset and Data Preprocessing

The dataset is a transaction-level extract from the World Bank Remittance Prices Worldwide (RPW) database, subset to United States-to-LAC corridors: 12,397 rows and 23 columns, one row per provider offering per corridor per quarter, spanning period 2016_2Q through 2025_1Q. speed actual and pickup method were lower-cased and stripped to merge near-duplicate categories caused by inconsistent capitalization; period was parsed into integer year and quarter columns for the trend analysis and for the temporal train/test split used throughout this report. firm_type was consolidated (the two Money Transfer Operator variants merged into one level) and the 11 sparse Non-Bank FI rows were dropped, leaving 12,386 rows for all modeling tasks below. Only the cc2 * fields have missing values (235 rows, 1.9%), used to engineer the is_non_transparent proxy flag; the free-text Standard Note is missing for 69.2% of rows (expected, since it is only populated when there is something unusual to disclose). Every other column, including the target cc1 total cost %, is fully populated.


# Proposed Methodology

This project's methodology is grounded in the course's model/variable selection framework (ISyE 7406, Prof. Yajun Mei). The midterm phase used information criteria (AIC/BIC via forward stepwise search), shrinkage-based selection (Ridge/LASSO), and PCA to identify a final predictor set: destination_name, firm_type_clean, payment_instrument_group, access point, receiving network coverage, speed actual, pickup method, cc1 fx margin, year, and is_non_transparent, encoded via one-hot (drop_first=True) for categoricals and standardized for numerics (42 encoded features). This final report builds on that feature set with three further analyses, all evaluated on an identical pre-2023 train / 2023+ test temporal split (8,228 train rows, 4,158 test rows): (1) a regression model comparison (Linear Regression, Ridge, LASSO, Random Forest) scored by RMSE, MAE, and R²; (2) a binary classification task that labels a transaction "high cost" when cc1 total cost % ≥ 3% (the SDG 10.c threshold), compared across Logistic Regression, Linear Discriminant Analysis (LDA), and Random Forest, scored by accuracy, precision, recall, and F1; and (3) unsupervised clustering of the 16 LAC destination countries by their mean cost profile (total cost, FX margin, fee, transparency rate), using both k-means (k chosen via elbow and silhouette score) and hierarchical (Ward-linkage) clustering. The same production LassoCV model (refit on all available data) is shipped in a deployed FastAPI/React web application described in its own section below.


# Analysis and Results


## Target Variable

The target cc1 total cost % has a mean of 5.34%, a median of 4.71%, and a standard deviation of 3.03%, with a range of −4.09% to 34.0%. It is right-skewed (skewness = 1.88, excess kurtosis = 6.33), driven by a small number of high-cost providers in outlier corridors. The 25th percentile (3.50%) sits just above the SDG 3% target, meaning roughly three-quarters of all transactions fail to meet it.

![Figure 1. Distribution of cc1 total cost % (histogram and boxplot), with the 3% SDG target marked.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/01_target_distribution.png)

*Figure 1. Distribution of cc1 total cost % (histogram and boxplot), with the 3% SDG target marked.*


## Cost by Destination Country

Destination is the single strongest driver of cost variation in the dataset. Cuba averages 12.0% and Guyana 8.9%, both more than double the SDG target. At the other end, Honduras (3.9%), Ecuador (4.0%), and El Salvador (4.0%) are closest to the threshold. Mexico, the largest-volume corridor, averages 4.6%. The roughly 8-percentage-point spread between the cheapest and most expensive corridors far exceeds the differences attributable to firm type or payment instrument, and motivates the corridor clustering analysis below.

![Figure 3. Mean cc1 total cost % by destination, sorted descending.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/03_cost_by_destination.png)

*Figure 3. Mean cc1 total cost % by destination, sorted descending.*


## Cost by Firm Type

Banks are not meaningfully cheaper than Money Transfer Operators (MTOs) in this dataset: recomputed directly from the modeling subset, Banks average 5.34% and MTOs average 5.34%, a negligible difference, and both sit well above the 3% SDG target. MTOs dominate the market by volume (10,707 of 12,386 modeling rows, 86%).

![Figure 2. cc1 total cost % by firm type.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/02_cost_by_firm_type.png)

*Figure 2. cc1 total cost % by firm type.*


## Cost by Payment Instrument

Payment instrument differentiates cost meaningfully: cash-based transfers show higher and more variable costs than bank account transfers or digital instruments, and credit card transfers carry a distinct premium, likely reflecting card processing fees.

![Figure 4. cc1 total cost % by payment instrument group.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/04_cost_by_payment_instrument.png)

*Figure 4. cc1 total cost % by payment instrument group.*


## Cost Over Time

Mean cost has been essentially flat across 2016-2025, fluctuating between 4.96% (2020) and 5.59% (2024), with no sustained downward trend toward the SDG target. year is a low-signal predictor on its own, retained mainly because it defines the temporal train/test split used throughout this report.

![Figure 5. Mean cc1 total cost % by year, 2016-2025.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/05_cost_by_year.png)

*Figure 5. Mean cc1 total cost % by year, 2016-2025.*


## Transparency Flag

The engineered is_non_transparent flag (cc2 total cost % missing as a proxy for non-disclosure) flags 1.9% of rows, which show a modestly lower mean cost than transparent rows (4.80% vs. 5.35%). The proxy is imperfect—81% of flagged rows carry a "send limits" disclosure note rather than an exchange-rate non-disclosure note—so it is retained as a candidate feature with this limitation stated explicitly.

![Figure 6. cc1 total cost % by transparency proxy (cc2 total cost % missing vs. present).](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/06_transparency_comparison.png)

*Figure 6. cc1 total cost % by transparency proxy (cc2 total cost % missing vs. present).*


## Correlation Structure

Among numeric fields, cc1 fx margin shows the strongest correlation with the target (r = 0.71), as expected since FX margin is a direct cost component. cc1 lcu fx rate and inter lcu bank fx correlate weakly (r ≈ 0.06-0.07) and are currency-scale fields excluded from the predictor set. All cc2 * fields correlate highly with the target by construction (r = 0.91 for cc2 total cost %) and are excluded as leakage variables.

![Figure 7. Pearson correlation heatmap of numeric RPW fields plus year.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/07_correlation_heatmap.png)

*Figure 7. Pearson correlation heatmap of numeric RPW fields plus year.*


## Feature Selection (Information Criteria, Shrinkage, and PCA)

Feature selection used three complementary methods, detailed in full in the midterm's EDA & Feature Selection report (r-models/reports/EDA_FeatureSelection_Group105), condensed here. Forward stepwise selection by BIC chose 9 of 10 candidate blocks, in order: cc1 fx margin, pickup method, payment_instrument_group, destination_name, access point, receiving network coverage, speed actual, year, is_non_transparent (firm_type_clean was the one block not selected). LassoCV (5-fold CV, alpha = 0.001) zeroed out 6 of 42 encoded features, with the largest surviving coefficients on pickup method_own/partner bank account, destination_name_Cuba, and access point_Bank branch,Internet. RidgeCV (alpha = 1.15) shrinks all coefficients without zeroing any out. PCA on the numeric block (cc1 fx margin, year, is_non_transparent) confirmed the three features are near-orthogonal, so dimensionality reduction was not useful. Both the information-criterion and shrinkage methods converge on the same dominant predictors—destination, payment instrument/pickup method, and FX margin—giving confidence the final feature set (destination_name, firm_type_clean, payment_instrument_group, access point, receiving network coverage, speed actual, pickup method, cc1 fx margin, year, is_non_transparent; 42 encoded features) is robust.

![Figure 8. AIC/BIC vs. number of features in forward stepwise selection.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/08_aic_bic_vs_nvars.png)

*Figure 8. AIC/BIC vs. number of features in forward stepwise selection.*

![Figure 9. LASSO regularization path (coefficients vs. alpha).](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/09_lasso_coef_path.png)

*Figure 9. LASSO regularization path (coefficients vs. alpha).*

![Figure 10. Final LassoCV coefficients, top 20 by magnitude.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/10_lasso_coefficients_bar.png)

*Figure 10. Final LassoCV coefficients, top 20 by magnitude.*

![Figure 11. PCA scree plot of the numeric candidate block.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures/11_pca_scree.png)

*Figure 11. PCA scree plot of the numeric candidate block.*


## Regression Model Comparison

Four regression models were fit on the identical final feature set (one-hot encoded categoricals + standardized numerics), trained on pre-2023 rows (n = 8,228) and evaluated on the 2023+ holdout (n = 4,158): plain Linear Regression, RidgeCV, LassoCV, and a Random Forest Regressor (400 trees). RMSE is the primary metric, with MAE and R² reported alongside.

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| Linear Regression | 1.9352 | 1.2216 | 0.6724 |
| LASSO | 1.9468 | 1.2159 | 0.6685 |
| Ridge | 1.9595 | 1.218 | 0.6641 |
| Random Forest | 1.9782 | 1.0409 | 0.6577 |

*Table 5. Regression model comparison, 2023+ holdout (sorted by test RMSE).*


Linear Regression achieves the lowest test RMSE (1.935, R² = 0.672), essentially tied with LASSO (1.947) and Ridge (1.960); the Random Forest has the highest test RMSE (1.978) but the lowest MAE, suggesting it fits the bulk of typical transactions slightly better while being less able to extrapolate to the handful of extreme-cost corridors (Cuba, Guyana) that drive RMSE's squared-error penalty. This is consistent with the shipped production model (LassoCV refit on all data, holdout RMSE ≈ 1.95, R² ≈ 0.67 as reported in train_model.py/model/metadata.json): linear, regularized models capture this dataset's structure about as well as a flexible nonparametric model does, which is itself informative—it suggests the relationship between the selected features and cost is close to additive and linear once destination, payment instrument, and FX margin are already encoded, with limited extra signal left for a tree ensemble to exploit.

![Figure 12. Residuals vs. fitted values and residual distribution, Linear Regression (best test-RMSE regression model), 2023+ holdout.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures_final/12_regression_residuals.png)

*Figure 12. Residuals vs. fitted values and residual distribution, Linear Regression (best test-RMSE regression model), 2023+ holdout.*

The residual plot for Linear Regression (Figure 12) shows residuals concentrated near zero with a long right tail, mirroring the right-skewed target distribution: a handful of very high-cost transactions (predominantly Cuba and Guyana) are systematically under-predicted, since no linear model can fully capture an outlier destination's level shift from the pooled coefficient estimate alone.

![Figure 13. Top 15 standardized LASSO coefficients by magnitude, compared to Ridge on the same features.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures_final/13_lasso_ridge_coefficients.png)

*Figure 13. Top 15 standardized LASSO coefficients by magnitude, compared to Ridge on the same features.*

| Feature | LASSO coef. | Ridge coef. |
|---|---|---|
| access point_Bank branch,Internet | 5.3421 | 4.8679 |
| destination_name_Cuba | 4.9537 | 4.7053 |
| pickup method_own/partner bank account | 3.9218 | 3.7035 |
| payment_instrument_group_Credit Card | 2.5929 | 2.5756 |
| pickup method_bank account (same/partner bank) | 2.3272 | 2.3127 |
| cc1 fx margin | 2.0185 | 2.0214 |
| destination_name_Guyana | 1.1708 | 1.2275 |
| destination_name_Haiti | 1.1285 | 1.1849 |
| receiving network coverage_Low | -0.8589 | -0.8951 |
| destination_name_Nicaragua | 0.7448 | 0.8067 |
| destination_name_Panama | 0.7101 | 0.7767 |
| destination_name_Mexico | -0.6844 | -0.6602 |
| payment_instrument_group_Cash | 0.6526 | 0.6667 |
| payment_instrument_group_Multiple | 0.6136 | 0.6256 |
| destination_name_Guatemala | -0.6119 | -0.5635 |

*Table 6. Top 15 standardized LASSO coefficients and matching Ridge coefficients.*


LASSO's largest standardized coefficients line up with the midterm's feature-selection findings (destination and pickup-method dummies, cc1 fx margin); Ridge's coefficients on the same features are directionally consistent but uniformly smaller in magnitude, reflecting its smoother L2 shrinkage versus LASSO's sparser L1 penalty.

![Figure 14. Random Forest regressor feature importance (mean decrease in impurity), top 15.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures_final/14_rf_feature_importance.png)

*Figure 14. Random Forest regressor feature importance (mean decrease in impurity), top 15.*

| Feature | Importance |
|---|---|
| cc1 fx margin | 0.6151 |
| year | 0.0556 |
| speed actual_next day | 0.0539 |
| destination_name_Cuba | 0.0367 |
| access point_Bank branch,Internet | 0.0357 |
| payment_instrument_group_Credit Card | 0.0355 |
| speed actual_less than one hour | 0.0212 |
| pickup method_own/partner bank account | 0.0134 |
| pickup method_cash | 0.0131 |
| access point_Internet | 0.0125 |
| speed actual_3-5 days | 0.01 |
| payment_instrument_group_Cash | 0.0083 |
| receiving network coverage_Medium | 0.0077 |
| payment_instrument_group_Multiple | 0.0076 |
| destination_name_Haiti | 0.0075 |

*Table 7. Random Forest regressor feature importance, top 15.*


The Random Forest's impurity-based feature importances point to the same handful of dominant predictors—cc1 fx margin and the Cuba/Guyana destination dummies rank highest—corroborating the linear models' coefficients with an entirely different, nonparametric method.


## Classification at the 3% SDG Threshold

The target was binarized at the SDG 10.c threshold (cc1 total cost % ≥ 3% = "high cost"); the resulting label is heavily imbalanced toward "high cost" (86.4% of the pre-2023 training rows, 85.4% of the 2023+ test rows), consistent with the EDA finding that three-quarters of transactions already exceed the target. Logistic Regression, LDA, and a Random Forest Classifier (400 trees) were trained on the same encoded feature set and the same temporal split.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.88 | 0.8991 | 0.9682 | 0.9324 |
| Logistic Regression | 0.8446 | 0.9161 | 0.9006 | 0.9083 |
| LDA | 0.8247 | 0.8986 | 0.8958 | 0.8972 |

*Table 8. Classification comparison at the 3% SDG threshold, 2023+ holdout.*


Random Forest is the best classifier by F1 (F1 = 0.932, accuracy = 0.880, precision = 0.899, recall = 0.968). Given the class imbalance, accuracy alone would be a misleading metric (a trivial "always predict high cost" classifier would already score roughly 85% accuracy); F1 and recall are the more meaningful comparison, and the Random Forest's higher recall indicates it misses fewer of the (minority) low-cost transactions that are most actionable for a sender trying to find a cheaper option.

![Figure 15. Confusion matrices by firm type, Random Forest (best-F1 classifier), 2023+ holdout.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures_final/15_confusion_by_firmtype.png)

*Figure 15. Confusion matrices by firm type, Random Forest (best-F1 classifier), 2023+ holdout.*

| Firm Type | n | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Bank | 701 | 0.8402 | 0.8651 | 0.9528 | 0.9068 |
| Money Transfer Operator | 3457 | 0.8881 | 0.9058 | 0.9711 | 0.9373 |

*Table 9. Classification performance by firm type, Random Forest.*


Broken out by firm type, the Random Forest classifier performs differently for Banks (F1 = 0.907, n = 701) versus Money Transfer Operators (F1 = 0.937, n = 3457), indicating the model is not simply learning a firm-type shortcut but is picking up on corridor/instrument-level cost signal that generalizes across provider types.

![Figure 16. Precision/recall/F1 by destination country, Random Forest, 2023+ holdout.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures_final/16_classification_by_destination.png)

*Figure 16. Precision/recall/F1 by destination country, Random Forest, 2023+ holdout.*

| Destination | n | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Cuba | 56 | 1.0 | 1.0 | 1.0 | 1.0 |
| Guyana | 222 | 0.9955 | 0.9955 | 1.0 | 0.9977 |
| Haiti | 142 | 0.9859 | 0.9833 | 1.0 | 0.9916 |
| Jamaica | 197 | 0.9492 | 0.9492 | 1.0 | 0.974 |
| Nicaragua | 194 | 0.9381 | 0.9505 | 0.983 | 0.9665 |
| Brazil | 270 | 0.9111 | 0.9101 | 1.0 | 0.9529 |
| Colombia | 394 | 0.901 | 0.9182 | 0.9775 | 0.9469 |
| El Salvador | 460 | 0.9022 | 0.9274 | 0.9623 | 0.9445 |
| Dominican Republic | 267 | 0.8652 | 0.8851 | 0.9585 | 0.9204 |
| Costa Rica | 266 | 0.8534 | 0.8802 | 0.9552 | 0.9161 |
| Guatemala | 267 | 0.8502 | 0.8824 | 0.9459 | 0.913 |
| Mexico | 446 | 0.8386 | 0.8602 | 0.9623 | 0.9084 |
| Ecuador | 327 | 0.8379 | 0.875 | 0.9418 | 0.9072 |
| Panama | 143 | 0.8112 | 0.8203 | 0.9633 | 0.8861 |
| Peru | 201 | 0.8259 | 0.8535 | 0.9178 | 0.8845 |
| Honduras | 306 | 0.7908 | 0.8058 | 0.9573 | 0.875 |

*Table 10. Classification performance by destination, Random Forest.*


Per-destination performance is uneven: Cuba has the highest F1 (1.000, n = 56), while Honduras has the lowest (F1 = 0.875, n = 306). The weakest destinations tend to be those with either very few 2023+ holdout observations or a cost distribution that straddles the 3% boundary closely, so a small number of misclassifications has an outsized effect on that corridor's F1.


## Corridor Clustering

The 16 LAC destination countries were aggregated to a country-level profile (mean total cost, mean FX margin, mean fee in local currency units, and transparency rate), standardized, and clustered with both k-means and hierarchical (Ward-linkage) methods. k for k-means was chosen by jointly inspecting the elbow (inertia) curve and the silhouette score across k = 2..7.

![Figure 17. K-means inertia (elbow) and silhouette score vs. number of clusters k, for the 16 LAC destinations.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures_final/17_kmeans_elbow_silhouette.png)

*Figure 17. K-means inertia (elbow) and silhouette score vs. number of clusters k, for the 16 LAC destinations.*

| k | Inertia | Silhouette |
|---|---|---|
| 2 | 38.259 | 0.5779 |
| 3 | 19.432 | 0.4442 |
| 4 | 9.946 | 0.3864 |
| 5 | 7.409 | 0.2912 |
| 6 | 5.348 | 0.2508 |
| 7 | 3.559 | 0.2604 |

*Table 11. K-means inertia and silhouette score by k.*


Silhouette score is maximized at k = 2 (silhouette = 0.578) and decreases monotonically for larger k, but at k=2 the partition is dominated by a single extreme outlier: Cuba (mean cost 11.99%, mean FX margin 2.84) forms its own cluster, versus a single large cluster of the remaining 15 destinations (mean cost 5.37%). This is a legitimate finding in its own right—Cuba's cost profile is simply far enough from every other destination's that no amount of recombination produces a better-separated two-group split—but it is not very informative about structure *among* the other 15 destinations. For that finer-grained interpretation, we also report the k=4 partition (lower, but still reasonable, silhouette) as supplementary detail.

![Figure 18. K-means clusters (k=2) of the 16 LAC destinations, mean FX margin vs. mean total cost %.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures_final/18_cluster_scatter.png)

*Figure 18. K-means clusters (k=2) of the 16 LAC destinations, mean FX margin vs. mean total cost %.*

| Destination | Mean Cost % | Mean FX Margin | Mean Fee (LCU) | Transparency Rate | k-means Cluster (k=2) | k-means Cluster (k=4) |
|---|---|---|---|---|---|---|
| Honduras | 3.91 | 0.27 | 7.27 | 0.983 | 0 | 0 |
| Ecuador | 3.95 | 0.0 | 7.89 | 0.983 | 0 | 0 |
| El Salvador | 3.97 | 0.0 | 7.94 | 0.98 | 0 | 0 |
| Peru | 4.22 | 0.6 | 7.23 | 0.989 | 0 | 0 |
| Guatemala | 4.53 | 1.34 | 6.38 | 0.982 | 0 | 0 |
| Panama | 4.61 | 0.0 | 9.21 | 0.966 | 0 | 0 |
| Mexico | 4.61 | 1.52 | 6.18 | 0.99 | 0 | 3 |
| Nicaragua | 4.63 | 0.0 | 9.25 | 0.974 | 0 | 0 |
| Haiti | 4.97 | 0.12 | 9.7 | 0.985 | 0 | 0 |
| Colombia | 5.79 | 2.51 | 6.57 | 0.988 | 0 | 3 |
| Costa Rica | 6.15 | 2.24 | 7.83 | 0.997 | 0 | 3 |
| Brazil | 6.33 | 3.21 | 6.23 | 0.999 | 0 | 3 |
| Dominican Republic | 6.62 | 2.39 | 8.45 | 0.95 | 0 | 1 |
| Jamaica | 7.21 | 3.38 | 7.67 | 0.976 | 0 | 1 |
| Guyana | 8.98 | 4.84 | 8.28 | 0.955 | 0 | 1 |
| Cuba | 11.99 | 2.84 | 18.3 | 1.0 | 1 | 2 |

*Table 12. Destination-level cost profile and cluster assignments.*


At k=4, the finer partition separates into four interpretable cost tiers (listed from cheapest to most expensive mean total cost): cluster 0 (mean cost 4.35%): Ecuador, El Salvador, Guatemala, Haiti, Honduras, Nicaragua, Panama, Peru; cluster 1 (mean cost 7.61%): Dominican Republic, Guyana, Jamaica; cluster 2 (mean cost 11.99%): Cuba; cluster 3 (mean cost 5.72%): Brazil, Colombia, Costa Rica, Mexico. The cheapest tier groups destinations with low FX margins and no (or near-zero) reported FX margin component; the most expensive non-Cuba tier (Brazil, Colombia, Costa Rica, Mexico and similar) combines moderate-to-high FX margins with high transaction volume; and Dominican Republic/Guyana/Jamaica form a distinct mid-to-high-cost tier with the highest FX margins outside of Cuba. This is consistent with the regression and classification results above: FX margin and destination are the dominant cost drivers, and the clustering shows that destinations group along almost exactly that same axis.

![Figure 19. Hierarchical (Ward linkage) clustering dendrogram of destination cost profiles.](/Users/yonnyleyvasuarez/repos/copilot-worktrees/isye-7406-g105/yllsuarez-ideal-eureka/r-models/reports/figures_final/19_dendrogram.png)

*Figure 19. Hierarchical (Ward linkage) clustering dendrogram of destination cost profiles.*

The hierarchical clustering dendrogram (Ward linkage, Figure 19) tells the same story from a different angle: Cuba joins the rest of the tree at by far the largest linkage distance (confirming it as a singleton outlier at low k), while the remaining destinations merge into recognizable sub-groups consistent with the k=4 k-means partition above, giving convergent evidence from two different clustering methods for the same underlying corridor groupings.


# Web Application

To put the modeling work in front of an actual money sender, the project ships a small full-stack web application: a FastAPI backend and a React/TypeScript single-page frontend. The backend (web-service/backend/main.py) loads the shipped production model (the LassoCV pipeline from r-models/model/model.pkl, refit on all available data) and its metadata.json at startup, and exposes two synchronous REST endpoints: GET /metadata, which returns the feature list, valid category values, and the historical mean FX margin by destination (used to pre-fill a sensible default), and POST /predict, which accepts a sender's inputs (destination, firm type, payment instrument, access point, receiving network coverage, speed, pickup method, FX margin, year, and a transparency flag) and returns the model's predicted cc1 total cost % (plus an estimated fee in USD if a transfer amount is supplied). There is no background task queue or async job system—the architecture is deliberately a simple, synchronous request/response flow, appropriate for a single fast model-inference call.

The frontend (web-service/frontend/src/App.tsx) is a single form that mirrors the backend's input schema: dropdowns for the categorical fields (populated from GET /metadata so the UI never drifts out of sync with what the model was trained on), a numeric field for cc1 fx margin pre-filled with the historical destination-level average default, and a submit action that calls POST /predict and renders the predicted cost percentage (and estimated fee) back to the user. Keeping the categories server-driven means that if the model is ever retrained on an updated dataset with a new destination or payment instrument category, the frontend does not need a code change to stay correct.

web-service/sync_model.sh is the glue between the two solutions in this monorepo: it copies the trained artifacts (model.pkl and metadata.json) from r-models/model/ into web-service/backend/model/, so the backend always serves whatever model r-models/train_model.py most recently produced, without needing the backend to import or depend on the r-models training code directly.


# Conclusions and Next Steps


## Lessons Learned

- Data quality is uneven across fields that look superficially similar: the cc1/cc2 comparator fields are mechanically correlated by construction, and naive inclusion of cc2 * fields would have silently leaked the target into the feature set. Catching this required reading the RPW database's own field definitions, not just running a correlation matrix.
- The is_non_transparent proxy is a good example of a feature that looked clean in isolation but turned out to be confounded (send-limit disclosures vs. true FX non-disclosure) once cross-tabulated against the free-text note field. Engineered features built from a single missingness pattern deserve this kind of secondary validation before being trusted.
- Linear/regularized models (LASSO, Ridge, plain OLS) and a 400-tree Random Forest landed on almost the same test RMSE in this project. That was initially surprising—we expected the Random Forest to meaningfully outperform on nonlinearities—but in retrospect it is explained by how much of the signal was already captured by the carefully selected categorical feature set (destination, payment instrument, pickup method) before any model-side nonlinearity could add value. A worthwhile extension would be engineering interaction terms (e.g. destination × payment instrument) explicitly for the linear models, to see whether that closes the remaining gap with the Random Forest or whether the Random Forest's flexibility is already capturing those interactions implicitly.
- With only 16 destination countries, clustering results are sensitive to the exact choice of k and to a single extreme outlier (Cuba) dominating the silhouette-optimal partition. Reporting both the statistically "optimal" k and a secondary, more interpretable k was a deliberate choice to avoid letting a single metric understate the richer structure visible in the dendrogram.
- If we were to start over, we would build the FastAPI/React web application earlier in the project rather than after the modeling work, since having a live, interactive view into the model's predictions surfaced a few feature-encoding edge cases (e.g. the normalization of speed actual and pickup method casing) that were easy to overlook while working purely from notebook output.


## Key Findings

- The mean total cost of US-to-LAC remittances (5.44%) has remained well above the UN SDG 10.c target of 3% throughout 2016-2025, with no meaningful downward trend.
- Destination country is the dominant driver of cost variation, with an 8+ percentage point spread between the cheapest (Honduras, 3.9%) and most expensive (Cuba, 12.0%) corridors.
- Linear Regression achieved the best regression performance on the 2023+ holdout (RMSE = 1.935, R² = 0.672), closely matched by LASSO and Ridge; the Random Forest did not meaningfully outperform the linear models, suggesting the relationship is close to additive once destination, payment instrument, and FX margin are encoded.
- At the 3% SDG threshold, Random Forest was the best classifier (F1 = 0.932), though performance varies meaningfully by destination, reflecting both class imbalance and genuine cross-corridor heterogeneity.
- Clustering the 16 destinations by cost profile isolates Cuba as a singleton outlier cluster at the silhouette-optimal k=2, while a finer k=4 partition reveals an interpretable low/medium/high-FX-margin tiering among the remaining 15 destinations, convergent across k-means and hierarchical methods.
- cc1 fx margin, payment instrument, pickup method, and destination remain the strongest predictors across every method applied in this project—information-criterion selection, shrinkage, Random Forest importance, and corridor clustering all converge on the same handful of drivers.
- A deployed FastAPI/React web application now lets a sender get a predicted cost for their own intended transfer, putting the project's modeling work directly in front of its intended end users.


# Appendix

Figure 1. Distribution of cc1 total cost % (histogram and boxplot), with the 3% SDG target marked.

Figure 2. cc1 total cost % by firm type.

Figure 3. Mean cc1 total cost % by destination, sorted descending.

Figure 4. cc1 total cost % by payment instrument group.

Figure 5. Mean cc1 total cost % by year, 2016-2025.

Figure 6. cc1 total cost % by transparency proxy (cc2 total cost % missing vs. present).

Figure 7. Pearson correlation heatmap of numeric RPW fields plus year.

Figure 8. AIC/BIC vs. number of features in forward stepwise selection.

Figure 9. LASSO regularization path (coefficients vs. alpha).

Figure 10. Final LassoCV coefficients, top 20 by magnitude.

Figure 11. PCA scree plot of the numeric candidate block.

Figure 12. Residuals vs. fitted values and residual distribution, Linear Regression (best test-RMSE regression model), 2023+ holdout.

Figure 13. Top 15 standardized LASSO coefficients by magnitude, compared to Ridge on the same features.

Figure 14. Random Forest regressor feature importance (mean decrease in impurity), top 15.

Figure 15. Confusion matrices by firm type, Random Forest (best-F1 classifier), 2023+ holdout.

Figure 16. Precision/recall/F1 by destination country, Random Forest, 2023+ holdout.

Figure 17. K-means inertia (elbow) and silhouette score vs. number of clusters k, for the 16 LAC destinations.

Figure 18. K-means clusters (k=2) of the 16 LAC destinations, mean FX margin vs. mean total cost %.

Figure 19. Hierarchical (Ward linkage) clustering dendrogram of destination cost profiles.


# Bibliography

- World Bank. Remittance Prices Worldwide Database. https://remittanceprices.worldbank.org/en/database
- United Nations. Sustainable Development Goal 10, Target 10.c. https://sdgs.un.org/goals/goal10
- Mei, Yajun. ISyE 7406 – Data Mining & Statistical Learning. Georgia Institute of Technology, Fall 2026.
- Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12 (2011): 2825-2830. https://scikit-learn.org
