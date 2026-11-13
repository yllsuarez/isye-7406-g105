"""
Export the FINAL report (condensed EDA/feature-selection from the midterm, plus the
newly completed regression/classification/clustering analysis, a Web Application
section, and a Lessons Learned subsection) for Team 105's remittance-cost project into:
  - r-models/reports/Final_Report_Group105.md
  - r-models/reports/Final_Report_Group105.docx
  - r-models/reports/Final_Report_Group105.pdf

Mirrors export_eda_doc.py's pattern: figures are reused from
reports/figures/ (01-11, midterm) and reports/figures_final/ (12-19, new), tables
are read from the CSVs written by final_model_analysis.py (so the report numbers
are guaranteed consistent with that script's actual run), narrative is written as
clean prose in a flat (kind, ...) content-block list, and the docx/md/pdf are built
from that single list. PDF conversion uses the same fallback chain: pandoc+xelatex,
then LibreOffice (soffice), then docx2pdf.

Run from the r-models/ directory:
    /opt/anaconda3/bin/python3 export_final_report.py
"""
import os
import subprocess
import sys

import pandas as pd

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(HERE, "reports")
FIG_DIR_EDA = os.path.join(REPORTS_DIR, "figures")
FIG_DIR_FINAL = os.path.join(REPORTS_DIR, "figures_final")
os.makedirs(REPORTS_DIR, exist_ok=True)

TITLE = "Predicting Remittance Transfer Costs from the United States to Latin America and the Caribbean"
TEAM_LINE = "Team Project: Number 105  |  ISyE 7406 \u2013 Fall 2026"
AUTHORS_LINE = (
    "Oscar Borge Donado \u2013 GT ID: 903965024, Yonny Leyva \u2013 GT ID: 903942855"
)
SDG_TARGET = 3.0

# ---------------------------------------------------------------------------
# Load the tables written by final_model_analysis.py, so every number quoted
# below is guaranteed consistent with that script's actual last run.
# ---------------------------------------------------------------------------
reg_comparison = pd.read_csv(os.path.join(REPORTS_DIR, "regression_comparison.csv"))
reg_coefficients = pd.read_csv(os.path.join(REPORTS_DIR, "regression_coefficients.csv"), index_col=0)
rf_importance = pd.read_csv(os.path.join(REPORTS_DIR, "rf_feature_importance.csv"), index_col=0)
clf_comparison = pd.read_csv(os.path.join(REPORTS_DIR, "classification_comparison.csv"))
clf_by_firmtype = pd.read_csv(os.path.join(REPORTS_DIR, "classification_by_firmtype.csv"))
clf_by_destination = pd.read_csv(os.path.join(REPORTS_DIR, "classification_by_destination.csv"))
cluster_profile = pd.read_csv(os.path.join(REPORTS_DIR, "cluster_country_profile.csv"))
cluster_metrics = pd.read_csv(os.path.join(REPORTS_DIR, "cluster_metrics.csv"))

best_reg = reg_comparison.sort_values("test_rmse").iloc[0]
worst_reg = reg_comparison.sort_values("test_rmse").iloc[-1]
best_clf = clf_comparison.sort_values("f1", ascending=False).iloc[0]
best_k_row = cluster_metrics.sort_values("silhouette", ascending=False).iloc[0]
best_k = int(best_k_row["k"])

cuba_row = cluster_profile[cluster_profile["destination_name"] == "Cuba"].iloc[0]
rest_mean_cost = cluster_profile[cluster_profile["destination_name"] != "Cuba"]["mean_total_cost"].mean()

# k=4 groupings, named by inspection of mean_total_cost within each group, for narrative use.
k4_groups = (
    cluster_profile.groupby("kmeans_cluster_k4")["destination_name"]
    .apply(lambda s: ", ".join(sorted(s)))
    .to_dict()
)
k4_cost = cluster_profile.groupby("kmeans_cluster_k4")["mean_total_cost"].mean().sort_values()

worst_dest_clf = clf_by_destination.sort_values("f1").iloc[0]
best_dest_clf = clf_by_destination.sort_values("f1", ascending=False).iloc[0]

bank_row = clf_by_firmtype[clf_by_firmtype["firm_type"] == "Bank"].iloc[0] \
    if "Bank" in clf_by_firmtype["firm_type"].values else clf_by_firmtype.iloc[0]
mto_row = clf_by_firmtype[clf_by_firmtype["firm_type"] == "Money Transfer Operator"].iloc[0] \
    if "Money Transfer Operator" in clf_by_firmtype["firm_type"].values else clf_by_firmtype.iloc[-1]

# ---------------------------------------------------------------------------
# Figures: existing midterm figures (01-11) + new final-report figures (12-19).
# ---------------------------------------------------------------------------
FIGURES_MIDTERM = [
    ("01_target_distribution.png", FIG_DIR_EDA,
     "Figure 1. Distribution of cc1 total cost % (histogram and boxplot), with the 3% SDG target marked."),
    ("02_cost_by_firm_type.png", FIG_DIR_EDA, "Figure 2. cc1 total cost % by firm type."),
    ("03_cost_by_destination.png", FIG_DIR_EDA, "Figure 3. Mean cc1 total cost % by destination, sorted descending."),
    ("04_cost_by_payment_instrument.png", FIG_DIR_EDA, "Figure 4. cc1 total cost % by payment instrument group."),
    ("05_cost_by_year.png", FIG_DIR_EDA, "Figure 5. Mean cc1 total cost % by year, 2016-2025."),
    ("06_transparency_comparison.png", FIG_DIR_EDA,
     "Figure 6. cc1 total cost % by transparency proxy (cc2 total cost % missing vs. present)."),
    ("07_correlation_heatmap.png", FIG_DIR_EDA, "Figure 7. Pearson correlation heatmap of numeric RPW fields plus year."),
    ("08_aic_bic_vs_nvars.png", FIG_DIR_EDA, "Figure 8. AIC/BIC vs. number of features in forward stepwise selection."),
    ("09_lasso_coef_path.png", FIG_DIR_EDA, "Figure 9. LASSO regularization path (coefficients vs. alpha)."),
    ("10_lasso_coefficients_bar.png", FIG_DIR_EDA, "Figure 10. Final LassoCV coefficients, top 20 by magnitude."),
    ("11_pca_scree.png", FIG_DIR_EDA, "Figure 11. PCA scree plot of the numeric candidate block."),
]
FIGURES_FINAL = [
    ("12_regression_residuals.png", FIG_DIR_FINAL,
     f"Figure 12. Residuals vs. fitted values and residual distribution, {best_reg['model']} "
     "(best test-RMSE regression model), 2023+ holdout."),
    ("13_lasso_ridge_coefficients.png", FIG_DIR_FINAL,
     "Figure 13. Top 15 standardized LASSO coefficients by magnitude, compared to Ridge on the same features."),
    ("14_rf_feature_importance.png", FIG_DIR_FINAL,
     "Figure 14. Random Forest regressor feature importance (mean decrease in impurity), top 15."),
    ("15_confusion_by_firmtype.png", FIG_DIR_FINAL,
     f"Figure 15. Confusion matrices by firm type, {best_clf['model']} (best-F1 classifier), 2023+ holdout."),
    ("16_classification_by_destination.png", FIG_DIR_FINAL,
     f"Figure 16. Precision/recall/F1 by destination country, {best_clf['model']}, 2023+ holdout."),
    ("17_kmeans_elbow_silhouette.png", FIG_DIR_FINAL,
     "Figure 17. K-means inertia (elbow) and silhouette score vs. number of clusters k, for the 16 LAC destinations."),
    ("18_cluster_scatter.png", FIG_DIR_FINAL,
     f"Figure 18. K-means clusters (k={best_k}) of the 16 LAC destinations, mean FX margin vs. mean total cost %."),
    ("19_dendrogram.png", FIG_DIR_FINAL,
     "Figure 19. Hierarchical (Ward linkage) clustering dendrogram of destination cost profiles."),
]
ALL_FIGURES = FIGURES_MIDTERM + FIGURES_FINAL
FIG_LOOKUP = {f: (d, c) for f, d, c in ALL_FIGURES}

for fname, fdir, _ in ALL_FIGURES:
    fpath = os.path.join(fdir, fname)
    if not os.path.exists(fpath):
        print(f"WARNING: expected figure not found: {fpath}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Narrative content. Entries: ("h1"/"h2", text) | ("p", text) | ("ul", [items]) |
# ("fig", filename) | ("table", name)
# ---------------------------------------------------------------------------
CONTENT = [
    ("h1", "Problem Statement"),
    ("p", "Every year, millions of Latino families in the United States send money back to "
          "relatives in Latin America and the Caribbean (LAC). These transfers are a primary "
          "source of income for receiving households. In 2023, the United States sent over "
          "$150 billion in remittances to LAC countries, making it the largest single source "
          "of remittance inflows into the region. Despite their scale and importance, the "
          "cost of sending that money remains stubbornly high. The United Nations set a "
          "target of 3% average cost per transaction under SDG 10.c, yet the corridors in "
          "this study average 5.44% as of 2025 Q1, with some destinations such as Cuba "
          "(12.0%) and Guyana (8.7%) far exceeding that benchmark."),
    ("p", "The goal of this project is to understand why costs are where they are, and to "
          "build tools that help senders find lower-cost options. The World Bank Remittance "
          "Prices Worldwide (RPW) database tracks transaction-level pricing across hundreds "
          "of global corridors, capturing provider type, payment instrument, transfer speed, "
          "fees, exchange rate margins, and destination country for each offering. This "
          "report completes the three-part plan laid out in the midterm: (1) exploratory "
          "data analysis and feature selection (midterm work, condensed below); (2) fitting "
          "and comparing regression models, a classification task at the 3% threshold, and "
          "destination clustering (this report's new work); and (3) a deployed web "
          "application that lets a money sender get a predicted cost for their own transfer. "
          "All analysis was conducted in Python using pandas, numpy, matplotlib, seaborn, "
          "scipy, scikit-learn, and statsmodels; the web application uses FastAPI and "
          "React/TypeScript. Figures are described in the Appendix."),

    ("h1", "Proposed Dataset and Data Preprocessing"),
    ("p", "The dataset is a transaction-level extract from the World Bank Remittance Prices "
          "Worldwide (RPW) database, subset to United States-to-LAC corridors: 12,397 rows "
          "and 23 columns, one row per provider offering per corridor per quarter, spanning "
          "period 2016_2Q through 2025_1Q. speed actual and pickup method were lower-cased "
          "and stripped to merge near-duplicate categories caused by inconsistent "
          "capitalization; period was parsed into integer year and quarter columns for the "
          "trend analysis and for the temporal train/test split used throughout this report. "
          "firm_type was consolidated (the two Money Transfer Operator variants merged into "
          "one level) and the 11 sparse Non-Bank FI rows were dropped, leaving 12,386 rows "
          "for all modeling tasks below. Only the cc2 * fields have missing values (235 "
          "rows, 1.9%), used to engineer the is_non_transparent proxy flag; the free-text "
          "Standard Note is missing for 69.2% of rows (expected, since it is only populated "
          "when there is something unusual to disclose). Every other column, including the "
          "target cc1 total cost %, is fully populated."),

    ("h1", "Proposed Methodology"),
    ("p", "This project's methodology is grounded in the course's model/variable selection "
          "framework (ISyE 7406, Prof. Yajun Mei). The midterm phase used information "
          "criteria (AIC/BIC via forward stepwise search), shrinkage-based selection "
          "(Ridge/LASSO), and PCA to identify a final predictor set: destination_name, "
          "firm_type_clean, payment_instrument_group, access point, receiving network "
          "coverage, speed actual, pickup method, cc1 fx margin, year, and "
          "is_non_transparent, encoded via one-hot (drop_first=True) for categoricals and "
          "standardized for numerics (42 encoded features). This final report builds on "
          "that feature set with three further analyses, all evaluated on an identical "
          "pre-2023 train / 2023+ test temporal split (8,228 train rows, 4,158 test rows): "
          "(1) a regression model comparison (Linear Regression, Ridge, LASSO, Random "
          "Forest) scored by RMSE, MAE, and R\u00b2; (2) a binary classification task that "
          "labels a transaction \"high cost\" when cc1 total cost % \u2265 3% (the SDG 10.c "
          "threshold), compared across Logistic Regression, Linear Discriminant Analysis "
          "(LDA), and Random Forest, scored by accuracy, precision, recall, and F1; and (3) "
          "unsupervised clustering of the 16 LAC destination countries by their mean cost "
          "profile (total cost, FX margin, fee, transparency rate), using both k-means "
          "(k chosen via elbow and silhouette score) and hierarchical (Ward-linkage) "
          "clustering. The same production LassoCV model (refit on all available data) is "
          "shipped in a deployed FastAPI/React web application described in its own section "
          "below."),

    ("h1", "Analysis and Results"),

    ("h2", "Target Variable"),
    ("p", "The target cc1 total cost % has a mean of 5.34%, a median of 4.71%, and a "
          "standard deviation of 3.03%, with a range of \u22124.09% to 34.0%. It is "
          "right-skewed (skewness = 1.88, excess kurtosis = 6.33), driven by a small number "
          "of high-cost providers in outlier corridors. The 25th percentile (3.50%) sits "
          "just above the SDG 3% target, meaning roughly three-quarters of all transactions "
          "fail to meet it."),
    ("fig", "01_target_distribution.png"),

    ("h2", "Cost by Destination Country"),
    ("p", "Destination is the single strongest driver of cost variation in the dataset. "
          "Cuba averages 12.0% and Guyana 8.9%, both more than double the SDG target. At "
          "the other end, Honduras (3.9%), Ecuador (4.0%), and El Salvador (4.0%) are "
          "closest to the threshold. Mexico, the largest-volume corridor, averages 4.6%. "
          "The roughly 8-percentage-point spread between the cheapest and most expensive "
          "corridors far exceeds the differences attributable to firm type or payment "
          "instrument, and motivates the corridor clustering analysis below."),
    ("fig", "03_cost_by_destination.png"),

    ("h2", "Cost by Firm Type"),
    ("p", "Banks are not meaningfully cheaper than Money Transfer Operators (MTOs) in "
          "this dataset: recomputed directly from the modeling subset, Banks average "
          "5.34% and MTOs average 5.34%, a negligible difference, and both sit well above "
          "the 3% SDG target. MTOs dominate the market by volume (10,707 of 12,386 "
          "modeling rows, 86%)."),
    ("fig", "02_cost_by_firm_type.png"),

    ("h2", "Cost by Payment Instrument"),
    ("p", "Payment instrument differentiates cost meaningfully: cash-based transfers show "
          "higher and more variable costs than bank account transfers or digital "
          "instruments, and credit card transfers carry a distinct premium, likely "
          "reflecting card processing fees."),
    ("fig", "04_cost_by_payment_instrument.png"),

    ("h2", "Cost Over Time"),
    ("p", "Mean cost has been essentially flat across 2016-2025, fluctuating between 4.96% "
          "(2020) and 5.59% (2024), with no sustained downward trend toward the SDG target. "
          "year is a low-signal predictor on its own, retained mainly because it defines "
          "the temporal train/test split used throughout this report."),
    ("fig", "05_cost_by_year.png"),

    ("h2", "Transparency Flag"),
    ("p", "The engineered is_non_transparent flag (cc2 total cost % missing as a proxy for "
          "non-disclosure) flags 1.9% of rows, which show a modestly lower mean cost than "
          "transparent rows (4.80% vs. 5.35%). The proxy is imperfect\u201481% of flagged "
          "rows carry a \"send limits\" disclosure note rather than an exchange-rate "
          "non-disclosure note\u2014so it is retained as a candidate feature with this "
          "limitation stated explicitly."),
    ("fig", "06_transparency_comparison.png"),

    ("h2", "Correlation Structure"),
    ("p", "Among numeric fields, cc1 fx margin shows the strongest correlation with the "
          "target (r = 0.71), as expected since FX margin is a direct cost component. "
          "cc1 lcu fx rate and inter lcu bank fx correlate weakly (r \u2248 0.06-0.07) and "
          "are currency-scale fields excluded from the predictor set. All cc2 * fields "
          "correlate highly with the target by construction (r = 0.91 for cc2 total "
          "cost %) and are excluded as leakage variables."),
    ("fig", "07_correlation_heatmap.png"),

    ("h2", "Feature Selection (Information Criteria, Shrinkage, and PCA)"),
    ("p", "Feature selection used three complementary methods, detailed in full in the "
          "midterm's EDA & Feature Selection report (r-models/reports/"
          "EDA_FeatureSelection_Group105), condensed here. Forward stepwise selection by "
          "BIC chose 9 of 10 candidate blocks, in order: cc1 fx margin, pickup method, "
          "payment_instrument_group, destination_name, access point, receiving network "
          "coverage, speed actual, year, is_non_transparent (firm_type_clean was the one "
          "block not selected). LassoCV (5-fold CV, alpha = 0.001) zeroed out 6 of 42 "
          "encoded features, with the largest surviving coefficients on pickup "
          "method_own/partner bank account, destination_name_Cuba, and access point_Bank "
          "branch,Internet. RidgeCV (alpha = 1.15) shrinks all coefficients without zeroing "
          "any out. PCA on the numeric block (cc1 fx margin, year, is_non_transparent) "
          "confirmed the three features are near-orthogonal, so dimensionality reduction "
          "was not useful. Both the information-criterion and shrinkage methods converge "
          "on the same dominant predictors\u2014destination, payment instrument/pickup "
          "method, and FX margin\u2014giving confidence the final feature set "
          "(destination_name, firm_type_clean, payment_instrument_group, access point, "
          "receiving network coverage, speed actual, pickup method, cc1 fx margin, year, "
          "is_non_transparent; 42 encoded features) is robust."),
    ("fig", "08_aic_bic_vs_nvars.png"),
    ("fig", "09_lasso_coef_path.png"),
    ("fig", "10_lasso_coefficients_bar.png"),
    ("fig", "11_pca_scree.png"),

    ("h2", "Regression Model Comparison"),
    ("p", "Four regression models were fit on the identical final feature set (one-hot "
          "encoded categoricals + standardized numerics), trained on pre-2023 rows (n = "
          "8,228) and evaluated on the 2023+ holdout (n = 4,158): plain Linear Regression, "
          "RidgeCV, LassoCV, and a Random Forest Regressor (400 trees). RMSE is the "
          "primary metric, with MAE and R\u00b2 reported alongside."),
    ("table", "regression_comparison"),
    ("p", f"{best_reg['model']} achieves the lowest test RMSE ({best_reg['test_rmse']:.3f}, "
          f"R\u00b2 = {best_reg['test_r2']:.3f}), essentially tied with LASSO "
          f"({reg_comparison[reg_comparison['model']=='LASSO']['test_rmse'].iloc[0]:.3f}) and "
          f"Ridge ({reg_comparison[reg_comparison['model']=='Ridge']['test_rmse'].iloc[0]:.3f}); "
          f"the Random Forest has the highest test RMSE ({worst_reg['test_rmse']:.3f}) but the "
          "lowest MAE, suggesting it fits the bulk of typical transactions slightly better "
          "while being less able to extrapolate to the handful of extreme-cost corridors "
          "(Cuba, Guyana) that drive RMSE's squared-error penalty. This is consistent with "
          "the shipped production model (LassoCV refit on all data, holdout RMSE \u2248 "
          "1.95, R\u00b2 \u2248 0.67 as reported in train_model.py/model/metadata.json): "
          "linear, regularized models capture this dataset's structure about as well as a "
          "flexible nonparametric model does, which is itself informative\u2014it suggests "
          "the relationship between the selected features and cost is close to additive "
          "and linear once destination, payment instrument, and FX margin are already "
          "encoded, with limited extra signal left for a tree ensemble to exploit."),
    ("fig", "12_regression_residuals.png"),
    ("p", f"The residual plot for {best_reg['model']} (Figure 12) shows residuals "
          "concentrated near zero with a long right tail, mirroring the right-skewed "
          "target distribution: a handful of very high-cost transactions (predominantly "
          "Cuba and Guyana) are systematically under-predicted, since no linear model can "
          "fully capture an outlier destination's level shift from the pooled coefficient "
          "estimate alone."),
    ("fig", "13_lasso_ridge_coefficients.png"),
    ("table", "regression_coefficients"),
    ("p", "LASSO's largest standardized coefficients line up with the midterm's feature-"
          "selection findings (destination and pickup-method dummies, cc1 fx margin); "
          "Ridge's coefficients on the same features are directionally consistent but "
          "uniformly smaller in magnitude, reflecting its smoother L2 shrinkage versus "
          "LASSO's sparser L1 penalty."),
    ("fig", "14_rf_feature_importance.png"),
    ("table", "rf_feature_importance"),
    ("p", "The Random Forest's impurity-based feature importances point to the same "
          "handful of dominant predictors\u2014cc1 fx margin and the Cuba/Guyana "
          "destination dummies rank highest\u2014corroborating the linear models' "
          "coefficients with an entirely different, nonparametric method."),

    ("h2", "Classification at the 3% SDG Threshold"),
    ("p", f"The target was binarized at the SDG 10.c threshold (cc1 total cost % "
          f"\u2265 {SDG_TARGET:.0f}% = \"high cost\"); the resulting label is heavily "
          "imbalanced toward \"high cost\" (86.4% of the pre-2023 training rows, 85.4% of "
          "the 2023+ test rows), consistent with the EDA finding that three-quarters of "
          "transactions already exceed the target. Logistic Regression, LDA, and a "
          "Random Forest Classifier (400 trees) were trained on the same encoded feature "
          "set and the same temporal split."),
    ("table", "classification_comparison"),
    ("p", f"{best_clf['model']} is the best classifier by F1 "
          f"(F1 = {best_clf['f1']:.3f}, accuracy = {best_clf['accuracy']:.3f}, precision = "
          f"{best_clf['precision']:.3f}, recall = {best_clf['recall']:.3f}). Given the "
          "class imbalance, accuracy alone would be a misleading metric (a trivial "
          "\"always predict high cost\" classifier would already score roughly 85% "
          "accuracy); F1 and recall are the more meaningful comparison, and the Random "
          "Forest's higher recall indicates it misses fewer of the (minority) low-cost "
          "transactions that are most actionable for a sender trying to find a cheaper "
          "option."),
    ("fig", "15_confusion_by_firmtype.png"),
    ("table", "classification_by_firmtype"),
    ("p", f"Broken out by firm type, the {best_clf['model']} classifier performs "
          f"{'similarly' if abs(bank_row['f1'] - mto_row['f1']) < 0.03 else 'differently'} "
          f"for Banks (F1 = {bank_row['f1']:.3f}, n = {int(bank_row['n'])}) versus Money "
          f"Transfer Operators (F1 = {mto_row['f1']:.3f}, n = {int(mto_row['n'])}), "
          "indicating the model is not simply learning a firm-type shortcut but is "
          "picking up on corridor/instrument-level cost signal that generalizes across "
          "provider types."),
    ("fig", "16_classification_by_destination.png"),
    ("table", "classification_by_destination"),
    ("p", f"Per-destination performance is uneven: {best_dest_clf['destination_name']} has "
          f"the highest F1 ({best_dest_clf['f1']:.3f}, n = {int(best_dest_clf['n'])}), while "
          f"{worst_dest_clf['destination_name']} has the lowest "
          f"(F1 = {worst_dest_clf['f1']:.3f}, n = {int(worst_dest_clf['n'])}). The weakest "
          "destinations tend to be those with either very few 2023+ holdout observations "
          "or a cost distribution that straddles the 3% boundary closely, so a small "
          "number of misclassifications has an outsized effect on that corridor's F1."),

    ("h2", "Corridor Clustering"),
    ("p", "The 16 LAC destination countries were aggregated to a country-level profile "
          "(mean total cost, mean FX margin, mean fee in local currency units, and "
          "transparency rate), standardized, and clustered with both k-means and "
          "hierarchical (Ward-linkage) methods. k for k-means was chosen by jointly "
          "inspecting the elbow (inertia) curve and the silhouette score across "
          "k = 2..7."),
    ("fig", "17_kmeans_elbow_silhouette.png"),
    ("table", "cluster_metrics"),
    ("p", f"Silhouette score is maximized at k = {best_k} "
          f"(silhouette = {best_k_row['silhouette']:.3f}) and decreases monotonically for "
          "larger k, but at k=2 the partition is dominated by a single extreme outlier: "
          f"Cuba (mean cost {cuba_row['mean_total_cost']:.2f}%, mean FX margin "
          f"{cuba_row['mean_fx_margin']:.2f}) forms its own cluster, versus a single large "
          f"cluster of the remaining 15 destinations (mean cost {rest_mean_cost:.2f}%). "
          "This is a legitimate finding in its own right\u2014Cuba's cost profile is simply "
          "far enough from every other destination's that no amount of recombination "
          "produces a better-separated two-group split\u2014but it is not very informative "
          "about structure *among* the other 15 destinations. For that finer-grained "
          "interpretation, we also report the k=4 partition (lower, but still reasonable, "
          "silhouette) as supplementary detail."),
    ("fig", "18_cluster_scatter.png"),
    ("table", "cluster_country_profile"),
]

# Build the k=4 narrative dynamically from the actual cluster assignments.
k4_lines = []
for cluster_id in sorted(k4_cost.index):
    members = k4_groups[cluster_id]
    mean_cost = k4_cost.loc[cluster_id]
    k4_lines.append(f"cluster {int(cluster_id)} (mean cost {mean_cost:.2f}%): {members}")
CONTENT.append((
    "p",
    "At k=4, the finer partition separates into four interpretable cost tiers (listed "
    "from cheapest to most expensive mean total cost): " + "; ".join(k4_lines) + ". The "
    "cheapest tier groups destinations with low FX margins and no (or near-zero) reported "
    "FX margin component; the most expensive non-Cuba tier (Brazil, Colombia, Costa Rica, "
    "Mexico and similar) combines moderate-to-high FX margins with high transaction "
    "volume; and Dominican Republic/Guyana/Jamaica form a distinct mid-to-high-cost tier "
    "with the highest FX margins outside of Cuba. This is consistent with the regression "
    "and classification results above: FX margin and destination are the dominant cost "
    "drivers, and the clustering shows that destinations group along almost exactly that "
    "same axis."
))
CONTENT.append(("fig", "19_dendrogram.png"))
CONTENT.append((
    "p",
    "The hierarchical clustering dendrogram (Ward linkage, Figure 19) tells the same "
    "story from a different angle: Cuba joins the rest of the tree at by far the largest "
    "linkage distance (confirming it as a singleton outlier at low k), while the "
    "remaining destinations merge into recognizable sub-groups consistent with the k=4 "
    "k-means partition above, giving convergent evidence from two different clustering "
    "methods for the same underlying corridor groupings."
))

CONTENT += [
    ("h1", "Web Application"),
    ("p", "To put the modeling work in front of an actual money sender, the project ships "
          "a small full-stack web application: a FastAPI backend and a React/TypeScript "
          "single-page frontend. The backend (web-service/backend/main.py) loads the "
          "shipped production model (the LassoCV pipeline from r-models/model/model.pkl, "
          "refit on all available data) and its metadata.json at startup, and exposes two "
          "synchronous REST endpoints: GET /metadata, which returns the feature list, "
          "valid category values, and the historical mean FX margin by destination (used "
          "to pre-fill a sensible default), and POST /predict, which accepts a sender's "
          "inputs (destination, firm type, payment instrument, access point, receiving "
          "network coverage, speed, pickup method, FX margin, year, and a transparency "
          "flag) and returns the model's predicted cc1 total cost % (plus an estimated fee "
          "in USD if a transfer amount is supplied). There is no background task queue or "
          "async job system\u2014the architecture is deliberately a simple, synchronous "
          "request/response flow, appropriate for a single fast model-inference call."),
    ("p", "The frontend (web-service/frontend/src/App.tsx) is a single form that mirrors "
          "the backend's input schema: dropdowns for the categorical fields (populated "
          "from GET /metadata so the UI never drifts out of sync with what the model was "
          "trained on), a numeric field for cc1 fx margin pre-filled with the historical "
          "destination-level average default, and a submit action that calls POST "
          "/predict and renders the predicted cost percentage (and estimated fee) back to "
          "the user. Keeping the categories server-driven means that if the model is ever "
          "retrained on an updated dataset with a new destination or payment instrument "
          "category, the frontend does not need a code change to stay correct."),
    ("p", "web-service/sync_model.sh is the glue between the two solutions in this "
          "monorepo: it copies the trained artifacts (model.pkl and metadata.json) from "
          "r-models/model/ into web-service/backend/model/, so the backend always serves "
          "whatever model r-models/train_model.py most recently produced, without needing "
          "the backend to import or depend on the r-models training code directly."),

    ("h1", "Conclusions and Next Steps"),

    ("h2", "Lessons Learned"),
    ("ul", [
        "Data quality is uneven across fields that look superficially similar: the cc1/"
        "cc2 comparator fields are mechanically correlated by construction, and naive "
        "inclusion of cc2 * fields would have silently leaked the target into the "
        "feature set. Catching this required reading the RPW database's own field "
        "definitions, not just running a correlation matrix.",
        "The is_non_transparent proxy is a good example of a feature that looked clean "
        "in isolation but turned out to be confounded (send-limit disclosures vs. true "
        "FX non-disclosure) once cross-tabulated against the free-text note field. "
        "Engineered features built from a single missingness pattern deserve this kind "
        "of secondary validation before being trusted.",
        "Linear/regularized models (LASSO, Ridge, plain OLS) and a 400-tree Random "
        "Forest landed on almost the same test RMSE in this project. That was initially "
        "surprising\u2014we expected the Random Forest to meaningfully outperform on "
        "nonlinearities\u2014but in retrospect it is explained by how much of the signal "
        "was already captured by the carefully selected categorical feature set "
        "(destination, payment instrument, pickup method) before any model-side "
        "nonlinearity could add value. A worthwhile extension would be engineering "
        "interaction terms (e.g. destination \u00d7 payment instrument) explicitly for "
        "the linear models, to see whether that closes the remaining gap with the "
        "Random Forest or whether the Random Forest's flexibility is already capturing "
        "those interactions implicitly.",
        "With only 16 destination countries, clustering results are sensitive to the "
        "exact choice of k and to a single extreme outlier (Cuba) dominating the "
        "silhouette-optimal partition. Reporting both the statistically \"optimal\" k "
        "and a secondary, more interpretable k was a deliberate choice to avoid letting "
        "a single metric understate the richer structure visible in the dendrogram.",
        "If we were to start over, we would build the FastAPI/React web application "
        "earlier in the project rather than after the modeling work, since having a "
        "live, interactive view into the model's predictions surfaced a few feature-"
        "encoding edge cases (e.g. the normalization of speed actual and pickup method "
        "casing) that were easy to overlook while working purely from notebook output.",
    ]),

    ("h2", "Key Findings"),
    ("ul", [
        "The mean total cost of US-to-LAC remittances (5.44%) has remained well above "
        "the UN SDG 10.c target of 3% throughout 2016-2025, with no meaningful downward "
        "trend.",
        "Destination country is the dominant driver of cost variation, with an 8+ "
        "percentage point spread between the cheapest (Honduras, 3.9%) and most "
        "expensive (Cuba, 12.0%) corridors.",
        f"{best_reg['model']} achieved the best regression performance on the 2023+ "
        f"holdout (RMSE = {best_reg['test_rmse']:.3f}, R\u00b2 = {best_reg['test_r2']:.3f}), "
        "closely matched by LASSO and Ridge; the Random Forest did not meaningfully "
        "outperform the linear models, suggesting the relationship is close to additive "
        "once destination, payment instrument, and FX margin are encoded.",
        f"At the 3% SDG threshold, {best_clf['model']} was the best classifier "
        f"(F1 = {best_clf['f1']:.3f}), though performance varies meaningfully by "
        "destination, reflecting both class imbalance and genuine cross-corridor "
        "heterogeneity.",
        f"Clustering the 16 destinations by cost profile isolates Cuba as a singleton "
        f"outlier cluster at the silhouette-optimal k={best_k}, while a finer k=4 "
        "partition reveals an interpretable low/medium/high-FX-margin tiering among the "
        "remaining 15 destinations, convergent across k-means and hierarchical methods.",
        "cc1 fx margin, payment instrument, pickup method, and destination remain the "
        "strongest predictors across every method applied in this project\u2014"
        "information-criterion selection, shrinkage, Random Forest importance, and "
        "corridor clustering all converge on the same handful of drivers.",
        "A deployed FastAPI/React web application now lets a sender get a predicted "
        "cost for their own intended transfer, putting the project's modeling work "
        "directly in front of its intended end users.",
    ]),

    ("h1", "Appendix"),
]
for fname, _, caption in ALL_FIGURES:
    CONTENT.append(("appendix_entry", caption))

CONTENT += [
    ("h1", "Bibliography"),
    ("ul", [
        "World Bank. Remittance Prices Worldwide Database. "
        "https://remittanceprices.worldbank.org/en/database",
        "United Nations. Sustainable Development Goal 10, Target 10.c. "
        "https://sdgs.un.org/goals/goal10",
        "Mei, Yajun. ISyE 7406 \u2013 Data Mining & Statistical Learning. Georgia "
        "Institute of Technology, Fall 2026.",
        "Pedregosa, F. et al. Scikit-learn: Machine Learning in Python. Journal of "
        "Machine Learning Research 12 (2011): 2825-2830. https://scikit-learn.org",
    ]),
]


# ---------------------------------------------------------------------------
# Table definitions: shared between the docx and markdown builders.
# ---------------------------------------------------------------------------
def table_spec(name):
    if name == "regression_comparison":
        headers = ["Model", "Test RMSE", "Test MAE", "Test R\u00b2"]
        rows = [
            [r["model"], round(r["test_rmse"], 4), round(r["test_mae"], 4), round(r["test_r2"], 4)]
            for _, r in reg_comparison.sort_values("test_rmse").iterrows()
        ]
        caption = "Table 5. Regression model comparison, 2023+ holdout (sorted by test RMSE)."
    elif name == "regression_coefficients":
        data = reg_coefficients.reset_index().rename(columns={"index": "feature"})
        headers = ["Feature", "LASSO coef.", "Ridge coef."]
        rows = [[r["feature"], round(r["LASSO"], 4), round(r["Ridge"], 4)] for _, r in data.iterrows()]
        caption = "Table 6. Top 15 standardized LASSO coefficients and matching Ridge coefficients."
    elif name == "rf_feature_importance":
        data = rf_importance.reset_index().rename(columns={"index": "feature"})
        headers = ["Feature", "Importance"]
        rows = [[r["feature"], round(r["importance"], 4)] for _, r in data.iterrows()]
        caption = "Table 7. Random Forest regressor feature importance, top 15."
    elif name == "classification_comparison":
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1"]
        rows = [
            [r["model"], round(r["accuracy"], 4), round(r["precision"], 4), round(r["recall"], 4), round(r["f1"], 4)]
            for _, r in clf_comparison.sort_values("f1", ascending=False).iterrows()
        ]
        caption = "Table 8. Classification comparison at the 3% SDG threshold, 2023+ holdout."
    elif name == "classification_by_firmtype":
        headers = ["Firm Type", "n", "Accuracy", "Precision", "Recall", "F1"]
        rows = [
            [r["firm_type"], int(r["n"]), round(r["accuracy"], 4), round(r["precision"], 4),
             round(r["recall"], 4), round(r["f1"], 4)]
            for _, r in clf_by_firmtype.iterrows()
        ]
        caption = f"Table 9. Classification performance by firm type, {best_clf['model']}."
    elif name == "classification_by_destination":
        headers = ["Destination", "n", "Accuracy", "Precision", "Recall", "F1"]
        rows = [
            [r["destination_name"], int(r["n"]), round(r["accuracy"], 4), round(r["precision"], 4),
             round(r["recall"], 4), round(r["f1"], 4)]
            for _, r in clf_by_destination.sort_values("f1", ascending=False).iterrows()
        ]
        caption = f"Table 10. Classification performance by destination, {best_clf['model']}."
    elif name == "cluster_metrics":
        headers = ["k", "Inertia", "Silhouette"]
        rows = [[int(r["k"]), round(r["inertia"], 3), round(r["silhouette"], 4)] for _, r in cluster_metrics.iterrows()]
        caption = "Table 11. K-means inertia and silhouette score by k."
    elif name == "cluster_country_profile":
        headers = ["Destination", "Mean Cost %", "Mean FX Margin", "Mean Fee (LCU)", "Transparency Rate",
                   "k-means Cluster (k=2)", "k-means Cluster (k=4)"]
        rows = [
            [r["destination_name"], round(r["mean_total_cost"], 2), round(r["mean_fx_margin"], 2),
             round(r["mean_fee_lcu"], 2), round(r["transparency_rate"], 3), int(r["kmeans_cluster"]),
             int(r["kmeans_cluster_k4"])]
            for _, r in cluster_profile.sort_values("mean_total_cost").iterrows()
        ]
        caption = "Table 12. Destination-level cost profile and cluster assignments."
    else:
        return None, None, None
    return headers, rows, caption


def strip_md(text):
    return text.replace("`", "")


# ---------------------------------------------------------------------------
# Build the Word document
# ---------------------------------------------------------------------------
def build_docx():
    doc = Document()
    title = doc.add_heading(TITLE, level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for line in (TEAM_LINE, AUTHORS_LINE):
        p = doc.add_paragraph(line)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for item in CONTENT:
        kind = item[0]
        if kind == "h1":
            doc.add_heading(item[1], level=1)
        elif kind == "h2":
            doc.add_heading(item[1], level=2)
        elif kind == "p":
            doc.add_paragraph(strip_md(item[1]))
        elif kind == "ul":
            for bullet in item[1]:
                doc.add_paragraph(strip_md(bullet), style="List Bullet")
        elif kind == "fig":
            fdir, caption = FIG_LOOKUP[item[1]]
            fpath = os.path.join(fdir, item[1])
            if os.path.exists(fpath):
                doc.add_picture(fpath, width=Inches(6.0))
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True
            cap.runs[0].font.size = Pt(9)
        elif kind == "table":
            add_table_docx(doc, item[1])
        elif kind == "appendix_entry":
            doc.add_paragraph(item[1])
    doc.save(os.path.join(REPORTS_DIR, "Final_Report_Group105.docx"))


def add_table_docx(doc, name):
    headers, rows, caption = table_spec(name)
    if headers is None:
        return
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Light Grid Accent 1"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = str(h)
    for row in rows:
        cells = table.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    cap = doc.add_paragraph(caption)
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(9)


# ---------------------------------------------------------------------------
# Build the parallel Markdown file (used as the pandoc -> PDF source)
# ---------------------------------------------------------------------------
def df_to_md_table(headers, rows):
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def table_to_markdown(name):
    headers, rows, caption = table_spec(name)
    if headers is None:
        return ""
    return df_to_md_table(headers, rows) + f"\n\n*{caption}*\n"


def build_markdown():
    lines = [f"# {TITLE}", "", f"*{TEAM_LINE}*", "", f"*{AUTHORS_LINE}*", ""]
    for item in CONTENT:
        kind = item[0]
        if kind == "h1":
            lines.append(f"\n# {item[1]}\n")
        elif kind == "h2":
            lines.append(f"\n## {item[1]}\n")
        elif kind == "p":
            lines.append(item[1] + "\n")
        elif kind == "ul":
            for b in item[1]:
                lines.append(f"- {b}")
            lines.append("")
        elif kind == "fig":
            fdir, caption = FIG_LOOKUP[item[1]]
            fpath = os.path.join(fdir, item[1])
            lines.append(f"![{caption}]({fpath})\n")
            lines.append(f"*{caption}*\n")
        elif kind == "table":
            lines.append(table_to_markdown(item[1]) + "\n")
        elif kind == "appendix_entry":
            lines.append(item[1] + "\n")
    md_path = os.path.join(REPORTS_DIR, "Final_Report_Group105.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return md_path


def build_pdf(md_path):
    pdf_path = os.path.join(REPORTS_DIR, "Final_Report_Group105.pdf")
    docx_path = os.path.join(REPORTS_DIR, "Final_Report_Group105.docx")

    try:
        env = os.environ.copy()
        env["PATH"] = "/usr/local/bin:" + env.get("PATH", "")
        result = subprocess.run(
            ["/opt/anaconda3/bin/pandoc", md_path, "-o", pdf_path,
             "--pdf-engine=/usr/local/bin/xelatex", "-V", "geometry:margin=1in"],
            cwd=REPORTS_DIR, capture_output=True, text=True, timeout=300, env=env,
        )
        if result.returncode == 0 and os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 2000:
            print("PDF generated via pandoc + xelatex.")
            return "pandoc+xelatex"
        else:
            print("pandoc+xelatex failed:", result.returncode, result.stderr[-2000:], file=sys.stderr)
    except Exception as e:
        print(f"pandoc+xelatex raised exception: {e}", file=sys.stderr)

    for soffice in ["soffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice"]:
        try:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", "--outdir", REPORTS_DIR, docx_path],
                capture_output=True, text=True, timeout=300,
            )
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 2000:
                print(f"PDF generated via LibreOffice ({soffice}).")
                return f"libreoffice ({soffice})"
        except Exception as e:
            print(f"LibreOffice attempt with {soffice} failed: {e}", file=sys.stderr)

    try:
        from docx2pdf import convert
        convert(docx_path, pdf_path)
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 2000:
            print("PDF generated via docx2pdf.")
            return "docx2pdf"
    except Exception as e:
        print(f"docx2pdf attempt failed: {e}", file=sys.stderr)

    print("ERROR: all PDF generation approaches failed.", file=sys.stderr)
    return None


if __name__ == "__main__":
    build_docx()
    md_path = build_markdown()
    method = build_pdf(md_path)

    docx_path = os.path.join(REPORTS_DIR, "Final_Report_Group105.docx")
    pdf_path = os.path.join(REPORTS_DIR, "Final_Report_Group105.pdf")
    print("\n--- Export summary ---")
    print("DOCX:", docx_path, os.path.getsize(docx_path) if os.path.exists(docx_path) else "MISSING")
    print("MD:  ", md_path, os.path.getsize(md_path) if os.path.exists(md_path) else "MISSING")
    print("PDF:", pdf_path, os.path.getsize(pdf_path) if os.path.exists(pdf_path) else "MISSING",
          f"(method: {method})")
