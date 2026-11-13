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
    ("p", "Millions of Latino families living in the United States send money home each "
          "year to relatives across Latin America and the Caribbean (LAC), and for many "
          "receiving households those remittances are the single largest source of income "
          "they have. In 2023 alone the U.S. sent over $150 billion to the region, more "
          "than any other single source country, yet the price of making that transfer "
          "has stayed stubbornly high. The UN's SDG 10.c calls for an average cost of 3% "
          "per transaction, but the corridors examined here average 5.44% as of 2025 Q1, "
          "and a handful of destinations, including Cuba (12.0%) and Guyana (8.7%), blow "
          "well past even that elevated average."),
    ("p", "We set out to understand what actually drives these costs and to build "
          "something senders can use to find cheaper options. Our data comes from the "
          "World Bank's Remittance Prices Worldwide (RPW) database, which tracks "
          "transaction-level pricing across hundreds of corridors worldwide, recording "
          "provider type, payment instrument, transfer speed, fees, exchange-rate margins, "
          "and destination country for each offering. This report finishes the three-part "
          "plan we outlined at the midterm: (1) exploratory data analysis and feature "
          "selection, condensed below since that work is already behind us; (2) fitting "
          "and comparing regression models, running a classification task at the 3% "
          "threshold, and clustering destinations, which is the new work in this report; "
          "and (3) shipping a web application that gives a sender a predicted cost for "
          "their own transfer. Everything here was built in Python: pandas, numpy, "
          "matplotlib, seaborn, scipy, scikit-learn, and statsmodels handled the analysis, "
          "while the web app runs on FastAPI and React/TypeScript. Figures referenced "
          "throughout are collected in the Appendix."),

    ("h1", "Proposed Dataset and Data Preprocessing"),
    ("p", "We work from a transaction-level extract of the World Bank's RPW database, "
          "subset down to United States-to-LAC corridors: 12,397 rows across 23 columns, "
          "one row per provider offering per corridor per quarter, covering the period "
          "2016_2Q through 2025_1Q. A few cleanup steps came first. speed actual and "
          "pickup method needed lower-casing and whitespace stripping to merge categories "
          "that were really duplicates but differed only in capitalization, and period "
          "was parsed into separate integer year and quarter columns, both for the trend "
          "analysis and for the temporal train/test split used throughout the rest of "
          "this report. We also consolidated firm_type (folding the two Money Transfer "
          "Operator variants into a single level) and dropped the 11 sparse Non-Bank FI "
          "rows, leaving 12,386 rows for every modeling task below. Missing values are "
          "concentrated almost entirely in the cc2 * fields (235 rows, 1.9%), which we "
          "use to engineer the is_non_transparent proxy flag; the free-text Standard Note "
          "is missing for 69.2% of rows, which is expected since it's only filled in when "
          "there's something unusual to disclose. Every other column, including the "
          "target cc1 total cost %, is fully populated."),

    ("h1", "Proposed Methodology"),
    ("p", "Our approach follows the model- and variable-selection framework taught in "
          "ISyE 7406 (Prof. Yajun Mei). Back at the midterm, we used information criteria "
          "(AIC/BIC via forward stepwise search), shrinkage-based selection (Ridge/LASSO), "
          "and PCA together to settle on a final predictor set\u2014destination_name, "
          "firm_type_clean, payment_instrument_group, access point, receiving network "
          "coverage, speed actual, pickup method, cc1 fx margin, year, and "
          "is_non_transparent\u2014encoded with one-hot (drop_first=True) for "
          "categoricals and standardization for numerics, giving 42 encoded features in "
          "total. This report builds on that same feature set with three new analyses, "
          "each evaluated on the identical pre-2023 train / 2023+ test temporal split "
          "(8,228 training rows, 4,158 test rows). First, we compare four regression "
          "models (Linear Regression, Ridge, LASSO, Random Forest) on RMSE, MAE, and "
          "R\u00b2. Second, we binarize the target into a \"high cost\" label whenever "
          "cc1 total cost % \u2265 3% (the SDG 10.c threshold) and compare Logistic "
          "Regression, Linear Discriminant Analysis (LDA), and Random Forest on accuracy, "
          "precision, recall, and F1. Third, we cluster the 16 LAC destination countries "
          "by their mean cost profile\u2014total cost, FX margin, fee, and transparency "
          "rate\u2014using both k-means (with k chosen via the elbow method and "
          "silhouette score) and hierarchical Ward-linkage clustering. The same "
          "production LassoCV model, refit on all available data, is what powers the "
          "deployed FastAPI/React web application described in its own section further "
          "below."),

    ("h1", "Analysis and Results"),

    ("h2", "Target Variable"),
    ("p", "Our target, cc1 total cost %, averages 5.34% with a median of 4.71% and a "
          "standard deviation of 3.03%, ranging from \u22124.09% all the way up to 34.0%. "
          "The distribution is noticeably right-skewed (skewness = 1.88, excess kurtosis "
          "= 6.33)\u2014a small set of outlier corridors with especially high-cost "
          "providers pulls the tail out. Even the 25th percentile, at 3.50%, sits just "
          "above the SDG 3% target, which tells us roughly three-quarters of all "
          "transactions still miss that benchmark."),
    ("fig", "01_target_distribution.png"),

    ("h2", "Cost by Destination Country"),
    ("p", "Destination turns out to be the single biggest driver of cost variation in "
          "this dataset. Cuba sits at 12.0% and Guyana at 8.9%\u2014both more than double "
          "the SDG target\u2014while Honduras (3.9%), Ecuador (4.0%), and El Salvador "
          "(4.0%) come closest to meeting it. Mexico, despite being the highest-volume "
          "corridor, averages 4.6%. That roughly 8-percentage-point gap between the "
          "cheapest and priciest corridors dwarfs anything attributable to firm type or "
          "payment instrument, which is part of what motivates the corridor clustering "
          "analysis later in this report."),
    ("fig", "03_cost_by_destination.png"),

    ("h2", "Cost by Firm Type"),
    ("p", "Banks turn out not to be meaningfully cheaper than Money Transfer Operators "
          "(MTOs): recomputed directly from the modeling subset, both average 5.34%, a "
          "negligible gap, and both land well above the 3% SDG target. MTOs dominate the "
          "market by sheer volume, accounting for 10,707 of the 12,386 modeling rows "
          "(86%)."),
    ("fig", "02_cost_by_firm_type.png"),

    ("h2", "Cost by Payment Instrument"),
    ("p", "Payment instrument does matter, though: cash-based transfers run higher and "
          "more variable than bank-account transfers or digital instruments, and "
          "credit-card transfers carry a clear premium on top of that\u2014most likely "
          "the card-processing fee passed through to the sender."),
    ("fig", "04_cost_by_payment_instrument.png"),

    ("h2", "Cost Over Time"),
    ("p", "Cost has stayed essentially flat from 2016 through 2025, bouncing between a "
          "low of 4.96% (2020) and a high of 5.59% (2024) with no sustained movement "
          "toward the SDG target in either direction. year turns out to be a fairly weak "
          "predictor on its own; we keep it mainly because it defines the temporal "
          "train/test split used everywhere else in this report."),
    ("fig", "05_cost_by_year.png"),

    ("h2", "Transparency Flag"),
    ("p", "The engineered is_non_transparent flag\u2014built from missing cc2 total "
          "cost % values as a proxy for non-disclosure\u2014applies to 1.9% of rows, and "
          "those flagged rows run modestly cheaper on average than transparent ones "
          "(4.80% vs. 5.35%). The proxy isn't perfect: 81% of flagged rows actually carry "
          "a \"send limits\" disclosure note rather than a true exchange-rate "
          "non-disclosure note, so we keep it as a candidate feature while being upfront "
          "about that limitation."),
    ("fig", "06_transparency_comparison.png"),

    ("h2", "Correlation Structure"),
    ("p", "Of the numeric fields, cc1 fx margin correlates most strongly with the target "
          "(r = 0.71), which makes sense since FX margin is a direct cost component. "
          "cc1 lcu fx rate and inter lcu bank fx correlate only weakly (r \u2248 "
          "0.06-0.07) and are currency-scale fields we exclude from the predictor set. "
          "All of the cc2 * fields correlate highly with the target by construction "
          "(r = 0.91 for cc2 total cost %, for instance) and are excluded outright as "
          "leakage variables."),
    ("fig", "07_correlation_heatmap.png"),

    ("h2", "Feature Selection (Information Criteria, Shrinkage, and PCA)"),
    ("p", "We relied on three complementary feature-selection methods, documented in "
          "full in the midterm's EDA & Feature Selection report (r-models/reports/"
          "EDA_FeatureSelection_Group105) and summarized here. Forward stepwise "
          "selection by BIC kept 9 of the 10 candidate blocks, entering them in this "
          "order: cc1 fx margin, pickup method, payment_instrument_group, "
          "destination_name, access point, receiving network coverage, speed actual, "
          "year, is_non_transparent\u2014firm_type_clean was the lone block it left out. "
          "LassoCV (5-fold CV, alpha = 0.001) zeroed out 6 of the 42 encoded features, "
          "with the largest surviving coefficients landing on pickup "
          "method_own/partner bank account, destination_name_Cuba, and access point_Bank "
          "branch,Internet. RidgeCV (alpha = 1.15), by contrast, shrinks every "
          "coefficient but zeroes out none of them. Running PCA on the numeric block "
          "(cc1 fx margin, year, is_non_transparent) confirmed those three features are "
          "close to orthogonal, so dimensionality reduction didn't buy us anything "
          "there. Encouragingly, the information-criterion and shrinkage methods "
          "converge on the same dominant predictors\u2014destination, payment "
          "instrument/pickup method, and FX margin\u2014which gives us confidence that "
          "the final feature set (destination_name, firm_type_clean, "
          "payment_instrument_group, access point, receiving network coverage, speed "
          "actual, pickup method, cc1 fx margin, year, is_non_transparent; 42 encoded "
          "features) is a solid one."),
    ("fig", "08_aic_bic_vs_nvars.png"),
    ("fig", "09_lasso_coef_path.png"),
    ("fig", "10_lasso_coefficients_bar.png"),
    ("fig", "11_pca_scree.png"),

    ("h2", "Regression Model Comparison"),
    ("p", "We fit four regression models on the identical final feature set\u2014"
          "one-hot encoded categoricals plus standardized numerics\u2014training on the "
          "pre-2023 rows (n = 8,228) and evaluating on the 2023+ holdout (n = 4,158): "
          "plain Linear Regression, RidgeCV, LassoCV, and a 400-tree Random Forest "
          "Regressor. RMSE is our primary metric here, with MAE and R\u00b2 reported "
          "alongside it for context."),
    ("table", "regression_comparison"),
    ("p", f"{best_reg['model']} comes out with the lowest test RMSE "
          f"({best_reg['test_rmse']:.3f}, R\u00b2 = {best_reg['test_r2']:.3f}), though "
          f"it's effectively tied with LASSO "
          f"({reg_comparison[reg_comparison['model']=='LASSO']['test_rmse'].iloc[0]:.3f}) and "
          f"Ridge ({reg_comparison[reg_comparison['model']=='Ridge']['test_rmse'].iloc[0]:.3f}). "
          f"The Random Forest posts the highest test RMSE ({worst_reg['test_rmse']:.3f}) "
          "but the lowest MAE of the four, which suggests it handles the bulk of "
          "ordinary transactions a bit better even though it struggles more to "
          "extrapolate to the handful of extreme-cost corridors (Cuba, Guyana) that "
          "dominate RMSE's squared-error penalty. That lines up with what we see from "
          "the shipped production model (LassoCV refit on all data, holdout RMSE "
          "\u2248 1.95, R\u00b2 \u2248 0.67 as reported in "
          "train_model.py/model/metadata.json): a linear, regularized model captures "
          "this dataset's structure about as well as a flexible nonparametric one does. "
          "That's informative in itself\u2014it suggests the relationship between our "
          "selected features and cost is close to additive and linear once destination, "
          "payment instrument, and FX margin are already encoded, leaving little extra "
          "signal for a tree ensemble to exploit."),
    ("fig", "12_regression_residuals.png"),
    ("p", f"Figure 12's residual plot for {best_reg['model']} tells a consistent story: "
          "residuals cluster near zero but trail off into a long right tail, mirroring "
          "the target's own right-skewed shape. A handful of very high-cost "
          "transactions, mostly from Cuba and Guyana, get systematically "
          "under-predicted, which makes sense since no linear model can fully absorb an "
          "outlier destination's level shift from a single pooled coefficient."),
    ("fig", "13_lasso_ridge_coefficients.png"),
    ("table", "regression_coefficients"),
    ("p", "LASSO's biggest standardized coefficients echo what we already found during "
          "feature selection at the midterm\u2014destination and pickup-method dummies, "
          "plus cc1 fx margin, lead the pack. Ridge's coefficients on those same "
          "features point in the same direction but are uniformly smaller, which fits "
          "with its smoother L2 shrinkage compared to LASSO's sparser L1 penalty."),
    ("fig", "14_rf_feature_importance.png"),
    ("table", "rf_feature_importance"),
    ("p", "The Random Forest's impurity-based importances single out the same handful "
          "of dominant predictors\u2014cc1 fx margin and the Cuba/Guyana destination "
          "dummies come out on top here too\u2014which corroborates the linear models' "
          "coefficients using a completely different, nonparametric approach."),

    ("h2", "Classification at the 3% SDG Threshold"),
    ("p", f"We binarized the target at the SDG 10.c threshold\u2014cc1 total cost % "
          f"\u2265 {SDG_TARGET:.0f}% counts as \"high cost\"\u2014and the resulting "
          "label skews heavily toward that class (86.4% of the pre-2023 training rows, "
          "85.4% of the 2023+ test rows), which tracks with the EDA finding that "
          "three-quarters of transactions already miss the target. Logistic Regression, "
          "LDA, and a 400-tree Random Forest Classifier were all trained on the same "
          "encoded feature set and temporal split."),
    ("table", "classification_comparison"),
    ("p", f"By F1, {best_clf['model']} comes out on top "
          f"(F1 = {best_clf['f1']:.3f}, accuracy = {best_clf['accuracy']:.3f}, precision = "
          f"{best_clf['precision']:.3f}, recall = {best_clf['recall']:.3f}). Accuracy "
          "alone would be misleading here given the class imbalance\u2014a classifier "
          "that just always guessed \"high cost\" would already land around 85%\u2014so "
          "F1 and recall matter more for comparing these models. The Random Forest's "
          "higher recall means it misses fewer of the minority low-cost transactions, "
          "which are exactly the ones a sender is trying to find."),
    ("fig", "15_confusion_by_firmtype.png"),
    ("table", "classification_by_firmtype"),
    ("p", f"Splitting the results by firm type, the {best_clf['model']} classifier "
          f"performs "
          f"{'similarly' if abs(bank_row['f1'] - mto_row['f1']) < 0.03 else 'differently'} "
          f"for Banks (F1 = {bank_row['f1']:.3f}, n = {int(bank_row['n'])}) and Money "
          f"Transfer Operators (F1 = {mto_row['f1']:.3f}, n = {int(mto_row['n'])}). That "
          "tells us the model isn't just learning a shortcut based on firm type\u2014"
          "it's actually picking up corridor- and instrument-level cost signal that "
          "holds up across provider types."),
    ("fig", "16_classification_by_destination.png"),
    ("table", "classification_by_destination"),
    ("p", f"Performance varies a fair amount by destination: "
          f"{best_dest_clf['destination_name']} posts the highest F1 "
          f"({best_dest_clf['f1']:.3f}, n = {int(best_dest_clf['n'])}), while "
          f"{worst_dest_clf['destination_name']} comes in lowest "
          f"(F1 = {worst_dest_clf['f1']:.3f}, n = {int(worst_dest_clf['n'])}). The "
          "weaker destinations tend to share one of two traits\u2014either very few "
          "2023+ holdout observations, or a cost distribution that sits right on the 3% "
          "boundary\u2014so a handful of misclassifications can swing that corridor's "
          "F1 quite a bit."),

    ("h2", "Corridor Clustering"),
    ("p", "We rolled the 16 LAC destination countries up into a country-level "
          "profile\u2014mean total cost, mean FX margin, mean fee in local currency "
          "units, and transparency rate\u2014standardized it, and ran both k-means and "
          "hierarchical (Ward-linkage) clustering on top. For k-means, we picked k by "
          "looking at the elbow (inertia) curve alongside the silhouette score across "
          "k = 2..7."),
    ("fig", "17_kmeans_elbow_silhouette.png"),
    ("table", "cluster_metrics"),
    ("p", f"Silhouette score peaks at k = {best_k} "
          f"(silhouette = {best_k_row['silhouette']:.3f}) and falls off steadily as k "
          "grows, but that k=2 split is really just driven by one extreme outlier: "
          f"Cuba (mean cost {cuba_row['mean_total_cost']:.2f}%, mean FX margin "
          f"{cuba_row['mean_fx_margin']:.2f}) ends up alone in its own cluster, against "
          f"a single large cluster holding the other 15 destinations (mean cost "
          f"{rest_mean_cost:.2f}%). That's a real finding\u2014Cuba's cost profile is "
          "simply far enough from everyone else's that no regrouping beats a clean "
          "two-way split\u2014but it doesn't tell us much about how the other 15 "
          "destinations relate to each other. For that finer-grained picture, we also "
          "report the k=4 partition below, which has a lower but still reasonable "
          "silhouette score."),
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
    "Looking at k=4 instead, the partition splits into four cost tiers that are much "
    "easier to interpret (ordered here from cheapest to most expensive mean total "
    "cost): " + "; ".join(k4_lines) + ". The cheapest tier is made up of destinations "
    "with low, often near-zero, FX margins; the priciest non-Cuba tier (Brazil, "
    "Colombia, Costa Rica, Mexico, and similar corridors) pairs moderate-to-high FX "
    "margins with high transaction volume; and Dominican Republic, Guyana, and Jamaica "
    "form their own mid-to-high-cost tier with the highest FX margins outside of Cuba. "
    "All of this lines up with the regression and classification results above\u2014FX "
    "margin and destination are the dominant cost drivers, and the clustering shows "
    "destinations grouping along almost that exact same axis."
))
CONTENT.append(("fig", "19_dendrogram.png"))
CONTENT.append((
    "p",
    "Figure 19's hierarchical dendrogram (Ward linkage) tells essentially the same "
    "story from another angle: Cuba doesn't join the rest of the tree until by far the "
    "largest linkage distance, confirming its status as a singleton outlier at low k, "
    "while the remaining destinations merge into recognizable sub-groups that line up "
    "with the k=4 k-means partition above. Two very different clustering methods "
    "landing on the same corridor groupings is reassuring convergent evidence."
))

CONTENT += [
    ("h1", "Web Application"),
    ("p", "To get this modeling work in front of an actual money sender, we built a "
          "small full-stack web application: a FastAPI backend paired with a "
          "React/TypeScript single-page frontend. On startup, the backend "
          "(web-service/backend/main.py) loads the shipped production model\u2014the "
          "LassoCV pipeline from r-models/model/model.pkl, refit on all available "
          "data\u2014along with its metadata.json, and it exposes two synchronous REST "
          "endpoints. GET /metadata returns the feature list, the valid category "
          "values, and the historical mean FX margin by destination, which we use to "
          "pre-fill a sensible default. POST /predict takes a sender's inputs "
          "(destination, firm type, payment instrument, access point, receiving "
          "network coverage, speed, pickup method, FX margin, year, and a transparency "
          "flag) and returns the model's predicted cc1 total cost %, plus an estimated "
          "fee in USD if a transfer amount was supplied. There's no background task "
          "queue or async job system here\u2014the architecture is intentionally a "
          "simple, synchronous request/response flow, which is all a single fast "
          "model-inference call really needs."),
    ("p", "The frontend (web-service/frontend/src/App.tsx) is just a single form "
          "mirroring the backend's input schema: dropdowns for the categorical fields, "
          "populated from GET /metadata so the UI can never drift out of sync with "
          "what the model was actually trained on; a numeric field for cc1 fx margin, "
          "pre-filled with the historical destination-level average; and a submit "
          "button that calls POST /predict and shows the predicted cost percentage "
          "(plus the estimated fee) back to the user. Because the categories are "
          "server-driven, if the model is ever retrained on an updated dataset with a "
          "new destination or payment instrument category, the frontend doesn't need a "
          "single line of code changed to stay correct."),
    ("p", "web-service/sync_model.sh is what glues the two halves of this monorepo "
          "together: it copies the trained artifacts (model.pkl and metadata.json) "
          "from r-models/model/ into web-service/backend/model/, so the backend always "
          "serves whichever model r-models/train_model.py produced most recently, "
          "without ever needing to import or depend on the r-models training code "
          "directly."),

    ("h1", "Conclusions and Next Steps"),

    ("h2", "Lessons Learned"),
    ("ul", [
        "Fields that look alike on the surface can hide very different data "
        "quality: the cc1/cc2 comparator fields are mechanically correlated by "
        "construction, and if we'd naively thrown the cc2 * fields into the feature "
        "set, we would have silently leaked the target. We only caught this by "
        "actually reading the RPW database's field definitions\u2014a correlation "
        "matrix alone wouldn't have flagged it.",
        "The is_non_transparent proxy is a good illustration of how a feature can "
        "look clean on its own but turn out to be confounded once you dig "
        "further\u2014cross-tabulating it against the free-text note field revealed "
        "it was mixing send-limit disclosures with true FX non-disclosure. Any "
        "engineered feature built from a single missingness pattern deserves this "
        "kind of second look before you trust it.",
        "We were initially surprised that the linear/regularized models (LASSO, "
        "Ridge, plain OLS) and a 400-tree Random Forest all landed on almost the "
        "same test RMSE\u2014we expected the Random Forest to meaningfully "
        "outperform on nonlinearities. In hindsight, it makes sense: most of the "
        "signal was already captured by the carefully selected categorical feature "
        "set (destination, payment instrument, pickup method) before any "
        "model-side nonlinearity had a chance to add value. One extension worth "
        "trying would be engineering explicit interaction terms (e.g. destination "
        "\u00d7 payment instrument) for the linear models, to see whether that "
        "closes the remaining gap or whether the Random Forest is already "
        "capturing those interactions on its own.",
        "With only 16 destination countries to work with, clustering results end "
        "up fairly sensitive to the exact choice of k, and a single extreme "
        "outlier (Cuba) ends up dominating the silhouette-optimal partition. We "
        "deliberately reported both the statistically \"optimal\" k and a "
        "secondary, more interpretable k so that one metric wouldn't understate "
        "the richer structure the dendrogram actually shows.",
        "If we were starting over, we'd build the FastAPI/React web application "
        "earlier in the project instead of tacking it on after the modeling work. "
        "Having a live, interactive view into the model's predictions surfaced a "
        "few feature-encoding edge cases\u2014like the casing normalization on "
        "speed actual and pickup method\u2014that were easy to miss while we were "
        "only looking at notebook output.",
    ]),

    ("h2", "Key Findings"),
    ("ul", [
        "Average US-to-LAC remittance costs (5.44%) have stayed well above the "
        "UN's SDG 10.c target of 3% throughout 2016-2025, with no meaningful move "
        "downward over that span.",
        "Destination country drives the most cost variation by far, with an 8+ "
        "percentage point gap separating the cheapest corridor (Honduras, 3.9%) "
        "from the most expensive (Cuba, 12.0%).",
        f"{best_reg['model']} delivered the best regression performance on the "
        f"2023+ holdout (RMSE = {best_reg['test_rmse']:.3f}, R\u00b2 = "
        f"{best_reg['test_r2']:.3f}), with LASSO and Ridge close behind; the "
        "Random Forest never meaningfully outperformed the linear models, which "
        "suggests the underlying relationship is close to additive once "
        "destination, payment instrument, and FX margin are encoded.",
        f"At the 3% SDG threshold, {best_clf['model']} came out as the strongest "
        f"classifier (F1 = {best_clf['f1']:.3f}), though its performance shifts "
        "meaningfully by destination, reflecting both class imbalance and "
        "genuine differences across corridors.",
        f"Clustering the 16 destinations by cost profile isolates Cuba as its "
        f"own outlier cluster at the silhouette-optimal k={best_k}, while a finer "
        "k=4 partition uncovers an interpretable low/medium/high-FX-margin "
        "tiering among the remaining 15 destinations\u2014a pattern that k-means "
        "and hierarchical clustering both agree on.",
        "cc1 fx margin, payment instrument, pickup method, and destination stand "
        "out as the strongest predictors no matter which method we "
        "apply\u2014information-criterion selection, shrinkage, Random Forest "
        "importance, and corridor clustering all point to the same handful of "
        "drivers.",
        "A deployed FastAPI/React web application now lets a sender plug in "
        "their own intended transfer and get a predicted cost back, putting the "
        "project's modeling work directly in front of the people it's meant for.",
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
