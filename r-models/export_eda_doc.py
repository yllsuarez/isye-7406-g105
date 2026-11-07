"""
Export the EDA + Feature Selection content (and only that content) from the Team 105
remittance-cost project into:
  - r-models/reports/EDA_FeatureSelection_Group105.docx
  - r-models/reports/EDA_FeatureSelection_Group105.md
  - r-models/reports/EDA_FeatureSelection_Group105.pdf

This script recomputes the small set of summary tables it needs directly from the
dataset (missingness, correlation matrix, VIF, final LassoCV coefficients, final
feature list) so that the report tables are guaranteed consistent with the notebook's
logic, and reuses the figures already rendered by eda_feature_selection.ipynb under
reports/figures/. It writes clean narrative prose (not code listings).

Run from the r-models/ directory:
    /opt/anaconda3/bin/python3 export_eda_doc.py
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "rpw_dataset_2011_2025_q1.csv")
FIG_DIR = os.path.join(HERE, "reports", "figures")
REPORTS_DIR = os.path.join(HERE, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

SDG_TARGET = 3.0

# ---------------------------------------------------------------------------
# Recompute the handful of tables/values the report needs, from scratch, so the
# report is self-consistent with the notebook's methodology.
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

for col in ["speed actual", "pickup method"]:
    df[col] = df[col].str.strip().str.lower()

period_split = df["period"].str.extract(r"(?P<year>\d{4})_(?P<quarter>\d)Q")
df["year"] = period_split["year"].astype(int)
df["quarter"] = period_split["quarter"].astype(int)
df["is_non_transparent"] = df["cc2 total cost %"].isnull()

missing = df.isnull().sum().to_frame("n_missing")
missing["pct_missing"] = (missing["n_missing"] / len(df) * 100).round(2)
missing = missing[missing["n_missing"] > 0].sort_values("n_missing", ascending=False)

numeric_cols = [
    "cc1 lcu fee", "cc1 lcu fx rate", "cc1 fx margin", "cc1 total cost %",
    "cc2 lcu fee", "cc2 lcu fx rate", "cc2 fx margin", "cc2 total cost %",
    "inter lcu bank fx", "year",
]
corr = df[numeric_cols].corr(method="pearson").round(2)

model_df = df.copy()
model_df["firm_type_clean"] = model_df["firm_type"].replace(
    {"Money Transfer Operator / Building Society": "Money Transfer Operator"}
)
model_df = model_df[model_df["firm_type_clean"] != "Non-Bank FI"].copy()

categorical_cols = [
    "firm_type_clean", "payment_instrument_group", "access point",
    "receiving network coverage", "destination_name", "speed actual", "pickup method",
]
numeric_cols_candidate = ["cc1 fx margin", "year"]

X_cat = pd.get_dummies(model_df[categorical_cols], drop_first=True)
X_num_raw = model_df[numeric_cols_candidate].copy()
X_flag = model_df[["is_non_transparent"]].astype(int)
scaler = StandardScaler()
X_num_scaled = pd.DataFrame(
    scaler.fit_transform(X_num_raw), columns=numeric_cols_candidate, index=model_df.index
)
X = pd.concat([X_num_scaled, X_flag, X_cat], axis=1)
y = model_df["cc1 total cost %"].values


def compute_vif(frame):
    vifs = {}
    for col in frame.columns:
        y_col = frame[col].values
        X_other = sm.add_constant(frame.drop(columns=[col]).values)
        fit = sm.OLS(y_col, X_other).fit()
        r2 = fit.rsquared
        vifs[col] = np.inf if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return pd.Series(vifs, name="VIF").sort_values(ascending=False)


numeric_block = X_num_raw.assign(is_non_transparent=X_flag["is_non_transparent"].astype(float))
vif_series = compute_vif(numeric_block).round(3)

alphas = np.logspace(-3, 2, 50)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
lasso_cv = LassoCV(alphas=alphas, cv=cv, max_iter=20000, random_state=42)
lasso_cv.fit(X.astype(float).values, y)
coef_series = pd.Series(lasso_cv.coef_, index=X.columns).sort_values(key=np.abs, ascending=False)
top_coefs = coef_series.head(15).round(4)
n_zero = int(np.sum(np.isclose(lasso_cv.coef_, 0.0)))

final_source_features = [
    "destination_name", "firm_type_clean", "payment_instrument_group", "access point",
    "receiving network coverage", "speed actual", "pickup method", "cc1 fx margin",
    "year", "is_non_transparent",
]

# ---------------------------------------------------------------------------
# Figure list, in the exact order the notebook renders/saves them.
# ---------------------------------------------------------------------------
FIGURES_EDA = [
    ("01_target_distribution.png", "Figure 1. Distribution of cc1 total cost % (histogram and boxplot), "
                                    "with the 3% SDG target marked."),
    ("02_cost_by_firm_type.png", "Figure 2. cc1 total cost % by firm type."),
    ("03_cost_by_destination.png", "Figure 3. Mean cc1 total cost % by destination, sorted descending."),
    ("04_cost_by_payment_instrument.png", "Figure 4. cc1 total cost % by payment instrument group."),
    ("05_cost_by_year.png", "Figure 5. Mean cc1 total cost % by year, 2016-2025."),
    ("06_transparency_comparison.png", "Figure 6. cc1 total cost % by transparency proxy "
                                        "(cc2 total cost % missing vs. present)."),
    ("07_correlation_heatmap.png", "Figure 7. Pearson correlation heatmap of numeric RPW fields plus year."),
]
FIGURES_FS = [
    ("08_aic_bic_vs_nvars.png", "Figure 8. AIC/BIC vs. number of features in forward stepwise selection."),
    ("09_lasso_coef_path.png", "Figure 9. LASSO regularization path (coefficients vs. alpha)."),
    ("10_lasso_coefficients_bar.png", "Figure 10. Final LassoCV coefficients, top 20 by magnitude."),
    ("11_pca_scree.png", "Figure 11. PCA scree plot of the numeric candidate block."),
]

for fname, _ in FIGURES_EDA + FIGURES_FS:
    fpath = os.path.join(FIG_DIR, fname)
    if not os.path.exists(fpath):
        print(f"WARNING: expected figure not found: {fpath}", file=sys.stderr)

# ---------------------------------------------------------------------------
# Narrative text blocks (reused/adapted from the notebook's markdown cells).
# Each entry is (level, text) where level 0 = body paragraph, 1/2 = heading levels,
# 'fig' = figure reference (handled separately), 'table' = named table to insert.
# ---------------------------------------------------------------------------
CONTENT = [
    ("h1", "Exploratory Data Analysis"),

    ("h2", "1.1 Data Overview"),
    ("p", "The dataset is a transaction-level extract from the World Bank Remittance "
          "Prices Worldwide (RPW) database, covering United States to Latin America and "
          "Caribbean (LAC) remittance corridors. It holds 12,397 rows and 23 columns, one "
          "row per provider offering per corridor per quarter, spanning period 2016_2Q "
          "through 2025_1Q. The extract is already restricted to the post-schema-change "
          "reporting period, so there is no pre-2016 data to account for and no "
          "schema-harmonization step is needed."),
    ("p", "Only the cc2 * fields (cc2 lcu fee, cc2 lcu fx rate, cc2 fx margin, "
          "cc2 total cost %) have missing values, each absent for exactly 235 rows "
          "(1.9% of rows). Standard Note is missing for a much larger share -- 8,585 rows "
          "(69.2%) -- but that is expected, since it is a free-text disclosure note that "
          "simply is not written when there is nothing unusual to disclose. Every other "
          "column, including the target cc1 total cost %, is fully populated."),
    ("table", "missingness"),

    ("h2", "1.2 Data Cleaning for EDA Purposes"),
    ("p", "The saved CSV is never modified; all cleaning happens in memory. speed actual "
          "and pickup method are lower-cased and stripped to merge near-duplicate "
          "categories caused by inconsistent capitalization (e.g. \"Less than one hour\" "
          "vs. \"less than one hour\"; \"Own/partner bank account\" vs. \"Own/partner Bank "
          "account\"). The period field (e.g. \"2016_2Q\") is parsed into integer year and "
          "quarter columns for the trend analysis below and for the temporal train/test "
          "split planned in downstream modeling work."),

    ("h2", "1.3 Target Variable: cc1 total cost %"),
    ("p", "The target variable, cc1 total cost %, is the total cost (as a percentage of "
          "the transferred amount) a sender pays using the cheapest ranked (cc1) product "
          "for a given firm/corridor/quarter. Its distribution is right-skewed, with a "
          "heavy tail driven by a small number of very expensive corridors and products. "
          "That lines up with the proposal's observation that Cuba (about 12.0% average) "
          "and Guyana (about 8.7% average) are high-cost outlier destinations sitting well "
          "above both the corridor average (5.44%) and the UN SDG 10.c target of 3%. Most "
          "of the mass falls between roughly 1% and 8%, but the heavy tail means a plain "
          "OLS model could end up sensitive to these few corridors, so regularization "
          "(Ridge/LASSO) and possibly a log-transform of the target are worth considering "
          "in later modeling."),
    ("fig", "01_target_distribution.png"),

    ("h2", "1.4 Cost by Firm Type, Destination, Payment Instrument, and Year"),
    ("p", "Banks tend to run notably higher and more variable total costs than Money "
          "Transfer Operators (MTOs), which dominate the market by volume (10,673 of "
          "12,397 rows). The \"Money Transfer Operator / Building Society\" and "
          "\"Non-Bank FI\" categories are sparse (34 and 11 rows respectively), making "
          "them candidates for consolidation or removal in the feature-selection subset."),
    ("fig", "02_cost_by_firm_type.png"),
    ("p", "Destination is clearly a strong driver of cost: Cuba and Guyana sit far above "
          "both the corridor average and the 3% SDG line, while several destinations -- "
          "Mexico among them, the highest-volume corridor -- sit close to or even below "
          "the SDG target. This spread is reason enough to include destination_name/"
          "corridor as a categorical predictor, and it is also what motivates the "
          "descriptive clustering step planned later in the project."),
    ("fig", "03_cost_by_destination.png"),
    ("p", "Cash-based transfers (the largest group, 3,780 rows) and card-based "
          "instruments show different cost profiles, with the \"Multiple\" group landing "
          "somewhere in between. That is why the cleaned payment_instrument_group (6 "
          "categories) is used rather than the raw 16-value payment instrument column, "
          "which contains compound strings such as \"Bank account transfer,Cash\" that do "
          "not one-hot encode meaningfully."),
    ("fig", "04_cost_by_payment_instrument.png"),
    ("p", "Mean cost moves only modestly from year to year across 2016-2025 and stays "
          "well above the 3% SDG target throughout, so there is no clear long-run "
          "downward trend toward the SDG goal in this corridor set. year is therefore "
          "treated as a secondary, lower-signal predictor, while still being retained for "
          "the planned pre-2023/2023+ temporal train/test split."),
    ("fig", "05_cost_by_year.png"),

    ("h2", "1.5 Transparency Investigation"),
    ("p", "The dataset has no explicit transparency boolean. Per the proposal, a "
          "transaction is considered non-transparent when the receiving-side exchange "
          "rate is not disclosed. We test a candidate proxy for this: cc2 total cost % is "
          "null (235 rows) wherever the second comparator product's cost -- which "
          "requires the FX rate to compute -- could not be computed or disclosed. Rows "
          "flagged non-transparent by this proxy are a small subset (about 1.9%), and the "
          "crosstab shows they are not well explained by routine disclosure notes. In "
          "fact the pattern runs opposite to what a simple \"missing note = transparent\" "
          "story would predict: 204 of 235 (87%) of the flagged rows do have a Standard "
          "Note present, versus only about 30% of the transparent group (3,608 of "
          "12,162). Looking at the note text itself, the dominant reason is \"Send limits "
          "on amount\" (191 of the 235 rows, 81%), which is a transaction-limit "
          "disclosure, not an exchange-rate-disclosure statement. So the cc2-null proxy "
          "is really picking up a mix of true non-disclosure and an unrelated send-limit "
          "restriction, making it a noisier signal for \"exchange rate not disclosed\" "
          "than the proposal's framing implies. The non-transparent group also shows a "
          "modestly different cost distribution. We retain is_non_transparent as an "
          "engineered binary candidate feature, but flag it explicitly as an imperfect "
          "proxy (not a ground-truth transparency flag) conflated with send-limit "
          "disclosures."),
    ("fig", "06_transparency_comparison.png"),

    ("h2", "1.6 Correlation Matrix"),
    ("p", "Pearson correlations were computed among all numeric RPW fields plus the "
          "engineered year column, to understand (a) mechanical multicollinearity "
          "between the cc1 and cc2 fields, (b) which numeric fields correlate most with "
          "the target, and (c) the implication for shrinkage-method choice downstream."),
    ("fig", "07_correlation_heatmap.png"),
    ("table", "correlation"),
    ("p", "cc1 total cost % and cc2 total cost % are strongly positively correlated (as "
          "are the lcu fee and fx margin pairs). This is mechanical: cc1 and cc2 are the "
          "two cheapest comparator products for the same firm/corridor/quarter, priced "
          "off the same underlying FX market and fee schedule, so they naturally move "
          "together. Because cc2 * fields are partially derived from, and highly "
          "predictive of, the target by construction, they are excluded from the "
          "predictor set in the Feature Selection section to avoid leakage. Among the "
          "remaining independent numeric fields, cc1 fx margin shows the strongest "
          "correlation with the target, which makes sense since fx margin is a direct "
          "fee component. cc1 lcu fx rate and inter lcu bank fx correlate weakly, "
          "consistent with them being currency-scale fields rather than true cost "
          "drivers, and year has very weak correlation with the target. Once the cc2 * "
          "leakage fields are dropped, the remaining numeric block is small and not "
          "highly collinear, so neither Ridge's L2 shrinkage nor LASSO's sparsity is "
          "strictly necessary to control numeric multicollinearity. The categorical "
          "encoding is another story: many one-hot dummies (e.g. per destination) "
          "reintroduce enough dimensionality that both methods are still worth "
          "comparing, and LASSO's ability to zero out redundant dummy levels is "
          "particularly useful there."),

    ("h2", "1.7 EDA Summary"),
    ("ul", [
        "The dataset is clean and complete except for the mechanically linked cc2 * "
        "fields (235 missing rows, 1.9%) and the free-text Standard Note (8,585 missing, "
        "69.2%).",
        "The target cc1 total cost % is right-skewed, with Cuba and Guyana as clear "
        "high-cost outlier destinations well above the 3% SDG 10.c target.",
        "destination_name/corridor and firm_type are strong categorical drivers of cost; "
        "payment_instrument_group also differentiates cost meaningfully.",
        "There is no strong long-run time trend in mean cost from 2016 to 2025.",
        "cc2 total cost % being null is a usable, if imperfect, transparency proxy.",
        "cc1 fx margin is the strongest independent numeric correlate of the target; "
        "cc1 lcu fee and inter lcu bank fx are currency-scale fields that need caution "
        "before use as cross-country-comparable numeric predictors.",
        "cc2 * fields must be excluded from the predictor set (target leakage); "
        "firm_type's rare categories (Non-Bank FI, MTO/Building Society) need "
        "consolidation or filtering for modeling.",
    ]),

    ("h1", "Feature Selection"),
    ("p", "This section is grounded in the course's model/variable selection "
          "methodology (ISyE 7406, Prof. Yajun Mei). Information criteria (AIC, BIC, "
          "Mallow's Cp) score a model with k parameters and maximized log-likelihood as "
          "IC = -2*log-likelihood + penalty*k; AIC uses penalty 1 (equivalently 2k), "
          "while BIC uses penalty log(n) -- a heavier penalty for larger samples that "
          "favors sparser models -- and Mallow's Cp is an analogous criterion under "
          "squared-error loss. Forward/backward/stepwise sequential search greedily adds "
          "or removes one variable (or block of variables) at a time, choosing at each "
          "step the change that most improves the information criterion, and stopping "
          "when no further improvement is available. Ridge regression (L2 penalty) "
          "solves argmin ||Y-X*beta||^2 + lambda*sum(beta_j^2), shrinking all "
          "coefficients smoothly toward zero but, generically, never exactly to zero. "
          "LASSO (L1 penalty) solves argmin ||Y-X*beta||^2 + lambda*sum(|beta_j|); "
          "because the L1 constraint region has corners located on the coordinate axes, "
          "the optimum frequently lands exactly on an axis, driving some coefficients to "
          "exactly zero -- i.e. LASSO performs implicit variable selection as a "
          "byproduct of regularization. PCA/PLS provide complementary dimensionality "
          "views: PCA finds orthogonal linear combinations of the numeric predictors "
          "that maximize explained variance (unsupervised, ignores the target), while "
          "PLS performs the analogous decomposition while maximizing covariance with "
          "the target (supervised)."),

    ("h2", "2.1 Candidate Feature Set and Encoding"),
    ("p", "Candidate categorical predictors: firm_type (consolidated -- the two MTO "
          "variants merged into one \"Money Transfer Operator\" level; the 11 Non-Bank FI "
          "rows dropped from the modeling subset as too sparse per the proposal), "
          "payment_instrument_group, access point, receiving network coverage, "
          "destination_name, normalized speed actual, and normalized pickup method. "
          "Candidate numeric predictors: cc1 fx margin, engineered year, and the "
          "engineered is_non_transparent flag."),
    ("p", "Excluded from the candidate numeric set: cc1 lcu fee and inter lcu bank fx "
          "are expressed in local currency units, so they are not comparable across the "
          "16 destination countries without normalization -- using them as raw "
          "cross-country numeric features would implicitly encode currency scale rather "
          "than a meaningful cost driver. Both are excluded here; a later modeling "
          "notebook could instead engineer a currency-normalized ratio (e.g. fee as a "
          "percentage of a reference transfer amount) if that data were available. All "
          "cc2 * fields are also excluded per the correlation-based leakage discussion "
          "above. After dropping the 11 Non-Bank FI rows and consolidating firm_type, "
          "12,386 rows remain for feature selection. Categorical predictors are one-hot "
          "encoded (pd.get_dummies, drop_first=True) and numeric predictors are "
          "standardized (StandardScaler), producing a 42-column encoded feature matrix."),

    ("h2", "2.2 Correlation / Multicollinearity Pruning"),
    ("p", "Correlation and Variance Inflation Factor (VIF, computed manually as "
          "1/(1-R^2) from regressing each numeric column on the others via OLS, which "
          "avoids any dependency issue with the statsmodels VIF helper) were re-examined "
          "for the small numeric candidate block only (cc1 fx margin, year, "
          "is_non_transparent); the cc1/cc2 mechanically tied fields were already "
          "excluded above, so there is no need to re-check them here."),
    ("table", "vif"),
    ("p", "All VIF values for the numeric block fall well under the common "
          "rule-of-thumb threshold of 10, confirming these three numeric/engineered "
          "features are not collinear with each other. VIF is not computed across the "
          "full one-hot-encoded dummy matrix, since it is only meaningful for "
          "continuous/near-continuous regressors; collinearity among dummy groups is "
          "instead handled naturally by drop_first=True reference-level encoding and by "
          "the shrinkage methods below."),

    ("h2", "2.3 Model/Variable Selection via Information Criteria"),
    ("p", "A greedy forward stepwise selection was run using statsmodels OLS, scoring "
          "each candidate addition by AIC and BIC, over a manageable set of candidate "
          "\"blocks\": the three numeric/engineered features individually, plus each "
          "categorical predictor's full dummy group added together, since adding a "
          "factor should add all of its dummy columns at once. The procedure stops "
          "adding blocks once BIC stops improving."),
    ("fig", "08_aic_bic_vs_nvars.png"),
    ("p", "destination_name, payment_instrument_group/pickup method, and cc1 fx margin "
          "are added earliest and contribute the largest BIC improvements, with later "
          "blocks contributing progressively smaller gains and the curve flattening "
          "once these dominant drivers are included -- consistent with the EDA findings "
          "above. Note that firm_type_clean is the one candidate block not in the "
          "selected list: it was tested last and did not improve BIC further at that "
          "point, so the greedy search stopped before adding it (see Section 2.6 for why "
          "it is nonetheless kept in the final recommended feature set)."),

    ("h2", "2.4 Shrinkage-Based Selection: Ridge vs. LASSO"),
    ("p", "RidgeCV and LassoCV were fit on the full encoded, scaled feature matrix "
          "(all numeric plus all one-hot dummy columns) to predict cc1 total cost %, "
          "using 5-fold cross-validation over a log-spaced alpha grid "
          "(np.logspace(-3, 2, 50))."),
    ("fig", "09_lasso_coef_path.png"),
    ("fig", "10_lasso_coefficients_bar.png"),
    ("table", "lasso_top"),
    ("p", f"LassoCV's cross-validated alpha zeroed out {n_zero} of the {X.shape[1]} "
          "encoded coefficients. RidgeCV, by contrast, shrinks all coefficients smoothly "
          "and keeps every feature in the model (no exact zeros, by construction) -- "
          "which tracks with the numeric block alone not being strongly collinear, "
          "though the large one-hot dummy block still benefits from some shrinkage to "
          "stabilize estimates across rare categories. LassoCV, at its cross-validated "
          "alpha, drives a number of lower-signal dummy levels (sparse destination or "
          "access-point categories) to exactly zero, performing the implicit feature "
          "selection that is LASSO's signature property, while clearly keeping "
          "high-magnitude coefficients such as cc1 fx margin and the strongest "
          "destination/payment-instrument dummies. This agrees qualitatively with the "
          "information-criterion-based forward selection above: the same handful of "
          "blocks (destination, payment instrument group/pickup method, fx margin) "
          "emerge as dominant drivers under both approaches, which is reassuring "
          "evidence that the selection is not an artifact of either single method."),

    ("h2", "2.5 Complementary View: PCA on the Numeric Block"),
    ("p", "PCA was applied only to the small numeric block (cc1 fx margin, year, "
          "is_non_transparent) as a quick diagnostic of whether dimensionality reduction "
          "would be worthwhile there; it was not applied to the full dummy matrix, since "
          "PCA on one-hot categorical encodings is not usually meaningful or "
          "interpretable."),
    ("fig", "11_pca_scree.png"),
    ("p", "With only three numeric/engineered features, already shown to be close to "
          "uncorrelated, PCA does not find much redundancy to compress: all three "
          "components are needed to explain close to 100% of the variance, and no "
          "single component dominates. Dimensionality reduction is not very useful "
          "here, which is expected given the numeric candidate block was intentionally "
          "kept small. PCA/PLS would likely earn its keep if a richer, less-curated "
          "numeric feature set were used instead."),

    ("h2", "2.6 Final Selected Feature List"),
    ("p", "Combining the information-criterion-based forward selection, the "
          "shrinkage-based evidence, the exclusions justified above (currency-scale "
          "cc1 lcu fee/inter lcu bank fx; leakage cc2 * fields), and the VIF check "
          "(all numeric VIFs well under 10), the following source columns "
          "(pre-encoding) are recommended as the feature set to hand off to the "
          "regression/classification modeling notebooks:"),
    ("ul", [f"`{c}`" for c in final_source_features]),
    ("p", "destination_name, payment_instrument_group/pickup method/access point, and "
          "cc1 fx margin emerged as the dominant predictors under both the "
          "information-criterion forward-selection procedure (added early, largest BIC "
          "improvement) and LASSO's shrinkage (largest-magnitude surviving "
          "coefficients) -- convergent evidence that they belong in the final model. "
          "receiving network coverage and speed actual also showed meaningful cost "
          "differentiation in the EDA and are retained as weaker but non-trivial "
          "predictors (several of their dummy levels survive LASSO's zeroing). "
          "firm_type_clean is a deliberate exception to this convergent-evidence "
          "pattern: the forward BIC stepwise search actually excluded it (tested last, "
          "did not improve BIC further), and LASSO keeps it with only a small nonzero "
          "coefficient (rank 34 of 36 surviving coefficients by magnitude). We retain it "
          "anyway because the plain EDA shows a clear, substantively large cost gap "
          "between Banks and Money Transfer Operators -- its effect is real, but it gets "
          "absorbed/diluted once destination_name and payment_instrument_group are "
          "already in the model. This is a domain-reasoning judgment call, not "
          "something the IC search or LASSO alone would have recommended. year is "
          "retained with low expected signal strength but is required anyway for the "
          "planned pre-2023/2023+ temporal train/test split. is_non_transparent is "
          "retained as a cheap-to-compute engineered flag, with the caveat that it is an "
          "imperfect proxy conflated with send-limit disclosures. cc1 lcu fee and "
          "inter lcu bank fx remain excluded (currency-scale, not cross-country "
          "comparable without further engineering -- a stated modeling assumption, not "
          "an empirically tested fact), and all cc2 * fields remain excluded (mechanical "
          "leakage of the target)."),
]


def strip_md(text):
    return text.replace("`", "")


# ---------------------------------------------------------------------------
# Build the Word document
# ---------------------------------------------------------------------------
def build_docx():
    doc = Document()
    title = doc.add_heading("EDA & Feature Selection — Remittance Transfer Costs (Team 105)", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("ISyE 7406 — Data Mining & Statistical Learning")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER

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
            fname, caption = next((f, c) for f, c in FIGURES_EDA + FIGURES_FS if f == item[1])
            doc.add_picture(os.path.join(FIG_DIR, fname), width=Inches(6.0))
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True
            cap.runs[0].font.size = Pt(9)
        elif kind == "table":
            add_table_docx(doc, item[1])
    doc.save(os.path.join(REPORTS_DIR, "EDA_FeatureSelection_Group105.docx"))


def add_table_docx(doc, name):
    if name == "missingness":
        data = missing.reset_index().rename(columns={"index": "column"})
        headers = ["Column", "# Missing", "% Missing"]
        rows = [[r["column"], int(r["n_missing"]), f"{r['pct_missing']}%"] for _, r in data.iterrows()]
        caption = "Table 1. Columns with missing values."
    elif name == "correlation":
        data = corr.reset_index().rename(columns={"index": "field"})
        headers = ["Field"] + list(corr.columns)
        rows = [[r["field"]] + [r[c] for c in corr.columns] for _, r in data.iterrows()]
        caption = "Table 2. Pearson correlation matrix (numeric fields + year)."
    elif name == "vif":
        data = vif_series.reset_index()
        data.columns = ["feature", "VIF"]
        headers = ["Feature", "VIF"]
        rows = [[r["feature"], round(r["VIF"], 3)] for _, r in data.iterrows()]
        caption = "Table 3. Variance Inflation Factor, numeric candidate block."
    elif name == "lasso_top":
        data = top_coefs.reset_index()
        data.columns = ["feature", "coefficient"]
        headers = ["Feature", "Standardized Coefficient"]
        rows = [[r["feature"], round(r["coefficient"], 4)] for _, r in data.iterrows()]
        caption = "Table 4. Top 15 LassoCV coefficients by |magnitude|."
    else:
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


def build_markdown():
    lines = ["# EDA & Feature Selection --- Remittance Transfer Costs (Team 105)", "",
             "*ISyE 7406 --- Data Mining & Statistical Learning*", ""]
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
            fname, caption = next((f, c) for f, c in FIGURES_EDA + FIGURES_FS if f == item[1])
            fig_abspath = os.path.join(FIG_DIR, fname)
            lines.append(f"![{caption}]({fig_abspath})\n")
            lines.append(f"*{caption}*\n")
        elif kind == "table":
            lines.append(table_to_markdown(item[1]) + "\n")
    md_path = os.path.join(REPORTS_DIR, "EDA_FeatureSelection_Group105.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    return md_path


def table_to_markdown(name):
    if name == "missingness":
        data = missing.reset_index().rename(columns={"index": "column"})
        headers = ["Column", "# Missing", "% Missing"]
        rows = [[r["column"], int(r["n_missing"]), f"{r['pct_missing']}%"] for _, r in data.iterrows()]
        caption = "Table 1. Columns with missing values."
    elif name == "correlation":
        data = corr.reset_index().rename(columns={"index": "field"})
        headers = ["Field"] + list(corr.columns)
        rows = [[r["field"]] + [r[c] for c in corr.columns] for _, r in data.iterrows()]
        caption = "Table 2. Pearson correlation matrix (numeric fields + year)."
    elif name == "vif":
        data = vif_series.reset_index()
        data.columns = ["feature", "VIF"]
        headers = ["Feature", "VIF"]
        rows = [[r["feature"], round(r["VIF"], 3)] for _, r in data.iterrows()]
        caption = "Table 3. Variance Inflation Factor, numeric candidate block."
    elif name == "lasso_top":
        data = top_coefs.reset_index()
        data.columns = ["feature", "coefficient"]
        headers = ["Feature", "Standardized Coefficient"]
        rows = [[r["feature"], round(r["coefficient"], 4)] for _, r in data.iterrows()]
        caption = "Table 4. Top 15 LassoCV coefficients by |magnitude|."
    else:
        return ""
    return df_to_md_table(headers, rows) + f"\n\n*{caption}*\n"


def build_pdf(md_path):
    pdf_path = os.path.join(REPORTS_DIR, "EDA_FeatureSelection_Group105.pdf")
    docx_path = os.path.join(REPORTS_DIR, "EDA_FeatureSelection_Group105.docx")

    # Approach 1: pandoc + xelatex, run from REPORTS_DIR so relative links aren't needed
    # (figure paths in the markdown are already absolute).
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

    # Approach 2: LibreOffice headless conversion from the docx.
    for soffice in ["soffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice"]:
        try:
            result = subprocess.run(
                [soffice, "--headless", "--convert-to", "pdf", "--outdir", REPORTS_DIR, docx_path],
                capture_output=True, text=True, timeout=300,
            )
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 2000:
                print(f"PDF generated via LibreOffice ({soffice}).")
                return f"libreoffice ({soffice})"
        except Exception as e:
            print(f"LibreOffice attempt with {soffice} failed: {e}", file=sys.stderr)

    # Approach 3: docx2pdf (macOS/Word-based), if installed.
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

    docx_path = os.path.join(REPORTS_DIR, "EDA_FeatureSelection_Group105.docx")
    pdf_path = os.path.join(REPORTS_DIR, "EDA_FeatureSelection_Group105.pdf")
    print("\n--- Export summary ---")
    print("DOCX:", docx_path, os.path.getsize(docx_path) if os.path.exists(docx_path) else "MISSING")
    print("MD:  ", md_path, os.path.getsize(md_path) if os.path.exists(md_path) else "MISSING")
    print("PDF:", pdf_path, os.path.getsize(pdf_path) if os.path.exists(pdf_path) else "MISSING",
          f"(method: {method})")
