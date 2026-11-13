"""Final-report analysis: regression comparison, classification at the 3% SDG
threshold, and destination-country clustering (Team 105 remittance-cost project).

Reuses the exact cleaning/feature logic from train_model.py (same candidate
feature set, same pre-2023/2023+ temporal split) so results are directly
comparable to the shipped production LassoCV model.

Outputs:
  - r-models/reports/figures_final/12_*.png ... 19_*.png
  - r-models/reports/regression_comparison.csv
  - r-models/reports/regression_coefficients.csv
  - r-models/reports/rf_feature_importance.csv
  - r-models/reports/classification_comparison.csv
  - r-models/reports/classification_by_firmtype.csv
  - r-models/reports/classification_by_destination.csv
  - r-models/reports/cluster_country_profile.csv
  - r-models/reports/cluster_metrics.csv

Run from r-models/:  /opt/anaconda3/bin/python3 final_model_analysis.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "rpw_dataset_2011_2025_q1.csv")
REPORTS_DIR = os.path.join(HERE, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures_final")
os.makedirs(FIG_DIR, exist_ok=True)

TARGET = "cc1 total cost %"
CATEGORICAL = [
    "destination_name",
    "firm_type_clean",
    "payment_instrument_group",
    "access point",
    "receiving network coverage",
    "speed actual",
    "pickup method",
]
NUMERIC = ["cc1 fx margin", "year", "is_non_transparent"]
FEATURES = CATEGORICAL + NUMERIC
SPLIT_YEAR = 2023
SDG_TARGET = 3.0
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Data loading / cleaning -- identical to train_model.py
# ---------------------------------------------------------------------------
def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    for col in ["speed actual", "pickup method"]:
        df[col] = df[col].str.lower().str.strip()
    df["year"] = df["period"].str.split("_").str[0].astype(int)
    df["quarter"] = df["period"].str.extract(r"_(\d)Q")[0].astype(int)
    df["firm_type_clean"] = df["firm_type"].replace(
        {"Money Transfer Operator / Building Society": "Money Transfer Operator"}
    )
    df = df[df["firm_type_clean"] != "Non-Bank FI"].copy()
    df["is_non_transparent"] = df["cc2 total cost %"].isna().astype(int)
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def rmse(y, pred) -> float:
    return float(np.sqrt(mean_squared_error(y, pred)))


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )


def encoded_feature_names(pre: ColumnTransformer) -> list:
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL))
    return cat_names + NUMERIC


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


df = load_clean()
train = df[df["year"] < SPLIT_YEAR].copy()
test = df[df["year"] >= SPLIT_YEAR].copy()
print(f"rows: {len(df)}  train (<{SPLIT_YEAR}): {len(train)}  test (>={SPLIT_YEAR}): {len(test)}")

# ===========================================================================
# 1. Regression model comparison
# ===========================================================================
def to_dense(mat):
    return mat.toarray() if hasattr(mat, "toarray") else mat


pre = make_preprocessor()
pre.fit(train[FEATURES])
Xtr = to_dense(pre.transform(train[FEATURES]))
Xte = to_dense(pre.transform(test[FEATURES]))
ytr, yte = train[TARGET].values, test[TARGET].values
feat_names = encoded_feature_names(pre)

regressors = {
    "Linear Regression": LinearRegression(),
    "Ridge": RidgeCV(alphas=np.logspace(-3, 2, 50), cv=5),
    "LASSO": LassoCV(alphas=np.logspace(-3, 2, 50), cv=5, random_state=RANDOM_STATE, max_iter=50000),
    "Random Forest": RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

reg_rows = []
reg_preds = {}
reg_fitted = {}
for name, model in regressors.items():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    reg_preds[name] = pred
    reg_fitted[name] = model
    reg_rows.append(
        {
            "model": name,
            "test_rmse": rmse(yte, pred),
            "test_mae": mean_absolute_error(yte, pred),
            "test_r2": r2_score(yte, pred),
        }
    )
reg_comparison = pd.DataFrame(reg_rows).sort_values("test_rmse").reset_index(drop=True)
reg_comparison.to_csv(os.path.join(REPORTS_DIR, "regression_comparison.csv"), index=False)
print("\nRegression comparison (test, 2023+):")
print(reg_comparison.round(4).to_string(index=False))

best_reg_name = reg_comparison.iloc[0]["model"]
best_reg_pred = reg_preds[best_reg_name]

# Figure 12: residual plot for the best model
residuals = yte - best_reg_pred
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(best_reg_pred, residuals, alpha=0.35, s=14, color="#2f6fa8")
axes[0].axhline(0, color="black", linewidth=1)
axes[0].set_xlabel("Predicted cc1 total cost %")
axes[0].set_ylabel("Residual (actual - predicted)")
axes[0].set_title(f"Residuals vs. fitted -- {best_reg_name}")
axes[1].hist(residuals, bins=30, color="#2f6fa8", edgecolor="white")
axes[1].set_xlabel("Residual")
axes[1].set_title("Residual distribution")
fig.tight_layout()
savefig(fig, "12_regression_residuals.png")

# Figure 13: LASSO and Ridge standardized coefficients (top 15 by |LASSO| magnitude)
lasso_coef = pd.Series(reg_fitted["LASSO"].coef_, index=feat_names)
ridge_coef = pd.Series(reg_fitted["Ridge"].coef_, index=feat_names)
top_idx = lasso_coef.abs().sort_values(ascending=False).head(15).index
coef_compare = pd.DataFrame({"LASSO": lasso_coef[top_idx], "Ridge": ridge_coef[top_idx]})
coef_compare.to_csv(os.path.join(REPORTS_DIR, "regression_coefficients.csv"))

fig, ax = plt.subplots(figsize=(8, 6))
coef_compare.sort_values("LASSO").plot.barh(ax=ax, color=["#d1495b", "#8c8c8c"])
ax.set_xlabel("Standardized coefficient")
ax.set_title("Top 15 LASSO coefficients vs. Ridge (same features)")
fig.tight_layout()
savefig(fig, "13_lasso_ridge_coefficients.png")

# Figure 14: Random Forest feature importance (top 15)
rf_importance = pd.Series(reg_fitted["Random Forest"].feature_importances_, index=feat_names)
rf_importance = rf_importance.sort_values(ascending=False).head(15)
rf_importance.to_csv(os.path.join(REPORTS_DIR, "rf_feature_importance.csv"), header=["importance"])

fig, ax = plt.subplots(figsize=(8, 6))
rf_importance.sort_values().plot.barh(ax=ax, color="#2a9d8f")
ax.set_xlabel("Mean decrease in impurity")
ax.set_title("Random Forest feature importance (top 15)")
fig.tight_layout()
savefig(fig, "14_rf_feature_importance.png")

# ===========================================================================
# 2. Classification at the 3% SDG threshold
# ===========================================================================
train["high_cost"] = (train[TARGET] >= SDG_TARGET).astype(int)
test["high_cost"] = (test[TARGET] >= SDG_TARGET).astype(int)
ytr_c, yte_c = train["high_cost"].values, test["high_cost"].values
print(f"\nhigh_cost base rate -- train: {ytr_c.mean():.3f}  test: {yte_c.mean():.3f}")

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "LDA": LinearDiscriminantAnalysis(),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

clf_rows = []
clf_preds = {}
clf_fitted = {}
for name, model in classifiers.items():
    model.fit(Xtr, ytr_c)
    pred = model.predict(Xte)
    clf_preds[name] = pred
    clf_fitted[name] = model
    clf_rows.append(
        {
            "model": name,
            "accuracy": accuracy_score(yte_c, pred),
            "precision": precision_score(yte_c, pred, zero_division=0),
            "recall": recall_score(yte_c, pred, zero_division=0),
            "f1": f1_score(yte_c, pred, zero_division=0),
        }
    )
clf_comparison = pd.DataFrame(clf_rows).sort_values("f1", ascending=False).reset_index(drop=True)
clf_comparison.to_csv(os.path.join(REPORTS_DIR, "classification_comparison.csv"), index=False)
print("\nClassification comparison (test, 2023+, >=3% = high cost):")
print(clf_comparison.round(4).to_string(index=False))

best_clf_name = clf_comparison.iloc[0]["model"]
best_clf_pred = clf_preds[best_clf_name]

# Figure 15: confusion matrices broken out by firm type (Bank vs. MTO), best classifier
firm_types = sorted(test["firm_type_clean"].unique())
fig, axes = plt.subplots(1, len(firm_types), figsize=(5 * len(firm_types), 4.2))
if len(firm_types) == 1:
    axes = [axes]
firmtype_rows = []
for ax, ft in zip(axes, firm_types):
    mask = test["firm_type_clean"].values == ft
    if mask.sum() == 0:
        continue
    cm = confusion_matrix(yte_c[mask], best_clf_pred[mask], labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["low (<3%)", "high (>=3%)"]).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"{ft} (n={int(mask.sum())})")
    firmtype_rows.append(
        {
            "firm_type": ft,
            "n": int(mask.sum()),
            "accuracy": accuracy_score(yte_c[mask], best_clf_pred[mask]),
            "precision": precision_score(yte_c[mask], best_clf_pred[mask], zero_division=0),
            "recall": recall_score(yte_c[mask], best_clf_pred[mask], zero_division=0),
            "f1": f1_score(yte_c[mask], best_clf_pred[mask], zero_division=0),
        }
    )
fig.suptitle(f"Confusion matrix by firm type -- {best_clf_name}")
fig.tight_layout()
savefig(fig, "15_confusion_by_firmtype.png")
pd.DataFrame(firmtype_rows).to_csv(os.path.join(REPORTS_DIR, "classification_by_firmtype.csv"), index=False)

# Figure 16: per-destination precision/recall/F1 for the best classifier
dest_rows = []
for dest in sorted(test["destination_name"].unique()):
    mask = test["destination_name"].values == dest
    if mask.sum() == 0:
        continue
    dest_rows.append(
        {
            "destination_name": dest,
            "n": int(mask.sum()),
            "accuracy": accuracy_score(yte_c[mask], best_clf_pred[mask]),
            "precision": precision_score(yte_c[mask], best_clf_pred[mask], zero_division=0),
            "recall": recall_score(yte_c[mask], best_clf_pred[mask], zero_division=0),
            "f1": f1_score(yte_c[mask], best_clf_pred[mask], zero_division=0),
        }
    )
dest_perf = pd.DataFrame(dest_rows).sort_values("f1")
dest_perf.to_csv(os.path.join(REPORTS_DIR, "classification_by_destination.csv"), index=False)

fig, ax = plt.subplots(figsize=(8, 7))
dest_perf.set_index("destination_name")[["precision", "recall", "f1"]].plot.barh(ax=ax)
ax.set_xlabel("Score")
ax.set_title(f"Classification performance by destination -- {best_clf_name}")
fig.tight_layout()
savefig(fig, "16_classification_by_destination.png")

# ===========================================================================
# 3. Clustering the 16 LAC destinations by cost profile
# ===========================================================================
country_profile = (
    df.groupby("destination_name")
    .agg(
        mean_total_cost=(TARGET, "mean"),
        mean_fx_margin=("cc1 fx margin", "mean"),
        mean_fee_lcu=("cc1 lcu fee", "mean"),
        transparency_rate=("is_non_transparent", lambda s: 1 - s.mean()),
        n_obs=(TARGET, "size"),
    )
    .reset_index()
)
cluster_features = ["mean_total_cost", "mean_fx_margin", "mean_fee_lcu", "transparency_rate"]
Xc = StandardScaler().fit_transform(country_profile[cluster_features])

# Figure 17: elbow (inertia) + silhouette vs. k
ks = range(2, 8)
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(Xc)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(Xc, km.labels_))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(list(ks), inertias, marker="o", color="#2f6fa8")
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].set_title("K-means elbow")
axes[1].plot(list(ks), sils, marker="o", color="#2a9d8f")
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette score")
axes[1].set_title("K-means silhouette")
fig.tight_layout()
savefig(fig, "17_kmeans_elbow_silhouette.png")

best_k = list(ks)[int(np.argmax(sils))]
km_final = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE).fit(Xc)
country_profile["kmeans_cluster"] = km_final.labels_

# Secondary, finer-grained partition (k=4) purely for interpretive detail: the
# silhouette-optimal k above tends to isolate the single most extreme outlier
# (Cuba) vs. everyone else, so a slightly larger k is reported alongside it to
# show the finer cost-profile gradations among the remaining destinations.
km_k4 = KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit(Xc)
country_profile["kmeans_cluster_k4"] = km_k4.labels_

cluster_metrics = pd.DataFrame({"k": list(ks), "inertia": inertias, "silhouette": sils})
cluster_metrics.to_csv(os.path.join(REPORTS_DIR, "cluster_metrics.csv"), index=False)

# Figure 18: scatter of countries on cost vs. fx margin, colored by k-means cluster;
# point size reflects transparency rate
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    country_profile["mean_fx_margin"],
    country_profile["mean_total_cost"],
    c=country_profile["kmeans_cluster"],
    s=80 + 40 * country_profile["transparency_rate"],
    cmap="tab10",
)
for _, row in country_profile.iterrows():
    ax.annotate(row["destination_name"], (row["mean_fx_margin"], row["mean_total_cost"]), fontsize=8,
                xytext=(4, 4), textcoords="offset points")
ax.set_xlabel("Mean cc1 fx margin")
ax.set_ylabel("Mean cc1 total cost %")
ax.set_title(f"K-means clusters (k={best_k}) of 16 LAC destinations")
fig.tight_layout()
savefig(fig, "18_cluster_scatter.png")

# Figure 19: hierarchical clustering dendrogram
Z = linkage(Xc, method="ward")
fig, ax = plt.subplots(figsize=(9, 5))
dendrogram(Z, labels=country_profile["destination_name"].tolist(), ax=ax, leaf_rotation=90)
ax.set_title("Hierarchical clustering (Ward linkage) of destination cost profiles")
ax.set_ylabel("Distance")
fig.tight_layout()
savefig(fig, "19_dendrogram.png")

country_profile = country_profile.round(4)
country_profile.to_csv(os.path.join(REPORTS_DIR, "cluster_country_profile.csv"), index=False)

print("\nCountry cluster profile:")
print(country_profile.sort_values("kmeans_cluster").to_string(index=False))
print(f"\nchosen k (max silhouette): {best_k}")
print("\nAll figures/tables written under r-models/reports/ and r-models/reports/figures_final/.")
