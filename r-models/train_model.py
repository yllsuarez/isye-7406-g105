"""Train the remittance cost model (LassoCV) and export artifacts.

Run from r-models/:  python train_model.py
Outputs r-models/model/model.pkl and r-models/model/metadata.json.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
DATA = HERE / "data" / "rpw_dataset_2011_2025_q1.csv"
OUT = HERE / "model"
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
SPLIT_YEAR = 2023  # train: year < 2023, test: year >= 2023


def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA)
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


def main() -> None:
    df = load_clean()
    train, test = df[df["year"] < SPLIT_YEAR], df[df["year"] >= SPLIT_YEAR]
    print(f"rows: {len(df)} train: {len(train)} test: {len(test)}")

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
    pipe = Pipeline(
        [
            ("prep", pre),
            (
                "lasso",
                LassoCV(
                    cv=5, alphas=np.logspace(-3, 2, 50), random_state=42, max_iter=50000
                ),
            ),
        ]
    )
    pipe.fit(train[FEATURES], train[TARGET])

    for name, part in [("train", train), ("holdout", test)]:
        pred = pipe.predict(part[FEATURES])
        print(
            f"{name}: RMSE={rmse(part[TARGET], pred):.4f}  "
            f"R2={r2_score(part[TARGET], pred):.4f}"
        )
    print(f"best alpha: {pipe.named_steps['lasso'].alpha_:.5f}")

    # Refit on all data so the shipped model benefits from the latest observations.
    pipe.fit(df[FEATURES], df[TARGET])

    OUT.mkdir(exist_ok=True)
    joblib.dump(pipe, OUT / "model.pkl")
    metadata = {
        "target": TARGET,
        "features": [
            {"name": c, "role": "categorical" if c in CATEGORICAL else "numeric"}
            for c in FEATURES
        ],
        "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL},
        "fx_margin_by_destination": {
            k: round(float(v), 4)
            for k, v in df.groupby("destination_name")["cc1 fx margin"].mean().items()
        },
        "fx_margin_overall": round(float(df["cc1 fx margin"].mean()), 4),
        "year_min": int(df["year"].min()),
        "year_max": int(df["year"].max()),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"saved artifacts to {OUT}")


if __name__ == "__main__":
    main()
