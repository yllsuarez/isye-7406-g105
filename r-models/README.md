# r-models

Model training for the remittance cost predictor (target: `cc1 total cost %`).
Starting point: `eda_feature_selection.ipynb` and `reports/EDA_FeatureSelection_Group105.md`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python train_model.py
```

Applies the EDA cleaning/engineering (merged near-duplicate categories, `year`,
`firm_type_clean`, `is_non_transparent`; no `cc2 *` leakage fields), one-hot
encodes the 7 categorical features, standardizes the 3 numeric ones and fits a
`LassoCV` (5-fold, alphas `logspace(-3, 2, 50)`) on `year < 2023`, reporting
RMSE / R² on the `year >= 2023` holdout. It then refits on all data and writes:

- `model/model.pkl` – the fitted sklearn pipeline
- `model/metadata.json` – feature list, category options, per-destination average
  `cc1 fx margin` (+ overall fallback), year range

Copy the artifacts to the web app with `../web-service/sync_model.sh`.
