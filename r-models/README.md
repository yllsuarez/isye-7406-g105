# r-models

R code for training and exporting the ML model(s).

## Setup

```r
install.packages("renv")
renv::restore()
```

## Usage

```r
source("train_model.R")
```

This trains the model and saves it to `model.pkl` (via `reticulate` +
`joblib`) so the Python web service can load it without needing R at
inference time. If your model type isn't easily portable (e.g. complex
`caret`/`mgcv` objects), export to PMML instead (see `pmml` package) or
save with `saveRDS()` and serve via an R `plumber` API.
