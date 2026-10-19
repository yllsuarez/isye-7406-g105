# train_model.R
#
# Train an ML model and export it in a portable format for the
# Python web service (../web-service) to consume.
#
# Example uses randomForest; swap in whatever algorithm the project needs.

library(randomForest)

# ---- 1. Load cleaned data -----------------------------------------------
# Expect data-cleaning/ to have produced a tidy CSV ready for modeling.
data_path <- file.path("..", "data-cleaning", "output", "clean_data.csv")
df <- read.csv(data_path)

# ---- 2. Train model -------------------------------------------------------
set.seed(42)
model <- randomForest(target ~ ., data = df, ntree = 500)

print(model)

# ---- 3. Export for Python ---------------------------------------------
# Option A: if using xgboost/lightgbm, save in their native format instead:
#   xgboost::xgb.save(model, "model.xgb")
#
# Option B: PMML export (works for many classic R model types):
#   library(pmml)
#   saveXML(pmml(model), "model.pmml")
#
# Option C: keep as native R object and serve via plumber API.
saveRDS(model, "model.rds")

cat("Model trained and saved to model.rds\n")
