import os
import json
import gzip
import cloudpickle
import mlflow
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, f1_score
)
from snowflake.snowpark import Session
from snowflake.snowpark.functions import when
from catboost import CatBoostRegressor, Pool
from mlflow.tracking import MlflowClient

# -----------------------------
# Snowflake connection
# -----------------------------
connection_parameters = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "role": os.environ["SNOWFLAKE_ROLE"],
    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
}
session = Session.builder.configs(connection_parameters).create()

# -----------------------------
# Load and join 3 tables
# -----------------------------
cust = session.table("ML_DB.TRAINING_DATA.CUSTOMER_SAMPLE")
demo = session.table("ML_DB.TRAINING_DATA.CUSTOMER_DEMOGRAPHICS_SAMPLE")
date = session.table("ML_DB.TRAINING_DATA.DATE_DIM_SAMPLE")

df = (
    cust.join(demo, cust["C_CURRENT_CDEMO_SK"] == demo["CD_DEMO_SK"])
        .join(date, cust["C_FIRST_SALES_DATE_SK"] == date["D_DATE_SK"])
        .with_column("AGE", 2025 - cust["C_BIRTH_YEAR"])
        .with_column("IS_WEEKEND", when(date["D_DAY_NAME"].isin(["Saturday", "Sunday"]), 1).otherwise(0))
        .select(
            "CD_GENDER",
            "CD_MARITAL_STATUS",
            "CD_EDUCATION_STATUS",
            "CD_CREDIT_RATING",
            "CD_DEP_COUNT",
            "CD_DEP_EMPLOYED_COUNT",
            "CD_DEP_COLLEGE_COUNT",
            "AGE",
            "IS_WEEKEND",
            "CD_PURCHASE_ESTIMATE"
        )
)

# -----------------------------
# Convert to Pandas and clean
# -----------------------------
pdf = df.to_pandas()
pdf.dropna(inplace=True)

# -----------------------------
# Feature Engineering
# -----------------------------
pdf["IS_MARRIED"] = pdf["CD_MARITAL_STATUS"].isin(["M", "D"]).astype(int)
pdf["HAS_COLLEGE_DEP"] = (pdf["CD_DEP_COLLEGE_COUNT"] > 0).astype(int)
pdf["TOTAL_DEP"] = pdf["CD_DEP_COUNT"] + pdf["CD_DEP_EMPLOYED_COUNT"] + pdf["CD_DEP_COLLEGE_COUNT"]
pdf["AGE_BIN"] = pd.cut(pdf["AGE"], bins=[0, 18, 30, 45, 60, 100], labels=["0", "1", "2", "3", "4"]).astype(str)

X = pdf.drop(columns=["CD_PURCHASE_ESTIMATE"])
y = pdf["CD_PURCHASE_ESTIMATE"]

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

# -----------------------------
# Train CatBoost Regressor
# -----------------------------
model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    loss_function="RMSE",
    early_stopping_rounds=20,
    verbose=100,
    random_seed=42
)
model.fit(train_pool, eval_set=test_pool)

# -----------------------------
# Predict and Evaluate
# -----------------------------
y_pred_cont = model.predict(test_pool)

# Binning for classification metrics
actual_bins = pd.qcut(y_test, q=3, labels=["Low", "Medium", "High"])
pred_bins = pd.qcut(y_pred_cont, q=3, labels=["Low", "Medium", "High"])

accuracy = accuracy_score(actual_bins, pred_bins)
f1 = f1_score(actual_bins, pred_bins, average="macro")
rmse = np.sqrt(mean_squared_error(y_test, y_pred_cont))
r2 = r2_score(y_test, y_pred_cont)

print(f"✅ RMSE: {rmse:.2f}, R²: {r2:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# -----------------------------
# Save Artifacts
# -----------------------------
os.makedirs("ml", exist_ok=True)
pipeline = {
    "model": model,
    "cat_features": cat_cols,
    "feature_order": list(X.columns)
}
with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(pipeline, f)

with open("ml/metrics.json", "w") as f:
    json.dump({
        "rmse": rmse,
        "r2": r2,
        "accuracy": accuracy,
        "f1": f1
    }, f, indent=2)

with open("ml/signature.json", "w") as f:
    json.dump({
        "inputs": list(X.columns),
        "output": "CD_PURCHASE_ESTIMATE",
        "model_type": model.__class__.__name__,
        "hyperparams": model.get_params()
    }, f, indent=2)

with open("ml/drift_baseline.json", "w") as f:
    json.dump(pdf.describe(include='all').to_dict(), f, indent=2)

# -----------------------------
# Log to MLflow
# -----------------------------
experiment_name = "snowflake-ml-model-regression"
mlflow.set_experiment(experiment_name)
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
existing_runs = client.search_runs(experiment.experiment_id)
version_number = len(existing_runs) + 1
run_name = f"catboost_regressor_v{version_number}"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tag("model_version", run_name)
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({
        "rmse": rmse,
        "r2_score": r2,
        "accuracy": accuracy,
        "f1_score": f1
    })

    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("ml/shap_summary.png")
    mlflow.log_artifact("ml/shap_summary.png")

    mlflow.log_artifact("ml/model.pkl.gz")
    mlflow.log_artifact("ml/metrics.json")
    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")

print(f"✅ Logged and saved model v{version_number}")
