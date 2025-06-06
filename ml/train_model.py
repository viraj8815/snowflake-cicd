import os
import gzip
import json
import cloudpickle
import mlflow
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from mlflow.models.signature import infer_signature
from snowflake.snowpark import Session
from snowflake.snowpark.functions import when
from catboost import CatBoostClassifier
from mlflow.tracking import MlflowClient

# -----------------------------
# Snowflake Connection
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
# Load and join 4 tables
# -----------------------------
customer = session.table("TPCDS_10TB.TPCDS_SF10TCL.CUSTOMER")
cdemo = session.table("TPCDS_10TB.TPCDS_SF10TCL.CUSTOMER_DEMOGRAPHICS")
ddim = session.table("TPCDS_10TB.TPCDS_SF10TCL.DATE_DIM")
csales = session.table("TPCDS_10TB.TPCDS_SF10TCL.CATALOG_SALES").limit(10000)

df = (
    csales.join(customer, csales["CS_BILL_CUSTOMER_SK"] == customer["C_CUSTOMER_SK"])
          .join(cdemo, customer["C_CURRENT_CDEMO_SK"] == cdemo["CD_DEMO_SK"])
          .join(ddim, csales["CS_SOLD_DATE_SK"] == ddim["D_DATE_SK"])
          .select(
              "CD_GENDER",
              "CD_MARITAL_STATUS",
              "CD_EDUCATION_STATUS",
              "CD_CREDIT_RATING",
              "CD_DEP_COUNT",
              "CD_DEP_EMPLOYED_COUNT",
              "CD_DEP_COLLEGE_COUNT",
              (2025 - customer["C_BIRTH_YEAR"]).alias("AGE"),
              when(ddim["D_DAY_NAME"].isin(["Saturday", "Sunday"]), 1).otherwise(0).alias("IS_WEEKEND"),
              csales["CS_NET_PAID"].alias("TOTAL_SPENT")
          )
)

# -----------------------------
# Convert to Pandas
# -----------------------------
pdf = df.to_pandas()
pdf.dropna(inplace=True)

# -----------------------------
# Feature Engineering
# -----------------------------
pdf["IS_MARRIED"] = pdf["CD_MARITAL_STATUS"].isin(["M", "D"]).astype(int)
pdf["HAS_COLLEGE_DEP"] = (pdf["CD_DEP_COLLEGE_COUNT"] > 0).astype(int)
pdf["TOTAL_DEP"] = (
    pdf["CD_DEP_COUNT"] +
    pdf["CD_DEP_EMPLOYED_COUNT"] +
    pdf["CD_DEP_COLLEGE_COUNT"]
)
pdf["AGE_BIN"] = pd.cut(pdf["AGE"], bins=[0, 18, 30, 45, 60, 100], labels=["0", "1", "2", "3", "4"]).astype(str)

# -----------------------------
# Binarize High Spender
# -----------------------------
q2 = pdf["TOTAL_SPENT"].quantile(0.66)
pdf["HIGH_SPENDER"] = (pdf["TOTAL_SPENT"] >= q2).astype(int)

X = pdf.drop(columns=["TOTAL_SPENT", "HIGH_SPENDER"])
y = pdf["HIGH_SPENDER"]

# -----------------------------
# Encode categorical
# -----------------------------
cat_cols = X.select_dtypes(include="object").columns.tolist()
X = pd.get_dummies(X, columns=cat_cols)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# -----------------------------
# Compute class weights
# -----------------------------
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(weights)}

# -----------------------------
# Hyperparameter Tuning
# -----------------------------
param_grid = {
    "depth": [6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05],
    "iterations": [500, 1000],
    "l2_leaf_reg": [1, 3, 5],
    "border_count": [32, 64],
}

base_model = CatBoostClassifier(
    loss_function="Logloss",
    early_stopping_rounds=20,
    class_weights=class_weights,
    random_seed=42,
    verbose=0
)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=30,
    scoring="accuracy",
    cv=3,
    verbose=2,
    n_jobs=-1
)

search.fit(X_train, y_train)
model = search.best_estimator_
print("✅ Best Params:", search.best_params_)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# -----------------------------
# Save Artifacts
# -----------------------------
os.makedirs("ml", exist_ok=True)
pipeline = {
    "model": model,
    "feature_order": list(X.columns)
}
with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(pipeline, f)

with open("ml/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "f1": f1}, f, indent=2)

with open("ml/signature.json", "w") as f:
    json.dump({
        "inputs": list(X.columns),
        "output": "HIGH_SPENDER",
        "model_type": model.__class__.__name__,
        "hyperparams": model.get_params()
    }, f, indent=2)

with open("ml/drift_baseline.json", "w") as f:
    json.dump(pdf.describe(include='all').to_dict(), f, indent=2)

# -----------------------------
# MLflow Logging
# -----------------------------
experiment_name = "snowflake-high-spender-predictor"
mlflow.set_experiment(experiment_name)

client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
existing_runs = client.search_runs(experiment.experiment_id)
version_number = len(existing_runs) + 1
run_name = f"high_spender_v{version_number}"

with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("model_version", run_name)
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig("ml/confusion_matrix.png")
    mlflow.log_artifact("ml/confusion_matrix.png")

    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("ml/shap_summary.png")
    mlflow.log_artifact("ml/shap_summary.png")

    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X.head(5)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        input_example=input_example,
        signature=signature
    )

    mlflow.log_artifact("ml/model.pkl.gz")
    mlflow.log_artifact("ml/metrics.json")
    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")

print(f"✅ Final Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Version: {version_number}")
