import os
import json
import gzip
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
from snowflake.snowpark import Session
from snowflake.snowpark.functions import when
from catboost import CatBoostClassifier
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

# -----------------------------
# Convert target to classes
# -----------------------------
pdf["PURCHASE_RANGE"] = pd.qcut(pdf["CD_PURCHASE_ESTIMATE"], q=3, labels=["Low", "Medium", "High"])
X = pdf.drop(columns=["CD_PURCHASE_ESTIMATE", "PURCHASE_RANGE"])
y = pdf["PURCHASE_RANGE"]

# -----------------------------
# Encode target
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, random_state=42)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# -----------------------------
# Compute class weights
# -----------------------------
class_weights_arr = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(class_weights_arr)}

# -----------------------------
# Define hyperparameter search
# -----------------------------
param_grid = {
    "depth": [6, 8, 10, 12],
    "learning_rate": [0.01, 0.03, 0.05],
    "iterations": [500, 1000, 1500, 2000],
    "l2_leaf_reg": [1, 3, 5, 7, 9],
    "border_count": [32, 64, 128],
}

base_model = CatBoostClassifier(
    loss_function="MultiClass",
    cat_features=cat_cols,
    early_stopping_rounds=10,
    class_weights=class_weights,
    verbose=0,
    random_seed=42
)

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=30,  # increase for deeper search
    scoring="accuracy",
    cv=3,
    verbose=2,
    n_jobs=-1
)

search.fit(X_train, y_train)
model = search.best_estimator_
print("✅ Best Params:", search.best_params_)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
print(f"✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# -----------------------------
# Save artifacts
# -----------------------------
os.makedirs("ml", exist_ok=True)
pipeline = {
    "model": model,
    "cat_features": cat_cols,
    "label_encoder": label_encoder,
    "feature_order": list(X.columns)
}
with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(pipeline, f)

with open("ml/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "f1": f1}, f, indent=2)

with open("ml/signature.json", "w") as f:
    json.dump({
        "inputs": list(X.columns),
        "output": "PURCHASE_RANGE",
        "model_type": model.__class__.__name__,
        "hyperparams": model.get_params()
    }, f, indent=2)

with open("ml/drift_baseline.json", "w") as f:
    json.dump(pdf.describe(include='all').to_dict(), f, indent=2)

# -----------------------------
# Log to MLflow
# -----------------------------
experiment_name = "snowflake-ml-model-classifier-optimized"
mlflow.set_experiment(experiment_name)
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
existing_runs = client.search_runs(experiment.experiment_id)
version_number = len(existing_runs) + 1
run_name = f"catboost_classifier_opt_v{version_number}"

with mlflow.start_run(run_name=run_name) as run:
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

    mlflow.log_artifact("ml/model.pkl.gz")
    mlflow.log_artifact("ml/metrics.json")
    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")

print(f"✅ Final Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Version: {version_number}")
