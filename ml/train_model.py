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
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from catboost import CatBoostClassifier

# Load all 3 datasets
customer = pd.read_csv("CUSTOMER_SAMPLE.csv")
demo = pd.read_csv("CUSTOMER_DEMOGRAPHICS_SAMPLE.csv")
date = pd.read_csv("DATE_DIM_SAMPLE.csv")

# Join datasets
df = customer.merge(demo, left_on="C_CURRENT_CDEMO_SK", right_on="CD_DEMO_SK") \
             .merge(date, left_on="C_FIRST_SALES_DATE_SK", right_on="D_DATE_SK")

# Feature Engineering
df["AGE"] = 2025 - df["C_BIRTH_YEAR"]
df["IS_WEEKEND"] = df["D_DAY_NAME"].isin(["Saturday", "Sunday"]).astype(int)
df["IS_MARRIED"] = df["CD_MARITAL_STATUS"].isin(["M", "D"]).astype(int)
df["HAS_COLLEGE_DEP"] = (df["CD_DEP_COLLEGE_COUNT"] > 0).astype(int)
df["TOTAL_DEP"] = df["CD_DEP_COUNT"] + df["CD_DEP_EMPLOYED_COUNT"] + df["CD_DEP_COLLEGE_COUNT"]
df["AGE_BIN"] = pd.cut(df["AGE"], bins=[0, 18, 30, 45, 60, 100], labels=["0", "1", "2", "3", "4"]).astype(str)
df["EMPLOYED_DEP_RATIO"] = (
    df["CD_DEP_EMPLOYED_COUNT"] / df["CD_DEP_COUNT"].replace(0, np.nan)
).fillna(0)

# Filter to Low vs High target
q1 = df["CD_PURCHASE_ESTIMATE"].quantile(1/3)
q3 = df["CD_PURCHASE_ESTIMATE"].quantile(2/3)
df = df[(df["CD_PURCHASE_ESTIMATE"] <= q1) | (df["CD_PURCHASE_ESTIMATE"] >= q3)].copy()
df["PURCHASE_RANGE"] = pd.cut(df["CD_PURCHASE_ESTIMATE"], [-np.inf, q1, np.inf], labels=["Low", "High"])

# Define input and target
X = df[[
    "AGE", "IS_WEEKEND", "IS_MARRIED", "HAS_COLLEGE_DEP", "TOTAL_DEP",
    "CD_DEP_COUNT", "CD_DEP_EMPLOYED_COUNT", "CD_DEP_COLLEGE_COUNT",
    "CD_GENDER", "CD_MARITAL_STATUS", "CD_EDUCATION_STATUS", "CD_CREDIT_RATING", "AGE_BIN",
    "EMPLOYED_DEP_RATIO"
]]
y = df["PURCHASE_RANGE"]

# Encode target and prepare inputs
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, random_state=42)

# Class weights
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weights = {i: w for i, w in enumerate(weights)}

# Random search params
param_grid = {
    "depth": [12],
    "learning_rate": [0.01],
    "iterations": [500],
    "l2_leaf_reg": [1],
    "border_count": [64],
}

model = CatBoostClassifier(
    loss_function="Logloss",
    early_stopping_rounds=10,
    class_weights=class_weights,
    verbose=0,
    random_seed=42
)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=30,
    scoring="accuracy",
    cv=3,
    verbose=2,
    n_jobs=-1
)
search.fit(X_train, y_train)
model = search.best_estimator_

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("✅ Best Params:", search.best_params_)
print(f"✅ Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# Save artifacts
os.makedirs("ml", exist_ok=True)
with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(model, f)

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
    json.dump(df.describe(include='all').to_dict(), f, indent=2)

# -----------------------------
# Log to MLflow
# -----------------------------
experiment_name = "snowflake-ml-model"
mlflow.set_experiment(experiment_name)

client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
existing_runs = mlflow.search_runs(experiment.experiment_id)
version_number = len(existing_runs) + 1
run_name = f"rf_model_v{version_number}"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tag("dataset_version", f"v{version_number}")
    mlflow.set_tag("model_version", run_name)

    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)

    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.savefig("ml/confusion_matrix.png")
    mlflow.log_artifact("ml/confusion_matrix.png")

    # SHAP summary
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("ml/shap_summary.png")
    mlflow.log_artifact("ml/shap_summary.png")

    # Log model
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X.head(5)
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example, signature=signature)

    # Additional artifacts
    mlflow.log_artifact("ml/model.pkl.gz")
    mlflow.log_artifact("ml/metrics.json")
    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")

print(f"✅ Logged as version {version_number} to MLflow")
