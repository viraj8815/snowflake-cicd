import os
import json
import gzip
import cloudpickle
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from snowflake.snowpark import Session
from snowflake.snowpark.functions import when
from catboost import CatBoostClassifier

# -----------------------------
# Snowflake connection
# -----------------------------
connection_parameters = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "role": os.environ["SNOWFLAKE_ROLE"],
    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"]
}
session = Session.builder.configs(connection_parameters).create()

# -----------------------------
# Load 5 tables from TPCDS
# -----------------------------
customer = session.table("ML_DB.TRAINING_DATA.CUSTOMER_SAMPLE")
cdemo = session.table("ML_DB.TRAINING_DATA.CUSTOMER_DEMOGRAPHICS_SAMPLE")
ddim = session.table("ML_DB.TRAINING_DATA.DATE_DIM_SAMPLE")
csales = session.table("ML_DB.TRAINING_DATA.CATALOG_SALES_SAMPLE")
item = session.table("ML_DB.TRAINING_DATA.ITEM_SAMPLE")

# -----------------------------
# Join tables using Snowpark
# -----------------------------
joined = (
    csales
    .join(customer, csales["CS_BILL_CUSTOMER_SK"] == customer["C_CUSTOMER_SK"])
    .join(cdemo, customer["C_CURRENT_CDEMO_SK"] == cdemo["CD_DEMO_SK"])
    .join(ddim, customer["C_FIRST_SALES_DATE_SK"] == ddim["D_DATE_SK"])
    .join(item, csales["CS_ITEM_SK"] == item["I_ITEM_SK"])
    .select(
        "CD_GENDER", "CD_MARITAL_STATUS", "CD_EDUCATION_STATUS", "CD_CREDIT_RATING",
        "CD_DEP_COUNT", "CD_DEP_EMPLOYED_COUNT", "CD_DEP_COLLEGE_COUNT",
        (2025 - customer["C_BIRTH_YEAR"]).alias("AGE"),
        when(ddim["D_DAY_NAME"].isin(["Saturday", "Sunday"]), 1).otherwise(0).alias("IS_WEEKEND"),
        "CS_EXT_LIST_PRICE", "CS_WHOLESALE_COST", "CS_SALES_PRICE", "CS_QUANTITY",
        "I_CATEGORY"
    )
)

# -----------------------------
# Convert to pandas
# -----------------------------
pdf = joined.to_pandas()
session.close()
pdf.dropna(inplace=True)

# -----------------------------
# Target creation
# -----------------------------
pdf["TOTAL_SPENT"] = pdf["CS_SALES_PRICE"] * pdf["CS_QUANTITY"]
pdf["SPENDER_CLASS"] = pd.qcut(pdf["TOTAL_SPENT"], q=[0, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"])

# -----------------------------
# Feature engineering
# -----------------------------
pdf["IS_MARRIED"] = pdf["CD_MARITAL_STATUS"].isin(["M", "D"]).astype(int)
pdf["HAS_COLLEGE_DEP"] = (pdf["CD_DEP_COLLEGE_COUNT"] > 0).astype(int)
pdf["TOTAL_DEP"] = pdf["CD_DEP_COUNT"] + pdf["CD_DEP_EMPLOYED_COUNT"] + pdf["CD_DEP_COLLEGE_COUNT"]
pdf["AGE_BIN"] = pd.cut(pdf["AGE"], bins=[0, 18, 30, 45, 60, 100], labels=["0", "1", "2", "3", "4"]).astype(str)

# -----------------------------
# Prepare features and target
# -----------------------------
X = pdf.drop(columns=["TOTAL_SPENT", "SPENDER_CLASS"])
y = pdf["SPENDER_CLASS"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, random_state=42)

# CatBoost expects categorical columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# -----------------------------
# Train the model (1 time only)
# -----------------------------
model = CatBoostClassifier(
    depth=10,
    learning_rate=0.05,
    iterations=1000,
    l2_leaf_reg=3,
    border_count=64,
    loss_function="MultiClass",
    cat_features=cat_cols,
    verbose=100,
    early_stopping_rounds=20,
    random_seed=42
)
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
print(f"✅ Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

# -----------------------------
# Save artifacts
# -----------------------------
os.makedirs("ml", exist_ok=True)
with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(model, f)

with open("ml/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "f1_score": f1}, f, indent=2)

with open("ml/signature.json", "w") as f:
    json.dump({
        "inputs": list(X.columns),
        "output": "SPENDER_CLASS",
        "model_type": model.__class__.__name__,
        "hyperparams": model.get_params()
    }, f, indent=2)

with open("ml/drift_baseline.json", "w") as f:
    json.dump(pdf.describe(include="all").to_dict(), f, indent=2)

# -----------------------------
# Log to MLflow
# -----------------------------
experiment_name = "snowflake-high-spender-rf"
mlflow.set_experiment(experiment_name)
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
version_number = len(client.search_runs(experiment.experiment_id)) + 1
run_name = f"catboost_highspender_v{version_number}"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tag("model_version", run_name)
    mlflow.log_params(model.get_params())
    mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig("ml/confusion_matrix.png")
    mlflow.log_artifact("ml/confusion_matrix.png")

    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X.head(5)
    mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=signature)

    mlflow.log_artifact("ml/model.pkl.gz")
    mlflow.log_artifact("ml/metrics.json")
    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")

print(f"✅ Final Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Version: {version_number}")



