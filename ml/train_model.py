import os
import json
import gzip
import cloudpickle
import mlflow
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from snowflake.snowpark import Session
from snowflake.snowpark.functions import when
from mlflow.models.signature import infer_signature

# -----------------------------
# Snowflake connection
# -----------------------------
connection_parameters = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "role": os.environ["SNOWFLAKE_ROLE"],
    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    "database": os.environ["SNOWFLAKE_DATABASE"],
    "schema": os.environ["SNOWFLAKE_SCHEMA"]
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
        .with_column("age", 2025 - cust["C_BIRTH_YEAR"])
        .with_column("is_weekend", when(date["D_DAY_NAME"].isin(["Saturday", "Sunday"]), 1).otherwise(0))
        .with_column("purchase_range",
                     when(demo["CD_PURCHASE_ESTIMATE"] < 500, "Low")
                     .when(demo["CD_PURCHASE_ESTIMATE"] < 1000, "Medium")
                     .otherwise("High"))
        .select(
            "CD_GENDER",
            "CD_MARITAL_STATUS",
            "CD_EDUCATION_STATUS",
            "CD_CREDIT_RATING",
            "CD_DEP_COUNT",
            "CD_DEP_EMPLOYED_COUNT",
            "CD_DEP_COLLEGE_COUNT",
            "age",
            "is_weekend",
            "purchase_range"
        )
        .limit(10000)
)

# -----------------------------
# Convert to Pandas
# -----------------------------
pdf = df.to_pandas()
pdf.dropna(inplace=True)

if len(pdf) == 0:
    raise ValueError("No rows available for training.")

print("📋 Columns in DataFrame:", pdf.columns.tolist())

X = pdf.drop("PURCHASE_RANGE", axis=1)
y = pdf["PURCHASE_RANGE"]
X = pd.get_dummies(X)

# -----------------------------
# Train model
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# Save artifacts locally
# -----------------------------
os.makedirs("ml", exist_ok=True)
with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(model, f)

with open("ml/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f, indent=2)

with open("ml/signature.json", "w") as f:
    json.dump({
        "inputs": list(X.columns),
        "output": "purchase_range",
        "model_type": "RandomForestClassifier"
    }, f, indent=2)

with open("ml/drift_baseline.json", "w") as f:
    json.dump(pdf.describe(include='all').to_dict(), f, indent=2)

# -----------------------------
# Log to MLflow
# -----------------------------
mlflow.set_tracking_uri("file:///C:/Users/HP/OneDrive/Documents/LOCAL/snowflake-cicd/mlruns")
mlflow.set_experiment("snowflake-ml-model")
with mlflow.start_run(run_name="rf_model_v1") as run:
    mlflow.log_params({
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("dataset_version", "v1.0")

    # Save and log confusion matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.savefig("ml/confusion_matrix.png")
    mlflow.log_artifact("ml/confusion_matrix.png")

    # Save and log SHAP summary
    explainer = shap.TreeExplainer(model)
    shap.summary_plot(explainer.shap_values(X_test), X_test, show=False)
    plt.savefig("ml/shap_summary.png")
    mlflow.log_artifact("ml/shap_summary.png")

    # Log model using sklearn flavor
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X.head(5), signature=signature)

    # Log other files
    mlflow.log_artifact("ml/model.pkl.gz")
    mlflow.log_artifact("ml/metrics.json")
    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")

print(f"✅ Model trained with {len(X.columns)} features. Accuracy = {accuracy:.4f}")
