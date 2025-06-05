import os, json, gzip, cloudpickle
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from snowflake.snowpark import Session
from snowflake.snowpark.functions import when
from sklearn.preprocessing import OrdinalEncoder
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from xgboost import XGBClassifier

# Snowflake connection
connection_parameters = {
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "role": os.environ["SNOWFLAKE_ROLE"],
    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"]
}
session = Session.builder.configs(connection_parameters).create()

# Load training tables
cust = session.table("ML_DB.TRAINING_DATA.CUSTOMER_SAMPLE")
demo = session.table("ML_DB.TRAINING_DATA.CUSTOMER_DEMOGRAPHICS_SAMPLE")
date = session.table("ML_DB.TRAINING_DATA.DATE_DIM_SAMPLE")

df = (
    cust.join(demo, cust["C_CURRENT_CDEMO_SK"] == demo["CD_DEMO_SK"])
        .join(date, cust["C_FIRST_SALES_DATE_SK"] == date["D_DATE_SK"])
        .with_column("age", 2025 - cust["C_BIRTH_YEAR"])
        .with_column("is_weekend", when(date["D_DAY_NAME"].isin(["Saturday", "Sunday"]), 1).otherwise(0))
        .with_column("purchase_range",
                     when(demo["CD_PURCHASE_ESTIMATE"] < 4000, "Low")
                     .when(demo["CD_PURCHASE_ESTIMATE"] < 7000, "Medium")
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

pdf = df.to_pandas()
pdf.dropna(inplace=True)
if len(pdf) == 0:
    raise ValueError("No data for training.")

X = pdf.drop("PURCHASE_RANGE", axis=1)
y = pdf["PURCHASE_RANGE"]

# Ordinal encode categoricals
cat_cols = X.select_dtypes(include=["object"]).columns
encoder = OrdinalEncoder()
X[cat_cols] = encoder.fit_transform(X[cat_cols])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, use_label_encoder=False, eval_metric="mlogloss", random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

# Save artifacts
os.makedirs("ml", exist_ok=True)
with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(model, f)
with open("ml/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "f1_score": f1}, f, indent=2)
with open("ml/signature.json", "w") as f:
    json.dump({
        "inputs": list(X.columns),
        "output": "purchase_range",
        "model_type": model.__class__.__name__,
        "hyperparams": model.get_params()
    }, f, indent=2)
with open("ml/drift_baseline.json", "w") as f:
    json.dump(pdf.describe(include='all').to_dict(), f, indent=2)

# MLflow logging
experiment_name = "snowflake-ml-model"
mlflow.set_experiment(experiment_name)
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
existing_runs = client.search_runs(experiment.experiment_id)
version_number = len(existing_runs) + 1
run_name = f"xgb_model_v{version_number}"

with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("dataset_version", f"v{version_number}")
    mlflow.set_tag("model_version", run_name)
    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.savefig("ml/confusion_matrix.png")
    mlflow.log_artifact("ml/confusion_matrix.png")

    # Log model
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X.head(5).astype(int)
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=input_example, signature=signature)

    # Log files
    mlflow.log_artifact("ml/model.pkl.gz")
    mlflow.log_artifact("ml/metrics.json")
    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")

print(f"✅ Trained XGBClassifier v{version_number} | Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")
