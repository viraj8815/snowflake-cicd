import os
import json
import gzip
import cloudpickle
import mlflow
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from snowflake.snowpark import Session
from snowflake.snowpark.functions import when
from catboost import CatBoostClassifier, Pool
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
# Target Variable
# -----------------------------
pdf["PURCHASE_RANGE"] = pd.qcut(pdf["CD_PURCHASE_ESTIMATE"], 3, labels=["Low", "Medium", "High"])
pdf.drop(columns=["CD_PURCHASE_ESTIMATE"], inplace=True)

print("📊 Class Distribution:\n", pdf["PURCHASE_RANGE"].value_counts())
print("📋 Columns in DataFrame:", pdf.columns.tolist())

X = pdf.drop("PURCHASE_RANGE", axis=1)
y = pdf["PURCHASE_RANGE"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, random_state=42)

# Categorical columns (CatBoost handles them natively)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# -----------------------------
# Train CatBoost Model
# -----------------------------
train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.03,
    eval_metric="MultiClass",
    early_stopping_rounds=20,
    verbose=100,
    random_seed=42
)

model.fit(train_pool, eval_set=test_pool)

# Predict and evaluate
y_pred = model.predict(test_pool)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

# -----------------------------
# Wrap model with data for saving
# -----------------------------
pipeline = {
    "model": model,
    "cat_features": cat_cols,
    "label_encoder": label_encoder,
    "feature_order": list(X.columns)
}

# -----------------------------
# Save artifacts locally
# -----------------------------
os.makedirs("ml", exist_ok=True)
with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(pipeline, f)

with open("ml/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "f1": f1}, f, indent=2)

with open("ml/signature.json", "w") as f:
    json.dump({
        "inputs": list(X.columns),
        "output": "purchase_range",
        "model_type": model.__class__.__name__,
        "hyperparams": model.get_params()
    }, f, indent=2)

with open("ml/drift_baseline.json", "w") as f:
    json.dump(pdf.describe(include='all').to_dict(), f, indent=2)

with open("ml/label_mapping.json", "w") as f:
    label_mapping = {label: int(code) for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    json.dump(label_mapping, f, indent=2)

# -----------------------------
# Log to MLflow
# -----------------------------
experiment_name = "snowflake-ml-model"
mlflow.set_experiment(experiment_name)
client = MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
existing_runs = client.search_runs(experiment.experiment_id)
version_number = len(existing_runs) + 1
run_name = f"catboost_model_v{version_number}"

with mlflow.start_run(run_name=run_name) as run:
    mlflow.set_tag("dataset_version", f"v{version_number}")
    mlflow.set_tag("model_version", run_name)

    mlflow.log_params(model.get_params())
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig("ml/confusion_matrix.png")
    mlflow.log_artifact("ml/confusion_matrix.png")

    # SHAP summary (CatBoost-specific)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("ml/shap_summary.png")
    mlflow.log_artifact("ml/shap_summary.png")

    # Log artifacts
    mlflow.log_artifact("ml/model.pkl.gz")
    mlflow.log_artifact("ml/metrics.json")
    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")
    mlflow.log_artifact("ml/label_mapping.json")

print(f"✅ Trained and logged {run_name} with accuracy = {accuracy:.4f}, f1 = {f1:.4f}")
