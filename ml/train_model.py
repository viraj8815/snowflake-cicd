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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from snowflake.snowpark import Session
from snowflake.snowpark.functions import when
from mlflow.models.signature import infer_signature
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
# Convert to Pandas
# -----------------------------
pdf = df.to_pandas()
pdf.dropna(inplace=True)

# Assign purchase_range using quantiles
pdf["PURCHASE_RANGE"] = pd.qcut(pdf["CD_PURCHASE_ESTIMATE"], 3, labels=["Low", "Medium", "High"])
pdf.drop(columns=["CD_PURCHASE_ESTIMATE"], inplace=True)

print("📊 Class Distribution:\n", pdf["PURCHASE_RANGE"].value_counts())
print("📋 Columns in DataFrame:", pdf.columns.tolist())

X = pdf.drop("PURCHASE_RANGE", axis=1)
y = pdf["PURCHASE_RANGE"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, random_state=42)

# -----------------------------
# Preprocessing
# -----------------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# Fit the preprocessor and transform manually
preprocessor.fit(X_train)
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# -----------------------------
# Train the model with early stopping
# -----------------------------
model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

model.fit(
    X_train_processed, y_train,
    eval_set=[(X_test_processed, y_test)],
    early_stopping_rounds=20,
    verbose=True
)

# Predict and evaluate
y_pred = model.predict(X_test_processed)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

# Wrap model and preprocessor for deployment
pipeline = make_pipeline(preprocessor, model)

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
run_name = f"xgb_model_v{version_number}"

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

    # SHAP summary
    explainer = shap.Explainer(model)
    shap.summary_plot(explainer(X_test_processed), X_test_processed, show=False)
    plt.savefig("ml/shap_summary.png")
    mlflow.log_artifact("ml/shap_summary.png")

    # Log artifacts
    mlflow.log_artifact("ml/model.pkl.gz")
    mlflow.log_artifact("ml/metrics.json")
    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")
    mlflow.log_artifact("ml/label_mapping.json")

print(f"✅ Trained and logged {run_name} with accuracy = {accuracy:.4f}, f1 = {f1:.4f}")
