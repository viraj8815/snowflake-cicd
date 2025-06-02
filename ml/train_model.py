import os
import pandas as pd
import mlflow
import mlflow.sklearn
import uuid
import cloudpickle, gzip, json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, when

# Snowflake connection using Snowpark
connection_parameters = {
    "user": os.environ["SNOWFLAKE_USER"],
    "password": os.environ["SNOWFLAKE_PASSWORD"],
    "account": os.environ["SNOWFLAKE_ACCOUNT"],
    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
    "database": os.environ["SNOWFLAKE_DATABASE"],
    "schema": os.environ["SNOWFLAKE_SCHEMA"],
    "role": os.environ["SNOWFLAKE_ROLE"]
}

session = Session.builder.configs(connection_parameters).create()

# Load and join tables
sales = session.table("ML_DB.TRAINING_DATA.STORE_CUSTOMER_SALES_SAMPLE")
cust = session.table("ML_DB.TRAINING_DATA.CUSTOMER_SAMPLE")
date = session.table("ML_DB.TRAINING_DATA.DATE_SAMPLE")

df = (
    sales.join(cust, sales["SS_CUSTOMER_SK"] == cust["C_CUSTOMER_SK"])
         .join(date, (sales["D_YEAR"] == date["D_YEAR"]) & ((date["D_MONTH_SEQ"] % 12) == sales["D_MONTH"]))
         .select(
             sales["SS_SALES_PRICE"],
             sales["SS_QUANTITY"],
             sales["SS_EXT_DISCOUNT_AMT"],
             sales["SS_NET_PROFIT"],
             sales["D_YEAR"],
             sales["D_MONTH"],
             sales["D_DAY"],
             cust["C_BIRTH_YEAR"],
             cust["C_CURRENT_CDEMO_SK"],
             date["D_DAY_NAME"]
         )
         .with_column("profit_ratio", sales["SS_NET_PROFIT"] / sales["SS_SALES_PRICE"])
         .with_column("age_group",
                      when(cust["C_BIRTH_YEAR"] <= 1980, "GenX")
                      .when(cust["C_BIRTH_YEAR"] <= 2000, "Millennial")
                      .otherwise("GenZ"))
         .with_column("is_weekend",
                      when(date["D_DAY_NAME"].isin(["Saturday", "Sunday"]), 1).otherwise(0))
)

# Convert to pandas and prepare label
pdf = df.to_pandas()
pdf.rename(columns={"C_CURRENT_CDEMO_SK": "label"}, inplace=True)
pdf.dropna(subset=["label"], inplace=True)

X = pdf.drop("label", axis=1)
y = pdf["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# MLflow experiment tracking
run_id = str(uuid.uuid4())
mlflow.set_tracking_uri("file:///ml/mlruns")
mlflow.set_experiment("snowflake-mlops-pipeline")

with mlflow.start_run(run_name=f"train-{run_id}"):
    mlflow.log_param("n_estimators", 500)
    mlflow.log_param("random_state", 42)

    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

    os.makedirs("ml", exist_ok=True)
    with gzip.open("ml/model.pkl.gz", "wb") as f:
        cloudpickle.dump(model, f)

    signature = {
        "inputs": list(X.columns),
        "output": "label",
        "model_type": "RandomForestClassifier"
    }
    with open("ml/signature.json", "w") as f:
        json.dump(signature, f, indent=2)

    baseline_stats = pdf.describe().to_dict()
    with open("ml/drift_baseline.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)

    metrics = { "accuracy": accuracy }
    with open("ml/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")
    mlflow.log_artifact("ml/metrics.json")

print("✅ Model trained using all 3 tables (no aliasing) and tracked via MLflow.")
