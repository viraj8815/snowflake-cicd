import os
import json
import gzip
import mlflow
import cloudpickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from snowflake.snowpark import Session
from snowflake.snowpark.functions import when

# -------------------------------
# Snowflake connection
# -------------------------------
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

# -------------------------------
# Load and Join 3 Tables
# -------------------------------
sales = session.table("ML_DB.TRAINING_DATA.CATALOG_SALES_SAMPLE")
cust = session.table("ML_DB.TRAINING_DATA.CUSTOMER_SAMPLE")
date = session.table("ML_DB.TRAINING_DATA.DATE_DIM_SAMPLE")

df = (
    sales.join(cust, sales["CS_BILL_CUSTOMER_SK"] == cust["C_CUSTOMER_SK"])
         .join(date, sales["CS_SOLD_DATE_SK"] == date["D_DATE_SK"])
         .select(
             sales["CS_SALES_PRICE"],
             sales["CS_QUANTITY"],
             sales["CS_EXT_DISCOUNT_AMT"],
             sales["CS_NET_PROFIT"],
             date["D_YEAR"],
             date["D_MONTH_SEQ"],
             date["D_DAY_NAME"],
             cust["C_BIRTH_YEAR"],
             cust["C_CURRENT_CDEMO_SK"]
         )
         .with_column("profit_ratio",
                      when(sales["CS_SALES_PRICE"] != 0,
                           sales["CS_NET_PROFIT"] / sales["CS_SALES_PRICE"])
                      .otherwise(0))
         .with_column("age_group",
                      when(cust["C_BIRTH_YEAR"] <= 1980, "GenX")
                      .when(cust["C_BIRTH_YEAR"] <= 2000, "Millennial")
                      .otherwise("GenZ"))
         .with_column("is_weekend",
                      when(date["D_DAY_NAME"].isin(["Saturday", "Sunday"]), 1)
                      .otherwise(0))
)

# -------------------------------
# Convert to Pandas
# -------------------------------
pdf = df.to_pandas()
pdf.rename(columns={"C_CURRENT_CDEMO_SK": "label"}, inplace=True)
pdf.dropna(subset=["label"], inplace=True)

if len(pdf) == 0:
    raise ValueError("❌ No rows available for training.")

X = pdf.drop("label", axis=1)
y = pdf["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# -------------------------------
# Train model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------------------
# MLflow Tracking
# -------------------------------
mlflow.set_experiment("snowflake-ml-model")
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, artifact_path="model")

# -------------------------------
# Save Artifacts
# -------------------------------
os.makedirs("ml", exist_ok=True)

with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(model, f)

with open("ml/signature.json", "w") as f:
    json.dump({
        "inputs": list(X.columns),
        "output": "label",
        "model_type": "RandomForestClassifier"
    }, f, indent=2)

with open("ml/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f, indent=2)

with open("ml/drift_baseline.json", "w") as f:
    json.dump(pdf.describe().to_dict(), f, indent=2)

print("✅ Model trained, tracked with MLflow, and saved to ./ml")
