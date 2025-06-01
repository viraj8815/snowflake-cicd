from snowflake.snowpark import Session
from snowflake.snowpark.functions import col, when
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import uuid, cloudpickle, gzip, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Snowflake connection
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

# Load tables from marketplace
cs = session.table("TPCDS_10TD.CATALOG_SALES")
cu = session.table("TPCDS_10TD.CUSTOMER")
dt = session.table("TPCDS_10TD.DATE_DIM")

# Join them together
df = (
    cs.join(cu, cs["cs_bill_customer_sk"] == cu["c_customer_sk"])
      .join(dt, cs["cs_sold_date_sk"] == dt["d_date_sk"])
      .select(
          cs["cs_sales_price"],
          cs["cs_quantity"],
          cs["cs_ext_discount_amt"],
          cs["cs_net_profit"],
          dt["d_year"],
          dt["d_month_seq"],
          dt["d_week_seq"],
          cu["c_birth_year"],
          cu["c_current_cdemo_sk"].alias("label")
      )
      .filter(cs["cs_sales_price"].is_not_null())
      .filter(cs["cs_quantity"].is_not_null())
      .filter(cs["cs_net_profit"].is_not_null())
      .limit(10000)
)

# Add derived features
df = df.with_column("profit_ratio", col("cs_net_profit") / col("cs_sales_price"))

# Convert to pandas
pandas_df = df.to_pandas()

# ML Training
pandas_df = pandas_df.dropna(subset=["label"])
X = pandas_df.drop("label", axis=1)
y = pandas_df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

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

    baseline_stats = pandas_df.describe().to_dict()
    with open("ml/drift_baseline.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)

    metrics = { "accuracy": accuracy }
    with open("ml/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    mlflow.log_artifact("ml/signature.json")
    mlflow.log_artifact("ml/drift_baseline.json")
    mlflow.log_artifact("ml/metrics.json")

print("✅ Model trained on 3 joined tables and tracked with MLflow.")
