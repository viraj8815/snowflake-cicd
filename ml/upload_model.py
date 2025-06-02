import os
import json
import snowflake.connector

conn = snowflake.connector.connect(
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"],
    role=os.environ["SNOWFLAKE_ROLE"]
)
cursor = conn.cursor()

try:
    with open("ml/metrics.json", "r") as f:
        metrics = json.load(f)

    current_accuracy = metrics["accuracy"]

    cursor.execute("SELECT MAX(ACCURACY) FROM MODEL_HISTORY WHERE IS_CHAMPION = TRUE")
    result = cursor.fetchone()
    champion_accuracy = result[0] if result[0] is not None else -1

    if current_accuracy > champion_accuracy:
        cursor.execute("SELECT MAX(TRY_CAST(SUBSTRING(VERSION, 2) AS INTEGER)) FROM MODEL_HISTORY")
        result = cursor.fetchone()
        max_version = result[0] if result[0] is not None else 0
        model_version = f"v{max_version + 1}"

        with open("ml/version.txt", "w") as v:
            v.write(model_version)

        os.rename("ml/model.pkl.gz", f"ml/model_{model_version}.pkl.gz")
        os.rename("ml/signature.json", f"ml/signature_{model_version}.json")
        os.rename("ml/drift_baseline.json", f"ml/drift_baseline_{model_version}.json")
        os.rename("ml/metrics.json", f"ml/metrics_{model_version}.json")

        for file in [
            f"ml/model_{model_version}.pkl.gz",
            f"ml/signature_{model_version}.json",
            f"ml/drift_baseline_{model_version}.json",
            f"ml/metrics_{model_version}.json"
        ]:
            cursor.execute(f"PUT file://{file} @ml_models_stage OVERWRITE=TRUE")

        cursor.execute(f"""
            INSERT INTO MODEL_HISTORY (VERSION, ACCURACY, IS_CHAMPION)
            SELECT '{model_version}', {current_accuracy}, FALSE;
        """)
        cursor.execute("UPDATE MODEL_HISTORY SET IS_CHAMPION = FALSE;")
        cursor.execute(f"UPDATE MODEL_HISTORY SET IS_CHAMPION = TRUE WHERE VERSION = '{model_version}';")

        udf_sql = f"""
CREATE OR REPLACE FUNCTION predict_customer_segment(
    sales_price FLOAT,
    quantity INT,
    discount_amt FLOAT,
    net_profit FLOAT,
    year INT,
    month INT,
    day INT,
    birth_year INT,
    profit_ratio FLOAT,
    is_weekend INT
)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
HANDLER = 'predict'
PACKAGES = ('scikit-learn', 'cloudpickle', 'numpy')
IMPORTS = ('@ml_models_stage/model_{model_version}.pkl.gz')
AS
$$
import cloudpickle, gzip, sys, os
model_path = os.path.join(sys._xoptions["snowflake_import_directory"], "model_{model_version}.pkl.gz")
with gzip.open(model_path, "rb") as f:
    model = cloudpickle.load(f)
def predict(sales_price, quantity, discount_amt, net_profit, year, month, day, birth_year, profit_ratio, is_weekend):
    return float(model.predict([[sales_price, quantity, discount_amt, net_profit, year, month, day, birth_year, profit_ratio, is_weekend]])[0])
$$;
"""
        cursor.execute(udf_sql)
        print("✅ New champion promoted and UDF updated.")
    else:
        print("⚠️ Model did not outperform current champion.")
finally:
    cursor.close()
    conn.close()
