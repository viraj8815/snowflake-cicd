# ✅ Full upload_model.py with corrected UDF formatting

import os
import json
import snowflake.connector
import re

# Connect to Snowflake
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
    # Load current model metrics
    with open("ml/metrics.json", "r") as f:
        metrics = json.load(f)

    current_accuracy = metrics["accuracy"]

    # Check current champion accuracy
    cursor.execute("""
        SELECT MAX(ACCURACY)
        FROM MODEL_HISTORY
        WHERE IS_CHAMPION = TRUE
    """)
    result = cursor.fetchone()
    champion_accuracy = result[0] if result[0] is not None else -1

    print(f"🔍 Champion accuracy: {champion_accuracy}")
    print(f"📊 Current model accuracy: {current_accuracy}")

    if current_accuracy > champion_accuracy:
        # Determine next version number (v1, v2, ...)
        cursor.execute("""
            SELECT MAX(TRY_CAST(SUBSTRING(VERSION, 2) AS INTEGER))
            FROM MODEL_HISTORY
        """)
        result = cursor.fetchone()
        max_version = result[0] if result[0] is not None else 0
        model_version = f"v{max_version + 1}"

        # Save version to file
        with open("ml/version.txt", "w") as v:
            v.write(model_version)

        print(f"🔢 New model version: {model_version}")

        # Rename artifacts
        os.rename("ml/model.pkl.gz", f"ml/model_{model_version}.pkl.gz")
        os.rename("ml/signature.json", f"ml/signature_{model_version}.json")
        os.rename("ml/drift_baseline.json", f"ml/drift_baseline_{model_version}.json")
        os.rename("ml/metrics.json", f"ml/metrics_{model_version}.json")

        # Upload files to Snowflake stage
        files_to_upload = [
            f"ml/model_{model_version}.pkl.gz",
            f"ml/signature_{model_version}.json",
            f"ml/drift_baseline_{model_version}.json",
            f"ml/metrics_{model_version}.json"
        ]

        stage_name = "@ml_models_stage"
        for file_path in files_to_upload:
            put_command = f"PUT file://{file_path} {stage_name} OVERWRITE=TRUE;"
            print(f"📦 Uploading {file_path} to Snowflake stage...")
            cursor.execute(put_command)

        print("✅ All artifacts uploaded successfully!")

        # Insert into MODEL_HISTORY and promote
        insert_sql = f"""
        INSERT INTO MODEL_HISTORY (VERSION, ACCURACY, IS_CHAMPION)
        SELECT '{model_version}', {current_accuracy}, FALSE;
        """
        cursor.execute(insert_sql)

        cursor.execute("UPDATE MODEL_HISTORY SET IS_CHAMPION = FALSE;")
        cursor.execute(f"""
        UPDATE MODEL_HISTORY
        SET IS_CHAMPION = TRUE
        WHERE VERSION = '{model_version}';
        """)
        print("🏆 New champion promoted.")

        # Auto-redeploy Python UDF to use new champion version
        udf_sql = f"""
CREATE OR REPLACE FUNCTION predict_category(
    price FLOAT,
    qty INT,
    discount FLOAT,
    profit FLOAT,
    year INT,
    month INT,
    day INT,
    closed_date_sk INT
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

model_path = os.path.join(
    sys._xoptions["snowflake_import_directory"],
    "model_{model_version}.pkl.gz"
)

with gzip.open(model_path, "rb") as f:
    model = cloudpickle.load(f)

def predict(price, qty, discount, profit, year, month, day, closed_date_sk):
    return float(model.predict([[price, qty, discount, profit, year, month, day, closed_date_sk]])[0])
$$;
        """
        cursor.execute(udf_sql)
        print("🔁 Auto-redeployed UDF for new champion model.")

    else:
        print("❌ Accuracy not improved. Skipping versioning and promotion.")

finally:
    cursor.close()
    conn.close()

    
