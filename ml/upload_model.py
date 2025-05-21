# ✅ Final upload_model.py with version comparison and numeric versioning (v1, v2, ...)

import os
import json
import snowflake.connector
import re

# Determine next version number (v1, v2, ...)
def get_next_version(cursor):
    cursor.execute("""
        SELECT MAX(TRY_CAST(SUBSTRING(VERSION, 2) AS INTEGER))
        FROM MODEL_HISTORY
    """)
    result = cursor.fetchone()
    max_version = result[0] if result[0] is not None else 0
    return f"v{max_version + 1}"

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
    # Generate version
    model_version = get_next_version(cursor)
    print(f"🔢 New model version: {model_version}")

    # Save version to file
    with open("ml/version.txt", "w") as v:
        v.write(model_version)

    # Define files
    files_to_upload = [
        f"ml/model_{model_version}.pkl.gz",
        f"ml/signature_{model_version}.json",
        f"ml/drift_baseline_{model_version}.json",
        f"ml/metrics_{model_version}.json"
    ]

    stage_name = "@ml_models_stage"

    # Upload files to Snowflake stage
    for file_path in files_to_upload:
        put_command = f"PUT file://{file_path} {stage_name} OVERWRITE=TRUE;"
        print(f"📦 Uploading {file_path} to Snowflake stage...")
        cursor.execute(put_command)

    print("✅ All artifacts uploaded successfully!")

    # Load metrics
    with open(f"ml/metrics_{model_version}.json", "r") as f:
        metrics = json.load(f)

    accuracy = metrics["accuracy"]

    # Check current champion accuracy
    cursor.execute("""
        SELECT MAX(ACCURACY)
        FROM MODEL_HISTORY
        WHERE IS_CHAMPION = TRUE
    """)
    result = cursor.fetchone()
    champion_accuracy = result[0] if result[0] is not None else -1

    print(f"🔍 Champion accuracy: {champion_accuracy}")
    print(f"📊 Current model accuracy: {accuracy}")

    if accuracy > champion_accuracy:
        print("✅ Accuracy is better. Promoting new model...")

        # Insert model as new champion
        insert_sql = f"""
        INSERT INTO MODEL_HISTORY (VERSION, ACCURACY, IS_CHAMPION)
        SELECT '{model_version}', {accuracy}, FALSE;
        """
        cursor.execute(insert_sql)

        cursor.execute("UPDATE MODEL_HISTORY SET IS_CHAMPION = FALSE;")
        cursor.execute(f"""
        UPDATE MODEL_HISTORY
        SET IS_CHAMPION = TRUE
        WHERE VERSION = '{model_version}';
        """)
        print("🏆 New champion promoted.")

    else:
        print("❌ Accuracy not improved. Skipping promotion.")

finally:
    cursor.close()
    conn.close()
