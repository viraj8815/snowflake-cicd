import os
import json
import snowflake.connector

# Load version from file
with open("ml/version.txt", "r") as v:
    model_version = v.read().strip()

files_to_upload = [
    f"ml/model_v{model_version}.pkl.gz",
    f"ml/signature_v{model_version}.json",
    f"ml/drift_baseline_v{model_version}.json",
    f"ml/metrics_v{model_version}.json"
]

stage_name = "@ml_models_stage"

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
    for file_path in files_to_upload:
        put_command = f"PUT file://{file_path} {stage_name} OVERWRITE=TRUE;"
        print(f"📦 Uploading {file_path} to Snowflake stage...")
        cursor.execute(put_command)

    print("✅ All artifacts uploaded successfully!")

    # Load metrics and log to Snowflake
    with open(f"ml/metrics_v{model_version}.json", "r") as f:
        metrics = json.load(f)

    version = metrics["version"]
    accuracy = metrics["accuracy"]

    insert_sql = f"""
    INSERT INTO MODEL_HISTORY (VERSION, ACCURACY, IS_CHAMPION)
    SELECT '{version}', {accuracy}, FALSE;
    """
    print("📝 Logging model to MODEL_HISTORY...")
    cursor.execute(insert_sql)

    # Champion selection logic
    print("🔍 Selecting new champion model...")
    cursor.execute("UPDATE MODEL_HISTORY SET IS_CHAMPION = FALSE;")
    cursor.execute("""
    UPDATE MODEL_HISTORY
    SET IS_CHAMPION = TRUE
    WHERE VERSION = (
      SELECT VERSION FROM MODEL_HISTORY
      ORDER BY ACCURACY DESC, UPLOADED_AT DESC
      LIMIT 1
    );
    """)
    print("🏆 Champion model selected automatically.")

finally:
    cursor.close()
    conn.close()
