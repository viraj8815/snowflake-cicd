# ✅ Updated upload_model.py to upload all versioned artifacts

import os
import snowflake.connector

# Load version from file
with open("ml/version.txt", "r") as v:
    model_version = v.read().strip()

# Define file list
files_to_upload = [
    f"ml/model_v{model_version}.pkl.gz",
    f"ml/signature_v{model_version}.json",
    f"ml/drift_baseline_v{model_version}.json",
    f"ml/metrics_v{model_version}.json"
]

# Snowflake stage
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
finally:
    cursor.close()
    conn.close()
