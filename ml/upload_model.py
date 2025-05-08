import os
import snowflake.connector

file_path = "ml/model.pkl.gz"
stage_name = "@ml_models_stage"

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
    put_command = f"PUT file://{file_path} {stage_name} OVERWRITE=TRUE;"
    print("📦 Uploading model to Snowflake stage...")
    cursor.execute(put_command)
    print("✅ Upload completed!")
finally:
    cursor.close()
    conn.close()