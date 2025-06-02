import os
import json
import snowflake.connector

# -----------------------------
# Snowflake connection
# -----------------------------
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

# -----------------------------
# Upload model + artifacts to stage
# -----------------------------
print("📦 Uploading model artifacts to @ml_models_stage...")
for file in ["ml/model.pkl.gz", "ml/metrics.json", "ml/signature.json", "ml/drift_baseline.json"]:
    cursor.execute(f"PUT file://{file} @ml_models_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
    print(f"✅ Uploaded {file}")

# -----------------------------
# Load accuracy from metrics.json
# -----------------------------
with open("ml/metrics.json") as f:
    metrics = json.load(f)
accuracy = float(metrics.get("accuracy", 0.0))

# -----------------------------
# Create history table if not exists
# -----------------------------
cursor.execute("""
    CREATE TABLE IF NOT EXISTS STAGE_DB.PUBLIC.MODEL_HISTORY (
        version STRING,
        accuracy FLOAT,
        deployed_on TIMESTAMP,
        is_champion BOOLEAN
    )
""")

# -----------------------------
# Determine next version
# -----------------------------
cursor.execute("SELECT MAX(TRY_CAST(version AS INT)) FROM STAGE_DB.PUBLIC.MODEL_HISTORY")
row = cursor.fetchone()
last_version = int(row[0]) if row[0] is not None else 0
version = str(last_version + 1)

# -----------------------------
# Insert new version as is_champion = FALSE for now
# -----------------------------
cursor.execute(f"""
    INSERT INTO STAGE_DB.PUBLIC.MODEL_HISTORY 
    (version, accuracy, deployed_on, is_champion)
    VALUES ('{version}', {accuracy}, CURRENT_TIMESTAMP(), FALSE)
""")
print(f"✅ Logged version {version} with accuracy {accuracy:.4f} to MODEL_HISTORY.")

# -----------------------------
# Champion logic — set only best version as TRUE
# -----------------------------
cursor.execute("""
    UPDATE STAGE_DB.PUBLIC.MODEL_HISTORY
    SET is_champion = FALSE
""")
cursor.execute("""
    UPDATE STAGE_DB.PUBLIC.MODEL_HISTORY
    SET is_champion = TRUE
    WHERE accuracy = (
        SELECT MAX(accuracy) FROM STAGE_DB.PUBLIC.MODEL_HISTORY
    )
""")
print("🏆 Champion model updated based on highest accuracy.")

# -----------------------------
# Deploy Python UDF
# -----------------------------
print("🔧 Creating or replacing Python UDF...")

cursor.execute("""
CREATE OR REPLACE FUNCTION infer_model(
    CS_SALES_PRICE FLOAT,
    CS_QUANTITY FLOAT,
    CS_EXT_DISCOUNT_AMT FLOAT,
    CS_NET_PROFIT FLOAT,
    D_YEAR INT,
    D_MONTH_SEQ INT,
    C_BIRTH_YEAR INT,
    PROFIT_RATIO FLOAT,
    IS_WEEKEND INT,
    AGE_GROUP_GENX INT,
    AGE_GROUP_MILLENNIAL INT,
    AGE_GROUP_GENZ INT,
    D_DAY_NAME_FRIDAY INT,
    D_DAY_NAME_MONDAY INT,
    D_DAY_NAME_SATURDAY INT,
    D_DAY_NAME_SUNDAY INT,
    D_DAY_NAME_THURSDAY INT,
    D_DAY_NAME_TUESDAY INT,
    D_DAY_NAME_WEDNESDAY INT
)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
HANDLER = 'predict'
PACKAGES = ('scikit-learn', 'cloudpickle', 'numpy')
IMPORTS = ('@ml_models_stage/model.pkl.gz')
AS
$$
import cloudpickle, gzip, os, sys
model_path = os.path.join(sys._xoptions["snowflake_import_directory"], "model.pkl.gz")
with gzip.open(model_path, "rb") as f:
    model = cloudpickle.load(f)

def predict(CS_SALES_PRICE, CS_QUANTITY, CS_EXT_DISCOUNT_AMT, CS_NET_PROFIT,
            D_YEAR, D_MONTH_SEQ, C_BIRTH_YEAR, PROFIT_RATIO, IS_WEEKEND,
            AGE_GROUP_GENX, AGE_GROUP_MILLENNIAL, AGE_GROUP_GENZ,
            D_DAY_NAME_FRIDAY, D_DAY_NAME_MONDAY, D_DAY_NAME_SATURDAY, D_DAY_NAME_SUNDAY,
            D_DAY_NAME_THURSDAY, D_DAY_NAME_TUESDAY, D_DAY_NAME_WEDNESDAY):
    features = [[
        CS_SALES_PRICE, CS_QUANTITY, CS_EXT_DISCOUNT_AMT, CS_NET_PROFIT,
        D_YEAR, D_MONTH_SEQ, C_BIRTH_YEAR, PROFIT_RATIO, IS_WEEKEND,
        AGE_GROUP_GENX, AGE_GROUP_MILLENNIAL, AGE_GROUP_GENZ,
        D_DAY_NAME_FRIDAY, D_DAY_NAME_MONDAY, D_DAY_NAME_SATURDAY, D_DAY_NAME_SUNDAY,
        D_DAY_NAME_THURSDAY, D_DAY_NAME_TUESDAY, D_DAY_NAME_WEDNESDAY
    ]]
    return float(model.predict(features)[0])
$$;
""")

print("✅ Python UDF deployed successfully.")

# -----------------------------
# Cleanup
# -----------------------------
cursor.close()
conn.close()

