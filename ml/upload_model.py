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
# Load accuracy from metrics.json
# -----------------------------
with open("ml/metrics.json") as f:
    metrics = json.load(f)
new_accuracy = float(metrics.get("accuracy", 0.0))

# -----------------------------
# Create MODEL_HISTORY table if not exists
# -----------------------------
cursor.execute("""
    CREATE TABLE IF NOT EXISTS STAGE_DB.PUBLIC.MODEL_HISTORY (
        version STRING,
        accuracy FLOAT,
        deployed_on TIMESTAMP,
        is_champion BOOLEAN,
        model_path STRING
    )
""")

# -----------------------------
# Set version = '1' manually for first run
# -----------------------------
version = "1"

# -----------------------------
# Upload artifacts with versioned names
# -----------------------------
print(f"📦 Uploading artifacts for version {version}...")

artifacts = {
    f"ml/model.pkl.gz": f"model_v{version}.pkl.gz",
    f"ml/metrics.json": f"metrics_v{version}.json",
    f"ml/signature.json": f"signature_v{version}.json",
    f"ml/drift_baseline.json": f"drift_baseline_v{version}.json"
}

for src, dest in artifacts.items():
    cursor.execute(f"PUT file://{src} @ml_models_stage/{dest} AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
    print(f"✅ Uploaded {dest}")

# -----------------------------
# Log v1 to MODEL_HISTORY
# -----------------------------
cursor.execute("UPDATE STAGE_DB.PUBLIC.MODEL_HISTORY SET is_champion = FALSE")

cursor.execute(f"""
    INSERT INTO STAGE_DB.PUBLIC.MODEL_HISTORY 
    (version, accuracy, deployed_on, is_champion, model_path)
    VALUES ('{version}', {new_accuracy}, CURRENT_TIMESTAMP(), TRUE, 'model_v{version}.pkl.gz')
""")
print(f"✅ Logged model version {version} with accuracy {new_accuracy:.4f} as champion.")

# -----------------------------
# Deploy UDF pointing to v1 model
# -----------------------------
print("🔧 Creating or replacing Python UDF...")

cursor.execute(f"""
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
IMPORTS = ('@ml_models_stage/model_v1.pkl.gz')
AS
$$
import cloudpickle, gzip, os, sys
model_path = os.path.join(sys._xoptions["snowflake_import_directory"], "model_v1.pkl.gz")
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

print("✅ UDF deployed using model_v1.pkl.gz")

# -----------------------------
# Done
# -----------------------------
cursor.close()
conn.close()
