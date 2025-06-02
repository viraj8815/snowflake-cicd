import os
import json
import snowflake.connector
import shutil

# -----------------------------
# Load accuracy
# -----------------------------
with open("ml/metrics.json") as f:
    metrics = json.load(f)
accuracy = float(metrics.get("accuracy", 0.0))

# -----------------------------
# Determine version number
# -----------------------------
history_file = "ml/model_history.json"
if os.path.exists(history_file):
    with open(history_file) as f:
        history = json.load(f)
else:
    history = []

last_version = int(history[-1]["version"]) if history else 0
version = last_version + 1

# -----------------------------
# Rename artifacts with version
# -----------------------------
versioned_artifacts = {
    f"ml/model_v{version}.pkl.gz": "ml/model.pkl.gz",
    f"ml/metrics_v{version}.json": "ml/metrics.json",
    f"ml/signature_v{version}.json": "ml/signature.json",
    f"ml/drift_baseline_v{version}.json": "ml/drift_baseline.json"
}

for dest, src in versioned_artifacts.items():
    shutil.copyfile(src, dest)
    print(f"✅ Created {dest}")

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
# Upload versioned artifacts to stage
# -----------------------------
print(f"📦 Uploading artifacts for version {version}...")
for filepath in versioned_artifacts.keys():
    cursor.execute(f"PUT file://{filepath} @ml_models_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
    print(f"✅ Uploaded {os.path.basename(filepath)}")

# -----------------------------
# Update history table in Snowflake
# -----------------------------
cursor.execute("""
    CREATE TABLE IF NOT EXISTS STAGE_DB.PUBLIC.MODEL_HISTORY (
        version STRING,
        accuracy FLOAT,
        deployed_on TIMESTAMP,
        is_champion BOOLEAN
    )
""")

cursor.execute("SELECT MAX(TRY_CAST(version AS INT)) FROM STAGE_DB.PUBLIC.MODEL_HISTORY")
row = cursor.fetchone()
existing_max_version = int(row[0]) if row[0] is not None else 0
new_version = existing_max_version + 1

cursor.execute(f"""
    INSERT INTO STAGE_DB.PUBLIC.MODEL_HISTORY 
    (version, accuracy, deployed_on, is_champion)
    VALUES ('{new_version}', {accuracy}, CURRENT_TIMESTAMP(), FALSE)
""")

# -----------------------------
# Set champion model (max accuracy)
# -----------------------------
cursor.execute("UPDATE STAGE_DB.PUBLIC.MODEL_HISTORY SET is_champion = FALSE")
cursor.execute("""
    UPDATE STAGE_DB.PUBLIC.MODEL_HISTORY
    SET is_champion = TRUE
    WHERE accuracy = (
        SELECT MAX(accuracy) FROM STAGE_DB.PUBLIC.MODEL_HISTORY
    )
""")

# -----------------------------
# If current version is the best, deploy UDF using static model.pkl.gz
# -----------------------------
cursor.execute("""
    SELECT version FROM STAGE_DB.PUBLIC.MODEL_HISTORY 
    WHERE is_champion = TRUE ORDER BY version DESC LIMIT 1
""")
champion_version = cursor.fetchone()[0]

if str(champion_version) == str(new_version):
    print("🏆 This is the best model so far. Updating model.pkl.gz for UDF...")
    shutil.copyfile(f"ml/model_v{version}.pkl.gz", "ml/model.pkl.gz")
    cursor.execute("PUT file://ml/model.pkl.gz @ml_models_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE")

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
    print("✅ UDF updated to use best performing model.")
else:
    print(f"⚠️ Model v{version} not deployed as it's not the highest accuracy.")

# -----------------------------
# Update local history for reference
# -----------------------------
history.append({"version": version, "accuracy": accuracy})
with open(history_file, "w") as f:
    json.dump(history, f, indent=2)

# -----------------------------
# Cleanup
# -----------------------------
cursor.close()
conn.close()

