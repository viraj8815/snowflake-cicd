import os
import json
import snowflake.connector
from snowflake.connector import DictCursor

# -------------------------------
# Load environment variables
# -------------------------------
sf_user = os.environ["SNOWFLAKE_USER"]
sf_password = os.environ["SNOWFLAKE_PASSWORD"]
sf_account = os.environ["SNOWFLAKE_ACCOUNT"]
sf_warehouse = os.environ["SNOWFLAKE_WAREHOUSE"]
sf_database = os.environ["SNOWFLAKE_DATABASE"]
sf_schema = os.environ["SNOWFLAKE_SCHEMA"]
sf_role = os.environ["SNOWFLAKE_ROLE"]

# -------------------------------
# Connect to Snowflake
# -------------------------------
conn = snowflake.connector.connect(
    user=sf_user,
    password=sf_password,
    account=sf_account,
    warehouse=sf_warehouse,
    database=sf_database,
    schema=sf_schema,
    role=sf_role
)
cs = conn.cursor(DictCursor)

# -------------------------------
# Upload artifacts to stage
# -------------------------------
stage = "@ml_models_stage"
files = ["model.pkl.gz", "signature.json", "metrics.json", "drift_baseline.json"]

for file in files:
    put_cmd = f"PUT file://ml/{file} {stage} OVERWRITE = TRUE"
    cs.execute(put_cmd)
    print(f"✅ Uploaded: {file} to {stage}")

# -------------------------------
# Load metrics and signature
# -------------------------------
with open("ml/metrics.json") as f:
    metrics = json.load(f)
accuracy = metrics.get("accuracy", None)
if accuracy is None:
    raise ValueError("⚠️ 'accuracy' missing in metrics.json")

# -------------------------------
# Create MODEL_HISTORY table
# -------------------------------
cs.execute("""
    CREATE TABLE IF NOT EXISTS MODEL_HISTORY (
        VERSION STRING,
        ACCURACY FLOAT,
        TIMESTAMP TIMESTAMP,
        IS_CHAMPION BOOLEAN
    )
""")

# -------------------------------
# Determine next version
# -------------------------------
cs.execute("SELECT MAX(VERSION) AS LAST_VERSION FROM MODEL_HISTORY WHERE VERSION LIKE 'v%'")
row = cs.fetchone()
if row["LAST_VERSION"]:
    last_version_num = int(row["LAST_VERSION"][1:])
    version = f"v{last_version_num + 1}"
else:
    version = "v1"

# -------------------------------
# Update champion tracking
# -------------------------------
cs.execute("UPDATE MODEL_HISTORY SET IS_CHAMPION = FALSE WHERE IS_CHAMPION = TRUE")
cs.execute("""
    INSERT INTO MODEL_HISTORY (VERSION, ACCURACY, TIMESTAMP, IS_CHAMPION)
    VALUES (%s, %s, CURRENT_TIMESTAMP, TRUE)
""", (version, accuracy))

print(f"✅ Logged model version: {version}, accuracy: {accuracy}, champion: TRUE")

# -------------------------------
# Create UDF in Snowflake
# -------------------------------
cs.execute(f"""
CREATE OR REPLACE FUNCTION INFER_RF_MODEL(
    VERSION STRING,
    CS_SALES_PRICE FLOAT,
    CS_QUANTITY FLOAT,
    CS_EXT_DISCOUNT_AMT FLOAT,
    CS_NET_PROFIT FLOAT,
    D_YEAR INT,
    D_MONTH_SEQ INT,
    C_BIRTH_YEAR INT,
    IS_WEEKEND INT,
    AGE_GROUP_GENX INT,
    AGE_GROUP_MILLENNIAL INT,
    AGE_GROUP_GENZ INT,
    D_DAY_NAME_SATURDAY INT,
    D_DAY_NAME_SUNDAY INT,
    D_DAY_NAME_OTHER INT
)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('cloudpickle', 'scikit-learn', 'pandas', 'numpy')
IMPORTS = ('@ml_models_stage/model.pkl.gz')
HANDLER = 'predict'
AS
$$
import cloudpickle, gzip, sys, os

model_path = os.path.join(sys._xoptions["snowflake_import_directory"], "model.pkl.gz")
with gzip.open(model_path, "rb") as f:
    model = cloudpickle.load(f)

def predict(VERSION, CS_SALES_PRICE, CS_QUANTITY, CS_EXT_DISCOUNT_AMT, CS_NET_PROFIT,
            D_YEAR, D_MONTH_SEQ, C_BIRTH_YEAR, IS_WEEKEND,
            AGE_GROUP_GENX, AGE_GROUP_MILLENNIAL, AGE_GROUP_GENZ,
            D_DAY_NAME_SATURDAY, D_DAY_NAME_SUNDAY, D_DAY_NAME_OTHER):
    row = [[
        CS_SALES_PRICE, CS_QUANTITY, CS_EXT_DISCOUNT_AMT, CS_NET_PROFIT,
        D_YEAR, D_MONTH_SEQ, C_BIRTH_YEAR, IS_WEEKEND,
        AGE_GROUP_GENX, AGE_GROUP_MILLENNIAL, AGE_GROUP_GENZ,
        D_DAY_NAME_SATURDAY, D_DAY_NAME_SUNDAY, D_DAY_NAME_OTHER
    ]]
    return float(model.predict(row)[0])
$$
""")

print("✅ Inference UDF INFER_RF_MODEL created.")
cs.close()
conn.close()
