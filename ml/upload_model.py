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
# Upload model artifacts to stage
# -------------------------------
stage = "@ml_models_stage"
files = ["model.pkl.gz", "signature.json", "metrics.json", "drift_baseline.json"]

for file in files:
    put_cmd = f"PUT file://ml/{file} {stage} OVERWRITE = TRUE"
    cs.execute(put_cmd)
    print(f"✅ Uploaded: {file} to {stage}")

# -------------------------------
# Load model metrics
# -------------------------------
with open("ml/metrics.json") as f:
    metrics = json.load(f)
accuracy = metrics.get("accuracy", None)
if accuracy is None:
    raise ValueError("⚠️ 'accuracy' missing in metrics.json")

# -------------------------------
# Create or update MODEL_HISTORY
# -------------------------------
cs.execute("""
    CREATE TABLE IF NOT EXISTS MODEL_HISTORY (
        VERSION STRING,
        ACCURACY FLOAT,
        TIMESTAMP TIMESTAMP,
        IS_CHAMPION BOOLEAN
    )
""")

# Find the latest version
cs.execute("SELECT MAX(VERSION) AS LAST_VERSION FROM MODEL_HISTORY WHERE VERSION LIKE 'v%'")
row = cs.fetchone()
if row["LAST_VERSION"]:
    last_version_num = int(row["LAST_VERSION"][1:])
    version = f"v{last_version_num + 1}"
else:
    version = "v1"

# Demote existing champion
cs.execute("UPDATE MODEL_HISTORY SET IS_CHAMPION = FALSE WHERE IS_CHAMPION = TRUE")

# Insert new version as champion
cs.execute(f"""
    INSERT INTO MODEL_HISTORY (VERSION, ACCURACY, TIMESTAMP, IS_CHAMPION)
    VALUES (%s, %s, CURRENT_TIMESTAMP, %s)
""", (version, accuracy, True))

print(f"✅ Logged model version: {version} | accuracy: {accuracy} | champion: TRUE")

# -------------------------------
# Recreate the inference UDF
# -------------------------------
print("🔁 Creating or replacing inference UDF...")

udf_sql = f"""
CREATE OR REPLACE FUNCTION infer_model(
    sales_price FLOAT,
    quantity FLOAT,
    discount_amt FLOAT,
    net_profit FLOAT,
    year INT,
    month_seq INT,
    birth_year INT,
    day_name STRING
)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('scikit-learn', 'cloudpickle', 'numpy')
IMPORTS = ('@ml_models_stage/model.pkl.gz')
HANDLER = 'predict'
AS
$$
import cloudpickle, gzip, os, sys

model_path = os.path.join(sys._xoptions["snowflake_import_directory"], "model.pkl.gz")
with gzip.open(model_path, "rb") as f:
    model = cloudpickle.load(f)

def predict(sales_price, quantity, discount_amt, net_profit, year, month_seq, birth_year, day_name):
    profit_ratio = net_profit / sales_price if sales_price != 0 else 0
    age_group = "GenZ"
    if birth_year <= 1980:
        age_group = "GenX"
    elif birth_year <= 2000:
        age_group = "Millennial"
    is_weekend = 1 if day_name in ["Saturday", "Sunday"] else 0

    features = [
        sales_price, quantity, discount_amt, net_profit,
        year, month_seq, birth_year, profit_ratio, is_weekend
    ]
    return float(model.predict([features])[0])
$$;
"""

cs.execute(udf_sql)
print("✅ Inference UDF `infer_model` created successfully.")

cs.close()
conn.close()
