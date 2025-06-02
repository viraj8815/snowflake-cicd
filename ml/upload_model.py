import os
import snowflake.connector
import cloudpickle
import gzip

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
# Upload model to stage
# -----------------------------
print("📦 Uploading model to stage...")
cursor.execute("""
    PUT file://ml/model.pkl.gz @ml_models_stage 
    AUTO_COMPRESS=FALSE 
    OVERWRITE=TRUE
""")
print("✅ Model uploaded to Snowflake stage.")

# -----------------------------
# Create or replace UDF
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


