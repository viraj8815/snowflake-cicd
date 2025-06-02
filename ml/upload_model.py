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
cursor.execute("PUT file://ml/model.pkl.gz @ml_models_stage auto_compress=false overwrite=true")
print("✅ Model uploaded to Snowflake stage.")

# -----------------------------
# Create or replace UDF
# -----------------------------
print("🔧 Creating or replacing Python UDF...")

cursor.execute("""
CREATE OR REPLACE FUNCTION infer_model(
    sales_price FLOAT,
    quantity FLOAT,
    discount_amt FLOAT,
    net_profit FLOAT,
    d_year INT,
    d_month_seq INT,
    c_birth_year INT,
    profit_ratio FLOAT,
    is_weekend INT,
    age_group_genx INT,
    age_group_millennial INT,
    age_group_genz INT,
    d_day_name_friday INT,
    d_day_name_monday INT,
    d_day_name_saturday INT,
    d_day_name_sunday INT,
    d_day_name_thursday INT,
    d_day_name_tuesday INT,
    d_day_name_wednesday INT
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

def predict(sales_price, quantity, discount_amt, net_profit, d_year, d_month_seq, c_birth_year,
            profit_ratio, is_weekend,
            age_group_genx, age_group_millennial, age_group_genz,
            d_day_name_friday, d_day_name_monday, d_day_name_saturday, d_day_name_sunday,
            d_day_name_thursday, d_day_name_tuesday, d_day_name_wednesday):
    features = [[
        sales_price, quantity, discount_amt, net_profit,
        d_year, d_month_seq, c_birth_year, profit_ratio, is_weekend,
        age_group_genx, age_group_millennial, age_group_genz,
        d_day_name_friday, d_day_name_monday, d_day_name_saturday, d_day_name_sunday,
        d_day_name_thursday, d_day_name_tuesday, d_day_name_wednesday
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
