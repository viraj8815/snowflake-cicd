import os
import json
import snowflake.connector
import cloudpickle
import gzip
import shutil

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
# Read accuracy from metrics.json
# -----------------------------
with open("ml/metrics.json") as f:
    metrics = json.load(f)
accuracy = float(metrics.get("accuracy", 0.0))

# -----------------------------
# Create MODEL_HISTORY if not exists
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
# Rename artifacts to versioned names
# -----------------------------
versioned_model = f"ml/model_v{version}.pkl.gz"
versioned_metrics = f"ml/metrics_v{version}.json"
versioned_signature = f"ml/signature_v{version}.json"
versioned_drift = f"ml/drift_baseline_v{version}.json"

shutil.copyfile("ml/model.pkl.gz", versioned_model)
shutil.copyfile("ml/metrics.json", versioned_metrics)
shutil.copyfile("ml/signature.json", versioned_signature)
shutil.copyfile("ml/drift_baseline.json", versioned_drift)

print(f"✅ Created versioned artifacts for version {version}")

# -----------------------------
# Upload all artifacts
# -----------------------------
print(f"📦 Uploading artifacts for version {version}...")
for file in [versioned_model, versioned_metrics, versioned_signature, versioned_drift]:
    cursor.execute(f"PUT file://{file} @ml_models_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE")
    print(f"✅ Uploaded {os.path.basename(file)}")

# -----------------------------
# Insert into MODEL_HISTORY
# -----------------------------
cursor.execute(f"""
    INSERT INTO STAGE_DB.PUBLIC.MODEL_HISTORY 
    (version, accuracy, deployed_on, is_champion)
    VALUES ('{version}', {accuracy}, CURRENT_TIMESTAMP(), FALSE)
""")

# -----------------------------
# Champion logic
# -----------------------------
cursor.execute("SELECT MAX(accuracy) FROM STAGE_DB.PUBLIC.MODEL_HISTORY")
best_accuracy = cursor.fetchone()[0]

if accuracy >= best_accuracy:
    print("🏆 This is the best model so far. Updating model.pkl.gz for UDF...")
    cursor.execute("UPDATE STAGE_DB.PUBLIC.MODEL_HISTORY SET is_champion = FALSE")
    cursor.execute(f"""
        UPDATE STAGE_DB.PUBLIC.MODEL_HISTORY 
        SET is_champion = TRUE 
        WHERE version = '{version}'
    """)
    # Copy current versioned model to model.pkl.gz for UDF
    shutil.copyfile(versioned_model, "ml/model.pkl.gz")
    cursor.execute("PUT file://ml/model.pkl.gz @ml_models_stage AUTO_COMPRESS=FALSE OVERWRITE=TRUE")

    # -----------------------------
    # Create or Replace UDF
    # -----------------------------
    print("🔧 Creating or replacing Python UDF...")

    cursor.execute("""
CREATE OR REPLACE FUNCTION infer_model(
    CD_DEP_COUNT INT,
    CD_DEP_EMPLOYED_COUNT INT,
    CD_DEP_COLLEGE_COUNT INT,
    AGE INT,
    IS_WEEKEND INT,
    CD_GENDER_F INT,
    CD_GENDER_M INT,
    CD_MARITAL_STATUS_D INT,
    CD_MARITAL_STATUS_M INT,
    CD_MARITAL_STATUS_S INT,
    CD_MARITAL_STATUS_U INT,
    CD_MARITAL_STATUS_W INT,
    CD_EDUCATION_STATUS_2_yr_Degree INT,
    CD_EDUCATION_STATUS_4_yr_Degree INT,
    CD_EDUCATION_STATUS_Advanced_Degree INT,
    CD_EDUCATION_STATUS_College INT,
    CD_EDUCATION_STATUS_Primary INT,
    CD_EDUCATION_STATUS_Secondary INT,
    CD_EDUCATION_STATUS_Unknown INT,
    CD_CREDIT_RATING_Good INT,
    CD_CREDIT_RATING_High_Risk INT,
    CD_CREDIT_RATING_Low_Risk INT,
    CD_CREDIT_RATING_Unknown INT
)
RETURNS STRING
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

def predict(*args):
    return model.predict([list(args)])[0]
$$;


    """)
    print("✅ UDF deployed with champion model.")
else:
    print("ℹ️ Model not deployed as it doesn't beat the champion accuracy.")

# -----------------------------
# Cleanup
# -----------------------------
cursor.close()
conn.close()
