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
        age FLOAT,
        is_weekend INT,
        CD_EDUCATION_STATUS_college INT,
        CD_EDUCATION_STATUS_primary INT,
        CD_EDUCATION_STATUS_secondary INT,
        CD_EDUCATION_STATUS_unknown INT,
        CD_CREDIT_RATING_high INT,
        CD_CREDIT_RATING_low INT,
        CD_CREDIT_RATING_medium INT
    )
    RETURNS STRING
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.10'
    HANDLER = 'predict'
    PACKAGES = ('scikit-learn', 'cloudpickle', 'numpy', 'pandas')
    IMPORTS = ('@ml_models_stage/model.pkl.gz')
    AS
    $$
    import cloudpickle, gzip, os, sys
    model_path = os.path.join(sys._xoptions["snowflake_import_directory"], "model.pkl.gz")
    with gzip.open(model_path, "rb") as f:
        model = cloudpickle.load(f)

    def predict(*args):
        features = [list(args)]
        return str(model.predict(features)[0])
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
