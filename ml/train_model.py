# ✅ Updated train_model.py to support v1, v2, ... versioning matching upload_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cloudpickle, gzip, os, json
import snowflake.connector
import re

# Connect to Snowflake
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

# Generate version number (v1, v2, ...)
def get_next_version():
    cursor.execute("SELECT MAX(TRY_CAST(SUBSTRING(VERSION, 2) AS INTEGER)) FROM MODEL_HISTORY")
    result = cursor.fetchone()
    max_version = result[0] if result[0] is not None else 0
    return f"v{max_version + 1}"

model_version = get_next_version()
with open("ml/version.txt", "w") as v:
    v.write(model_version)

# Fetch training data
query = """
SELECT
    SS_SALES_PRICE,
    SS_QUANTITY,
    SS_EXT_DISCOUNT_AMT,
    SS_NET_PROFIT,
    D_YEAR,
    D_MONTH,
    D_DAY,
    S_CLOSED_DATE_SK,
    I_CATEGORY_ID
FROM ML_DB.TRAINING_DATA.STORE_CUSTOMER_SALES_SAMPLE
LIMIT 10000;
"""
df = pd.read_sql(query, conn)
conn.close()
df = df.dropna(subset=["I_CATEGORY_ID"])

X = df.drop("I_CATEGORY_ID", axis=1)
y = df["I_CATEGORY_ID"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
os.makedirs("ml", exist_ok=True)
model_path = f"ml/model_{model_version}.pkl.gz"
with gzip.open(model_path, "wb") as f:
    cloudpickle.dump(model, f)

# Save signature
signature = {
    "inputs": list(X.columns),
    "output": "I_CATEGORY_ID",
    "model_type": "RandomForestClassifier"
}
with open(f"ml/signature_{model_version}.json", "w") as f:
    json.dump(signature, f, indent=2)

# Save drift baseline
baseline_stats = df.describe().to_dict()
with open(f"ml/drift_baseline_{model_version}.json", "w") as f:
    json.dump(baseline_stats, f, indent=2)

# Save evaluation metrics
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
metrics = {
    "version": model_version,
    "accuracy": accuracy
}
with open(f"ml/metrics_{model_version}.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Model {model_version} trained and saved with metrics, signature, and baseline.")


