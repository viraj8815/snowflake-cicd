# ✅ Finalized train_model.py (always save as base files; versioning handled in upload_model.py)

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import cloudpickle, gzip, os, json
import snowflake.connector

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

# Save base artifacts (no version yet)
os.makedirs("ml", exist_ok=True)
with gzip.open("ml/model.pkl.gz", "wb") as f:
    cloudpickle.dump(model, f)

signature = {
    "inputs": list(X.columns),
    "output": "I_CATEGORY_ID",
    "model_type": "RandomForestClassifier"
}
with open("ml/signature.json", "w") as f:
    json.dump(signature, f, indent=2)

baseline_stats = df.describe().to_dict()
with open("ml/drift_baseline.json", "w") as f:
    json.dump(baseline_stats, f, indent=2)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
metrics = {
    "accuracy": accuracy
}
with open("ml/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Model trained and saved with base metrics, signature, and baseline.")


