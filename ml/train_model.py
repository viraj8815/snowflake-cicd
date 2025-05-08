# ml/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import gzip
import cloudpickle
import os
import snowflake.connector
import joblib

# Connect to Snowflake to fetch data
conn = snowflake.connector.connect(
    user=os.environ["SNOWFLAKE_USER"],
    password=os.environ["SNOWFLAKE_PASSWORD"],
    account=os.environ["SNOWFLAKE_ACCOUNT"],
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
    database=os.environ["SNOWFLAKE_DATABASE"],
    schema=os.environ["SNOWFLAKE_SCHEMA"],
    role=os.environ["SNOWFLAKE_ROLE"]
)

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

# ✅ Drop rows where target is null
df = df.dropna(subset=["I_CATEGORY_ID"])

# Train model
X = df.drop("I_CATEGORY_ID", axis=1)
y = df["I_CATEGORY_ID"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
os.makedirs("ml", exist_ok=True)
with gzip.open("ml/model.pkl.gz", "wb") as f:
    joblib.dump(model, f)

print("✅ Model trained and saved to ml/model.pkl.gz")
