import os
import json
import gzip
import cloudpickle
import snowflake.connector

# Load metrics from saved file
with open("ml/metrics.json") as f:
    metrics = json.load(f)

# Accuracy and F1 thresholds
MIN_ACCURACY = 0.85
MIN_F1 = 0.80

# Required artifacts
required_files = [
    "ml/model.pkl.gz",
    "ml/metrics.json",
    "ml/signature.json",
    "ml/drift_baseline.json",
    "ml/confusion_matrix.png"
]

# File size threshold (100MB)
MAX_MODEL_SIZE_BYTES = 100 * 1024 * 1024

def test_accuracy_threshold():
    assert metrics["accuracy"] >= MIN_ACCURACY, f"❌ Accuracy too low: {metrics['accuracy']}"
    assert metrics["f1_score"] >= MIN_F1, f"❌ F1 score too low: {metrics['f1_score']}"

def test_champion_comparison():
    conn = snowflake.connector.connect(
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
        database=os.environ["SNOWFLAKE_DATABASE"],
        schema=os.environ["SNOWFLAKE_SCHEMA"]
    )
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(ACCURACY) FROM MODEL_HISTORY WHERE IS_CHAMPION = TRUE")
    result = cursor.fetchone()
    if result and result[0] is not None:
        prev_acc = result[0]
        assert metrics["accuracy"] > prev_acc, (
            f"❌ Accuracy {metrics['accuracy']} not better than current champion {prev_acc}"
        )
    cursor.close()
    conn.close()

def test_model_file_exists_and_loadable():
    assert os.path.exists("ml/model.pkl.gz"), "❌ model.pkl.gz not found"
    try:
        with gzip.open("ml/model.pkl.gz", "rb") as f:
            _ = cloudpickle.load(f)
    except Exception as e:
        raise AssertionError(f"❌ Failed to load model file: {e}")

def test_required_artifacts_exist():
    for file in required_files:
        assert os.path.exists(file), f"❌ Required artifact missing: {file}"

def test_model_file_size():
    size = os.path.getsize("ml/model.pkl.gz")
    assert size < MAX_MODEL_SIZE_BYTES, f"❌ Model file too large: {size} bytes"

# Run all tests
if __name__ == "__main__":
    test_accuracy_threshold()
    test_champion_comparison()
    test_model_file_exists_and_loadable()
    test_required_artifacts_exist()
    test_model_file_size()
    print("✅ All CI tests passed successfully.")
