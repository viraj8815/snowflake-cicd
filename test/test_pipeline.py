import os
import re
import json
import gzip
import cloudpickle
import snowflake.connector
from pathlib import Path

# Constants
MIN_ACCURACY = 0.85
MIN_F1 = 0.80
MAX_MODEL_SIZE_BYTES = 100 * 1024 * 1024

# Ensure local ml/ directory exists
Path("ml").mkdir(exist_ok=True)

# Connect to Snowflake
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

# Step 1: Get list of all files from stage
cursor.execute("LIST @ml_models_stage")
files = cursor.fetchall()
filenames = [row[0] for row in files]

# Step 2: Determine latest versioned files
def get_latest_versioned_file(prefix, extension=".json"):
    versioned = [
        (f, int(re.search(rf"{prefix}_v(\d+)", f).group(1)))
        for f in filenames if re.search(rf"{prefix}_v(\d+)", f) and f.endswith(extension)
    ]
    return max(versioned, key=lambda x: x[1])[0] if versioned else None

metrics_file = get_latest_versioned_file("metrics")
drift_file = get_latest_versioned_file("drift_baseline")
signature_file = get_latest_versioned_file("signature")
model_file = get_latest_versioned_file("model", extension=".pkl.gz")

# Step 3: Download required files
def download_file(file_path):
    if file_path:
        relative_path = file_path.split("/")[-1]  # removes any prefix
        cursor.execute(f"GET @ml_models_stage/{relative_path} file://ml/")
        print(f"✅ Downloaded: {relative_path}")
    else:
        raise FileNotFoundError(f"❌ No versioned file found for {file_path}")

download_file(metrics_file)
download_file(drift_file)
download_file(signature_file)
download_file(model_file)

# Step 4: Load metrics
metrics_local = Path("ml") / Path(metrics_file).name
with open(metrics_local) as f:
    metrics = json.load(f)

# -------------------------------
# CI TESTS
# -------------------------------
def test_accuracy_threshold():
    assert "accuracy" in metrics, "❌ 'accuracy' key missing in metrics.json"
    assert "f1_score" in metrics, "❌ 'f1_score' key missing in metrics.json"
    
    assert metrics["accuracy"] >= MIN_ACCURACY, f"❌ Accuracy too low: {metrics['accuracy']}"
    assert metrics["f1_score"] >= MIN_F1, f"❌ F1 score too low: {metrics['f1_score']}"

def test_champion_comparison():
    cursor.execute("SELECT MAX(ACCURACY) FROM MODEL_HISTORY WHERE IS_CHAMPION = TRUE")
    result = cursor.fetchone()
    if result and result[0] is not None:
        prev_acc = result[0]
        assert metrics["accuracy"] >= prev_acc, (
            f"❌ Accuracy {metrics['accuracy']} not better than or equal to current champion {prev_acc}"
        )

def test_model_file_exists_and_loadable():
    path = f"ml/{Path(model_file).name}"
    assert os.path.exists(path), "❌ Model file not found"
    try:
        with gzip.open(path, "rb") as f:
            _ = cloudpickle.load(f)
    except Exception as e:
        raise AssertionError(f"❌ Failed to load model file: {e}")

def test_required_artifacts_exist():
    required = [
        Path("ml") / Path(metrics_file).name,
        Path("ml") / Path(drift_file).name,
        Path("ml") / Path(signature_file).name,
        Path("ml") / Path(model_file).name,
    ]
    for file in required:
        assert os.path.exists(file), f"❌ Required artifact missing: {file}"

def test_model_file_size():
    path = f"ml/{Path(model_file).name}"
    size = os.path.getsize(path)
    assert size < MAX_MODEL_SIZE_BYTES, f"❌ Model file too large: {size} bytes"

# Run all tests
if __name__ == "__main__":
    test_accuracy_threshold()
    test_champion_comparison()
    test_model_file_exists_and_loadable()
    test_required_artifacts_exist()
    test_model_file_size()
    print("✅ All CI tests passed successfully.")
