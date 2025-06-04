import os
import zipfile
import shutil
from pathlib import Path

# Replace with your actual paths
ZIP_PATH = r"C:\Users\HP\Downloads\mlflow-logs.zip"
DEST_DIR = r"C:\Users\HP\OneDrive\Documents\LOCAL\snowflake-cicd\mlruns"

# ✅ 1. Remove old MLflow logs (Optional but recommended)
for sub in Path(DEST_DIR).iterdir():
    if sub.is_dir():
        print(f"🗑️ Removing: {sub}")
        shutil.rmtree(sub)

# ✅ 2. Extract new MLflow logs
print(f"📦 Extracting {ZIP_PATH} to {DEST_DIR}")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DEST_DIR)

# ✅ 3. Fix all artifact_uri paths inside meta.yaml
print("🔧 Fixing artifact_uri paths...")
for meta_file in Path(DEST_DIR).rglob("meta.yaml"):
    meta_path = meta_file.resolve()
    run_dir = meta_path.parent
    artifacts_dir = run_dir / "artifacts"
    new_uri = artifacts_dir.resolve().as_uri()

    new_lines = []
    with open(meta_file, "r") as f:
        for line in f:
            if line.strip().startswith("artifact_uri:"):
                new_lines.append(f"artifact_uri: {new_uri}\n")
            else:
                new_lines.append(line)

    with open(meta_file, "w") as f:
        f.writelines(new_lines)

print("✅ Artifacts extracted and URIs fixed.")
