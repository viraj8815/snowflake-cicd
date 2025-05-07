# ml/train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# ✅ Step 1: Load sample data (Pima Indians dataset)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
data = pd.read_csv(url, header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ✅ Step 2: Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ✅ Step 3: Save model
os.makedirs("ml", exist_ok=True)  # Make sure the ml/ directory exists
with open("ml/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to ml/model.pkl")
