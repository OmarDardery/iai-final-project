import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

df = pd.read_csv("dataset.csv")
feature_cols = [f"f{i}" for i in range(99)]
X = df[feature_cols].values
y = df["label"].values

print(f"Dataset: {len(df)} rows, {len(df['label'].unique())} classes")
print(df["label"].value_counts().to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=1000,
        random_state=42,
        verbose=True,
    )),
])

print("\nTraining MLP ...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Pose Classifier — Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("confusion_matrix.png saved")

joblib.dump(pipeline, "pose_model.pkl")
print("pose_model.pkl saved")
