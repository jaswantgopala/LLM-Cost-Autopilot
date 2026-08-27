"""
Phase 2, Step 3: Train a classifier to predict complexity tier from prompt features.
"""

import sys
import os
import csv
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from core.complexity import extract_features, features_to_vector, FEATURE_NAMES

DATA_PATH = "data/labeled_prompts.csv"
MODEL_OUT_PATH = "data/complexity_classifier.pkl"


def load_dataset(path: str):
    prompts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row["prompt"])
            labels.append(int(row["tier"]))
    return prompts, labels


def build_feature_matrix(prompts):
    X = []
    for p in prompts:
        features = extract_features(p)
        X.append(features_to_vector(features))
    return np.array(X, dtype=float)


def main():
    print("Loading dataset...")
    prompts, labels = load_dataset(DATA_PATH)
    print(f"Loaded {len(prompts)} labeled examples")

    print("Extracting features...")
    X = build_feature_matrix(prompts)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n" + "=" * 60)
    print("Model 1: Logistic Regression")
    print("=" * 60)
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_scaled, y_train)
    logreg_preds = logreg.predict(X_test_scaled)
    logreg_acc = accuracy_score(y_test, logreg_preds)
    print(f"Accuracy: {logreg_acc:.2%}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, logreg_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, logreg_preds, target_names=["Tier1", "Tier2", "Tier3"]))

    print("\n" + "=" * 60)
    print("Model 2: Random Forest")
    print("=" * 60)
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)  # tree models don't need scaling
    rf_preds = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_preds)
    print(f"Accuracy: {rf_acc:.2%}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, rf_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, rf_preds, target_names=["Tier1", "Tier2", "Tier3"]))

    print("\nFeature Importances (Random Forest):")
    for name, importance in sorted(zip(FEATURE_NAMES, rf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:30s} {importance:.4f}")

    # Pick the better model to save
    if rf_acc >= logreg_acc:
        best_model, best_name, best_acc = rf, "random_forest", rf_acc
        best_scaler = None  # RF doesn't need scaling
    else:
        best_model, best_name, best_acc = logreg, "logistic_regression", logreg_acc
        best_scaler = scaler

    print(f"\n{'='*60}")
    print(f"Best model: {best_name} ({best_acc:.2%} accuracy)")
    print(f"{'='*60}")

    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    with open(MODEL_OUT_PATH, "wb") as f:
        pickle.dump({
            "model": best_model,
            "model_type": best_name,
            "scaler": best_scaler,
            "feature_names": FEATURE_NAMES,
            "accuracy": best_acc,
        }, f)
    print(f"Saved best model to {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()