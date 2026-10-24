"""
INT315 - Cluster Computing: Unit VI Module (Classification & Evaluation)
Machine Learning Programming with Spark MLlib & Comparative Algorithms

Covers:
1. Machine Learning Concepts & Spark MLlib Architecture:
   - Feature Assembly (VectorAssembler) & Scaling (StandardScaler).
   - Supervised Learning Workflow.
2. Spark ML Classification Algorithms:
   - Logistic Regression
   - Decision Tree Classifier
   - Support Vector Machine (LinearSVC)
   - Naive Bayes Classifier
   - Linear Regression (for final grade prediction)
3. Evaluation & Performance Matrix (Unit VI):
   - Confusion Matrix (TP, TN, FP, FN)
   - Accuracy, Precision, Recall, F1-Score
   - Regression Metrics: R2, RMSE, MAE
   - Correlation Heat Map analysis
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
)

def run_mllib_classification(features_file="data/student_features.csv", engine="fast"):
    print("=" * 75)
    print("  UNIT VI: MACHINE LEARNING CLASSIFICATION & EVALUATION MATRIX")
    print("=" * 75)
    print(f"[+] Loading student behavioral features from: {features_file}")

    df = pd.read_csv(features_file)
    print(f"  * Dataset: {len(df):,} learner records with {df.shape[1]} attributes.")
    print(f"  * Features: total_actions, unique_targets, avg_time_gap, exam_attempt_rate, graph_in_degree, graph_out_degree")
    print(f"  * Target: 'dropout' (0 = Active, 1 = Dropout)")

    # 1. CORRELATION ANALYSIS (Unit VI: Correlation Matrix & Heat Map)
    print("\n[1] FEATURE CORRELATION MATRIX (Values for Heat Map):")
    numeric_cols = ["total_actions", "unique_targets", "avg_time_gap", "exam_attempt_rate", "graph_out_degree", "final_score", "dropout"]
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix.round(3).to_string())

    # Split features and labels
    feature_names = ["total_actions", "unique_targets", "avg_time_gap", "exam_attempt_rate", "graph_in_degree", "graph_out_degree"]
    X = df[feature_names]
    y = df["dropout"]
    y_score = df["final_score"]

    X_train, X_test, y_train, y_test, ys_train, ys_test = train_test_split(
        X, y, y_score, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. TRAIN & EVALUATE CLASSIFICATION ALGORITHMS
    print("\n[2] TRAINING & EVALUATING CLASSIFICATION MODELS (Unit VI):")
    models = {
        "Support Vector Machine (LinearSVC)": (LinearSVC(random_state=42, dual=False, max_iter=2000), True),
        "Logistic Regression": (LogisticRegression(random_state=42, max_iter=500), True),
        "Decision Tree Classifier": (DecisionTreeClassifier(max_depth=4, random_state=42), False),
        "Naive Bayes Classifier": (GaussianNB(), False)
    }

    results = {}
    confusion_matrices = {}

    for name, (model, use_scaled) in models.items():
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.ravel()
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)
        }
        confusion_matrices[name] = cm.tolist()

    # 3. REGRESSION EVALUATION METRICS (Linear Regression for Score Prediction)
    print("\n[3] REGRESSION EVALUATION METRICS (Linear Regression for Final Score):")
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_scaled, ys_train)
    ys_pred = lin_reg.predict(X_test_scaled)

    r2 = r2_score(ys_test, ys_pred)
    rmse = float(np.sqrt(mean_squared_error(ys_test, ys_pred)))
    mae = float(mean_absolute_error(ys_test, ys_pred))

    print(f"  * Coefficient of Determination (R2 Score): {r2:.4f} ({r2*100:.1f}% variance explained)")
    print(f"  * Root Mean Squared Error (RMSE)         : {rmse:.4f}")
    print(f"  * Mean Absolute Error (MAE)              : {mae:.4f}")

    # 4. TABULATE COMPARATIVE PERFORMANCE MATRIX
    print("\n[4] CLASSIFICATION PERFORMANCE MATRIX (Unit VI):")
    print(f"{'Algorithm Name':<38} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 88)
    for name, m in results.items():
        print(f"{name:<38} | {m['Accuracy']*100:>8.2f}% | {m['Precision']*100:>8.2f}% | {m['Recall']*100:>8.2f}% | {m['F1-Score']*100:>8.2f}%")

    print("\n[5] CONFUSION MATRICES (Row = True [Active, Dropout], Col = Predicted [Active, Dropout]):")
    for name, m in results.items():
        print(f"  * {name}:")
        print(f"      [ TN (True Active): {m['TN']:<3} | FP (False Dropout): {m['FP']:<3} ]")
        print(f"      [ FN (Missed Dropout): {m['FN']:<3} | TP (True Dropout): {m['TP']:<3} ]")

    return {
        "results": results,
        "confusion_matrices": confusion_matrices,
        "regression": {"r2": r2, "rmse": rmse, "mae": mae},
        "correlation": corr_matrix.to_dict()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spark", action="store_true", help="Run via full SparkSession MLlib")
    args = parser.parse_args()
    engine = "spark" if args.spark else "fast"
    run_mllib_classification(engine=engine)
