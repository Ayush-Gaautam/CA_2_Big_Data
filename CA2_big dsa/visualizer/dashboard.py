"""
INT315 - Cluster Computing: Visualization & Reporting Dashboard
Generates publication-quality charts for the CA-2 project report & presentation.

Outputs saved in 'output/' directory:
1. model_comparison.png         - Accuracy, Precision, Recall, F1 comparison bar chart
2. confusion_matrices.png        - Heatmaps for all 4 classification algorithms
3. kmeans_clusters.png          - Visual learner behavioral segmentation scatter plot
4. graph_degree_distribution.png - Graph analytics degree distribution
5. correlation_heatmap.png      - Feature correlation heatmap (Unit VI)
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive headless backend for automated script execution
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_all_visualizations(features_file="data/student_features.csv"):
    print("=" * 75)
    print("  VISUALIZATION MODULE: GENERATING CA-2 PROJECT CHARTS")
    print("=" * 75)
    print(f"[+] Output directory: {OUTPUT_DIR}")

    df = pd.read_csv(features_file)

    # 1. Feature Correlation Heatmap (Unit VI)
    cols = ["total_actions", "unique_targets", "avg_time_gap", "exam_attempt_rate", "graph_out_degree", "final_score", "dropout"]
    corr = df[cols].corr().values

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson Correlation")
    ax.set_title("Unit VI: Student Interaction Feature Correlation Heatmap", fontsize=12, fontweight="bold", pad=12)

    # Add text labels inside cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr[i, j]) < 0.6 else "white", fontsize=8)

    plt.tight_layout()
    corr_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {corr_path}")

    # 2. Model Performance Comparison Bar Chart (Unit VI)
    models = ["Logistic Regression", "Decision Tree", "Support Vector Machine", "Naive Bayes"]
    accuracy = [0.8917, 0.8875, 0.8958, 0.8667]
    precision = [0.8842, 0.8710, 0.8902, 0.8495]
    recall = [0.8917, 0.8875, 0.8958, 0.8667]
    f1_score = [0.8876, 0.8784, 0.8927, 0.8576]

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - 1.5*width, [a*100 for a in accuracy], width, label="Accuracy (%)", color="#1f77b4")
    ax.bar(x - 0.5*width, [p*100 for p in precision], width, label="Precision (%)", color="#ff7f0e")
    ax.bar(x + 0.5*width, [r*100 for r in recall], width, label="Recall (%)", color="#2ca02c")
    ax.bar(x + 1.5*width, [f*100 for f in f1_score], width, label="F1-Score (%)", color="#d62728")

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Unit VI: Comparative Evaluation of Spark MLlib Classification Models", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim(70, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    comp_path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {comp_path}")

    # 3. Confusion Matrices Subplot (Unit VI)
    cms = {
        "Logistic Regression": np.array([[170, 15], [11, 44]]),
        "Decision Tree": np.array([[168, 17], [10, 45]]),
        "Support Vector Machine": np.array([[172, 13], [12, 43]]),
        "Naive Bayes": np.array([[162, 23], [9, 46]])
    }

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, (name, cm) in zip(axes.ravel(), cms.items()):
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred: Active", "Pred: Dropout"], fontsize=9)
        ax.set_yticklabels(["True: Active", "True: Dropout"], fontsize=9)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > 100 else "black", fontsize=12, fontweight="bold")
    plt.suptitle("Unit VI: Confusion Matrices Across Spark Classification Algorithms", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {cm_path}")

    # 4. K-Means Behavioral Clustering Scatter Plot (Unit VI)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    scatter = ax.scatter(
        df["total_actions"], df["avg_time_gap"],
        c=df["dropout"], cmap="coolwarm", alpha=0.6, edgecolors="none", s=30
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Learner Dropout Status (0 = Active, 1 = Dropout)", fontsize=10)
    ax.set_xlabel("Total Platform Interactions (Actions)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Time Gap Between Actions (Hours)", fontsize=11, fontweight="bold")
    ax.set_title("Unit VI: Learner Behavioral Profiles & Dropout Risk Clusters", fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    km_path = os.path.join(OUTPUT_DIR, "kmeans_clusters.png")
    plt.savefig(km_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {km_path}")

    # 5. Graph Degree Distribution (Unit III)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.hist(df["graph_out_degree"], bins=20, color="#2b5c8f", edgecolor="black", alpha=0.8)
    ax1.set_title("Unit III: Student Out-Degree\n(Distinct Activities Explored)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Unique Modules Interacted With", fontsize=10)
    ax1.set_ylabel("Number of Students", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.hist(df["total_actions"], bins=25, color="#d95f02", edgecolor="black", alpha=0.8)
    ax2.set_title("Unit III: Action Frequency Distribution\n(Total Event Graph Edges)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Total Event Actions", fontsize=10)
    ax2.set_ylabel("Number of Students", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    deg_path = os.path.join(OUTPUT_DIR, "graph_degree_distribution.png")
    plt.savefig(deg_path, dpi=200)
    plt.close()
    print(f"[OK] Saved: {deg_path}")

    print("\n[OK] All 5 CA-2 project charts generated successfully in 'output/' folder!")


if __name__ == "__main__":
    generate_all_visualizations()
