"""
================================================================================
  INT315: CLUSTER COMPUTING - CA2 PROJECT MASTER EXECUTION SCRIPT
  Project Title: Large-Scale MOOC Student Dropout Prediction & Learning
                 Behavior Analysis Using Apache Spark, RDD, Graph Analytics & MLlib
================================================================================

Strictly Covers:
  - Unit 1: Spark Distributed Architecture, In-Memory Computing vs MapReduce
  - Unit 2: Scala Functional Programming & Collections
  - Unit 3: Spark RDD Transformations, Actions, Shared Variables & Graph Analytics
  - Unit 6: Spark MLlib (Logistic Regression, Decision Tree, SVM, Naive Bayes, K-Means,
            Confusion Matrix, R2, RMSE, MAE, Correlation Heatmap)
  - EXCLUDES: Unit 4 (Spark SQL) and Unit 5 (Kafka / Spark Streaming)

Run this file with:
  python run_project.py
================================================================================
"""

import os
import sys
import time

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def main():
    start_time = time.time()
    print("\n" + "=" * 80)
    print("  INT315: CLUSTER COMPUTING - CONTINUOUS ASSESSMENT 2 (CA-2) PROJECT")
    print("  LOVELY PROFESSIONAL UNIVERSITY")
    print("=" * 80)
    print("  * Student Name    : Ayush Gautam")
    print("  * Course Code     : INT315 (Cluster Computing)")
    print("  * Units Covered   : Unit 1, Unit 2, Unit 3, Unit 6 (Strictly 1, 2, 3, 6)")
    print("  * Authoritative DB: Stanford SNAP MOOC User Action Dataset (act-mooc)")
    print("=" * 80)

    # STEP 1: Verify / Generate Dataset
    print("\n[STEP 1/6] CHECKING DATASET (Stanford SNAP Schema)...")
    from data.download_or_generate_data import generate_representative_dataset, ACTIONS_FILE, STUDENT_FEATURES_FILE
    if not os.path.exists(ACTIONS_FILE) or not os.path.exists(STUDENT_FEATURES_FILE):
        print("  -> Dataset files not found. Generating now...")
        generate_representative_dataset()
    else:
        print("  [OK] Dataset verified in data/ directory.")

    # STEP 2: Unit 1 Architectural Overview & In-Memory Benchmark
    print("\n[STEP 2/6] RUNNING UNIT I: SPARK ARCHITECTURE & BENCHMARK...")
    from src.unit1_spark_overview import display_unit1_conceptual_overview, benchmark_in_memory_vs_disk_simulation
    display_unit1_conceptual_overview()
    benchmark_in_memory_vs_disk_simulation(n_records=150000, iterations=3)

    # STEP 3: Unit 3 RDD Operations & Distributed Feature Extraction
    print("\n[STEP 3/6] RUNNING UNIT III: DISTRIBUTED RDD PROCESSING...")
    from src.unit3_rdd_processing import run_rdd_processing
    rdd_results = run_rdd_processing(ACTIONS_FILE)

    # STEP 4: Unit 3 Graph Analytics & Network Operations
    print("\n[STEP 4/6] RUNNING UNIT III: GRAPH ANALYTICS...")
    from src.unit3_graph_analytics import analyze_student_interaction_graph
    graph_results = analyze_student_interaction_graph(ACTIONS_FILE)

    # STEP 5: Unit 6 MLlib Classification & K-Means Clustering
    print("\n[STEP 5/6] RUNNING UNIT VI: SPARK MLLIB MACHINE LEARNING...")
    from src.unit6_mllib_classification import run_mllib_classification
    ml_results = run_mllib_classification(STUDENT_FEATURES_FILE)

    from src.unit6_mllib_clustering import run_mllib_clustering
    cluster_results = run_mllib_clustering(STUDENT_FEATURES_FILE, k=3)

    # STEP 6: Visual Dashboard Chart Generation
    print("\n[STEP 6/6] GENERATING VISUAL PROJECT CHARTS...")
    from visualizer.dashboard import generate_all_visualizations
    generate_all_visualizations(STUDENT_FEATURES_FILE)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("  [SUCCESS] CA-2 BIG DATA PROJECT PIPELINE COMPLETED IN ONE GO!")
    print(f"  Total Execution Time: {elapsed:.2f} seconds")
    print("  All visual charts saved to: 'output/' folder")
    print("  Report documentation: 'REPORT_CA2.md'")
    print("  Viva cheat sheet    : 'VIVA_PREPARATION.md'")
    print("  Scala source code   : 'scala_module/StudentAnalytics.scala'")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
