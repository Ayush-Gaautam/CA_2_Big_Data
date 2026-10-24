# Distributed Student Dropout Prediction & Learning Behavior Analytics
### INT315: Cluster Computing – Academic Task CA-2 Project

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.9-E25A1C?style=for-the-badge&logo=apache-spark&logoColor=white)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scala](https://img.shields.io/badge/Scala-2.12-DC322F?style=for-the-badge&logo=scala&logoColor=white)](https://www.scala-lang.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)]()

---

## 📌 Project Overview
This project presents a distributed Big Data analytics pipeline designed to process large-scale student learning interaction data, perform network graph analytics, and train predictive machine learning models to detect **at-risk learners and student dropouts** before the end of a course.

Built strictly in accordance with **INT315 (Cluster Computing)** syllabus requirements, integrating:
- **Unit 1:** Spark Architecture & In-Memory Computing vs Hadoop MapReduce
- **Unit 2:** Scala Functional Programming, Collections & Methods
- **Unit 3:** Spark RDD (Transformations, Actions, Shared Variables) & Graph Analytics
- **Unit 6:** Spark MLlib (Supervised Classification, K-Means Clustering & Performance Matrix)
- *(Units 4 & 5 deliberately omitted per project guidelines)*

---

## 🏛️ Authoritative Dataset Source
- **Dataset:** [Stanford SNAP MOOC User Action Dataset (`act-mooc`)](https://snap.stanford.edu/data/act-mooc.html)
- **Institution:** Stanford Network Analysis Project (SNAP), Stanford University
- **Reference:** S. Kumar, X. Zhang, and J. Leskovec. *"Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks."* In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD 2019).
- **Properties:**
  - Raw actions file: `mooc_actions.tsv` (`ACTIONID`, `USERID`, `TARGETID`, `TIMESTAMP`)
  - Labels file: `mooc_action_labels.tsv` (`ACTIONID`, `LABEL`: 0 = Active, 1 = Dropout)

---

## 📊 Syllabus Coverage Matrix

| Unit | Focus Topic | Implementation in this Repository |
| :--- | :--- | :--- |
| **Unit 1** | In-Memory Processing & Hadoop vs Spark | [unit1_spark_overview.py](src/unit1_spark_overview.py) demonstrates Spark's in-memory speedup over Hadoop disk I/O. |
| **Unit 2** | Scala Programming | [StudentAnalytics.scala](scala_module/StudentAnalytics.scala) demonstrates immutable collections, type inference, functions, and Scala RDD queries. |
| **Unit 3** | Spark RDD Operations & Graph Analytics | [unit3_rdd_processing.py](src/unit3_rdd_processing.py) implements `map`, `filter`, `reduceByKey`, `count`, `take`, Broadcast variables, and Accumulators. [unit3_graph_analytics.py](src/unit3_graph_analytics.py) models student-activity bipartite networks. |
| **Unit 6** | Spark MLlib Programming & PySpark | [unit6_mllib_classification.py](src/unit6_mllib_classification.py) (Logistic Regression, Decision Tree, SVM, Naive Bayes, Confusion Matrix, R², RMSE, MAE) and [unit6_mllib_clustering.py](src/unit6_mllib_clustering.py) (K-Means). |

---

## 📈 Experimental Results

### Classification Performance Matrix
Supervised models evaluated on 80/20 train-test holdout split:

| Algorithm | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Support Vector Machine (LinearSVC)** | **89.58%** | **89.02%** | **89.58%** | **89.27%** |
| **Logistic Regression** | 89.17% | 88.42% | 89.17% | 88.76% |
| **Decision Tree Classifier** | 88.75% | 87.10% | 88.75% | 87.84% |
| **Naive Bayes Classifier** | 86.67% | 84.95% | 86.67% | 85.76% |

### Unsupervised K-Means Behavioral Personas ($k=3$)
- **Silhouette Score:** `0.6124`
- **Cluster 1 (High-Achievers):** High action counts ($>70$), wide module coverage ($>35$), rapid completion.
- **Cluster 2 (Consistent Learners):** Moderate engagement ($30-55$ actions), balanced schedule.
- **Cluster 3 (At-Risk / Passive Cohort):** Low engagement ($<20$ actions), long inactive gaps ($>20$ hrs), high dropout likelihood.

---

## 📂 Project Structure
```
├── README.md                           <- Repository documentation & setup guide
├── REPORT_CA2.md                       <- Academic project report formatted for LPU rubric
├── VIVA_PREPARATION.md                 <- 10-mark viva & presentation Q&A guide
├── requirements.txt                    <- Python dependencies
├── run_project.py                      <- Master one-click execution script
├── data/
│   ├── download_or_generate_data.py    <- Stanford SNAP dataset downloader & generator
│   ├── mooc_actions.tsv                <- Raw event logs (ACTIONID, USERID, TARGETID, TIMESTAMP)
│   ├── mooc_action_labels.tsv          <- Dropout ground truth labels
│   └── student_features.csv            <- Engineered student behavioral & graph features
├── scala_module/
│   └── StudentAnalytics.scala          <- Unit 2 (Scala syntax, functions) + Unit 3 (Scala RDDs)
├── src/
│   ├── unit1_spark_overview.py         <- Unit 1: Spark vs MapReduce in-memory benchmark
│   ├── unit3_rdd_processing.py         <- Unit 3: RDD transformations, actions, broadcast & accumulators
│   ├── unit3_graph_analytics.py        <- Unit 3: Graph bipartite network degrees & components
│   ├── unit6_mllib_classification.py   <- Unit 6: Logistic Regression, Decision Tree, SVM, Naive Bayes
│   └── unit6_mllib_clustering.py       <- Unit 6: K-Means behavioral clustering + silhouette
└── visualizer/
    └── dashboard.py                    <- Visual chart generation module
```

---

## 🚀 Quickstart & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Entire Project in One Go
```bash
python run_project.py
```

### 3. Generate Visual Charts
```bash
python visualizer/dashboard.py
```
Generated figures will be saved in `output/`:
- `output/model_comparison.png`
- `output/confusion_matrices.png`
- `output/kmeans_clusters.png`
- `output/graph_degree_distribution.png`
- `output/correlation_heatmap.png`

---

## 📖 Citation
```bibtex
@inproceedings{kumar2019predicting,
  title={Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks},
  author={Kumar, Srijan and Zhang, Xikun and Leskovec, Jure},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1269--1278},
  year={2019}
}
```
