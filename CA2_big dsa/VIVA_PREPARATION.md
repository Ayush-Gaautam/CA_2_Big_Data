# INT315 – CLUSTER COMPUTING: CA-2 VIVA & PRESENTATION CHEAT SHEET
**Student Guide for 10-Mark Presentation and Viva Voce**

---

### Q1: What is the title and objective of your CA-2 project?
> **Answer:**  
> The project title is **"Distributed Student Dropout Prediction and Learning Behavior Analytics Using Apache Spark, RDD, Graph Analytics, and MLlib"**.  
> The objective is to build a distributed Big Data analytics pipeline using Apache Spark to process large-scale student learning interaction logs, model engagement relationships using graph analytics, and train machine learning models (both classification and clustering) to detect at-risk students who are likely to drop out.

---

### Q2: What dataset did you use and what is its authoritative source?
> **Answer:**  
> I used the **Stanford SNAP MOOC User Action Dataset (`act-mooc`)** from the **Stanford Network Analysis Project (SNAP)** at Stanford University.  
> It was published in **ACM SIGKDD 2019** by S. Kumar, X. Zhang, and J. Leskovec.  
> It captures learner actions, activity targets (lectures, quizzes, assignments), timestamps, and binary dropout labels.

---

### Q3: Which syllabus units did you use and why did you not use Units 4 & 5?
> **Answer:**  
> I strictly used **Unit 1, Unit 2, Unit 3, and Unit 6**:
> - **Unit 1:** Spark Distributed Architecture & In-Memory Computing vs Hadoop MapReduce.
> - **Unit 2:** Scala Programming (immutable collections, functions, type inference).
> - **Unit 3:** Spark RDD (Transformations, Actions, Broadcast Variables, Accumulators) & Graph Analytics (bipartite network degrees).
> - **Unit 6:** Spark MLlib (Logistic Regression, Decision Tree, LinearSVC, Naive Bayes, K-Means clustering, and evaluation metrics).
> 
> I deliberately excluded **Unit 4 (Spark SQL)** and **Unit 5 (Kafka / Spark Streaming)** per project instructions, focusing purely on core resilient distributed datasets (RDDs), graph analytics, and MLlib.

---

### Q4: Explain Unit 1: How is Apache Spark superior to Hadoop MapReduce?
> **Answer:**  
> 1. **In-Memory Computation:** Hadoop writes all intermediate map output to HDFS disks before the reduce phase. Spark caches intermediate RDDs directly in executor RAM across iterations.
> 2. **Iterative Algorithms:** Machine learning algorithms (e.g., Logistic Regression, K-Means) require repeated passes over the data. Hadoop spends 90% of time on disk I/O, whereas Spark runs in RAM, achieving **up to 100x speedup**.
> 3. **DAG Execution:** Spark uses a Directed Acyclic Graph (DAG) engine that optimizes the entire query plan before launching jobs.

---

### Q5: Explain Unit 2: Where and how did you use Scala?
> **Answer:**  
> In `scala_module/StudentAnalytics.scala`:
> - **Immutability vs Mutability:** Demonstrated immutable `List` and `Map` collections alongside mutable `ArrayBuffer`.
> - **Type Inference:** Demonstrated `val` (immutable variable) vs `var` (mutable variable).
> - **Functional Expressions:** Used higher-order functions and concise lambda syntax like `_ + _` to aggregate action counts per student using `reduceByKey`.

---

### Q6: Explain Unit 3: What is an RDD? What are Transformations vs Actions?
> **Answer:**  
> - **RDD (Resilient Distributed Dataset):** Spark's fundamental data abstraction. It is an immutable, distributed collection of objects partitioned across cluster nodes with automatic fault tolerance via a lineage graph.
> - **Transformations:** Operations that produce a new RDD from an existing one using **Lazy Evaluation** (not executed immediately).  
>   *Examples in my code:* `map()`, `filter()`, `reduceByKey()`, `distinct()`, `join()`.
> - **Actions:** Operations that trigger DAG execution and return a concrete value or write to storage.  
>   *Examples in my code:* `count()`, `first()`, `take(5)`, `collect()`.

---

### Q7: Explain Unit 3: What are Shared Variables in Spark?
> **Answer:**  
> Normally, when Spark runs a closure on remote worker nodes, it copies variables to each worker. Shared Variables provide special shared state:
> 1. **Broadcast Variables (`sc.broadcast`):** Read-only data cached on every worker machine rather than shipping a copy with every single task. In our project, used to broadcast activity metadata.
> 2. **Accumulators (`sc.accumulator`):** Variables that worker tasks can only "add" to (write-only for workers, readable by driver). In our project, used to count anomalous or corrupted records across cluster workers.

---

### Q8: Explain Unit 3: How does Graph Analytics apply to this project?
> **Answer:**  
> The interaction log naturally forms a **bipartite directed graph**:
> - **Vertices ($V$):** Students on one side, Course Activities (videos, quizzes, exams) on the other.
> - **Directed Edges ($E$):** Student $\to$ Activity interactions.
> - **Out-Degree:** Measures how many unique learning objects each student explored (engagement breadth). Low out-degree indicates disengaged learners.
> - **In-Degree:** Measures how many students accessed each module (popularity / bottleneck detection).

---

### Q9: Explain Unit 6: Which ML models did you evaluate and what were the results?
> **Answer:**  
> I trained and compared four Spark MLlib classification models on an 80/20 train-test split:
> 1. **Support Vector Machine (LinearSVC):** ~89.6% Accuracy, best overall margin separator.
> 2. **Logistic Regression:** ~89.2% Accuracy, robust probabilistic baseline.
> 3. **Decision Tree:** ~88.8% Accuracy, highest interpretability with clear decision rules.
> 4. **Naive Bayes:** ~86.7% Accuracy, fast conditional probability calculation.

---

### Q10: What is a Confusion Matrix and why is it important for dropout prediction?
> **Answer:**  
> A confusion matrix breaks down predictions into four quadrants:
> - **True Positives (TP):** Students who dropped out and were correctly detected.
> - **True Negatives (TN):** Active students correctly identified as active.
> - **False Positives (FP):** Active students incorrectly flagged as dropping out (Type I error).
> - **False Negatives (FN):** Students who dropped out but were missed by the model (Type II error).
> 
> *In dropout intervention, **Recall** ($\frac{TP}{TP + FN}$) is critical because missing an at-risk student (False Negative) means they fail or leave the course without support!*

---

### Q11: Why did you also use K-Means Clustering?
> **Answer:**  
> Classification tells us *who* will drop out, but **K-Means Clustering** (unsupervised learning) discovers *why* by grouping learners into three natural behavioral personas:
> - **Cluster 1:** High-Engagement Achievers (high actions, wide activity coverage, fast completion).
> - **Cluster 2:** Consistent Moderate Learners (steady pacing, regular activity).
> - **Cluster 3:** At-Risk Inactive Learners (low action counts, long delays between logins).
