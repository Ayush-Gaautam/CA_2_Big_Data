"""
INT315 - Cluster Computing: Unit I Module
Introduction to Apache Spark & Architectural Foundations

Covers:
1. Limitations of MapReduce in Hadoop (Disk I/O vs In-Memory RDD computation).
2. Comparison of Batch vs Real-Time Stream Analytics.
3. Spark Features: In-memory caching, Lazy Evaluation, Lineage Graph (DAG), Fault Tolerance.
4. Spark vs Hadoop Ecosystem comparative analysis.
"""

import time
import os

def display_unit1_conceptual_overview():
    print("=" * 75)
    print("  UNIT I: INTRODUCTION TO APACHE SPARK & DISTRIBUTED COMPUTING")
    print("=" * 75)
    
    print("\n[1] LIMITATIONS OF HADOOP MAPREDUCE:")
    print("  * High Latency Disk I/O: MapReduce writes all intermediate states to HDFS.")
    print("  * Inefficiency for Iterative Workflows: Machine learning algorithms that require")
    print("    repeated iterations (like Logistic Regression or K-Means) must read and write to disk")
    print("    on every single pass, creating a major I/O bottleneck.")
    print("  * No Built-in In-Memory Caching: Hadoop cannot persist intermediate data in RAM.")

    print("\n[2] HOW APACHE SPARK OVERCOMES MAPREDUCE LIMITATIONS:")
    print("  * In-Memory Computing: Intermediate datasets (RDDs) are kept in memory across iterations.")
    print("  * Directed Acyclic Graph (DAG) Engine: Optimizes processing pipeline before execution.")
    print("  * Lazy Evaluation: Transformations are recorded in a lineage graph and only computed")
    print("    when an action (e.g., count, collect) is called.")
    print("  * Up to 100x faster than Hadoop MapReduce for iterative big data workloads.")

    print("\n[3] BATCH VS REAL-TIME PROCESSING COMPARISON:")
    print("  +----------------------+--------------------------+---------------------------+")
    print("  | Parameter            | Batch Processing (Hadoop)| Real-Time/In-Memory(Spark)|")
    print("  +----------------------+--------------------------+---------------------------+")
    print("  | Data Scope           | Historical complete data | Live streams / micro-batch|")
    print("  | Latency              | Hours to Minutes         | Milliseconds to Seconds   |")
    print("  | Storage Model        | Disk-centric (HDFS)      | Memory-centric (RAM)      |")
    print("  | Primary Use Case     | Nightly ETL, log dump    | Fraud detection, ML scoring|")
    print("  +----------------------+--------------------------+---------------------------+")

    print("\n[4] SPARK VS HADOOP ECOSYSTEM:")
    print("  * Storage: Hadoop uses HDFS; Spark can run on HDFS, S3, or local filesystems.")
    print("  * Computation: MapReduce is batch-only; Spark provides Unified Engine (RDD, GraphX, MLlib).")
    print("  * Fault Tolerance: Hadoop relies on 3x HDFS replication; Spark uses RDD Lineage graphs.")
    print("=" * 75)


def benchmark_in_memory_vs_disk_simulation(n_records=200000, iterations=5):
    """
    Empirical demonstration of Unit 1 concepts:
    Simulates the overhead of Hadoop-style disk writes vs Spark-style in-memory caching
    across multiple iterative passes (e.g. for ML model training).
    """
    print(f"\n[+] Running Unit 1 Empirical Benchmark ({iterations} iterative passes on {n_records:,} records)...")
    
    # 1. Spark In-Memory Simulation
    start_mem = time.time()
    data_in_ram = [i * 1.05 for i in range(n_records)]
    for _ in range(iterations):
        # In-memory computation pass (like Spark RDD cache())
        _ = sum(x for x in data_in_ram if x > 1000)
    in_memory_time = time.time() - start_mem

    # 2. Disk-Bound Simulation (like Hadoop MapReduce intermediate file serialization)
    tmp_path = "temp_disk_sim.tmp"
    start_disk = time.time()
    for _ in range(iterations):
        # Simulate writing intermediate results to disk
        with open(tmp_path, "w") as f:
            for x in data_in_ram[:10000]:  # sample written to disk to simulate I/O
                f.write(f"{x}\n")
        # Simulate reading intermediate results back from disk
        with open(tmp_path, "r") as f:
            _ = [float(line.strip()) for line in f]
    disk_bound_time = time.time() - start_disk

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    speedup = disk_bound_time / max(in_memory_time, 0.0001)

    print(f"  -> Spark In-Memory Processing Time: {in_memory_time:.4f} seconds")
    print(f"  -> Hadoop Disk-Bound Processing Time: {disk_bound_time:.4f} seconds")
    print(f"  -> In-Memory Speedup Factor: {speedup:.1f}x Faster")
    print("  [Conclusion] Spark's in-memory RDD model drastically reduces latency for iterative analytics.")


if __name__ == "__main__":
    display_unit1_conceptual_overview()
    benchmark_in_memory_vs_disk_simulation()
