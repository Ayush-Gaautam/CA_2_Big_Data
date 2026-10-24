package mooc.analytics

/**
 * ============================================================================
 * INT315: CLUSTER COMPUTING - CA2 PROJECT
 * UNIT II & UNIT III: SCALA PROGRAMMING & RDD ANALYTICS
 * ============================================================================
 * 
 * Demonstrates:
 * 1. Scala Core Features:
 *    - Immutability vs Mutability
 *    - Type Inference (val vs var)
 *    - Higher-Order Functions & Lambda expressions (_ + _)
 *    - Functional Collections (List, Map, Set)
 * 2. Spark RDD Operations in Scala:
 *    - RDD Creation (sc.textFile, sc.parallelize)
 *    - RDD Transformations: map(), filter(), reduceByKey(), groupByKey()
 *    - RDD Actions: count(), take(), collect()
 *    - Graph degree aggregation (Student -> Activities)
 * ============================================================================
 */

import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.ArrayBuffer

object StudentAnalytics {

  // --- UNIT II: SCALA FUNCTIONS & IMMUTABILITY DEMONSTRATION ---
  
  // Named function with explicit return type and pattern matching
  def classifyEngagement(actionCount: Int): String = {
    if (actionCount >= 50) "High Engagement"
    else if (actionCount >= 25) "Moderate Engagement"
    else "At-Risk / Low Engagement"
  }

  // Pure mathematical higher-order function
  def calculateActivityRate(actions: Int, hours: Double): Double = {
    if (hours > 0.0) actions / hours else 0.0
  }

  def main(args: Array[String]): Unit = {
    println("===============================================================")
    println("  INT315 - UNIT II: SCALA LANGUAGE FEATURES DEMO")
    println("===============================================================")

    // 1. Basic Data Types & Type Inference (val = immutable, var = mutable)
    val courseName: String = "INT315 - Cluster Computing"
    val academicYear: Int = 2026
    val passThreshold: Double = 50.0
    var activeWorkerCount: Int = 4 // mutable variable

    println(s"Course: $courseName | Year: $academicYear | Workers: $activeWorkerCount")

    // 2. Mutable vs Immutable Collections
    // Immutable List and Map (Functional standard in Scala)
    val activityTypes: List[String] = List("Lecture", "Quiz", "Assignment", "Forum", "Exam")
    val activityWeights: Map[String, Double] = Map(
      "Lecture" -> 1.0,
      "Quiz" -> 2.5,
      "Assignment" -> 5.0,
      "Forum" -> 1.5,
      "Exam" -> 10.0
    )

    // Mutable Collection (used only when local modification is strictly needed)
    val flaggedStudentBuffer: ArrayBuffer[Int] = ArrayBuffer()
    flaggedStudentBuffer += 101
    flaggedStudentBuffer += 105

    // 3. Higher-Order Functions: map, filter, fold/reduce
    val highValueActivities = activityWeights.filter(kv => kv._2 >= 3.0)
    println(s"[Unit 2] High Value Activities (weight >= 3.0): $highValueActivities")

    // Lambda expressions with shorthand syntax (_ + _)
    val totalWeight = activityWeights.values.foldLeft(0.0)(_ + _)
    println(s"[Unit 2] Total Course Activity Weight Sum: $totalWeight")

    println("\n===============================================================")
    println("  INT315 - UNIT III: SPARK RDD PROCESSING IN SCALA")
    println("===============================================================")

    // Initializing Spark Context for local cluster simulation
    val conf = new SparkConf()
      .setAppName("MOOCStudentBehaviorAnalytics")
      .setMaster("local[*]")
    val sc = new SparkContext(conf)

    try {
      // 1. RDD Creation from text file
      val actionsFilePath = "data/mooc_actions.tsv"
      println(s"[Unit 3] Reading action logs from: $actionsFilePath")
      val rawLinesRDD = sc.textFile(actionsFilePath)

      // 2. RDD Transformation: filter out header line
      val header = rawLinesRDD.first()
      val dataRDD = rawLinesRDD.filter(line => line != header)

      val totalActionsCount = dataRDD.count() // RDD Action
      println(s"[Unit 3 Action] Total raw actions loaded: $totalActionsCount")

      // 3. RDD Transformations: map & reduceByKey
      // Format: ACTIONID \t USERID \t TARGETID \t TIMESTAMP
      // Extract Pair RDD: (UserId, 1) to count total interactions per student
      val userActionPairs = dataRDD.map(line => {
        val parts = line.split("\t")
        val userId = parts(1).toInt
        (userId, 1)
      })

      // Aggregate action counts per student across distributed partitions
      val studentActionCounts = userActionPairs.reduceByKey(_ + _)

      // 4. Graph Out-Degree Analysis: Count distinct targets interacted per user
      val userTargetPairs = dataRDD.map(line => {
        val parts = line.split("\t")
        val userId = parts(1).toInt
        val targetId = parts(2).toInt
        (userId, targetId)
      }).distinct()

      val studentDistinctTargets = userTargetPairs
        .map(pair => (pair._1, 1))
        .reduceByKey(_ + _)

      // Join RDDs: (UserId, (TotalActions, DistinctTargets))
      val studentGraphProfile = studentActionCounts.join(studentDistinctTargets)

      // 5. RDD Actions: take(), collect()
      println("\n[Unit 3 Action] Sample 5 Student Engagement & Graph Out-Degrees:")
      studentGraphProfile.take(5).foreach { case (userId, (totalAct, distinctTargets)) =>
        val category = classifyEngagement(totalAct)
        println(s"  -> Student ID: $userId | Total Actions: $totalAct | Unique Targets: $distinctTargets | Status: $category")
      }

      // Filter at-risk students with low action count (< 20)
      val atRiskStudentsRDD = studentActionCounts.filter(record => record._2 < 20)
      val atRiskCount = atRiskStudentsRDD.count()
      println(s"\n[Unit 3 Action] Total At-Risk Learners Identified (< 20 actions): $atRiskCount")

    } finally {
      sc.stop()
      println("\n[OK] SparkContext successfully stopped.")
    }
  }
}
