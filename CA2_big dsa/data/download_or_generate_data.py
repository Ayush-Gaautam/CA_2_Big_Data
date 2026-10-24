"""
Data Acquisition & Generation Module
Source: Stanford Network Analysis Project (SNAP) - MOOC User Action Dataset
Reference: S. Kumar, X. Zhang, J. Leskovec. "Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks", ACM SIGKDD 2019.
URL: https://snap.stanford.edu/data/act-mooc.html
"""

import os
import csv
import random
import urllib.request
import tarfile

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIONS_FILE = os.path.join(DATA_DIR, "mooc_actions.tsv")
LABELS_FILE = os.path.join(DATA_DIR, "mooc_action_labels.tsv")
STUDENT_FEATURES_FILE = os.path.join(DATA_DIR, "student_features.csv")


def generate_representative_dataset(num_users=1200, num_activities=60, actions_per_user_range=(15, 80)):
    """
    Generates a realistic dataset matching the official Stanford SNAP act-mooc schema:
    - mooc_actions.tsv: ACTIONID \\t USERID \\t TARGETID \\t TIMESTAMP
    - mooc_action_labels.tsv: ACTIONID \\t LABEL (0 = Active/Completed, 1 = Dropout)
    - student_features.csv: Aggregated behavioral and graph-ready features for MLlib
    """
    print(f"[+] Generating representative Stanford SNAP MOOC dataset in: {DATA_DIR}")
    random.seed(42)

    actions = []
    labels = []
    student_records = []

    action_id = 0
    base_timestamp = 1400000000.0  # Unix timestamp baseline

    for user_id in range(1, num_users + 1):
        # Determine dropout probability based on latent student engagement profile
        # Profile: 0 = high engagement, 1 = moderate, 2 = struggling / disengaged
        profile = random.choices([0, 1, 2], weights=[0.45, 0.35, 0.20])[0]

        if profile == 0:  # High engagement
            n_actions = random.randint(50, 95)
            unique_targets = random.randint(25, num_activities)
            avg_time_gap = random.uniform(1800, 7200)  # 30 mins to 2 hours
            exam_attempt_rate = random.uniform(0.75, 1.0)
            dropout_prob = 0.08
            base_score = random.uniform(75, 98)
        elif profile == 1:  # Moderate engagement
            n_actions = random.randint(25, 55)
            unique_targets = random.randint(12, 30)
            avg_time_gap = random.uniform(7200, 28800)  # 2 to 8 hours
            exam_attempt_rate = random.uniform(0.40, 0.74)
            dropout_prob = 0.35
            base_score = random.uniform(50, 78)
        else:  # Disengaged / At-risk
            n_actions = random.randint(8, 25)
            unique_targets = random.randint(3, 12)
            avg_time_gap = random.uniform(28800, 86400)  # 8 to 24 hours
            exam_attempt_rate = random.uniform(0.05, 0.39)
            dropout_prob = 0.82
            base_score = random.uniform(15, 52)

        is_dropout = 1 if random.random() < dropout_prob else 0

        # Generate individual action records
        current_time = base_timestamp + random.uniform(0, 10000)
        user_targets = random.sample(range(1, num_activities + 1), min(unique_targets, num_activities))

        for _ in range(n_actions):
            target_id = random.choice(user_targets)
            current_time += random.expovariate(1.0 / avg_time_gap)
            action_label = 1 if is_dropout else 0

            actions.append((action_id, user_id, target_id, round(current_time, 2)))
            labels.append((action_id, action_label))
            action_id += 1

        # Calculate student-level summary for MLlib modeling
        # (Out-degree in graph = total activity targets interacted with)
        out_degree = unique_targets
        in_degree = n_actions

        student_records.append({
            "student_id": user_id,
            "total_actions": n_actions,
            "unique_targets": unique_targets,
            "avg_time_gap": round(avg_time_gap / 3600.0, 2),  # in hours
            "exam_attempt_rate": round(exam_attempt_rate, 3),
            "graph_in_degree": in_degree,
            "graph_out_degree": out_degree,
            "final_score": round(base_score, 2),
            "dropout": is_dropout
        })

    # Write mooc_actions.tsv
    with open(ACTIONS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ACTIONID", "USERID", "TARGETID", "TIMESTAMP"])
        writer.writerows(actions)

    # Write mooc_action_labels.tsv
    with open(LABELS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["ACTIONID", "LABEL"])
        writer.writerows(labels)

    # Write student_features.csv
    with open(STUDENT_FEATURES_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "student_id", "total_actions", "unique_targets", "avg_time_gap",
            "exam_attempt_rate", "graph_in_degree", "graph_out_degree",
            "final_score", "dropout"
        ])
        writer.writeheader()
        writer.writerows(student_records)

    print(f"[OK] Successfully created {len(actions)} actions across {num_users} students.")
    print(f"[OK] mooc_actions.tsv -> {ACTIONS_FILE}")
    print(f"[OK] mooc_action_labels.tsv -> {LABELS_FILE}")
    print(f"[OK] student_features.csv -> {STUDENT_FEATURES_FILE}")


if __name__ == "__main__":
    generate_representative_dataset()
