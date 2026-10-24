import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, silhouette_score
)

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="EARLY ALERT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# COLORS
# ============================================================

NAVY = "#16324F"
BLUE = "#2563EB"
CYAN = "#0891B2"
TEAL = "#0F766E"
PURPLE = "#7C3AED"
ORANGE = "#EA580C"
PINK = "#DB2777"
GREEN = "#16A34A"
RED = "#DC2626"
TEXT = "#1E293B"
MUTED = "#64748B"
BORDER = "#DCE5F0"

# ============================================================
# DESIGN
# ============================================================

st.markdown(f"""
<style>

.stApp {{
    background:
        radial-gradient(circle at 5% 5%, rgba(37,99,235,.10), transparent 25%),
        radial-gradient(circle at 95% 5%, rgba(124,58,237,.10), transparent 25%),
        linear-gradient(135deg,#F8FAFC,#F2F7FF,#FAF7FF);
    color:{TEXT};
}}

.block-container {{
    max-width:1450px;
    padding-top:1.5rem;
}}

section[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,#FFFFFF,#F3F8FF,#FAF6FF);
    border-right:1px solid #D7E2EF;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] {{
    gap:5px;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background:#FFFFFF;
    border:1px solid #E5EAF2;
    border-radius:12px;
    padding:7px 10px;
    margin-bottom:4px;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background:#EAF2FF;
    border-color:#BDD6FF;
}}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {{
    color:#334155 !important;
    font-weight:650 !important;
}}

h1,h2,h3 {{
    color:{NAVY} !important;
    font-weight:800 !important;
}}

.hero {{
    padding:32px 36px;
    border-radius:24px;
    border:1px solid #D4E3F5;
    background:
        radial-gradient(circle at 90% 10%,rgba(124,58,237,.15),transparent 28%),
        radial-gradient(circle at 5% 100%,rgba(8,145,178,.14),transparent 28%),
        linear-gradient(135deg,#FFFFFF,#F2F8FF,#FAF5FF);
    box-shadow:0 12px 32px rgba(15,23,42,.07);
    margin-bottom:25px;
}}

.hero-title {{
    color:{NAVY};
    font-size:2.4rem;
    font-weight:900;
    letter-spacing:-1px;
}}

.hero-subtitle {{
    color:#52637A;
    margin-top:7px;
    font-size:1rem;
}}

.badge {{
    display:inline-block;
    margin-top:14px;
    padding:6px 13px;
    border-radius:999px;
    color:#1D4ED8;
    background:#E8F0FF;
    border:1px solid #C7DBFF;
    font-size:.75rem;
    font-weight:750;
}}

.card {{
    background:#FFFFFF;
    border:1px solid {BORDER};
    border-radius:17px;
    padding:18px;
    box-shadow:0 6px 20px rgba(15,23,42,.05);
}}

.card-label {{
    color:{MUTED};
    font-size:.78rem;
    font-weight:700;
}}

.card-value {{
    color:{NAVY};
    font-size:1.65rem;
    font-weight:850;
    margin-top:5px;
}}

.section-title {{
    color:{NAVY};
    font-size:1.3rem;
    font-weight:820;
    margin-top:20px;
}}

.section-subtitle {{
    color:{MUTED};
    font-size:.85rem;
    margin-bottom:12px;
}}

.warning {{
    border-left:5px solid {RED};
    background:#FFF5F5;
    padding:18px;
    border-radius:12px;
}}

.success {{
    border-left:5px solid {GREEN};
    background:#F0FDF4;
    padding:18px;
    border-radius:12px;
}}

div[data-testid="stPlotlyChart"] {{
    background:#FFFFFF;
    border:1px solid {BORDER};
    border-radius:17px;
    padding:7px;
}}

div[data-testid="stDataFrame"] {{
    border:1px solid {BORDER};
    border-radius:14px;
}}

div[data-testid="stMetric"] {{
    background:#FFFFFF;
    border:1px solid {BORDER};
    border-radius:15px;
    box-shadow:0 5px 17px rgba(15,23,42,.05);
}}

.footer {{
    text-align:center;
    color:#94A3B8;
    font-size:.75rem;
    margin-top:35px;
    padding:20px;
    border-top:1px solid {BORDER};
}}

</style>
""", unsafe_allow_html=True)

# ============================================================
# PATHS
# ============================================================

BASE = os.path.dirname(os.path.abspath(__file__))

FEATURE_FILE = os.path.join(BASE, "data", "student_features.csv")
ACTION_FILE = os.path.join(BASE, "data", "mooc_actions.tsv")
LABEL_FILE = os.path.join(BASE, "data", "mooc_action_labels.tsv")

FEATURE_COLUMNS = [
    "total_actions",
    "unique_targets",
    "avg_time_gap",
    "exam_attempt_rate",
    "graph_in_degree",
    "graph_out_degree",
    "final_score"
]

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    features = pd.read_csv(FEATURE_FILE)
    actions = pd.read_csv(ACTION_FILE, sep="\t")
    labels = pd.read_csv(LABEL_FILE, sep="\t")
    return features, actions, labels

features, actions, labels = load_data()

features.columns = [str(c).strip() for c in features.columns]

for col in FEATURE_COLUMNS:
    features[col] = pd.to_numeric(features[col], errors="coerce")

features["dropout"] = pd.to_numeric(
    features["dropout"], errors="coerce"
)

features = features.dropna(
    subset=FEATURE_COLUMNS + ["dropout"]
).copy()

features["dropout"] = features["dropout"].astype(int)

X = features[FEATURE_COLUMNS]
y = features["dropout"]

# ============================================================
# MODELS
# ============================================================

@st.cache_resource
def train_models(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=.20,
        random_state=42,
        stratify=y
    )

    models = {

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                random_state=42,
                max_iter=500
            ))
        ]),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=4,
                random_state=42
            ),

        "Linear SVC": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                random_state=42,
                dual=False,
                max_iter=2000
            ))
        ]),

        "Naive Bayes":
            GaussianNB()
    }

    trained = {}
    metrics = {}
    matrices = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        metrics[name] = {
            "Accuracy": accuracy_score(
                y_test, prediction
            ),
            "Precision": precision_score(
                y_test, prediction,
                zero_division=0
            ),
            "Recall": recall_score(
                y_test, prediction,
                zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, prediction,
                zero_division=0
            )
        }

        matrices[name] = confusion_matrix(
            y_test,
            prediction
        )

        trained[name] = model

    return trained, metrics, matrices


models, model_metrics, confusion_matrices = train_models(
    X, y
)

# ============================================================
# COMPONENTS
# ============================================================

def hero(title, subtitle, badge):

    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
        <div class="badge">{badge}</div>
    </div>
    """, unsafe_allow_html=True)


def section(title, subtitle=""):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )

    if subtitle:
        st.markdown(
            f'<div class="section-subtitle">{subtitle}</div>',
            unsafe_allow_html=True
        )


def card(label, value, icon=""):

    st.markdown(f"""
    <div class="card">
        <div class="card-label">{icon} {label}</div>
        <div class="card-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def chart_style(fig, height=420):

    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            color=TEXT
        ),
        margin=dict(
            l=30,
            r=25,
            t=60,
            b=30
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,.8)"
        )
    )

    return fig

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div style="
        padding:18px 5px 10px;
        color:#16324F;
        font-size:1.6rem;
        font-weight:900;
    ">
        🎓 EARLY ALERT
    </div>

    <div style="
        color:#64748B;
        font-size:.85rem;
        margin-bottom:20px;
    ">
        Learner Risk Analytics
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "📊 Data Explorer",
            "🤖 Student Prediction",
            "📈 Model Lab",
            "🕸️ Graph Analytics",
            "🔬 K-Means Lab"
        ],
        label_visibility="visible"
    )

# ============================================================
# OVERVIEW
# ============================================================

if page == "🏠 Overview":

    hero(
        "EARLY ALERT",
        "Intelligent Learner Risk & Behaviour Analytics",
        "LEARNER ANALYTICS"
    )

    total_students = features["student_id"].nunique()
    total_interactions = len(actions)
    dropout_count = int(features["dropout"].sum())
    dropout_rate = features["dropout"].mean() * 100

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card("Learners", f"{total_students:,}", "👥")

    with c2:
        card("Interactions", f"{total_interactions:,}", "⚡")

    with c3:
        card("Dropout Learners", f"{dropout_count:,}", "⚠️")

    with c4:
        card("Dropout Rate", f"{dropout_rate:.2f}%", "📊")

    left, right = st.columns(2)

    with left:

        section(
            "Learner Status",
            "Active and dropout learner distribution"
        )

        status = (
            features["dropout"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        status.columns = [
            "Dropout",
            "Learners"
        ]

        status["Status"] = status["Dropout"].map({
            0: "Active",
            1: "Dropout"
        })

        fig = px.bar(
            status,
            x="Status",
            y="Learners",
            color="Status",
            text="Learners",
            color_discrete_map={
                "Active": CYAN,
                "Dropout": RED
            }
        )

        fig.update_traces(
            textposition="outside"
        )

        st.plotly_chart(
            chart_style(fig, 390),
            use_container_width=True
        )

    with right:

        section(
            "Model Accuracy",
            "Classification accuracy comparison"
        )

        model_df = (
            pd.DataFrame(model_metrics)
            .T
            .reset_index()
            .rename(columns={"index": "Model"})
        )

        fig = px.bar(
            model_df,
            x="Model",
            y="Accuracy",
            color="Model",
            text=model_df["Accuracy"].map(
                lambda x: f"{x*100:.2f}%"
            ),
            color_discrete_sequence=[
                BLUE,
                PURPLE,
                TEAL,
                ORANGE
            ]
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_yaxes(
            tickformat=".0%",
            range=[0, 1]
        )

        st.plotly_chart(
            chart_style(fig, 390),
            use_container_width=True
        )

    section(
        "Learner Behaviour Snapshot",
        "Sample learner feature records"
    )

    st.dataframe(
        features.head(10),
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "📊 Data Explorer":

    hero(
        "DATA EXPLORER",
        "Explore learner and interaction records",
        "DATA INSIGHTS"
    )

    tab1, tab2, tab3 = st.tabs([
        "👥 Learner Features",
        "⚡ MOOC Actions",
        "🏷️ Action Labels"
    ])

    with tab1:

        section(
            "Learner Features",
            f"{len(features):,} learner records"
        )

        search = st.text_input(
            "Search Student ID",
            placeholder="Example: 25"
        )

        display = features.copy()

        if search.strip():

            try:

                sid = int(search)

                display = display[
                    display["student_id"] == sid
                ]

            except ValueError:

                st.warning(
                    "Student ID must be numeric."
                )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    with tab2:

        section(
            "MOOC Interaction Records",
            f"{len(actions):,} records"
        )

        st.dataframe(
            actions.head(500),
            use_container_width=True,
            hide_index=True
        )

    with tab3:

        section(
            "Action Labels",
            f"{len(labels):,} labelled records"
        )

        st.dataframe(
            labels.head(500),
            use_container_width=True,
            hide_index=True
        )

# ============================================================
# STUDENT PREDICTION
# ============================================================

elif page == "🤖 Student Prediction":

    hero(
        "STUDENT PREDICTION",
        "Inspect learner behaviour and model predictions",
        "INDIVIDUAL RISK ANALYSIS"
    )

    student_ids = sorted(
        features["student_id"].unique()
    )

    selected = st.selectbox(
        "Select Student",
        student_ids
    )

    student = features[
        features["student_id"] == selected
    ].iloc[0]

    section(
        f"Learner {selected}",
        "Learner behaviour profile"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card(
            "Total Actions",
            f"{student['total_actions']:.0f}",
            "⚡"
        )

    with c2:
        card(
            "Unique Activities",
            f"{student['unique_targets']:.0f}",
            "📚"
        )

    with c3:
        card(
            "Average Time Gap",
            f"{student['avg_time_gap']:.2f}",
            "⏱️"
        )

    with c4:
        card(
            "Exam Attempt Rate",
            f"{student['exam_attempt_rate']:.2f}",
            "📝"
        )

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        card(
            "Graph In-Degree",
            f"{student['graph_in_degree']:.0f}",
            "⬅️"
        )

    with c6:
        card(
            "Graph Out-Degree",
            f"{student['graph_out_degree']:.0f}",
            "➡️"
        )

    with c7:
        card(
            "Final Score",
            f"{student['final_score']:.2f}",
            "🏆"
        )

    with c8:

        status = (
            "Dropout"
            if int(student["dropout"]) == 1
            else "Active"
        )

        card(
            "Recorded Status",
            status,
            "📌"
        )

    section(
        "Model Predictions",
        "Predictions from the classification models"
    )

    student_input = pd.DataFrame(
        [student[FEATURE_COLUMNS].values],
        columns=FEATURE_COLUMNS
    )

    predictions = []

    for name, model in models.items():

        pred = int(
            model.predict(student_input)[0]
        )

        predictions.append({
            "Model": name,
            "Prediction":
                "Dropout"
                if pred == 1
                else "Active"
        })

    prediction_df = pd.DataFrame(
        predictions
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True
    )

    dropout_votes = int(
        (
            prediction_df["Prediction"]
            == "Dropout"
        ).sum()
    )

    active_votes = (
        len(prediction_df)
        - dropout_votes
    )

    if dropout_votes > active_votes:

        st.markdown(f"""
        <div class="warning">
            <b style="color:{RED};">
                ⚠️ Majority dropout prediction
            </b>
            <br><br>
            {dropout_votes} of
            {len(prediction_df)}
            models predict dropout.
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="success">
            <b style="color:{GREEN};">
                ✓ Majority active prediction
            </b>
            <br><br>
            {active_votes} of
            {len(prediction_df)}
            models predict active.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# MODEL LAB
# ============================================================

elif page == "📈 Model Lab":

    hero(
        "MODEL LAB",
        "Compare classification performance",
        "MACHINE LEARNING"
    )

    metric_df = (
        pd.DataFrame(model_metrics)
        .T
        .reset_index()
        .rename(columns={"index": "Model"})
    )

    display = metric_df.copy()

    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]:

        display[col] = (
            display[col] * 100
        ).round(2)

    section(
        "Classification Performance",
        "Accuracy, precision, recall and F1-score"
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    metric = st.selectbox(
        "Select Performance Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score"
        ]
    )

    fig = px.bar(
        metric_df,
        x="Model",
        y=metric,
        color="Model",
        text=metric_df[metric].map(
            lambda x: f"{x*100:.2f}%"
        ),
        color_discrete_sequence=[
            BLUE,
            PURPLE,
            TEAL,
            ORANGE
        ]
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_yaxes(
        tickformat=".0%",
        range=[0, 1]
    )

    st.plotly_chart(
        chart_style(fig, 430),
        use_container_width=True
    )

    section(
        "Confusion Matrix",
        "Actual class versus predicted class"
    )

    selected_model = st.selectbox(
        "Select Model",
        list(confusion_matrices.keys())
    )

    matrix = confusion_matrices[
        selected_model
    ]

    matrix_fig = go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                x=[
                    "Predicted Active",
                    "Predicted Dropout"
                ],
                y=[
                    "Actual Active",
                    "Actual Dropout"
                ],
                text=matrix,
                texttemplate="%{text}",
                colorscale=[
                    [0, "#EFF6FF"],
                    [.4, "#93C5FD"],
                    [.7, "#3B82F6"],
                    [1, "#1D4ED8"]
                ]
            )
        ]
    )

    matrix_fig.update_layout(
        title=selected_model,
        xaxis_title="Prediction",
        yaxis_title="Actual"
    )

    st.plotly_chart(
        chart_style(matrix_fig, 430),
        use_container_width=True
    )

# ============================================================
# GRAPH ANALYTICS
# ============================================================

elif page == "🕸️ Graph Analytics":

    hero(
        "GRAPH ANALYTICS",
        "Explore learner connectivity and engagement",
        "GRAPH INSIGHTS"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card(
            "Learners",
            f"{features['student_id'].nunique():,}",
            "👥"
        )

    with c2:
        card(
            "Interactions",
            f"{len(actions):,}",
            "🔗"
        )

    with c3:
        card(
            "Average In-Degree",
            f"{features['graph_in_degree'].mean():.2f}",
            "⬅️"
        )

    with c4:
        card(
            "Average Out-Degree",
            f"{features['graph_out_degree'].mean():.2f}",
            "➡️"
        )

    section(
        "Learner Connectivity",
        "Distribution of graph out-degree"
    )

    fig = px.histogram(
        features,
        x="graph_out_degree",
        nbins=30,
        color_discrete_sequence=[PURPLE],
        title="Activity Connectivity Across Learners"
    )

    fig.update_traces(
        marker_line_width=0
    )

    st.plotly_chart(
        chart_style(fig, 430),
        use_container_width=True
    )

    section(
        "Highly Connected Learners",
        "Learners with high graph engagement"
    )

    top = (
        features[
            [
                "student_id",
                "graph_in_degree",
                "graph_out_degree",
                "total_actions"
            ]
        ]
        .sort_values(
            "graph_out_degree",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        top,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# K-MEANS
# ============================================================

elif page == "🔬 K-Means Lab":

    hero(
        "K-MEANS LAB",
        "Discover learner behaviour groups",
        "BEHAVIOUR SEGMENTATION"
    )

    data = features[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()

    scaled = scaler.fit_transform(data)

    k = st.slider(
        "Number of Clusters",
        2,
        5,
        3
    )

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        scaled
    )

    cluster_data = features.copy()

    cluster_data["Cluster"] = cluster_labels

    silhouette = silhouette_score(
        scaled,
        cluster_labels
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        card(
            "Clusters",
            str(k),
            "🧩"
        )

    with c2:
        card(
            "Silhouette Score",
            f"{silhouette:.4f}",
            "📐"
        )

    with c3:
        card(
            "Learners",
            f"{len(cluster_data):,}",
            "👥"
        )

    summary = (
        cluster_data
        .groupby("Cluster")
        .agg(
            Learners=("student_id", "count"),
            Mean_Actions=("total_actions", "mean"),
            Mean_Activities=("unique_targets", "mean"),
            Dropout_Rate=("dropout", "mean")
        )
        .reset_index()
    )

    summary["Dropout_Rate"] *= 100

    summary = summary.round(2)

    section(
        "Cluster Summary",
        "Behaviour characteristics of learner groups"
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    section(
        "Learner Behaviour Map",
        "Learners grouped by activity and final score"
    )

    fig = px.scatter(
        cluster_data,
        x="total_actions",
        y="final_score",
        color="Cluster",
        hover_data=[
            "student_id",
            "unique_targets",
            "dropout"
        ],
        color_discrete_sequence=[
            BLUE,
            PURPLE,
            TEAL,
            ORANGE,
            PINK
        ],
        title="Learner Behaviour Clusters"
    )

    st.plotly_chart(
        chart_style(fig, 480),
        use_container_width=True
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    🎓 EARLY ALERT &nbsp;•&nbsp;
    Intelligent Learner Risk & Behaviour Analytics
</div>
""", unsafe_allow_html=True)