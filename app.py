import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="UCI Air Quality Classification Dashboard | ML Assignment 2",
    page_icon="🌫️",
    layout="wide"
)

st.title("🌫️ UCI Air Quality Pollution Alert Classification System")
st.markdown(
    "**BITS Pilani M.Tech (AIML) — Machine Learning Assignment 2**  \n"
    "Interactive evaluation of 5 pre-trained classification models predicting **High Carbon Monoxide (CO > 2.0 mg/m³) Pollution Alerts** "
    "from multi-sensor chemical and meteorological measurements."
)

MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/svm.pkl"
}

@st.cache_resource
def load_saved_artifacts():
    imputer = joblib.load("model/imputer.pkl")
    scaler = joblib.load("model/scaler.pkl")
    feature_names = joblib.load("model/feature_names.pkl")
    loaded_models = {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
    return imputer, scaler, feature_names, loaded_models

try:
    imputer, scaler, feature_names, models = load_saved_artifacts()
except Exception as err:
    st.error(
        f"⚠️ Pre-trained model artifacts not found in `model/`. "
        f"Please run `python model/train_models.py` first. Details: {err}"
    )
    st.stop()

# Sidebar Controls
st.sidebar.header("1. Upload Holdout Test Dataset")
uploaded_csv = st.sidebar.file_uploader(
    "Upload Test CSV (13 Sensor/Temporal Features + 'target')",
    type=["csv"]
)

st.sidebar.header("2. Select Classification Model")
selected_model_name = st.sidebar.selectbox(
    "Choose a Pre-Trained Classifier:",
    list(MODEL_PATHS.keys())
)

# Load Test Data
if uploaded_csv is not None:
    test_df = pd.read_csv(uploaded_csv)
    st.sidebar.success("✅ Custom Test CSV Loaded!")
elif os.path.exists("test_data.csv"):
    test_df = pd.read_csv("test_data.csv")
    st.sidebar.info("ℹ️ Using repository `test_data.csv`. Upload any test CSV above to evaluate.")
else:
    st.warning("Please upload `test_data.csv` from the sidebar to view model evaluations.")
    st.stop()

required_cols = feature_names + ["target"]
missing_cols = [c for c in required_cols if c not in test_df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
    st.stop()

with st.expander("🔍 Inspect Test Dataset Sample & Feature Summary", expanded=False):
    st.write(f"**Test Instances:** `{test_df.shape[0]}` rows | **Features:** `{len(feature_names)}` columns + `target`")
    st.dataframe(test_df.head(10), use_container_width=True)

# Prepare Features & Target (Inference Only - No Model Retraining)
X_test_raw = test_df[feature_names].replace(-200, np.nan).replace(-200.0, np.nan)
y_test = test_df["target"].astype(int)

X_test_imputed = imputer.transform(X_test_raw)
X_test_scaled = scaler.transform(X_test_imputed)

def evaluate_model(clf, X_scaled, y_true):
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    metrics_dict = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4)
    }
    return metrics_dict, y_pred

# Evaluate Currently Selected Model
active_model = models[selected_model_name]
active_metrics, y_pred_active = evaluate_model(active_model, X_test_scaled, y_test)

st.subheader(f"📌 Evaluation Metrics — {selected_model_name}")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Accuracy", f"{active_metrics['Accuracy']:.4f}")
m2.metric("AUC Score", f"{active_metrics['AUC']:.4f}")
m3.metric("Precision", f"{active_metrics['Precision']:.4f}")
m4.metric("Recall", f"{active_metrics['Recall']:.4f}")
m5.metric("F1 Score", f"{active_metrics['F1']:.4f}")
m6.metric("MCC Score", f"{active_metrics['MCC']:.4f}")

# Confusion Matrix and Classification Report
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("🧩 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred_active)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=["Normal CO (0)", "High CO Alert (1)"],
        yticklabels=["Normal CO (0)", "High CO Alert (1)"],
        ax=ax
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(f"Confusion Matrix ({selected_model_name})")
    st.pyplot(fig)

with right_col:
    st.subheader("📑 Detailed Classification Report")
    report = classification_report(
        y_test,
        y_pred_active,
        target_names=["Normal Air (0)", "High CO Alert (1)"],
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

# Full 5-Model Comparison Table on Uploaded Test Data
st.subheader("📊 All 5 Classification Models — Comparative Performance on Test Data")
comparison_list = []
for model_label, clf_obj in models.items():
    m_scores, _ = evaluate_model(clf_obj, X_test_scaled, y_test)
    row = {"ML Model Name": model_label}
    row.update(m_scores)
    comparison_list.append(row)

comparison_df = pd.DataFrame(comparison_list)
st.dataframe(comparison_df, use_container_width=True)