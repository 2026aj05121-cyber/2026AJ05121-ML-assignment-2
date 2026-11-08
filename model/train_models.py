import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Ensure output directory exists
os.makedirs("model", exist_ok=True)

# 2. Load UCI Air Quality Dataset (Supports both .xlsx and .csv)
dataset_path_xlsx = "AirQualityUCI.xlsx"
dataset_path_csv = "AirQualityUCI.csv"

if os.path.exists(dataset_path_xlsx):
    print(f"[INFO] Loading dataset from {dataset_path_xlsx}...")
    df = pd.read_excel(dataset_path_xlsx)
elif os.path.exists(dataset_path_csv):
    print(f"[INFO] Loading dataset from {dataset_path_csv}...")
    df = pd.read_csv(dataset_path_csv, sep=";", decimal=",")
else:
    raise FileNotFoundError(
        "Please place 'AirQualityUCI.xlsx' in the project root directory before running."
    )

# Clean empty rows/columns if present
df = df.dropna(how="all").dropna(axis=1, how="all")

# 3. Preprocess UCI Air Quality Specific Sentinel Values (-200 represents missing data)
df = df.replace(-200, np.nan).replace(-200.0, np.nan)

# Extract Temporal Features (Hour and Month) from Date and Time columns
if "Date" in df.columns:
    df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.month
    df = df.drop(columns=["Date"])

if "Time" in df.columns:
    def parse_hour(val):
        if pd.isna(val):
            return np.nan
        if hasattr(val, "hour"):
            return val.hour
        try:
            return int(str(val).split(":")[0].split(".")[0])
        except Exception:
            return np.nan

    df["Hour"] = df["Time"].apply(parse_hour)
    df = df.drop(columns=["Time"])

# Drop NMHC(GT) only if >85% missing, while keeping at least 12 features
if "NMHC(GT)" in df.columns and df["NMHC(GT)"].isna().mean() > 0.85:
    df = df.drop(columns=["NMHC(GT)"])

# 4. Define Binary Classification Target based on True Hourly CO Concentration: CO(GT)
# Target = 1 (High CO Pollution Alert: CO(GT) > 2.0 mg/m^3), Target = 0 (Normal Air Quality: CO(GT) <= 2.0 mg/m^3)
target_source_col = "CO(GT)"
df = df.dropna(subset=[target_source_col]).copy()
df["target"] = (df[target_source_col] > 2.0).astype(int)
df = df.drop(columns=[target_source_col])

# Ensure all remaining feature columns are numeric
feature_cols = [c for c in df.columns if c != "target"]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows where more than half of the sensor readings are missing
df = df.dropna(thresh=int(len(feature_cols) * 0.6)).reset_index(drop=True)

print(f"[INFO] Cleaned Dataset Instances: {df.shape[0]} (>= 500 required)")
print(f"[INFO] Feature Count: {len(feature_cols)} (>= 12 required)")
print(f"[INFO] Features: {feature_cols}")
print(f"[INFO] Target Class Distribution:\n{df['target'].value_counts()}")

# 5. Stratified Train-Test Split (80% Train, 20% Test)
X = df[feature_cols]
y = df["target"]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Impute missing sensor values using Median Imputation fitted strictly on Training Set
imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train_raw), columns=feature_cols, index=X_train_raw.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test_raw), columns=feature_cols, index=X_test_raw.index)

# Save clean holdout test_data.csv (with all 13 input features + target column) for Streamlit App upload
test_df = X_test_imputed.copy()
test_df["target"] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print(f"[INFO] Exported 'test_data.csv' with shape: {test_df.shape}")

# 6. Feature Standardization (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Save preprocessing artifacts
joblib.dump(imputer, "model/imputer.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(feature_cols, "model/feature_names.pkl")

# 7. Initialize Required 5 Classification Models
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=42),
        "model/logistic_regression.pkl"
    ),
    "Decision Tree": (
        DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=42),
        "model/decision_tree.pkl"
    ),
    "KNN": (
        KNeighborsClassifier(n_neighbors=9, weights="distance"),
        "model/knn.pkl"
    ),
    "Naive Bayes": (
        GaussianNB(),
        "model/naive_bayes.pkl"
    ),
    "Support Vector Machine": (
        SVC(kernel="rbf", C=2.0, probability=True, random_state=42),
        "model/svm.pkl"
    )
}

# 8. Train, Evaluate, and Persist All 5 Models
summary_rows = []

for name, (clf, filepath) in models.items():
    clf.fit(X_train_scaled, y_train)
    joblib.dump(clf, filepath)

    y_pred = clf.predict(X_test_scaled)
    y_prob = clf.predict_proba(X_test_scaled)[:, 1]

    summary_rows.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    })

results_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 82)
print("BITS PILANI M.TECH (AIML) - ML ASSIGNMENT 2: AIR QUALITY UCI EVALUATION METRICS")
print("=" * 82)
print(results_df.to_string(index=False))
print("=" * 82)