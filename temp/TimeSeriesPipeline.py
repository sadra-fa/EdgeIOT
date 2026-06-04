
# ============================================================
# IMPORTANT FIX FOR TRANSFORMERS + KERAS
# ============================================================

import os

os.environ["TRANSFORMERS_NO_TF"] = "1"

# ============================================================
# IMPORTS
# ============================================================

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from collections import Counter

from scipy.sparse import issparse

from sklearn.preprocessing import (
    StandardScaler,
    label_binarize
)

from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering

from sklearn.ensemble import (
    RandomForestClassifier,
    IsolationForest
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    accuracy_score
)

from xgboost import XGBClassifier, plot_importance

import shap

from transformers import AutoTokenizer


# ============================================================
# ROC CURVE FUNCTION
# ============================================================

def plot_multiclass_roc(model, X_test, y_test_bin, model_name):

    y_score = model.predict_proba(X_test)

    plt.figure(figsize=(8, 6))

    for i in range(y_test_bin.shape[1]):

        fpr, tpr, _ = roc_curve(
            y_test_bin[:, i],
            y_score[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"Class {i} (AUC={roc_auc:.2f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title(f"ROC Curve - {model_name}")

    plt.legend()

    plt.show()


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("RealLife.csv")

print("Original Dataset Shape:", df.shape)

df = df.drop_duplicates()

df = df.fillna(df.mean(numeric_only=True))

print("Cleaned Dataset Shape:", df.shape)


# ============================================================
# FEATURE TYPES
# ============================================================

num_cols = df.select_dtypes(include="number").columns
cat_cols = df.select_dtypes(include="object").columns

print("\nNumerical Columns:")
print(num_cols.tolist())

print("\nCategorical Columns:")
print(cat_cols.tolist())


# ============================================================
# ONE HOT ENCODING
# ============================================================

X_processed = pd.get_dummies(
    df,
    columns=["location", "behaviour"]
)

print("\nProcessed Dataset Shape:")
print(X_processed.shape)


# ============================================================
# NORMALIZATION
# ============================================================

sensor_cols = [
    "temperature",
    "humidity",
    "noise_db",
    "light_intensity",
    "motion_level"
]

scaler = StandardScaler()

X_processed[sensor_cols] = scaler.fit_transform(
    X_processed[sensor_cols]
)

print("\nNormalization Completed.")


# ============================================================
# TIME-SERIES SENSOR VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 5))

plt.plot(
    X_processed["temperature"][:200],
    label="Temperature"
)

plt.plot(
    X_processed["humidity"][:200],
    label="Humidity"
)

plt.plot(
    X_processed["noise_db"][:200],
    label="Noise"
)

plt.title("Smart Home Sensor Time-Series")

plt.xlabel("Time")

plt.ylabel("Normalized Value")

plt.legend()

plt.show()


# ============================================================
# PCA FOR VISUALIZATION
# ============================================================

X_dense = (
    X_processed.toarray()
    if issparse(X_processed)
    else X_processed
)

pca = PCA(n_components=2)

X_pca_2d = pca.fit_transform(X_dense)


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

clusterer = AgglomerativeClustering(
    n_clusters=3,
    linkage="ward"
)

pseudo_labels = clusterer.fit_predict(X_dense)

X_labeled = X_processed.copy()

X_labeled["label"] = pseudo_labels

X_labeled.to_csv(
    "dataset_with_labels_Hierarchical.csv",
    index=False
)

print("\nHierarchical Clustering Completed.")


# ============================================================
# PCA CLUSTER VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=pseudo_labels,
    s=20
)

plt.title("Hierarchical Clustering (Ward)")

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.colorbar(label="Cluster")

plt.show()


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X = X_processed
y = pseudo_labels

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

plt.show()

# ============================================================
# XGBOOST
# ============================================================

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train)

y_proba = xgb.predict_proba(X_test)

y_pred_xgb = np.argmax(y_proba, axis=1)

y_confidence = np.max(y_proba, axis=1)

xgb_accuracy = accuracy_score(
    y_test,
    y_pred_xgb
)

print("\nXGBoost Report")
print(classification_report(y_test, y_pred_xgb))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_xgb
)

plt.title("XGBoost Confusion Matrix")

plt.show()


# ============================================================
# XGBOOST FEATURE IMPORTANCE
# ============================================================

plot_importance(
    xgb,
    max_num_features=15
)

plt.title("XGBoost Feature Importance")

plt.show()


# ============================================================
# ROC CURVES
# ============================================================

classes = np.unique(y)

y_test_bin = label_binarize(
    y_test,
    classes=classes
)

plot_multiclass_roc(
    xgb,
    X_test,
    y_test_bin,
    "XGBoost"
)


# ============================================================
# MODEL COMPARISON GRAPH
# ============================================================

models = [
    "Decision Tree",
    "Random Forest",
    "XGBoost"
]

scores = [
    xgb_accuracy
]

plt.figure(figsize=(8, 5))

plt.bar(models, scores)

plt.ylim(0, 1)

plt.ylabel("Accuracy")

plt.title("Model Performance Comparison")

plt.show()


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

print("\nRunning SHAP Explainability...")

try:

    explainer = shap.Explainer(xgb)

    shap_values = explainer(X_test)

    print("SHAP Analysis Completed.")

    shap.plots.beeswarm(
        shap_values,
        max_display=15
    )

except Exception as e:

    print("SHAP Error:", e)

    print("Skipping SHAP visualization...")


# ============================================================
# TIME-SERIES WINDOW CREATION
# ============================================================

WINDOW_SIZE = 5

def create_time_series_windows(
    df,
    sensor_columns,
    window_size=5
):

    windows = []

    for i in range(len(df) - window_size):

        window = (
            df.iloc[i:i+window_size][sensor_columns]
            .values
            .flatten()
        )

        windows.append(window)

    return np.array(windows)


X_sequence = create_time_series_windows(
    X_processed,
    sensor_cols,
    WINDOW_SIZE
)

print("\nTime-Series Shape:")
print(X_sequence.shape)


# ============================================================
# ANOMALY DETECTION
# ============================================================

iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

iso_model.fit(X_sequence)

anomaly_scores = iso_model.decision_function(
    X_sequence
)

anomaly_labels = iso_model.predict(
    X_sequence
)

anomaly_labels = np.where(
    anomaly_labels == -1,
    "ABNORMAL",
    "NORMAL"
)

print("\nAnomaly Detection Completed.")


# ============================================================
# ANOMALY VISUALIZATION
# ============================================================

plt.figure(figsize=(14, 5))

colors = [
    "red" if x == "ABNORMAL"
    else "blue"
    for x in anomaly_labels
]

plt.scatter(
    range(len(anomaly_scores)),
    anomaly_scores,
    c=colors,
    s=10
)

plt.title("Time-Series Anomaly Detection")

plt.xlabel("Sequence")

plt.ylabel("Anomaly Score")

plt.show()


# ============================================================
# SEMANTIC TOKENIZATION
# ============================================================

def generate_semantic_tokens(sample):

    tokens = []

    if sample["temperature"] > 1:
        tokens.append("[TEMP_HIGH]")

    elif sample["temperature"] < -1:
        tokens.append("[TEMP_LOW]")

    if sample["humidity"] > 1:
        tokens.append("[HUMIDITY_HIGH]")

    elif sample["humidity"] < -1:
        tokens.append("[HUMIDITY_LOW]")

    if sample["noise_db"] > 1:
        tokens.append("[NOISE_SPIKE]")

    if sample["motion_level"] > 1:
        tokens.append("[MOTION_ACTIVE]")

    if sample["light_intensity"] < -1:
        tokens.append("[LOW_LIGHT]")

    return tokens


# ============================================================
# ADD ANOMALY TOKEN
# ============================================================

def add_anomaly_token(tokens, anomaly_status):

    if anomaly_status == "ABNORMAL":

        tokens.append("[ANOMALY_ALERT]")

    return tokens


# ============================================================
# STANDARDIZED OUTPUT
# ============================================================

def standardize_output(
    prediction_label,
    confidence,
    semantic_tokens
):

    return {

        "schema_version": "2.0",

        "task_id": "smart_home_edge_ai",

        "timestamp": datetime.utcnow().isoformat(),

        "prediction": {

            "label": int(prediction_label),

            "confidence": float(
                round(confidence, 3)
            )
        },

        "semantic_tokens": semantic_tokens
    }


# ============================================================
# FINAL OUTPUT GENERATION
# ============================================================

final_outputs = []

for i in range(len(X_test)):

    sample = X_test.iloc[i]

    semantic_tokens = generate_semantic_tokens(
        sample
    )

    anomaly_status = (
        anomaly_labels[i]
        if i < len(anomaly_labels)
        else "NORMAL"
    )

    semantic_tokens = add_anomaly_token(
        semantic_tokens,
        anomaly_status
    )

    result = standardize_output(
        prediction_label=y_pred_xgb[i],
        confidence=y_confidence[i],
        semantic_tokens=semantic_tokens
    )

    final_outputs.append(result)


# ============================================================
# PRINT FINAL OUTPUT
# ============================================================

print("\nFINAL OUTPUT EXAMPLE\n")

print(
    json.dumps(
        final_outputs[0],
        indent=2
    )
)


# ============================================================
# SEMANTIC TOKEN DISTRIBUTION
# ============================================================

all_tokens = []

for output in final_outputs:

    all_tokens.extend(
        output["semantic_tokens"]
    )

token_counts = Counter(all_tokens)

plt.figure(figsize=(10, 5))

plt.bar(
    token_counts.keys(),
    token_counts.values()
)

plt.xticks(rotation=45)

plt.title("Semantic Sensor Token Distribution")

plt.ylabel("Frequency")

plt.show()


# ============================================================
# OFFLINE TRANSFORMER TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

semantic_text = " ".join(
    final_outputs[0]["semantic_tokens"]
)

tokens = tokenizer(semantic_text)

print("\nTOKENIZED OUTPUT\n")

print(tokens)


# ============================================================
# SAVE OUTPUTS
# ============================================================

with open(
    "semantic_outputs.json",
    "w"
) as f:

    json.dump(
        final_outputs,
        f,
        indent=2
    )

print("\nSemantic outputs saved successfully.")


# ============================================================
# SYSTEM SUMMARY
# ============================================================

print("""

========================================================
SMART HOME EDGE AI PIPELINE
========================================================

1. Smart-Home Sensor Stream
2. Preprocessing & Normalization
3. Hierarchical Clustering
4. XGBoost Classification
5. Time-Series Anomaly Detection
6. Semantic Sensor Tokenization
7. Offline Transformer-Compatible Encoding
8. Smart Home Decision / Alert

========================================================

""")

from transformers import pipeline
# ============================================================
# OFFLINE TINY TRANSFORMER REASONING
# ============================================================

print("\nRunning Offline Tiny Transformer...\n")

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)

semantic_input = " ".join(
    final_outputs[0]["semantic_tokens"]
)

prompt = f"""
Smart home semantic events:
{semantic_input}

Generate smart-home reasoning:
"""

result = generator(
    prompt,
    max_length=60,
    num_return_sequences=1
)

generated_reasoning = result[0]["generated_text"]

print("\nTRANSFORMER OUTPUT:\n")

print(generated_reasoning)

