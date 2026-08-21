import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# Hugging Face client
import os
from huggingface_hub import InferenceClient

# Load pipeline + RandomForest + label encoder
preprocessor = joblib.load("preprocessor_full.pkl")
rf_model = joblib.load("random_forest_multi_smote.pkl")
label_encoder = joblib.load("label_encoder_full.pkl")

# Derived feature engineering
def add_derived_features(df):
    df["byte_ratio"] = df["sbytes"] / (df["dbytes"] + 1)
    df["pkt_size_ratio"] = df["smean"] / (df["dmean"] + 1)
    df["traffic_rate"] = df["sload"] / (df["dload"] + 1)
    df["ttl_diff"] = df["sttl"] - df["dttl"]
    df["srv_dst_ratio"] = df["ct_srv_dst"] / (df["ct_dst_src_ltm"] + 1)
    return df

# Severity weights per category
severity_weights = {
    "Fuzzers": 65, "Reconnaissance": 70, "Normal": 0,
    "DoS": 90, "Exploits": 95, "Shellcode": 85,
    "Worms": 95, "Backdoor": 90, "Analysis": 60,
}

# Example per-class thresholds 
class_thresholds = {
    "Fuzzers": 0.45, "Reconnaissance": 0.5, "DoS": 0.6,
    "Exploits": 0.55, "Shellcode": 0.5, "Worms": 0.4,
    "Backdoor": 0.55, "Analysis": 0.45, "Normal": 0.9
}

# Hugging Face LLaMA‑3.1 client setup
client = InferenceClient("meta-llama/Llama-3.1-8B-Instruct", token=os.environ["HF_TOKEN"])

def ask_cyber_ai(question):
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": "You are a cybersecurity assistant. Only answer questions about cyberattacks. Answer clearly, concisely, and fully within 500 tokens."},
            {"role": "user", "content": question}
        ],
        max_tokens=500,
        temperature=0.2
    )
    return response.choices[0].message["content"]

# Streamlit UI
st.title("Cyber Attack Detection")
st.sidebar.header("Settings")

global_threshold = st.sidebar.slider("Global Severity Threshold", 0.0, 1.0, 0.7)

# Cyberattack Q&A sidebar
st.sidebar.subheader("Cyberattack Q&A")
user_question = st.sidebar.text_input("Ask about cyberattacks:")

if user_question:
    try:
        answer = ask_cyber_ai(user_question)
        st.sidebar.write("**Answer:**")
        st.sidebar.write(answer)
    except Exception as e:
        st.sidebar.error(f"AI Q&A unavailable: {e}")

uploaded_file = st.file_uploader("Upload testing set (CSV)", type="csv")

if uploaded_file:
    raw_data = pd.read_csv(uploaded_file)

    # Derived features
    raw_data = add_derived_features(raw_data)

    # Separate features and target
    X = raw_data.drop(columns=["label", "attack_cat"], errors="ignore")
    y = raw_data["attack_cat"] if "attack_cat" in raw_data.columns else None

    # Preprocess for RandomForest
    X_proc = preprocessor.transform(X)

    # Predictions
    rf_proba = rf_model.predict_proba(X_proc)
    preds = []
    probs = []

    for prob in rf_proba:
        pred_idx = np.argmax(prob)
        pred_class = label_encoder.classes_[pred_idx]
        confidence = prob[pred_idx]

        # Apply per-class threshold
        min_thresh = class_thresholds.get(pred_class, global_threshold)
        if confidence < min_thresh:
            pred_class = "Normal"
        preds.append(pred_class)
        probs.append(confidence)

    
    # Streaming predictions
    st.write("Streaming predictions...")
    for pred, prob in zip(preds, probs):
        base_severity = severity_weights.get(pred, 0)
        final_score = (base_severity / 100.0) * prob

        st.write(f"Prediction: {pred}, Severity Score: {final_score:.2f}")
        if final_score > global_threshold:
            st.error(f"🚨 ALERT: {pred} attack detected with severity {final_score:.2f}")
        time.sleep(0.5)

    # Evaluation if labels exist
    if y is not None:
        st.subheader("Evaluation Metrics")
        st.text("Classification Report:")
        st.text(classification_report(y, preds, zero_division=0))

        cm = confusion_matrix(y, preds, labels=label_encoder.classes_)
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_, ax=ax)
        st.pyplot(fig)
    else:
        st.write("No attack_cat column found — predictions only.")
