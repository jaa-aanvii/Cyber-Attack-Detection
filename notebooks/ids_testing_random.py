# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# -----------------------------
# Load UNSW testing set
# -----------------------------
df_test = pd.read_csv("data/UNSW_NB15_testing-set.csv")

# Separate features and target
X_test = df_test.drop(columns=["label", "attack_cat"], errors="ignore")
y_test = df_test["attack_cat"]

# Derived features (must match training!)
X_test["byte_ratio"] = X_test["sbytes"] / (X_test["dbytes"] + 1)
X_test["pkt_size_ratio"] = X_test["smean"] / (X_test["dmean"] + 1)
X_test["traffic_rate"] = X_test["sload"] / (X_test["dload"] + 1)
X_test["ttl_diff"] = X_test["sttl"] - X_test["dttl"]
X_test["srv_dst_ratio"] = X_test["ct_srv_dst"] / (X_test["ct_dst_src_ltm"] + 1)

# -----------------------------
# Load preprocessor + label encoder + trained model
# -----------------------------
preprocessor = joblib.load("preprocessor_full.pkl")
label_encoder = joblib.load("label_encoder_full.pkl")
rf_model = joblib.load("random_forest_multi_smote.pkl")

# -----------------------------
# Preprocess test set
# -----------------------------
X_test_proc = preprocessor.transform(X_test)

# -----------------------------
# Predictions
# -----------------------------
y_pred_encoded = rf_model.predict(X_test_proc)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# -----------------------------
# Evaluation
# -----------------------------
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Test Macro F1:", f1_score(y_test, y_pred, average="macro", zero_division=0))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (RandomForest Baseline)")
plt.show()

