import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("data/UNSW_NB15_training-set.csv")

# Separate features and target
X = df.drop(columns=["label", "attack_cat"])
y = df["attack_cat"]

# Separate features and target
X = df.drop(columns=["label", "attack_cat"])
y = df["attack_cat"]

# Derived features
X["byte_ratio"] = X["sbytes"] / (X["dbytes"] + 1)
X["pkt_size_ratio"] = X["smean"] / (X["dmean"] + 1)
X["traffic_rate"] = X["sload"] / (X["dload"] + 1)
X["ttl_diff"] = X["sttl"] - X["dttl"]
X["srv_dst_ratio"] = X["ct_srv_dst"] / (X["ct_dst_src_ltm"] + 1)

# Load existing label encoder + preprocessor
label_encoder = joblib.load("label_encoder_full.pkl")
preprocessor = joblib.load("preprocessor_full.pkl")

# Encode target labels
y_encoded = label_encoder.transform(y)


# In[3]:


# Split into train, val, test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Preprocess with existing preprocessor
X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

# Convert sparse to dense for SMOTE
X_train_proc = X_train_proc.toarray()


# In[4]:


# Apply SMOTE (lighter strategy)
smote = SMOTE(random_state=42, sampling_strategy="not majority")
X_train_res, y_train_res = smote.fit_resample(X_train_proc, y_train)

print("Before SMOTE:\n", pd.Series(y_train).value_counts())
print("\nAfter SMOTE:\n", pd.Series(y_train_res).value_counts())


# In[5]:


# RandomForest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

# Train
rf_model.fit(X_train_res, y_train_res)

# Save model
joblib.dump(rf_model, "models/random_forest_multi_smote.pkl", compress=3)
print("RandomForest trained with SMOTE and saved successfully!")


# In[ ]:


# Predict on processed test set
y_pred_encoded = rf_model.predict(X_test_proc)
y_pred = label_encoder.inverse_transform(y_pred_encoded)

# Convert y_test back to original labels for comparison
y_test_labels = label_encoder.inverse_transform(y_test)

print("\nTest Accuracy:", accuracy_score(y_test_labels, y_pred))
print("Test Macro F1:", f1_score(y_test_labels, y_pred, average="macro", zero_division=0))
print("\nClassification Report:\n", classification_report(y_test_labels, y_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (RandomForest with SMOTE)")
plt.show()

