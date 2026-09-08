import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# Load dataset
df = pd.read_csv("data/UNSW_NB15_training-set.csv")

# Separate features (drop target + binary label)
X = df.drop(columns=["label", "attack_cat"])

# Derived features
X["byte_ratio"] = X["sbytes"] / (X["dbytes"] + 1)
X["pkt_size_ratio"] = X["smean"] / (X["dmean"] + 1)
X["traffic_rate"] = X["sload"] / (X["dload"] + 1)
X["ttl_diff"] = X["sttl"] - X["dttl"]
X["srv_dst_ratio"] = X["ct_srv_dst"] / (X["ct_dst_src_ltm"] + 1)

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

# Fit preprocessor on full dataset
preprocessor.fit(X)

# Save preprocessor for reuse
joblib.dump(preprocessor, "models/preprocessor_full.pkl")
print("Preprocessor saved successfully!")


# In[3]:


from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd

# Target column
y = df["attack_cat"]

# Encode target labels into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder for reuse
joblib.dump(label_encoder, "models/label_encoder_full.pkl")
print("Label encoder saved successfully!")

