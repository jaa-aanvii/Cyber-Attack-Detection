
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ==================================================
# 1. PATHS AND SELECTED FEATURES
# ==================================================

CONFIG_PATH = Path("config/feature_schema.json")
RESULTS_DIR = Path("results")
LABEL_GROUPS_PATH = RESULTS_DIR / "label_groups.json"

# IMPORTANT:
# Set this to the directory containing your PREPROCESSED CSVs.
# This default searches CSVs inside results/ recursively.
PREPROCESSED_ROOT = RESULTS_DIR

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "IAT",
    "Max",
    "Tot size",
    "Magnitue",
    "AVG",
    "Tot sum",
    "Min",
    "Header_Length",
    "Protocol Type",
    "rst_count",
    "Variance",
    "urg_count",
    "flow_duration",
    "Rate",
    "Srate",
]

RANDOM_STATE = 42
MAX_ROWS_PER_FILE = 5000  # Increase for a larger training run.

# ==================================================
# 2. LOAD LABEL MAPPING
# ==================================================


def load_label_groups():
    if not LABEL_GROUPS_PATH.exists():
        raise FileNotFoundError(
            f"Label mapping not found: {LABEL_GROUPS_PATH}"
        )

    with open(LABEL_GROUPS_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    if isinstance(mapping, dict) and "label_groups" in mapping:
        mapping = mapping["label_groups"]

    if isinstance(mapping, list):
        return mapping

    if isinstance(mapping, dict):
        return mapping

    raise ValueError(
        "label_groups.json must contain a list or a JSON object."
    )



# ==================================================
# 3. LOAD PREPROCESSED DATA
# ==================================================

def main():
    label_groups = load_label_groups()

    if not PREPROCESSED_ROOT.exists():
        raise FileNotFoundError(
            f"Preprocessed data directory not found: {PREPROCESSED_ROOT}"
        )

    print("Loading preprocessed dataset...", flush=True)
    print("Data directory:", PREPROCESSED_ROOT)
    print("Selected feature count:", len(FEATURES))

    frames = []
    skipped_files = 0

    # Skip our generated evaluation CSVs if rerunning the script.
    excluded_files = {
        "class_distribution.csv",
        "validation_classification_report.csv",
        "validation_confusion_matrix.csv",
    }

    for file in sorted(PREPROCESSED_ROOT.rglob("*.csv")):
        if file.name in excluded_files:
            continue

        try:
            df = pd.read_csv(
                file,
                nrows=MAX_ROWS_PER_FILE,
                low_memory=False
            )
            df.columns = df.columns.str.strip()

            # All 15 selected features must exist.
            missing = [col for col in FEATURES if col not in df.columns]

            if missing:
                skipped_files += 1
                print(
                    f"Skipped {file}: missing columns {missing}",
                    flush=True
                )
                continue

            # Prefer a label column if one is already present.
            existing_label_col = next(
                (
                    col for col in ["group_label", "label", "Label"]
                    if col in df.columns
                ),
                None
            )

            if existing_label_col:
                df["target"] = df[existing_label_col].astype(str)
            else:
                # Otherwise derive the label from the folder name.
                folder_label = label_groups.get(file.parent.name)

                if folder_label is None:
                    skipped_files += 1
                    print(
                        f"Skipped {file}: no label column and folder "
                        f"'{file.parent.name}' is not in label_groups.",
                        flush=True
                    )
                    continue

                df["target"] = folder_label

            # Retain exactly the selected features and target.
            df = df[FEATURES + ["target"]].copy()

            for col in FEATURES:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=FEATURES + ["target"], inplace=True)

            if not df.empty:
                frames.append(df)
                print(
                    f"Loaded {file.name}: {len(df)} rows",
                    flush=True
                )

        except Exception as e:
            skipped_files += 1
            print(f"Could not read {file}: {e}", flush=True)

    if not frames:
        raise RuntimeError(
            "No usable preprocessed CSVs were loaded. Check the "
            "preprocessed folder, feature headers, and label mapping."
        )

    data = pd.concat(frames, ignore_index=True)

    print("\nTotal rows loaded:", len(data))
    print("Skipped files:", skipped_files)

    # ==================================================
    # 4. CLASS IMBALANCE CHECK
    # ==================================================

    print("\n========== CLASS IMBALANCE ==========")

    counts = data["target"].value_counts()
    percentages = data["target"].value_counts(normalize=True) * 100

    distribution = pd.DataFrame({
        "count": counts,
        "percentage": percentages.round(2),
    })

    print(distribution.to_string())

    largest = counts.max()
    smallest = counts.min()
    imbalance_ratio = largest / smallest

    print("\nLargest class:", largest)
    print("Smallest class:", smallest)
    print("Largest/smallest ratio:", round(imbalance_ratio, 2))

    if imbalance_ratio >= 3:
        print("Potential substantial class imbalance (ratio >= 3:1).")
    else:
        print("No substantial imbalance by the 3:1 screening threshold.")

    distribution.to_csv(RESULTS_DIR / "class_distribution.csv")

    # Every class needs enough examples for stratified splitting.
    if counts.min() < 3:
        raise ValueError(
            "At least one class has fewer than 3 rows. "
            "More examples are needed for train/validation/test splits."
        )

    # ==================================================
    # 5. TRAIN / VALIDATION / TEST SPLIT
    # ==================================================

    X = data[FEATURES]
    y = data["target"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
        stratify=y_trainval
    )

    print("\nTrain rows:", len(X_train))
    print("Validation rows:", len(X_val))
    print("Test rows:", len(X_test))

    print("\nTraining class counts:")
    print(y_train.value_counts().to_string())

    # ==================================================
    # 6. APPLY SMOTE TO TRAINING DATA ONLY
    # ==================================================

    print("\nClass counts BEFORE SMOTE:")
    print(y_train.value_counts().to_string())

    smote = SMOTE(
        sampling_strategy="not majority",
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )

    print("\nApplying SMOTE to training data...", flush=True)

    start = time.time()
    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train,
        y_train,
    )
    smote_time = time.time() - start

    print("\nClass counts AFTER SMOTE:")
    print(y_train_resampled.value_counts().to_string())
    print("SMOTE time (seconds):", round(smote_time, 2))

    # ==================================================
    # 7. TRAIN RANDOM FOREST ON RESAMPLED DATA
    # ==================================================

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight=None,
    )

    print("\nTraining Random Forest with SMOTE...", flush=True)

    start = time.time()
    model.fit(X_train_resampled, y_train_resampled)
    training_time = time.time() - start

    # ==================================================
    # 7. VALIDATION METRICS
    # ==================================================

    start = time.time()
    val_pred = model.predict(X_val)
    inference_time = time.time() - start

    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(
        y_val, val_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_val, val_pred, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        y_val, val_pred, average="macro", zero_division=0
    )

    print("\n========== VALIDATION RESULTS ==========")
    print("Accuracy:", round(accuracy, 4))
    print("Macro precision:", round(precision, 4))
    print("Macro recall:", round(recall, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Training time (seconds):", round(training_time, 2))
    print("Inference time (seconds):", round(inference_time, 2))

    print("\nPer-class report:")
    print(classification_report(y_val, val_pred, zero_division=0))

    labels = model.classes_
    cm = confusion_matrix(y_val, val_pred, labels=labels)

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # Multiclass one-vs-rest false positive rate per class.
    fpr = {}

    for i, label in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        fpr[str(label)] = (
            float(fp / (fp + tn)) if (fp + tn) else 0.0
        )

    print("\nPer-class false positive rates:")
    print(fpr)

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        RESULTS_DIR / "validation_confusion_matrix.csv"
    )

    report = classification_report(
        y_val, val_pred, zero_division=0, output_dict=True
    )
    pd.DataFrame(report).transpose().to_csv(
        RESULTS_DIR / "validation_classification_report.csv"
    )

    with open(
        RESULTS_DIR / "validation_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump({
            "accuracy": accuracy,
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": macro_f1,
            "training_time_seconds": training_time,
            "validation_inference_time_seconds": inference_time,
            "per_class_false_positive_rates": fpr,
            "features": FEATURES,
        }, f, indent=4)

    # ==================================================
    # 8. FINAL TEST METRICS
    # ==================================================

    test_pred = model.predict(X_test)

    print("\n========== HELD-OUT TEST RESULTS ==========")
    print("Test accuracy:", round(accuracy_score(y_test, test_pred), 4))
    print("Test macro F1:", round(
        f1_score(y_test, test_pred, average="macro", zero_division=0),
        4
    ))

    # ==================================================
    # 9. SAVE MODEL
    # ==================================================

    model_path = OUTPUT_DIR / "random_forest_smote.joblib"
    joblib.dump(model, model_path)

    print("\nModel saved to:", model_path)
    print("Reports saved to:", RESULTS_DIR)
    print("Class distribution saved to results/class_distribution.csv")
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
