"""
Orchestrator for CICIoT2023 feature selection targeting the grouped
8-class label (7 attack categories + Benign).

Steps:
  1. Stream train.csv, build a stratified sample capped per 8-class group.
  2. Drop columns already flagged REMOVE by create_feature_audit.py.
  3. Stratified train/val/test split -> saved to disk (canonical split,
     reused by mutual_information.py / rf_importance.py / rfe.py, and
     later by the RF training script).
  4. Call into mutual_information.py, rf_importance.py, rfe.py and combine
     their rankings into one comparison table.

Run create_feature_audit.py first -- this script reads its output to know
which columns to drop. mutual_information.py / rf_importance.py / rfe.py
can each also be run standalone afterward (they just reload the saved
split files) -- this script is what ties all three together.
"""

import pandas as pd
import zipfile
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mutual_information import compute_mi
from rf_importance import compute_rf_importance
from rfe import compute_rfe

ZIP_PATH = "data/raw/CICIoT2023.zip"
TRAIN_FILE = "train.csv"
AUDIT_PATH = "results/feature_audit.csv"
OUTPUT_DIR = "results"
LABEL_COL_OUT = "group_label"

CHUNK_SIZE = 500_000
MAX_ROWS_PER_GROUP = 150_000   # cap per 8-class group -> up to ~1.2M rows total
RANDOM_STATE = 42

LABEL_TO_GROUP = {
    "DDoS-RSTFINFlood": "DDoS", "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS", "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    "DoS-UDP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-HTTP_Flood": "DoS",
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
    "Recon-PingSweep": "Recon", "Recon-OSScan": "Recon", "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon", "Recon-HostDiscovery": "Recon",
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    "BrowserHijacking": "Web", "Backdoor_Malware": "Web", "XSS": "Web",
    "Uploading_Attack": "Web", "SqlInjection": "Web", "CommandInjection": "Web",
    "DictionaryBruteForce": "BruteForce",
    "BenignTraffic": "Benign",
}


def get_train_path(zip_file):
    return next(f for f in zip_file.namelist() if f.endswith(TRAIN_FILE))


def load_stratified_sample():
    """Stream through train.csv, capping rows per 8-class group, return one combined df."""
    audit = pd.read_csv(AUDIT_PATH, index_col=0)
    remove_cols = audit[
        (audit["initial_status"] == "REMOVE") & (~audit["is_label"])
    ].index.tolist()

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        train_path = get_train_path(zip_file)
        with zip_file.open(train_path) as f:
            header_cols = pd.read_csv(f, nrows=0).columns.tolist()
        label_col = next(c for c in header_cols if c.lower() == "label")
        dtype_map = {c: "float32" for c in header_cols if c != label_col}

        groups = sorted(set(LABEL_TO_GROUP.values()))
        group_frames = {g: [] for g in groups}
        group_counts = {g: 0 for g in groups}

        with zip_file.open(train_path) as f:
            reader = pd.read_csv(f, dtype=dtype_map, chunksize=CHUNK_SIZE)
            for i, chunk in enumerate(reader):
                chunk = chunk.copy()
                chunk[LABEL_COL_OUT] = chunk[label_col].map(LABEL_TO_GROUP)
                chunk = chunk.dropna(subset=[LABEL_COL_OUT])

                for group, sub in chunk.groupby(LABEL_COL_OUT):
                    remaining = MAX_ROWS_PER_GROUP - group_counts[group]
                    if remaining <= 0:
                        continue
                    take = sub.sample(n=min(len(sub), remaining), random_state=RANDOM_STATE)
                    group_frames[group].append(take)
                    group_counts[group] += len(take)

                print(f"  chunk {i + 1} | running per-group totals: {group_counts}")

                if all(c >= MAX_ROWS_PER_GROUP for c in group_counts.values()):
                    print("All group caps reached, stopping early.")
                    break

    non_empty = [pd.concat(frames, ignore_index=True) for frames in group_frames.values() if frames]
    sample_df = pd.concat(non_empty, ignore_index=True)
    sample_df = sample_df.drop(columns=[c for c in remove_cols if c in sample_df.columns])
    return sample_df, label_col


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading stratified sample from CICIoT2023 (streams the full file once)...")
    df, raw_label_col = load_stratified_sample()
    print(f"\nSample loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(df[LABEL_COL_OUT].value_counts())

    X = df.drop(columns=[raw_label_col, LABEL_COL_OUT])
    y_raw = df[LABEL_COL_OUT]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Save the canonical split -- mutual_information.py / rf_importance.py /
    # rfe.py (and later the RF training script) all reload these rather than
    # re-sampling/re-splitting independently.
    X_train.assign(**{LABEL_COL_OUT: le.inverse_transform(y_train)}).to_csv(
        f"{OUTPUT_DIR}/split_train.csv", index=False
    )
    X_val.assign(**{LABEL_COL_OUT: le.inverse_transform(y_val)}).to_csv(
        f"{OUTPUT_DIR}/split_val.csv", index=False
    )
    X_test.assign(**{LABEL_COL_OUT: le.inverse_transform(y_test)}).to_csv(
        f"{OUTPUT_DIR}/split_test.csv", index=False
    )
    print(f"\nSplit sizes -> train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")

    with open(f"{OUTPUT_DIR}/label_groups.json", "w") as f:
        json.dump(list(le.classes_), f, indent=2)

    # --- Run the three selection methods ---
    print("\nRunning Mutual Information...")
    mi_ranking = compute_mi(X_train, y_train)

    print("Running RF importance (permutation + impurity)...")
    perm_ranking, impurity_ranking, _ = compute_rf_importance(X_train, y_train, X_val, y_val)

    print("Running RFE...")
    rfe_result = compute_rfe(X_train, y_train)

    # --- Combine into one comparison table ---
    comparison = pd.DataFrame({
        "mi_score": mi_ranking.reindex(X_train.columns),
        "mi_rank": mi_ranking.rank(ascending=False).reindex(X_train.columns),
        "rf_perm_importance": perm_ranking.reindex(X_train.columns),
        "rf_perm_rank": perm_ranking.rank(ascending=False).reindex(X_train.columns),
        "rf_impurity_importance": impurity_ranking.reindex(X_train.columns),
        "rfe_selected": rfe_result["selected"].reindex(X_train.columns),
        "rfe_ranking": rfe_result["ranking"].reindex(X_train.columns),
    }, index=X_train.columns).sort_values("mi_rank")

    comparison.to_csv(f"{OUTPUT_DIR}/feature_selection_comparison.csv")

    print("\n" + "=" * 60)
    print("FEATURE SELECTION COMPLETE")
    print("=" * 60)
    print(f"\nComparison table saved to: {OUTPUT_DIR}/feature_selection_comparison.csv")
    print("\nTop 15 by Mutual Information:")
    print(mi_ranking.head(15))
    print("\nTop 15 by RF Permutation Importance:")
    print(perm_ranking.head(15))
    print("\nFeatures selected by RFE:")
    print(comparison[comparison["rfe_selected"]].index.tolist())


if __name__ == "__main__":
    main()
