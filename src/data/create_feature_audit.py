"""
Full-dataset feature audit for CICIoT2023.

Unlike the original version (which only read the first 10,000 rows), this
streams through the entire train.csv in chunks so 'constant' / missing /
label-distribution stats reflect the whole dataset, not just whatever
happened to be at the top of the file.

Memory-safe: never holds more than one chunk + a small running sample in
memory at once. Only exact stats needed for 'constant' detection (min==max)
are computed over the FULL dataset; 'unique value count' is reported as an
approximation from a running sample (exact nunique over millions of float
rows isn't worth the memory it costs).
"""

import pandas as pd
import zipfile
import os
from collections import defaultdict

ZIP_PATH ="data/raw/CICIoT2023.zip"
TRAIN_FILE = "train.csv"
OUTPUT_AUDIT = "results/feature_audit.csv"
OUTPUT_LABEL_DIST = "results/label_distribution.csv"

CHUNK_SIZE = 500_000          # rows per chunk while streaming
SAMPLE_TARGET_ROWS = 200_000  # rows kept for approx-unique-value estimate

# Official CICIoT2023 fine-grained label -> 8-class grouping
# (7 attack categories + Benign). Update this if your dataset's exact
# label strings differ (the script will warn you about any it can't map).
LABEL_TO_GROUP = {
    # DDoS
    "DDoS-RSTFINFlood": "DDoS", "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS", "DDoS-ICMP_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    # DoS
    "DoS-UDP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS", "DoS-HTTP_Flood": "DoS",
    # Mirai
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai", "Mirai-udpplain": "Mirai",
    # Recon
    "Recon-PingSweep": "Recon", "Recon-OSScan": "Recon", "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon", "Recon-HostDiscovery": "Recon",
    # Spoofing
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    # Web-based
    "BrowserHijacking": "Web", "Backdoor_Malware": "Web", "XSS": "Web",
    "Uploading_Attack": "Web", "SqlInjection": "Web", "CommandInjection": "Web",
    # Brute Force
    "DictionaryBruteForce": "BruteForce",
    # Benign
    "BenignTraffic": "Benign",
}


def get_train_path(zip_file):
    return next(f for f in zip_file.namelist() if f.endswith(TRAIN_FILE))


def main():
    os.makedirs("results", exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        train_path = get_train_path(zip_file)
        print(f"Reading: {train_path}")

        with zip_file.open(train_path) as f:
            header_cols = pd.read_csv(f, nrows=0).columns.tolist()

        label_col = next(c for c in header_cols if c.lower() == "label")
        dtype_map = {c: "float32" for c in header_cols if c != label_col}

        total_rows = 0
        missing_count = pd.Series(0, index=header_cols, dtype="int64")
        col_min, col_max = {}, {}
        label_counts_fine = defaultdict(int)
        label_counts_group = defaultdict(int)
        unmapped_labels = set()
        sample_frames = []
        sample_rows_so_far = 0

        with zip_file.open(train_path) as f:
            reader = pd.read_csv(f, dtype=dtype_map, chunksize=CHUNK_SIZE)
            for i, chunk in enumerate(reader):
                total_rows += len(chunk)
                missing_count += chunk.isnull().sum()

                for col in header_cols:
                    if col == label_col:
                        continue
                    cmin, cmax = chunk[col].min(), chunk[col].max()
                    col_min[col] = cmin if col not in col_min else min(col_min[col], cmin)
                    col_max[col] = cmax if col not in col_max else max(col_max[col], cmax)

                vc = chunk[label_col].value_counts()
                for label, count in vc.items():
                    label_counts_fine[label] += int(count)
                    group = LABEL_TO_GROUP.get(label)
                    if group is None:
                        unmapped_labels.add(label)
                        group = "UNMAPPED"
                    label_counts_group[group] += int(count)

                if sample_rows_so_far < SAMPLE_TARGET_ROWS:
                    take = min(len(chunk), 20_000)
                    sample_frames.append(chunk.sample(n=take, random_state=42))
                    sample_rows_so_far += take

                print(f"  chunk {i + 1} processed | cumulative rows: {total_rows:,}")

    if unmapped_labels:
        print("\nWARNING: labels found that aren't in LABEL_TO_GROUP:")
        for lbl in sorted(unmapped_labels):
            print(f"  - {lbl}")
        print("Fix the mapping in this script before running feature selection.\n")

    sample_df = pd.concat(sample_frames, ignore_index=True) if sample_frames else pd.DataFrame(columns=header_cols)

    audit_rows = []
    for col in header_cols:
        is_label = col == label_col
        missing = int(missing_count[col])
        row = {
            "feature": col,
            "data_type": "str" if is_label else "float32",
            "missing_count": missing,
            "missing_percentage": round(missing / total_rows * 100, 4) if total_rows else 0.0,
            "is_label": is_label,
            "approx_unique_values": sample_df[col].nunique(dropna=False) if col in sample_df.columns else None,
        }
        if is_label:
            row["constant"] = False
        else:
            row["constant"] = bool(col in col_min and col_min[col] == col_max[col])
        row["initial_status"] = "REMOVE" if (is_label or row["constant"]) else "INVESTIGATE"
        audit_rows.append(row)

    audit = pd.DataFrame(audit_rows).set_index("feature")
    audit.to_csv(OUTPUT_AUDIT)

    label_dist = pd.DataFrame({
        "fine_label": list(label_counts_fine.keys()),
        "count": list(label_counts_fine.values()),
    }).sort_values("count", ascending=False)
    label_dist["group"] = label_dist["fine_label"].map(LABEL_TO_GROUP).fillna("UNMAPPED")
    label_dist.to_csv(OUTPUT_LABEL_DIST, index=False)

    group_dist = pd.Series(label_counts_group).sort_values(ascending=False)

    print("=" * 60)
    print("FULL-DATASET FEATURE AUDIT COMPLETE")
    print("=" * 60)
    print(f"\nTotal rows scanned: {total_rows:,}")
    print(f"Total features: {len(header_cols)}")
    print(f"\nInitial status:\n{audit['initial_status'].value_counts()}")
    print(f"\n8-class group distribution (full dataset):\n{group_dist}")
    print(f"\nFull audit saved to: {OUTPUT_AUDIT}")
    print(f"Label distribution saved to: {OUTPUT_LABEL_DIST}")


if __name__ == "__main__":
    main()
