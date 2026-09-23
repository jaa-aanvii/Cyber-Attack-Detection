"""
Exploratory look at CICIoT2023's train.csv.

This is NOT the formal audit (see create_feature_audit.py, which streams
the *entire* dataset and saves persistent output the rest of the pipeline
depends on). This script is for a quick, console-only sanity check on a
decent-sized sample before running anything full-scale: shape, dtypes,
duplicate features/rows, and whether the raw label strings actually match
LABEL_TO_GROUP (used in select_features.py) -- so you don't burn time
streaming the full file only to get a pile of UNMAPPED warnings.

Sample is drawn from across the WHOLE file (not just the first N rows)
via lightweight per-chunk sampling, since train.csv isn't guaranteed to
be shuffled -- the first 10k rows could all be from one capture session.
"""

import pandas as pd
import zipfile

ZIP_PATH = "data/raw/CICIoT2023.zip"
TRAIN_FILE = "train.csv"
SAMPLE_SIZE = 50_000     # total rows to inspect, spread across the file
CHUNK_SIZE = 200_000     # rows read per chunk while sampling
RANDOM_STATE = 42

# Same mapping used in select_features.py / create_feature_audit.py --
# kept here too so this script can independently sanity-check raw label
# strings before anything downstream runs.
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


def load_spread_sample():
    """Sample spread across the whole file, not just the top."""
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        train_path = get_train_path(zip_file)
        print(f"Reading: {train_path}")

        frames = []
        rows_collected = 0

        with zip_file.open(train_path) as f:
            reader = pd.read_csv(f, chunksize=CHUNK_SIZE)
            for chunk in reader:
                if rows_collected >= SAMPLE_SIZE:
                    break
                remaining = SAMPLE_SIZE - rows_collected
                take_n = min(len(chunk), max(1, remaining // 4))  # spread across several chunks
                frames.append(chunk.sample(n=take_n, random_state=RANDOM_STATE))
                rows_collected += take_n

    return pd.concat(frames, ignore_index=True)


def main():
    df = load_spread_sample()

    print("=" * 60)
    print("CICIoT2023 DATASET INSPECTION (spread sample)")
    print("=" * 60)

    print("\nShape of loaded sample:")
    print(df.shape)

    print("\nColumn names:")
    for i, column in enumerate(df.columns, start=1):
        print(f"{i}. {column}")

    print("\nData types:")
    print(df.dtypes)

    # Missing values
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    missing_report = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage,
    })
    print("\nMissing Value Report:")
    print(missing_report[missing_report["missing_count"] > 0])

    # Constant values (within THIS sample only -- create_feature_audit.py
    # confirms this properly across the full dataset; treat this as a hint)
    constant_features = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print("\nConstant Features (in this sample -- verify against the full audit):")
    print(constant_features if constant_features else "None found.")

    print("\nNumber of unique values per column:")
    print(df.nunique())

    print("\nFirst 5 rows:")
    print(df.head())

    # Duplicate feature pairs -- columns identical to EACH OTHER.
    # Different from a constant column: two columns can both vary and
    # still always match each other (redundant, not useless).
    print("\nDuplicate Feature Pairs:")
    duplicate_features = []
    columns = df.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if df[columns[i]].equals(df[columns[j]]):
                duplicate_features.append((columns[i], columns[j]))
    if duplicate_features:
        for f1, f2 in duplicate_features:
            print(f"{f1} == {f2}")
    else:
        print("None found.")

    # Duplicate rows
    duplicate_rows = df.duplicated().sum()
    print("\nDuplicate Rows:")
    print(duplicate_rows)
    print(f"Duplicate row percentage: {(duplicate_rows / len(df)) * 100:.2f}%")

    # Label sanity check against LABEL_TO_GROUP (used later in select_features.py)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col:
        raw_labels = set(df[label_col].unique())
        unmapped = raw_labels - LABEL_TO_GROUP.keys()

        print(f"\nUnique label values found in sample ({len(raw_labels)}):")
        for lbl in sorted(raw_labels):
            print(f"  {lbl}  ->  {LABEL_TO_GROUP.get(lbl, 'UNMAPPED')}")

        if unmapped:
            print(f"\nWARNING: {len(unmapped)} label(s) not in LABEL_TO_GROUP:")
            for lbl in sorted(unmapped):
                print(f"  - {lbl}")
            print("Fix LABEL_TO_GROUP in select_features.py / create_feature_audit.py "
                  "before running the full pipeline.")
        else:
            print("\nAll labels in this sample are mapped. (A larger/full run "
                  "may still surface labels not seen in this sample.)")
    else:
        print("\nWARNING: no 'label' column found.")


if __name__ == "__main__":
    main()
