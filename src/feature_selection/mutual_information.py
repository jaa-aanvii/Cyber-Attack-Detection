"""
Mutual Information scoring for CICIoT2023 feature selection.

Standalone runnable (loads results/split_train.csv directly) or import
`compute_mi` from select_features.py to reuse an already-loaded X_train/y_train.
"""

import pandas as pd
from sklearn.feature_selection import mutual_info_classif

RESULTS_DIR = "results"
LABEL_COL = "group_label"
RANDOM_STATE = 42


def compute_mi(X_train, y_train):
    """Return a Series of MI scores per feature, sorted descending."""
    scores = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    return pd.Series(scores, index=X_train.columns, name="mi_score").sort_values(ascending=False)


def _load_train_split():
    df = pd.read_csv(f"{RESULTS_DIR}/split_train.csv")
    X_train = df.drop(columns=[LABEL_COL])
    y_train = df[LABEL_COL]
    return X_train, y_train


def main():
    X_train, y_train = _load_train_split()
    print(f"Loaded train split: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")

    mi_ranking = compute_mi(X_train, y_train)
    mi_ranking.to_csv(f"{RESULTS_DIR}/mi_ranking.csv", header=True)

    print("\nTop 20 features by Mutual Information:")
    print(mi_ranking.head(20))
    print(f"\nSaved full ranking to: {RESULTS_DIR}/mi_ranking.csv")


if __name__ == "__main__":
    main()
