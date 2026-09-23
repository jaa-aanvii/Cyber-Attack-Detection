"""
RF-based feature importance for CICIoT2023 -- both permutation importance
(computed on a held-out validation split, more trustworthy) and impurity
importance (biased toward high-cardinality numeric columns like IAT/Rate/
Magnitue, kept only as a secondary reference).

Standalone runnable (loads results/split_train.csv + split_val.csv) or
import `compute_rf_importance` from select_features.py.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

RESULTS_DIR = "results"
LABEL_COL = "group_label"
RANDOM_STATE = 42
N_ESTIMATORS = 150
MAX_DEPTH = 20
N_REPEATS = 5


def compute_rf_importance(X_train, y_train, X_val, y_val):
    """
    Trains a probe RF (not the final tuned classifier -- just for scoring)
    and returns (perm_importance_series, impurity_importance_series, probe_model).
    """
    probe_rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    probe_rf.fit(X_train, y_train)

    perm_result = permutation_importance(
        probe_rf, X_val, y_val,
        n_repeats=N_REPEATS, random_state=RANDOM_STATE, n_jobs=-1,
    )
    perm_ranking = pd.Series(
        perm_result.importances_mean, index=X_train.columns, name="rf_perm_importance"
    ).sort_values(ascending=False)

    impurity_ranking = pd.Series(
        probe_rf.feature_importances_, index=X_train.columns, name="rf_impurity_importance"
    ).sort_values(ascending=False)

    return perm_ranking, impurity_ranking, probe_rf


def _load_splits():
    train_df = pd.read_csv(f"{RESULTS_DIR}/split_train.csv")
    val_df = pd.read_csv(f"{RESULTS_DIR}/split_val.csv")
    X_train, y_train = train_df.drop(columns=[LABEL_COL]), train_df[LABEL_COL]
    X_val, y_val = val_df.drop(columns=[LABEL_COL]), val_df[LABEL_COL]
    return X_train, y_train, X_val, y_val


def main():
    X_train, y_train, X_val, y_val = _load_splits()
    print(f"Loaded train split: {X_train.shape[0]:,} rows | val split: {X_val.shape[0]:,} rows")

    print("Training probe RF...")
    perm_ranking, impurity_ranking, _ = compute_rf_importance(X_train, y_train, X_val, y_val)

    perm_ranking.to_csv(f"{RESULTS_DIR}/rf_perm_importance.csv", header=True)
    impurity_ranking.to_csv(f"{RESULTS_DIR}/rf_impurity_importance.csv", header=True)

    print("\nTop 20 by permutation importance (primary ranking):")
    print(perm_ranking.head(20))
    print("\nTop 20 by impurity importance (reference only, biased toward high-cardinality cols):")
    print(impurity_ranking.head(20))
    print(f"\nSaved to: {RESULTS_DIR}/rf_perm_importance.csv, rf_impurity_importance.csv")


if __name__ == "__main__":
    main()
