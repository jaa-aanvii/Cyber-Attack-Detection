"""
Recursive Feature Elimination for CICIoT2023.

RFE retrains a model at every elimination step, so it's run on a smaller
stratified subsample of the training split rather than the full thing.

Standalone runnable (loads results/split_train.csv) or import
`compute_rfe` from select_features.py.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

RESULTS_DIR = "results"
LABEL_COL = "group_label"
RANDOM_STATE = 42
SUBSAMPLE_ROWS = 40_000
N_FEATURES_TO_SELECT = 15
STEP = 2
N_ESTIMATORS = 100
MAX_DEPTH = 15


def compute_rfe(X_train, y_train, n_features=N_FEATURES_TO_SELECT):
    """
    Subsamples X_train/y_train (stratified) down to SUBSAMPLE_ROWS, then runs
    RFE. Returns a DataFrame indexed by feature with 'selected' (bool) and
    'ranking' (1 = selected, higher = eliminated earlier).
    """
    if len(X_train) > SUBSAMPLE_ROWS:
        X_rfe, _, y_rfe, _ = train_test_split(
            X_train, y_train,
            train_size=SUBSAMPLE_ROWS,
            random_state=RANDOM_STATE, stratify=y_train,
        )
    else:
        X_rfe, y_rfe = X_train, y_train

    estimator = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=STEP)
    rfe.fit(X_rfe, y_rfe)

    return pd.DataFrame({
        "selected": rfe.support_,
        "ranking": rfe.ranking_,
    }, index=X_train.columns).sort_values("ranking")


def _load_train_split():
    df = pd.read_csv(f"{RESULTS_DIR}/split_train.csv")
    X_train = df.drop(columns=[LABEL_COL])
    y_train = df[LABEL_COL]
    return X_train, y_train


def main():
    X_train, y_train = _load_train_split()
    print(f"Loaded train split: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"Running RFE on a {min(SUBSAMPLE_ROWS, len(X_train)):,}-row stratified subsample...")

    rfe_result = compute_rfe(X_train, y_train)
    rfe_result.to_csv(f"{RESULTS_DIR}/rfe_ranking.csv")

    print(f"\nFeatures selected by RFE ({N_FEATURES_TO_SELECT}):")
    print(rfe_result[rfe_result["selected"]].index.tolist())
    print(f"\nSaved full ranking to: {RESULTS_DIR}/rfe_ranking.csv")


if __name__ == "__main__":
    main()
