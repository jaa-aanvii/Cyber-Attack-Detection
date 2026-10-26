"""Prepare the extracted CICIoT2023 folder dataset for model training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SCHEMA = Path("config/feature_schema.json")
DEFAULT_OUTPUT = Path("data/processed")
CHUNK_SIZE = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_schema(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        schema = json.load(file)
    features = schema.get("features")
    if not isinstance(features, list) or len(features) != 15:
        raise ValueError("The schema must contain exactly 15 features.")
    return schema


def resolve_group(folder_name: str, label_groups: dict[str, str]) -> str:
    normalized = {key.casefold(): value for key, value in label_groups.items()}
    try:
        return normalized[folder_name.casefold()]
    except KeyError as error:
        raise ValueError(f"No label mapping exists for folder: {folder_name}") from error


def append_frame(frame: pd.DataFrame, output_path: Path, first_write: bool) -> bool:
    frame.to_csv(
        output_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )
    return False


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    dataset_root = (args.dataset_root or Path(schema["dataset_root"])).expanduser()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")

    features = schema["features"]
    label_groups = schema["label_groups"]
    split = schema["split"]
    rng = np.random.default_rng(split["random_state"])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_dir / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    for path in output_paths.values():
        if path.exists():
            path.unlink()

    csv_files = sorted(dataset_root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found below {dataset_root}")

    first_write = {name: True for name in output_paths}
    expected_columns = set(features)
    processed_rows = 0

    for csv_path in csv_files:
        group_label = resolve_group(csv_path.parent.name, label_groups)
        header = pd.read_csv(csv_path, nrows=0)
        missing = expected_columns.difference(header.columns)
        if missing:
            raise ValueError(
                f"{csv_path} is missing schema columns: {sorted(missing)}"
            )

        for chunk in pd.read_csv(csv_path, usecols=features, chunksize=CHUNK_SIZE):
            chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
            if chunk.empty:
                continue
            chunk["group_label"] = group_label
            random_values = rng.random(len(chunk))
            train_end = split["train"]
            validation_end = train_end + split["validation"]
            masks = {
                "train": random_values < train_end,
                "validation": (random_values >= train_end)
                & (random_values < validation_end),
                "test": random_values >= validation_end,
            }
            for name, mask in masks.items():
                if mask.any():
                    first_write[name] = append_frame(
                        chunk.loc[mask],
                        output_paths[name],
                        first_write[name],
                    )
            processed_rows += len(chunk)

        print(f"Processed: {csv_path}")

    if processed_rows == 0:
        raise ValueError("No usable rows were found in the dataset.")
    print(f"Processed rows: {processed_rows:,}")
    for name, path in output_paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
