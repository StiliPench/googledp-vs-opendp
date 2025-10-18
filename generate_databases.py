import numpy as np
import pandas as pd
from pathlib import Path

TARGET_DIRS = [
    Path("googledp_cpp"),
    Path("opendp_rust"),
]

DATASETS = [
    {"name": "Small (10k rows)",   "file": "small_dataset.csv",  "rows": 10_000},
    {"name": "Medium (100k rows)", "file": "medium_dataset.csv", "rows": 100_000},
    {"name": "Large (1M rows)",    "file": "large_dataset.csv",  "rows": 1_000_000},
]

NUM_NUMERIC_COLS = 5
NUMERIC_RANGE = (-100.0, 100.0)
RANDOM_STATE = 42

def ensure_dirs(dirs):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def dataset_exists_everywhere(fname: str) -> bool:
    return all((d / fname).exists() for d in TARGET_DIRS)

def save_to_targets(df: pd.DataFrame, filename: str):
    for d in TARGET_DIRS:
        out = d / filename
        if out.exists():
            print(f"[skip] {out} already exists")
        else:
            df.to_csv(out, index=False)
            print(f"[ok]   wrote {out}")

def generate_df(n_rows: int, n_cols: int, value_range: tuple[float, float], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"value_col_{i+1}": rng.uniform(value_range[0], value_range[1], n_rows).astype(float)
            for i in range(n_cols)}
    return pd.DataFrame(data)

def main():
    ensure_dirs(TARGET_DIRS)
    print("Checking datasets…")
    for ds in DATASETS:
        fname = ds["file"]
        rows = ds["rows"]
        if dataset_exists_everywhere(fname):
            print(f"[done] {fname} already present in all target dirs")
            continue

        print(f"Generating '{fname}' with {rows} rows × {NUM_NUMERIC_COLS} cols")
        df = generate_df(rows, NUM_NUMERIC_COLS, NUMERIC_RANGE, RANDOM_STATE)
        save_to_targets(df, fname)

    print("All done.")

if __name__ == "__main__":
    main()
