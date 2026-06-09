"""
merge_csv.py
------------
Merges all CSV files produced by ulg_to_csv_20hz.py (located in csv/)
into a single merged_data.csv.

Between consecutive files, 1 second of zero-padding rows is inserted
(at the same 50 ms tick rate) so flight segments are visually separated
when plotted.

The output timestamp is continuous — each file's timestamps are shifted
so they follow on from the previous segment + the 1 s gap.

Usage:
    python merge_csv.py
    python merge_csv.py --input csv/ --rate 20
"""

import sys
import argparse
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit(
        "ERROR: pandas / numpy is not installed.\n"
        "Install with:  pip install pandas numpy"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GAP_DURATION_S = 10.0        # seconds of zeros inserted between files
OUTPUT_FILENAME = "merged_data.csv"


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV produced by ulg_to_csv_20hz.py, indexed by timestamp_s."""
    df = pd.read_csv(path, index_col="timestamp_s")
    df = df.sort_index()
    return df


def make_gap(columns: list[str], dt: float) -> pd.DataFrame:
    """
    Build a block of zero rows spanning GAP_DURATION_S at step dt.
    Timestamps start at 0 — they will be shifted by the caller.
    """
    n_ticks = int(round(GAP_DURATION_S / dt))
    # arange gives n_ticks rows: dt, 2*dt, ..., n_ticks*dt
    times = np.round(np.arange(1, n_ticks + 1) * dt, decimals=9)
    gap = pd.DataFrame(0.0, index=times, columns=columns)
    gap.index.name = "timestamp_s"
    return gap


def merge_csvs(input_dir: Path, rate_hz: float) -> Path:
    """
    Load every CSV in *input_dir* (sorted by name), insert zero-gap rows
    between them, and write merged_data.csv into *input_dir*.

    Returns the path of the written file.
    """
    dt = 1.0 / rate_hz

    csv_files = sorted(input_dir.glob("*.csv"))
    # Exclude a previously generated merged file so re-runs stay clean
    csv_files = [f for f in csv_files if f.name != OUTPUT_FILENAME]

    if not csv_files:
        sys.exit(f"ERROR: no CSV files found in '{input_dir}'.")

    print(f"Found {len(csv_files)} file(s) to merge:")
    for f in csv_files:
        print(f"  {f.name}")

    segments: list[pd.DataFrame] = []
    cursor = 0.0        # running timestamp offset in seconds

    for i, path in enumerate(csv_files):
        df = load_csv(path)

        # Validate columns are consistent across files
        if segments:
            prev_cols = set(segments[0].columns)
            curr_cols = set(df.columns)
            if prev_cols != curr_cols:
                print(
                    f"  WARNING: column mismatch in '{path.name}'. "
                    "Missing columns will be filled with NaN."
                )

        # Re-index timestamps to be continuous from cursor
        # (original timestamps are replaced; only relative spacing matters)
        original_duration = df.index[-1] - df.index[0]
        new_times = np.round(cursor + (df.index - df.index[0]), decimals=9)
        df.index = new_times
        df.index.name = "timestamp_s"

        segments.append(df)
        cursor += original_duration

        # Insert gap after every file except the last
        if i < len(csv_files) - 1:
            gap = make_gap(df.columns.tolist(), dt)
            gap.index = np.round(cursor + gap.index.values, decimals=9)
            gap.index.name = "timestamp_s"
            segments.append(gap)
            cursor = gap.index[-1]   # advance cursor to end of gap

    merged = pd.concat(segments)
    merged.index.name = "timestamp_s"

    out_path = input_dir / OUTPUT_FILENAME
    merged.to_csv(out_path, float_format="%.9g")

    total_duration = merged.index[-1] - merged.index[0]
    print(
        f"\nmerged_data.csv written: {out_path}\n"
        f"  {len(csv_files)} flight segment(s)\n"
        f"  {len(csv_files) - 1} gap(s) of {GAP_DURATION_S:.1f} s\n"
        f"  {len(merged)} total rows  |  "
        f"total duration ≈ {total_duration:.2f} s  |  "
        f"{len(merged.columns)} columns"
    )
    return out_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Merge all per-flight CSVs in a folder into merged_data.csv, "
            "inserting 1 s of zeros between each file for visual separation."
        )
    )
    parser.add_argument(
        "--input",
        default="csv",
        help="Directory containing the per-flight CSV files (default: csv/)"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="Control-loop frequency used when generating the CSVs, in Hz (default: 20)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        sys.exit(f"ERROR: directory '{input_dir}' does not exist.")

    merge_csvs(input_dir, rate_hz=args.rate)