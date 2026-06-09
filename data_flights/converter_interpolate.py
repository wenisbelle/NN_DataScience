"""
ulg_to_csv.py
-------------
Extracts specific topics/fields from a PX4 ULog (.ulg) file and exports
them to a single CSV file, time-aligned by timestamp.

Dependencies:
    pip install pyulog pandas

Usage:
    python ulg_to_csv.py <path_to_file.ulg> [output.csv]

    If the output filename is omitted, the CSV will be saved next to the
    input file with the same base name (e.g. flight.ulg → flight.csv).
"""

import sys
import argparse
from pathlib import Path

try:
    from pyulog import ULog
except ImportError:
    sys.exit(
        "ERROR: pyulog is not installed.\n"
        "Install it with:  pip install pyulog"
    )

try:
    import pandas as pd
except ImportError:
    sys.exit(
        "ERROR: pandas is not installed.\n"
        "Install it with:  pip install pandas"
    )


# ---------------------------------------------------------------------------
# Field specification
# Each entry: (topic_name, raw_field_name, output_column_name)
# ---------------------------------------------------------------------------
FIELDS = [
    # actuator_motors
    ("actuator_motors", "control[0]", "actuator_motors.control.00"),
    ("actuator_motors", "control[1]", "actuator_motors.control.01"),
    ("actuator_motors", "control[2]", "actuator_motors.control.02"),
    ("actuator_motors", "control[3]", "actuator_motors.control.03"),
    # vehicle_attitude
    ("vehicle_attitude", "q[0]",      "vehicle_attitude.q.00"),
    ("vehicle_attitude", "q[1]",      "vehicle_attitude.q.01"),
    ("vehicle_attitude", "q[2]",      "vehicle_attitude.q.02"),
    ("vehicle_attitude", "q[3]",      "vehicle_attitude.q.03"),
    # vehicle_angular_velocity
    ("vehicle_angular_velocity", "xyz[0]", "vehicle_angular_velocity.xyz.00"),
    ("vehicle_angular_velocity", "xyz[1]", "vehicle_angular_velocity.xyz.01"),
    ("vehicle_angular_velocity", "xyz[2]", "vehicle_angular_velocity.xyz.02"),
    # vehicle_local_position
    ("vehicle_local_position", "ax", "vehicle_local_position.ax"),
    ("vehicle_local_position", "ay", "vehicle_local_position.ay"),
    ("vehicle_local_position", "az", "vehicle_local_position.az"),
    ("vehicle_local_position", "vx", "vehicle_local_position.vx"),
    ("vehicle_local_position", "vy", "vehicle_local_position.vy"),
    ("vehicle_local_position", "vz", "vehicle_local_position.vz"),
]


def load_topic(ulog: ULog, topic: str) -> pd.DataFrame | None:
    """Return a DataFrame for *topic* (indexed by timestamp in seconds), or None."""
    matches = [d for d in ulog.data_list if d.name == topic]
    if not matches:
        print(f"  WARNING: topic '{topic}' not found in the log.")
        return None

    data = matches[0].data          # dict {field_name: np.ndarray}
    timestamps_us = data["timestamp"]   # microseconds (uint64)

    df = pd.DataFrame({"timestamp_s": timestamps_us / 1e6})

    # Map every requested field for this topic
    for _, raw_field, col_name in [f for f in FIELDS if f[0] == topic]:
        if raw_field in data:
            df[col_name] = data[raw_field]
        else:
            print(f"  WARNING: field '{raw_field}' not found in topic '{topic}'.")
            df[col_name] = float("nan")

    df = df.set_index("timestamp_s").sort_index()
    return df


def ulg_to_csv(ulg_path: str, csv_path: str | None = None) -> Path:
    """
    Read *ulg_path*, extract the configured fields, and write a CSV to
    *csv_path* (defaults to same directory / same stem as the input).

    Returns the Path of the written CSV.
    """
    ulg_path = Path(ulg_path)
    if not ulg_path.exists():
        sys.exit(f"ERROR: file not found: {ulg_path}")

    if csv_path is None:
        csv_path = ulg_path.with_suffix(".csv")
    csv_path = Path(csv_path)

    print(f"Reading: {ulg_path}")
    ulog = ULog(str(ulg_path))

    # Collect one DataFrame per unique topic
    topics = sorted({f[0] for f in FIELDS})
    topic_dfs = {}
    for topic in topics:
        df = load_topic(ulog, topic)
        if df is not None:
            topic_dfs[topic] = df

    if not topic_dfs:
        sys.exit("ERROR: none of the requested topics were found in the log.")

    # Merge all topics on timestamp using outer join, then sort
    merged = None
    for df in topic_dfs.values():
        if merged is None:
            merged = df
        else:
            merged = merged.join(df, how="outer")

    merged = merged.sort_index()
    merged.index.name = "timestamp_s"

    # Interpolate NaNs that arise from topics logging at different rates.
    # 'index' mode uses the actual timestamp values for linear interpolation,
    # so uneven sampling is handled correctly.
    # Leading/trailing NaNs (before the first or after the last sample of a
    # topic) are forward/backward filled so no NaN remains in the output.
    merged = merged.interpolate(method="index")
    merged = merged.ffill().bfill()

    # Write CSV
    merged.to_csv(csv_path, float_format="%.9g")
    print(f"CSV written: {csv_path}  ({len(merged)} rows, {len(merged.columns)} columns)")
    return csv_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch export selected fields from PX4 ULog (.ulg) files to CSV."
    )
    parser.add_argument(
        "--input", 
        default="ulogs", 
        help="Directory containing the .ulg files (default: ulogs/)"
    )
    parser.add_argument(
        "--output", 
        default="csv", 
        help="Directory to save the resulting CSVs (default: csv/)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Validate input directory
    if not input_dir.is_dir():
        sys.exit(f"ERROR: The input directory '{input_dir}' does not exist.")

    # Ensure the output directory exists (creates it if it doesn't)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .ulg files in the input directory
    ulg_files = list(input_dir.glob("*.ulg"))

    if not ulg_files:
        print(f"No .ulg files found in '{input_dir}/'")
    else:
        print(f"Found {len(ulg_files)} log file(s). Starting batch processing...")
        
        # Iterate through the files
        for ulg_file in ulg_files:
            # Construct the target CSV file path: csv/filename.csv
            csv_file = output_dir / ulg_file.with_suffix(".csv").name
            
            try:
                ulg_to_csv(ulg_file, csv_file)
            except Exception as e:
                # Catching exceptions so one corrupted log doesn't crash the whole batch
                print(f"ERROR processing {ulg_file}: {e}")