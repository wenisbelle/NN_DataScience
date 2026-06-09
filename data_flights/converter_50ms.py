"""
ulg_to_csv_20hz.py
------------------
Extracts specific topics/fields from a PX4 ULog (.ulg) file and exports
them to a single CSV file sampled at a fixed 20 Hz (50 ms) control-loop rate.

Sampling strategy — "last value before the tick":
    For every 50 ms tick t, each column takes the most recent logged sample
    whose timestamp is strictly <= t.  This is equivalent to what a real
    control loop would observe if it read sensor/estimator data at t.
    Ticks that fall before the first sample of a topic are left as NaN.

Dependencies:
    pip install pyulog pandas

Usage:
    python ulg_to_csv_20hz.py --input ulogs/ --output csv/
    python ulg_to_csv_20hz.py --input ulogs/ --output csv/ --rate 50
        (--rate overrides the loop frequency in Hz; default 20)
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
    import numpy as np
except ImportError:
    sys.exit(
        "ERROR: pandas / numpy is not installed.\n"
        "Install with:  pip install pandas numpy"
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
    """
    Return a DataFrame for *topic* indexed by timestamp in seconds, or None.
    Only the columns defined in FIELDS for this topic are included.
    """
    matches = [d for d in ulog.data_list if d.name == topic]
    if not matches:
        print(f"  WARNING: topic '{topic}' not found in the log.")
        return None

    data = matches[0].data
    timestamps_us = data["timestamp"]   # microseconds (uint64)

    df = pd.DataFrame({"timestamp_s": timestamps_us / 1e6})

    for _, raw_field, col_name in [f for f in FIELDS if f[0] == topic]:
        if raw_field in data:
            df[col_name] = data[raw_field]
        else:
            print(f"  WARNING: field '{raw_field}' not found in topic '{topic}'.")
            df[col_name] = float("nan")

    df = df.set_index("timestamp_s").sort_index()
    # Remove duplicate timestamps (keep last), which can appear in some logs
    df = df[~df.index.duplicated(keep="last")]
    return df


def resample_to_control_loop(
    merged_raw: pd.DataFrame,
    dt: float = 0.05,
) -> pd.DataFrame:
    """
    Given a raw outer-joined DataFrame (index = timestamp_s, NaNs where a
    topic had no sample), produce a new DataFrame whose rows are evenly
    spaced at *dt* seconds, each cell holding the most recent value logged
    at or before that tick — i.e. a zero-order hold / last-value-before-tick.

    Ticks that precede the very first sample of a column remain NaN.

    Parameters
    ----------
    merged_raw : pd.DataFrame
        Output of the outer-join merge; index is timestamp in seconds.
    dt : float
        Control-loop period in seconds (default 0.05 → 20 Hz).

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame indexed by the tick timestamps.
    """
    t_start = merged_raw.index[0]
    t_end   = merged_raw.index[-1]

    # Build the uniform tick grid, rounded to dt precision to avoid float drift
    n_ticks = int(np.floor((t_end - t_start) / dt))
    ticks   = np.round(t_start + np.arange(n_ticks + 1) * dt, decimals=9)

    # Forward-fill the raw data onto the tick grid using merge_asof:
    #   for each tick, pick the last raw row with timestamp <= tick.
    # merge_asof requires both sides sorted — raw index already is.
    ticks_df = pd.DataFrame({"timestamp_s": ticks}).set_index("timestamp_s")

    # Reindex: insert tick timestamps into the raw index, forward-fill, then
    # select only the tick rows. This is O(N log N) and handles the
    # "before first sample → NaN" case naturally.
    combined = merged_raw.reindex(
        merged_raw.index.union(ticks_df.index)
    ).sort_index()

    # Forward-fill propagates the last observed value to every tick;
    # ticks before the first sample of each column stay NaN.
    combined = combined.ffill()

    # Keep only the tick rows
    resampled = combined.loc[ticks_df.index[1:]]
    resampled.index.name = "timestamp_s"
    return resampled


def ulg_to_csv(ulg_path: str, csv_path: str | None = None, rate_hz: float = 20.0) -> Path:
    """
    Read *ulg_path*, extract configured fields, resample to a fixed control-loop
    rate using last-value-before-tick, and write a CSV to *csv_path*.

    Parameters
    ----------
    ulg_path : str | Path
    csv_path : str | Path | None
        Defaults to same directory/stem as the input with .csv suffix.
    rate_hz  : float
        Control-loop frequency in Hz (default 20 → 50 ms ticks).

    Returns
    -------
    Path of the written CSV.
    """
    ulg_path = Path(ulg_path)
    if not ulg_path.exists():
        sys.exit(f"ERROR: file not found: {ulg_path}")

    if csv_path is None:
        csv_path = ulg_path.with_suffix(".csv")
    csv_path = Path(csv_path)

    dt = 1.0 / rate_hz
    print(f"Reading: {ulg_path}  (resampling to {rate_hz} Hz / {dt*1000:.1f} ms ticks)")
    ulog = ULog(str(ulg_path))

    # ── 1. Load each topic into its own DataFrame ──────────────────────────
    topics = sorted({f[0] for f in FIELDS})
    topic_dfs = {}
    for topic in topics:
        df = load_topic(ulog, topic)
        if df is not None:
            topic_dfs[topic] = df

    if not topic_dfs:
        sys.exit("ERROR: none of the requested topics were found in the log.")

    # ── 2. Outer-join into one raw DataFrame (NaNs between topic samples) ──
    merged_raw = None
    for df in topic_dfs.values():
        merged_raw = df if merged_raw is None else merged_raw.join(df, how="outer")
    merged_raw = merged_raw.sort_index()

    # ── 3. Resample to the control-loop grid ───────────────────────────────
    resampled = resample_to_control_loop(merged_raw, dt=dt)

    # ── 4. Write CSV ────────────────────────────────────────────────────────
    resampled.to_csv(csv_path, float_format="%.9g")
    print(
        f"CSV written: {csv_path}  "
        f"({len(resampled)} ticks × {len(resampled.columns)} columns, "
        f"dt={dt*1000:.1f} ms)"
    )
    return csv_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Batch export PX4 ULog fields to CSV resampled at a fixed "
            "control-loop rate (last value before each tick)."
        )
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
    parser.add_argument(
        "--rate",
        type=float,
        default=20.0,
        help="Control-loop frequency in Hz (default: 20 → 50 ms ticks)"
    )
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        sys.exit(f"ERROR: The input directory '{input_dir}' does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)

    ulg_files = list(input_dir.glob("*.ulg"))

    if not ulg_files:
        print(f"No .ulg files found in '{input_dir}/'")
    else:
        print(f"Found {len(ulg_files)} log file(s). Starting batch processing...")
        for ulg_file in ulg_files:
            csv_file = output_dir / ulg_file.with_suffix(".csv").name
            try:
                ulg_to_csv(ulg_file, csv_file, rate_hz=args.rate)
            except Exception as e:
                print(f"ERROR processing {ulg_file}: {e}")