import os
import re
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import List, Tuple

from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# 'broken' is treated like a removal style correction
# 'remove_broken' deletes the meter's open-ended row from the running broken list
# (date edits to broken rows are done in the maintenance notebook instead)
ALLOWED_CORRECTION_SOLUTIONS = {"broken", "remove_broken", "remove", "div100", "no_interp"}
BROKEN_STATUS_VALUES = {"broken", "repaired", "under_renovation", "no_meter"}

###################################################################
###################################################################

def _parse_strict_date_series(series, col_name):
    """
    Parse date columns safely from csv without guessing. 
    (Does not alter csv itself).

    Allowed formats:
    - YYYY-MM-DD
    - YYYY-MM-DD HH:MM:SS
    - M/D/YYYY
    - M/D/YYYY HH:MM:SS

    Blank values become NaT.
    Any other format raises an error.
    """
    allowed_formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%m/%d/%Y %H:%M:%S",
    ]

    s = series.copy()

    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    s = s.astype("string").str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA, "<NA>": pd.NA})

    parsed = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    remaining = s.notna()

    for fmt in allowed_formats:
        trial = pd.to_datetime(s.where(remaining), format=fmt, errors="coerce")
        matched = trial.notna()
        parsed.loc[matched] = trial.loc[matched]
        remaining = remaining & (~matched)

    if remaining.any():
        bad_values = sorted(s.loc[remaining].dropna().unique().tolist())
        raise ValueError(
            f"Unrecognized date format in column '{col_name}'. "
            f"Fix these values: {bad_values[:10]}"
        )

    return parsed
    

###################################################################
###################################################################

def fill_missing_timestamps(df, freq):
    
    """
    Make DataFrame's datetime column aligned to a complete time range at the specified frequency.

    Parameters:
        df (pd.DataFrame): A DataFrame with a 'datetime' column.
        freq (str): Frequency of time stamps (e.g., "15min").
    
    Returns:
        full_df (pd.DataFrame): A DataFrame whose 'datetime' column has full time stamps.
    """
        
    # Step 1: Determine the full time range
    start, end = df['datetime'].min(), df['datetime'].max()
    
    # Step 2: Create the full range of timestamps
    full_timestamps = pd.date_range(start=start, end=end, freq=freq)
    
    # Step 3: Merge DataFrame with full timestamps
    # Create a base DataFrame with the full timestamps
    full_df = pd.DataFrame({'datetime': full_timestamps})
    
    # Merge with df on 'datetime'
    full_df = full_df.merge(df, on='datetime', how='left')
    
    # # drop the last row: first timestamp (00:00:00) of the next day
    # full_df = full_df.drop(full_df.index[-1])
    
    return full_df



###################################################################
###################################################################

def select_meter_types_from_info(
    data,
    meter_info_file,
    allowed_end_uses=("main", "submeter"),
    excluded_meters=None,
    datetime_col="datetime",
):
    """
    Select meter columns using meter_info.csv and return a datetime indexed
    dataframe plus a meter_name to end_use map.

    This shared loader is used by both 1.kwh_end_points and meter_maintenance
    so meter names and plot title labels are normalized consistently.

    Parameters:
    data : pd.DataFrame
        Wide dataframe containing a datetime column and one column per meter.
    meter_info_file : str
        CSV containing at least meter_name and end_use.
    allowed_end_uses : iterable[str]
        End-use values to retain. Defaults to main and submeter.
    excluded_meters : iterable[str] or None
        Optional meter names to exclude, such as PV meters.
    datetime_col : str
        Name of the datetime column in data.

    Returns:
    selected_data : pd.DataFrame
        Selected meter data with a sorted DatetimeIndex.
    meter_type_map : dict
        Normalized meter name mapped to normalized end_use.
    selected_meter_info : pd.DataFrame
        Normalized meter information for meters present in selected_data.
    """
    if data is None:
        raise ValueError("data is None.")
    if datetime_col not in data.columns:
        raise ValueError(f"data is missing datetime column: {datetime_col}")

    selected_data = data.copy()

    normalized_columns = []
    for col in selected_data.columns:
        if col == datetime_col:
            normalized_columns.append(datetime_col)
        else:
            normalized_columns.append(str(col).strip().lower())

    duplicate_columns = pd.Index(normalized_columns)[
        pd.Index(normalized_columns).duplicated()
    ].unique().tolist()
    if duplicate_columns:
        raise ValueError(
            "Meter columns become duplicated after normalization: "
            + ", ".join(map(str, duplicate_columns))
        )

    selected_data.columns = normalized_columns

    meter_info = pd.read_csv(meter_info_file, usecols=["meter_name", "end_use"]).copy()
    meter_info["meter_name"] = (
        meter_info["meter_name"]
        .astype("string")
        .str.strip()
        .str.lower()
    )
    meter_info["end_use"] = (
        meter_info["end_use"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    allowed = {
        str(value).strip().lower()
        for value in allowed_end_uses
        if str(value).strip() != ""
    }
    excluded = {
        str(value).strip().lower()
        for value in (excluded_meters or [])
        if str(value).strip() != ""
    }

    selected_meter_info = meter_info[
        meter_info["end_use"].isin(allowed)
        & ~meter_info["meter_name"].isin(excluded)
    ].copy()
    selected_meter_info = selected_meter_info.drop_duplicates(
        subset="meter_name",
        keep="last",
    )

    meter_type_map = (
        selected_meter_info
        .set_index("meter_name")["end_use"]
        .to_dict()
    )

    selected_meter_columns = [
        meter_name
        for meter_name in selected_data.columns
        if meter_name != datetime_col and meter_name in meter_type_map
    ]

    selected_data = selected_data[
        [datetime_col] + selected_meter_columns
    ].copy()
    selected_data[datetime_col] = pd.to_datetime(
        selected_data[datetime_col],
        errors="coerce",
    )
    selected_data = (
        selected_data
        .dropna(subset=[datetime_col])
        .set_index(datetime_col)
        .sort_index()
    )

    selected_meter_info = selected_meter_info[
        selected_meter_info["meter_name"].isin(selected_meter_columns)
    ].reset_index(drop=True)

    selected_counts = pd.Series(
        [meter_type_map[meter] for meter in selected_meter_columns],
        dtype="string",
    ).value_counts()

    count_text = ", ".join(
        f"{int(selected_counts.get(end_use, 0))} {end_use}"
        for end_use in sorted(allowed)
    )
    print(f"Selected {len(selected_meter_columns)} meters: {count_text}.")

    return selected_data, meter_type_map, selected_meter_info



###################################################################
###################################################################

def resolve_analysis_window(index, start_time, end_time):
    """
    Check requested analysis timestamps to timestamps that
    actually exist in the data index. This reduces failures when a user
    selects times that are close to, but not exactly in, the filled index.
    """
    if index is None or len(index) == 0:
        raise ValueError("Cannot resolve analysis window because the index is empty.")

    idx = pd.DatetimeIndex(index).sort_values().unique()
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    if start_time > end_time:
        raise ValueError("start_time must be <= end_time.")

    if end_time < idx[0] or start_time > idx[-1]:
        raise ValueError("Requested analysis window falls outside the data index.")

    start_pos = idx.searchsorted(start_time, side="left")
    end_pos = idx.searchsorted(end_time, side="right") - 1

    if start_pos >= len(idx):
        raise ValueError("Resolved start_time is outside the available index.")
    if end_pos < 0:
        raise ValueError("Resolved end_time is outside the available index.")
    if start_pos > end_pos:
        raise ValueError("Resolved analysis window contains no timestamps.")

    resolved_start = idx[start_pos]
    resolved_end = idx[end_pos]

    return resolved_start, resolved_end





###################################################################
###################################################################

def apply_special_meter_corrections(data, special_meters_file):
    """
    Apply special corrections to selected meters based on a CSV config file.
    Correctly handles multiple periods per meter, including overlapping periods.

    Harvest/current behavior:
    - safely skips if the file path is None, empty, or missing
    - expects the same core columns as Aurora:
      meter_name, solution, start_datetime, end_datetime
    - supports at least: div100, remove
    - treats blank start_datetime as the beginning of the available data index
    - treats blank end_datetime as the end of the available data index
    - clips correction windows to the available data index at runtime

    Returns a copy of the DataFrame.
    """
    data_corrected = data.copy()

    if special_meters_file is None or not os.path.exists(special_meters_file):
        print("No special meter file provided. Skipping manual corrections.")
        return data_corrected

    if data_corrected.empty or len(data_corrected.index) == 0:
        return data_corrected

    bad_periods = pd.read_csv(special_meters_file, low_memory=False)

    required_cols = {"meter_name", "solution", "start_datetime", "end_datetime"}
    missing_cols = required_cols.difference(set(bad_periods.columns))
    if missing_cols:
        raise ValueError(
            "Special meter file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    data_start = pd.to_datetime(data_corrected.index.min())
    data_end = pd.to_datetime(data_corrected.index.max())

    # Clean up
    bad_periods["meter_name"] = bad_periods["meter_name"].astype(str).str.strip()
    bad_periods["solution"] = bad_periods["solution"].astype(str).str.strip().str.lower()
    bad_periods["start_datetime"] = pd.to_datetime(bad_periods["start_datetime"], errors="coerce")
    bad_periods["end_datetime"] = pd.to_datetime(bad_periods["end_datetime"], errors="coerce")

    # Filter to meters that exist in data
    bad_periods = bad_periods[bad_periods["meter_name"].isin(data_corrected.columns)]

    for _, row in bad_periods.iterrows():
        meter = row["meter_name"]
        start = row["start_datetime"]
        end = row["end_datetime"]
        solution = row["solution"]

        if pd.isna(start):
            start = data_start
        if pd.isna(end):
            end = data_end

        start = max(pd.to_datetime(start), data_start)
        end = min(pd.to_datetime(end), data_end)

        if start > end:
            continue

        # mask points in the period
        mask = (data_corrected.index >= start) & (data_corrected.index <= end)

        if solution == "div100":
            data_corrected.loc[mask, meter] = data_corrected.loc[mask, meter] / 100.0
        elif solution in {"remove", "broken"}:
            data_corrected.loc[mask, meter] = np.nan

    return data_corrected


def export_removed_special_meter_data(
    data,
    special_meters_file,
    output_csv=None,
    export_solutions=None,
):
    """
    Export raw meter readings that fall inside remove/broken correction windows.

    This function does not modify the input data. It preserves the raw values
    that would later be removed from the working dataframe.

    Parameters:
        data (pd.DataFrame): Wide meter dataframe with a DatetimeIndex.
        special_meters_file (str): CSV file with correction rows.
        output_csv (str or None): Optional CSV output path.
        export_solutions (set/list/tuple or None): Solutions to export.
            Defaults to {"remove", "broken"}.

    Returns:
        pd.DataFrame: Long-format exported rows with correction metadata.
    """
    export_cols = [
        "datetime",
        "meter_name",
        "meter_reading",
        "solution",
        "correction_start",
        "correction_end",
        "issue_type/status",
        "description",
    ]

    if export_solutions is None:
        export_solutions = {"remove", "broken"}

    export_solutions = {
        str(value).strip().lower()
        for value in export_solutions
        if str(value).strip() != ""
    }

    if data is None or data.empty or len(data.index) == 0:
        export_df = pd.DataFrame(columns=export_cols)
        if output_csv is not None and str(output_csv).strip() != "":
            output_dir = os.path.dirname(str(output_csv))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            export_df.to_csv(output_csv, index=False)
            print(f"Removed special meter data exported to {output_csv}")
        return export_df

    if special_meters_file is None or not os.path.exists(special_meters_file):
        print("No special meter file provided. Skipping removed-data export.")
        return pd.DataFrame(columns=export_cols)

    correction_df = pd.read_csv(special_meters_file, low_memory=False)

    required_cols = {"meter_name", "solution", "start_datetime", "end_datetime"}
    missing_cols = required_cols.difference(set(correction_df.columns))
    if missing_cols:
        raise ValueError(
            "Special meter file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    if "issue_type" in correction_df.columns and "issue_type/status" not in correction_df.columns:
        correction_df = correction_df.rename(columns={"issue_type": "issue_type/status"})
    if "status" in correction_df.columns and "issue_type/status" not in correction_df.columns:
        correction_df = correction_df.rename(columns={"status": "issue_type/status"})

    if "issue_type/status" not in correction_df.columns:
        correction_df["issue_type/status"] = ""
    if "description" not in correction_df.columns:
        correction_df["description"] = ""

    correction_df = correction_df.copy()
    correction_df["meter_name"] = correction_df["meter_name"].astype(str).str.strip()
    correction_df["solution"] = correction_df["solution"].astype(str).str.strip().str.lower()
    correction_df["start_datetime"] = pd.to_datetime(correction_df["start_datetime"], errors="coerce")
    correction_df["end_datetime"] = pd.to_datetime(correction_df["end_datetime"], errors="coerce")
    correction_df["issue_type/status"] = correction_df["issue_type/status"].fillna("").astype(str).str.strip()
    correction_df["description"] = correction_df["description"].fillna("").astype(str).str.strip()

    correction_df = correction_df[
        correction_df["meter_name"].isin(data.columns)
        & correction_df["solution"].isin(export_solutions)
    ].copy()

    if correction_df.empty:
        export_df = pd.DataFrame(columns=export_cols)
        if output_csv is not None and str(output_csv).strip() != "":
            output_dir = os.path.dirname(str(output_csv))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            export_df.to_csv(output_csv, index=False)
            print(f"Removed special meter data exported to {output_csv}")
        return export_df

    data_start = pd.to_datetime(data.index.min())
    data_end = pd.to_datetime(data.index.max())

    export_frames = []

    for _, row in correction_df.iterrows():
        meter = row["meter_name"]
        start = row["start_datetime"]
        end = row["end_datetime"]

        if pd.isna(start):
            start = data_start
        if pd.isna(end):
            end = data_end

        start = max(pd.to_datetime(start), data_start)
        end = min(pd.to_datetime(end), data_end)

        if start > end:
            continue

        meter_series = data.loc[(data.index >= start) & (data.index <= end), meter]
        if meter_series.empty:
            continue

        export_piece = meter_series.rename("meter_reading").reset_index()
        export_piece.columns = ["datetime", "meter_reading"]
        export_piece = export_piece.dropna(subset=["meter_reading"])
        if export_piece.empty:
            continue
        export_piece["meter_name"] = meter
        export_piece["solution"] = row["solution"]
        export_piece["correction_start"] = row["start_datetime"]
        export_piece["correction_end"] = row["end_datetime"]
        export_piece["issue_type/status"] = row["issue_type/status"]
        export_piece["description"] = row["description"]

        export_frames.append(export_piece)

    if export_frames:
        export_df = pd.concat(export_frames, ignore_index=True, sort=False)
        export_df = export_df.reindex(columns=export_cols)
        export_df = export_df.sort_values(
            by=["meter_name", "datetime", "solution"],
            na_position="last",
        ).reset_index(drop=True)
    else:
        export_df = pd.DataFrame(columns=export_cols)

    if output_csv is not None and str(output_csv).strip() != "":
        output_dir = os.path.dirname(str(output_csv))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        export_df.to_csv(output_csv, index=False)
        print(f"Removed special meter data exported to {output_csv}")

    return export_df



###################################################################
###################################################################

def find_special_meters(data, r2_threshold=0.9, stuck_thres_start=50, stuck_thres_end=50):
    """
    Identify bad meters + detect restart events.

    Returns:
        df_bad_meters: columns ['meter_name', 'r2', 'info']
        df_restarts: columns ['meter_name', 'previous_valid_time', 'restart_time']
    """
    bad_meters = []
    restart_rows = []   # collect all restart events

    for meter in data.columns:
        series = data[meter]
        y = series.values
        t = series.index

        # --------------------------
        # Condition 0: all NaN
        # --------------------------
        if np.isnan(y).all():
            bad_meters.append((meter, 0.0, "all NaN"))
            continue

        # ------------------------------------------------
        # Condition 1: detect restarts & count occurrences
        #   - use a cleaned copy with consecutive duplicates removed
        #   - original series is NOT modified (needed later for stuck detection)
        # ------------------------------------------------
        series_valid = series.dropna()
        restarts = 0

        if len(series_valid) >= 2:
            # Remove consecutive duplicates, keep only the first of each run
            clean_series = series_valid.mask(series_valid.eq(series_valid.shift()))
            clean_series = clean_series.dropna()

            if len(clean_series) >= 2:
                vals = clean_series.values
                times = clean_series.index

                prev_val = vals[0]
                prev_time = times[0]

                for val, time in zip(vals[1:], times[1:]):
                    if val < prev_val:
                        restarts += 1
                        restart_rows.append({
                            "meter_name": meter,
                            "previous_valid_time": prev_time,
                            "restart_time": time
                        })
                    prev_val = val
                    prev_time = time

        if restarts > 0:
            bad_meters.append((meter, -3.0, f"restart {restarts} times"))
            continue

        # --------------------------
        # Condition 2: NaN at edges
        # --------------------------
        if np.isnan(y[0]) and np.isnan(y[-1]):
            bad_meters.append((meter, -1.0, "missing start and end"))
            continue
        elif np.isnan(y[0]):
            bad_meters.append((meter, -1.0, "missing start"))
            continue
        elif np.isnan(y[-1]):
            bad_meters.append((meter, -1.0, "missing end"))
            continue

        # --------------------------
        # Condition 3: stuck at start
        #   (uses original y so stuck lengths are correct)
        # --------------------------
        start_len = 1
        while start_len < len(y) and y[start_len] == y[0]:
            start_len += 1
        if start_len >= stuck_thres_start:
            bad_meters.append((meter, -2.0, f"stuck {start_len} pts at start"))
            continue

        # --------------------------
        # Condition 3b: stuck at end
        # --------------------------
        end_len = 1
        while end_len < len(y) and y[-end_len - 1] == y[-1]:
            end_len += 1
        if end_len >= stuck_thres_end:
            bad_meters.append((meter, -2.0, f"stuck {end_len} pts at end"))
            continue

        # --------------------------
        # Condition 4 & 5: Too few points or R² too low
        # --------------------------
        x = np.arange(len(y)).reshape(-1, 1)
        mask = ~np.isnan(y)
        x_clean, y_clean = x[mask], y[mask]

        if len(y_clean) < 2:
            bad_meters.append((meter, -4.0, "too few points"))
            continue

        model = LinearRegression().fit(x_clean, y_clean)
        y_pred = model.predict(x_clean)
        r2 = r2_score(y_clean, y_pred)

        if r2 < r2_threshold:
            bad_meters.append((meter, r2, "low R²"))

    # --------------------------
    # Prepare output DataFrames
    # --------------------------

    # Sort by R² before building df_bad_meters
    bad_meters.sort(key=lambda x: (x[1] if not np.isnan(x[1]) else -999))
    
    # df_bad_meters
    if not bad_meters:
        df_bad_meters = pd.DataFrame(columns=["meter_name", "r2", "info"])
    else:
        df_bad_meters = pd.DataFrame(bad_meters, columns=["meter_name", "r2", "info"])

    # formatting r2
    def format_r2(x):
        if x in [0.0, -1.0, -2.0, -3.0, -4.0]:
            return str(int(x))
        else:
            return round(float(x), 4)

    if not df_bad_meters.empty:
        df_bad_meters["r2"] = df_bad_meters["r2"].apply(format_r2)

    # df_restarts
    if not restart_rows:
        df_restarts = pd.DataFrame(columns=[
            "meter_name", "previous_valid_time", "restart_time"
        ])
    else:
        df_restarts = pd.DataFrame(restart_rows, columns=[
            "meter_name", "previous_valid_time", "restart_time"
        ])

    return df_bad_meters, df_restarts



###################################################################
###################################################################

def suggest_meter_issues(data, stuck_run_threshold=50, jump_multiplier=20, max_rows_per_meter=5):
    """
    Suggest candidate meter issues to help automate the old manual flagging process.

    Returns a DataFrame with columns:
        meter_name, issue_type, start_datetime, end_datetime, description, suggestion
    """
    suggestions = []

    for meter in data.columns:
        series = data[meter].dropna()
        if series.empty:
            suggestions.append({
                "meter_name": meter,
                "issue_type": "all_nan",
                "start_datetime": pd.NaT,
                "end_datetime": pd.NaT,
                "description": "Meter has no valid values in the selected window.",
                "suggestion": "review"
            })
            continue

        # Stuck runs
        run_start = None
        prev_val = None
        prev_time = None
        run_len = 0
        added_for_meter = 0

        for t, val in series.items():
            if prev_val is not None and val == prev_val:
                if run_start is None:
                    run_start = prev_time
                    run_len = 2
                else:
                    run_len += 1
            else:
                if run_start is not None and run_len >= stuck_run_threshold:
                    suggestions.append({
                        "meter_name": meter,
                        "issue_type": "stuck_run",
                        "start_datetime": run_start,
                        "end_datetime": prev_time,
                        "description": f"Consecutive repeated values for {run_len} points.",
                        "suggestion": "remove_or_no_interp"
                    })
                    added_for_meter += 1
                    if added_for_meter >= max_rows_per_meter:
                        break
                run_start = None
                run_len = 1

            prev_val = val
            prev_time = t

        if added_for_meter < max_rows_per_meter and run_start is not None and run_len >= stuck_run_threshold:
            suggestions.append({
                "meter_name": meter,
                "issue_type": "stuck_run",
                "start_datetime": run_start,
                "end_datetime": prev_time,
                "description": f"Consecutive repeated values for {run_len} points.",
                "suggestion": "remove_or_no_interp"
            })

        diffs = series.diff()

        # restarts / drops
        neg_diffs = diffs[diffs < 0]
        for t, diff_val in neg_diffs.head(max_rows_per_meter).items():
            prev_idx = series.index[series.index.get_loc(t) - 1] if series.index.get_loc(t) > 0 else pd.NaT
            suggestions.append({
                "meter_name": meter,
                "issue_type": "restart_or_drop",
                "start_datetime": prev_idx,
                "end_datetime": t,
                "description": f"Negative jump of {diff_val}.",
                "suggestion": "review_restart"
            })

    if not suggestions:
        return pd.DataFrame(columns=[
            "meter_name", "issue_type", "start_datetime", "end_datetime", "description", "suggestion"
        ])

    return pd.DataFrame(suggestions).sort_values(
        by=["meter_name", "start_datetime", "issue_type"],
        na_position="last"
    ).reset_index(drop=True)



###################################################################
###################################################################


# BROKEN METER MASTER/CANDIDATE WORKFLOW
###################################################################
###################################################################

# shared text normalizer for the broken meter workflow.
# lower=False keeps the original capitalization (used for free-text descriptions).
def _normalize_text(value, underscore=False, lower=True):
    if pd.isna(value):
        return ""

    value = str(value).strip()
    if lower:
        value = value.lower()

    if underscore:
        value = re.sub(r"[\s\-]+", "_", value)
    else:
        value = re.sub(r"\s+", " ", value)

    return value

# TODO: fix, have a date parser function added now
# loader for the broken meter workbook with simple normalization and day-first dates.
def load_broken_meter_workbook(broken_meters_file, dayfirst=False):
    """
    Load and normalize the running broken meter CSV.

    Expected columns:
        meter_name, start_date, end_date, status, description, data_source, updated_data
    """
    cols = [
        "meter_name", "start_date", "end_date", "status",
        "description", "data_source", "updated_data"
    ]

    if broken_meters_file is None or str(broken_meters_file).strip() == "":
        return pd.DataFrame(columns=cols)

    if not os.path.exists(broken_meters_file):
        print(f"Broken meter file not found: {broken_meters_file}")
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(broken_meters_file)

    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    df = df.rename(columns={"updated_date": "updated_data"})

    required_cols = {"meter_name", "start_date", "end_date", "status"}
    missing_cols = required_cols.difference(set(df.columns))
    if missing_cols:
        raise ValueError(
            "Broken meter file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    for col in ["description", "data_source", "updated_data"]:
        if col not in df.columns:
            df[col] = ""

    df["meter_name"] = df["meter_name"].apply(_normalize_text)
    df["status"] = df["status"].apply(lambda x: _normalize_text(x, underscore=True))
    df["description"] = df["description"].apply(lambda x: _normalize_text(x, lower=False))
    df["data_source"] = df["data_source"].apply(_normalize_text)
    df["updated_data"] = _parse_strict_date_series(df["updated_data"], "updated_data")
    df["start_date"] = _parse_strict_date_series(df["start_date"], "start_date")
    df["end_date"] = _parse_strict_date_series(df["end_date"], "end_date")

    df = df[df["meter_name"] != ""].copy()
    df = df[df["status"].isin(BROKEN_STATUS_VALUES)].copy()

    return df.reset_index(drop=True)



# loader for the candidate workbook used by the review workflow.
def load_existing_candidates(candidate_file):
    cols = [
        "meter_name",
        "solution",
        "start_datetime",
        "end_datetime",
        "issue_type",
        "description",
        "suggestion",
        "r2",
        "approved",
    ]

    if candidate_file is None or not os.path.exists(candidate_file):
        return pd.DataFrame(columns=cols)

    df = pd.read_csv(candidate_file, low_memory=False)

    if "issue_type/status" in df.columns and "issue_type" not in df.columns:
        df = df.rename(columns={"issue_type/status": "issue_type"})

    if "details" in df.columns and "description" not in df.columns:
        df = df.rename(columns={"details": "description"})

    if "detected_r2" in df.columns and "r2" not in df.columns:
        df = df.rename(columns={"detected_r2": "r2"})

    for col in cols:
        if col not in df.columns:
            df[col] = np.nan if col not in {"solution", "description", "suggestion"} else ""

    df = df[cols].copy()
    df["meter_name"] = df["meter_name"].apply(_normalize_text)
    df["start_datetime"] = _parse_strict_date_series(df["start_datetime"], "start_datetime")
    df["end_datetime"] = _parse_strict_date_series(df["end_datetime"], "end_datetime")
    df["approved"] = pd.to_numeric(df["approved"], errors="coerce").fillna(0).astype(int)
    df["solution"] = df["solution"].fillna("").astype(str).str.strip().str.lower()
    df["issue_type"] = df["issue_type"].fillna("").astype(str).str.strip().str.lower()

    return df.reset_index(drop=True)



# deletes previously generated candidate review files so this run starts fresh.
def clear_candidate_files(candidate_file):
    """
    Remove the auto-generated candidate review file and its restarts sibling
    (<candidate_file base>_restarts.csv) so update_special_meter_candidates_workbook
    regenerates both from scratch on this run.
    """
    restart_file = os.path.splitext(candidate_file)[0] + "_restarts.csv"

    for f in (candidate_file, restart_file):
        if f and os.path.exists(f):
            os.remove(f)
            print(f"Cleared previous candidate file: {f}")
        else:
            print(f"No existing candidate file to clear: {f}")



# simple dedupe key builder for merged master/candidate rows.
def _build_simple_key(df):
    temp = df.copy()

    for col in ["meter_name", "solution", "issue_type/status", "description"]:
        if col not in temp.columns:
            temp[col] = ""
        temp[col] = temp[col].fillna("").astype(str).str.strip().str.lower()

    for col in ["start_datetime", "end_datetime"]:
        if col not in temp.columns:
            temp[col] = pd.NaT
        temp[col] = pd.to_datetime(temp[col])

    start_text = temp["start_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("<na_start>")
    end_text = temp["end_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S").fillna("<na_end>")

    return (
        temp["meter_name"] + "||" +
        temp["solution"] + "||" +
        temp["issue_type/status"] + "||" +
        temp["description"] + "||" +
        start_text + "||" +
        end_text
    )


# adds approved "broken" candidate rows to the running broken meter csv (in its M/D/YYYY format)
def add_candidates_to_broken_list(broken_candidates, broken_meters_file, data_source="harvest"):
    def to_text(value):
        if pd.isna(value):
            return ""
        value = pd.Timestamp(value)
        text = f"{value.month}/{value.day}/{value.year}"
        if value != value.normalize():
            text += value.strftime(" %H:%M:%S")
        return text

    if os.path.exists(broken_meters_file):
        broken_raw = pd.read_csv(broken_meters_file, dtype=str, keep_default_na=False)
        existing = load_broken_meter_workbook(broken_meters_file)
        existing_keys = set(zip(existing["meter_name"], existing["start_date"]))
    else:
        broken_raw = pd.DataFrame(columns=[
            "meter_name", "start_date", "end_date", "status",
            "description", "data_source", "updated_data"
        ])
        existing_keys = set()

    today = pd.Timestamp.today().normalize()
    new_rows = []
    for _, row in broken_candidates.iterrows():
        start = pd.to_datetime(row["start_datetime"], errors="coerce")
        # skip meters already listed as broken from the same start
        if (row["meter_name"], start) in existing_keys:
            continue
        new_rows.append({
            "meter_name": row["meter_name"],
            "start_date": to_text(start),
            "end_date": to_text(row["end_datetime"]),
            "status": "broken",
            "description": row["description"],
            "data_source": data_source,
            "updated_data": to_text(today),
        })

    if new_rows:
        broken_raw = pd.concat([broken_raw, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
        broken_raw.to_csv(broken_meters_file, index=False)

    print(f"{len(new_rows)} approved broken candidate(s) added to {broken_meters_file}")



# deletes the open-ended broken row each approved "remove_broken" candidate overlaps
def remove_candidates_from_broken_list(remove_candidates, broken_meters_file):
    """
    For each candidate, find the meter's broken-list row with a blank end_date that
    overlaps the candidate window and delete it. Every candidate must match exactly
    one open row, otherwise nothing is written and an error is raised (approved rows
    leave the candidate file next run, so a skipped one would be lost). Rows with an
    end date are never touched here; edit those in the maintenance notebook.

    A timestamped backup of the broken list is saved next to it before overwriting.
    """
    if not os.path.exists(broken_meters_file):
        raise ValueError(f"Broken meter file not found: {broken_meters_file}")

    # raw text keeps the file's own formatting when written back
    broken_raw = pd.read_csv(broken_meters_file, dtype=str, keep_default_na=False)
    broken_raw.columns = [str(col).strip().lower().replace(" ", "_") for col in broken_raw.columns]
    meter_names = broken_raw["meter_name"].apply(_normalize_text)
    statuses = broken_raw["status"].apply(lambda x: _normalize_text(x, underscore=True))
    starts = _parse_strict_date_series(broken_raw["start_date"], "start_date")
    ends = _parse_strict_date_series(broken_raw["end_date"], "end_date")

    drop_idx = set()
    errors = []
    for _, row in remove_candidates.iterrows():
        meter = _normalize_text(row["meter_name"])
        cand_start = pd.to_datetime(row["start_datetime"], errors="coerce")
        cand_end = pd.to_datetime(row["end_datetime"], errors="coerce")

        # blank candidate dates count as open, so they overlap anything on that side
        overlaps = (
            meter_names.eq(meter)
            & statuses.isin(BROKEN_STATUS_VALUES)
            & (starts.isna() | pd.isna(cand_end) | (starts <= cand_end))
            & (ends.isna() | pd.isna(cand_start) | (ends >= cand_start))
        )
        open_matches = broken_raw.index[overlaps & ends.isna()]

        if len(open_matches) == 1:
            drop_idx.add(open_matches[0])
        elif len(open_matches) > 1:
            errors.append(f"{meter}: more than one open-ended broken row overlaps the candidate window")
        elif overlaps.any():
            errors.append(
                f"{meter}: the overlapping broken row has an end date; "
                "edit or remove it in the maintenance notebook instead"
            )
        else:
            errors.append(f"{meter}: no open-ended broken row overlaps the candidate window")

    if errors:
        raise ValueError(
            "Could not apply remove_broken candidates (nothing was changed):\n- "
            + "\n- ".join(errors)
        )

    if drop_idx:
        stamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.splitext(broken_meters_file)[0] + f"_backup_{stamp}.csv"
        broken_raw_original = pd.read_csv(broken_meters_file, dtype=str, keep_default_na=False)
        broken_raw_original.to_csv(backup_file, index=False)
        print(f"Broken meter list backed up to {backup_file}")

        broken_raw_original.drop(index=sorted(drop_idx)).to_csv(broken_meters_file, index=False)

    print(f"{len(drop_idx)} open-ended broken row(s) removed from {broken_meters_file}")



# master sheet sync for official corrections, approved candidates, and broken-meter rows.
def sync_meter_corrections_master_sheet(
    candidate_file,
    broken_meters_file,
    meter_corrections_file,
    data_source="harvest",
):
    """
    Update the master corrections file. Rows already in it stay every run, except
    broken-list rows, which are rebuilt from the running broken meter file so edits,
    end dates and deletions there carry over.

    The updated file is:
    - existing master rows that did not come from the broken list
      (approved candidates from earlier runs, stuck_run rows, hand-added rows)
    - approved candidate rows (approved == 1 and solution filled in); rows with
      solution "broken" are added to the running broken meter file instead (with data_source),
      and rows with solution "remove_broken" delete the overlapping open-ended row from it
    - every row from the running broken meter file
    """
    master_cols = [
        "meter_name",
        "solution",
        "start_datetime",
        "end_datetime",
        "issue_type/status",
        "description",
    ]
    master_df = pd.DataFrame(columns=master_cols)

    if os.path.exists(meter_corrections_file):
        old_master = pd.read_csv(meter_corrections_file, low_memory=False).reindex(columns=master_cols)
        old_master["start_datetime"] = pd.to_datetime(old_master["start_datetime"], errors="coerce")
        old_master["end_datetime"] = pd.to_datetime(old_master["end_datetime"], errors="coerce")

        # broken-list rows are identified by their status and replaced below
        from_broken_list = (
            old_master["issue_type/status"].fillna("").astype(str).str.strip().str.lower()
            .isin(BROKEN_STATUS_VALUES)
        )
        master_df = old_master[~from_broken_list].copy()

    existing_candidates = load_existing_candidates(candidate_file)

    approved_candidates = existing_candidates[
        (existing_candidates["approved"] == 1) &
        (existing_candidates["solution"] != "")
    ].copy()

    # approved "broken" candidates go to the running broken list, and reach the master sheet from there
    is_broken = approved_candidates["solution"].eq("broken")
    if is_broken.any():
        add_candidates_to_broken_list(approved_candidates[is_broken], broken_meters_file, data_source)
    approved_candidates = approved_candidates[~is_broken]

    # approved "remove_broken" candidates delete their open-ended row from the running list
    is_remove_broken = approved_candidates["solution"].eq("remove_broken")
    if is_remove_broken.any():
        remove_candidates_from_broken_list(approved_candidates[is_remove_broken], broken_meters_file)
    approved_candidates = approved_candidates[~is_remove_broken]

    broken_df = load_broken_meter_workbook(broken_meters_file)

    if not approved_candidates.empty:
        approved_candidates = approved_candidates.rename(
            columns={"issue_type": "issue_type/status"}
        )
        approved_candidates = approved_candidates[
            ["meter_name", "solution", "start_datetime", "end_datetime", "issue_type/status", "description"]
        ].copy()
        master_df = pd.concat([master_df, approved_candidates], ignore_index=True, sort=False)

    broken_rows = []
    for _, row in broken_df.iterrows():
        broken_rows.append({
            "meter_name": row["meter_name"],
            "solution": "remove",
            "start_datetime": pd.to_datetime(row["start_date"], errors="coerce"),
            "end_datetime": pd.to_datetime(row["end_date"], errors="coerce"),
            "issue_type/status": row["status"],
            "description": row["description"],
        })

    if broken_rows:
        master_df = pd.concat([master_df, pd.DataFrame(broken_rows)], ignore_index=True, sort=False)

    if not master_df.empty:
        master_df = master_df.copy()
        master_df["_key"] = _build_simple_key(master_df)
        master_df = master_df.drop_duplicates(subset="_key", keep="last").drop(columns="_key")
        master_df = master_df.sort_values(
            by=["meter_name", "start_datetime", "solution", "issue_type/status"],
            na_position="last"
        ).reset_index(drop=True)

    output_dir = os.path.dirname(str(meter_corrections_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    master_df.to_csv(meter_corrections_file, index=False)

    print(f"Master correction file saved to {meter_corrections_file}")
    return master_df



# adds auto-approved stuck_run rows straight to the master corrections file
def add_stuck_runs_to_master_sheet(stuck_rows, meter_corrections_file):
    master_cols = [
        "meter_name",
        "solution",
        "start_datetime",
        "end_datetime",
        "issue_type/status",
        "description",
    ]

    new_rows = stuck_rows.rename(columns={"issue_type": "issue_type/status"}).reindex(columns=master_cols)

    if os.path.exists(meter_corrections_file):
        master_df = pd.read_csv(meter_corrections_file, low_memory=False).reindex(columns=master_cols)
    else:
        master_df = pd.DataFrame(columns=master_cols)

    master_df = pd.concat([master_df, new_rows], ignore_index=True, sort=False)
    master_df["start_datetime"] = pd.to_datetime(master_df["start_datetime"], errors="coerce")
    master_df["end_datetime"] = pd.to_datetime(master_df["end_datetime"], errors="coerce")

    master_df["_key"] = _build_simple_key(master_df)
    master_df = master_df.drop_duplicates(subset="_key", keep="first").drop(columns="_key")
    master_df = master_df.sort_values(
        by=["meter_name", "start_datetime", "solution", "issue_type/status"],
        na_position="last"
    ).reset_index(drop=True)

    master_df.to_csv(meter_corrections_file, index=False)
    print(f"{len(new_rows)} stuck_run row(s) added to master correction file {meter_corrections_file}")



# candidate workbook updater for unresolved review rows and new detected issues
def update_special_meter_candidates_workbook(
    data,
    candidate_file,
    broken_meters_file,
    window_start,
    window_end,
    df_bad_meters=None,
    df_restarts=None,
    meter_corrections_file=None,
    stuck_solution="remove",
):
    """
    Rules:
    - keep unresolved existing candidate rows, except stale rows that are now
      fully explained by broken-meter windows
    - remove rows where approved == 1
    - generate new candidate rows from the current data
    - convert missing-start / missing-end summary issues into real timeframe rows
    - split missing_start_and_end into two rows
    - exclude all-NaN rows when the broken interval covers the full review window
    - exclude interval rows fully inside broken-meter intervals
    - do not create blue missing rows for meters broken across the full window
    - stuck_run rows skip review: if meter_corrections_file is given they are written
      straight to the master sheet with solution stuck_solution, and left out of the candidate file
    - rows overlapping an open-ended broken-meter row get suggestion "review_remove_broken"
      (approve with solution remove_broken if the meter is not actually broken)
    """
    window_start = pd.to_datetime(window_start)
    window_end = pd.to_datetime(window_end)

    candidate_cols = [
        "meter_name",
        "solution",
        "start_datetime",
        "end_datetime",
        "issue_type",
        "description",
        "suggestion",
        "r2",
        "approved",
    ]

    def summary_info_to_issue_type(info):
        info = str(info).strip().lower()

        if info == "missing start":
            return "missing_start"
        if info == "missing end":
            return "missing_end"
        if info == "missing start and end":
            return "missing_start_and_end"
        if info == "all nan":
            return "all_nan"
        if info == "too few points":
            return "too_few_points"
        if info == "low r²" or info == "low r2":
            return "low_r2"

        return "summary_review"

    def build_edge_missing_rows(meter_name, info_value, r2_value):
        rows = []

        if meter_name not in data.columns:
            return rows

        meter_series = data[meter_name].copy()
        meter_series.index = pd.to_datetime(meter_series.index, errors="coerce")
        meter_series = meter_series.sort_index().loc[window_start:window_end]

        if meter_series.empty:
            return rows

        first_valid = meter_series.first_valid_index()
        last_valid = meter_series.last_valid_index()
        window_index = meter_series.index

        if first_valid is None or last_valid is None:
            return rows

        first_pos = window_index.get_loc(first_valid)
        last_pos = window_index.get_loc(last_valid)

        if isinstance(first_pos, slice) or isinstance(last_pos, slice):
            return rows

        if isinstance(first_pos, np.ndarray):
            first_pos = int(first_pos[0])
        if isinstance(last_pos, np.ndarray):
            last_pos = int(last_pos[0])

        if info_value in {"missing start", "missing start and end"} and first_pos > 0:
            last_missing_before_first_valid = window_index[first_pos - 1]
            rows.append({
                "meter_name": meter_name,
                "solution": "",
                "start_datetime": window_start,
                "end_datetime": last_missing_before_first_valid,
                "issue_type": "missing_start",
                "description": "No valid data from analysis window start to last missing timestamp before first valid reading.",
                "suggestion": "review_summary",
                "r2": r2_value,
                "approved": 0,
            })

        if info_value in {"missing end", "missing start and end"} and last_pos < len(window_index) - 1:
            first_missing_after_last_valid = window_index[last_pos + 1]
            rows.append({
                "meter_name": meter_name,
                "solution": "",
                "start_datetime": first_missing_after_last_valid,
                "end_datetime": window_end,
                "issue_type": "missing_end",
                "description": "No valid data from first missing timestamp after last valid reading to analysis window end.",
                "suggestion": "review_summary",
                "r2": r2_value,
                "approved": 0,
            })

        return rows

    existing_candidates = load_existing_candidates(candidate_file)
    unresolved_candidates = existing_candidates[
        existing_candidates["approved"] != 1
    ].copy()

    broken_df = load_broken_meter_workbook(broken_meters_file)

    broken_intervals = []
    for _, row in broken_df.iterrows():
        clipped_start = window_start if pd.isna(row["start_date"]) else max(pd.to_datetime(row["start_date"]), window_start)
        clipped_end = window_end if pd.isna(row["end_date"]) else min(pd.to_datetime(row["end_date"]), window_end)
        if clipped_start <= clipped_end:
            broken_intervals.append((row["meter_name"], clipped_start, clipped_end))

    fully_broken_meters = {
        meter_name
        for meter_name, broken_start, broken_end in broken_intervals
        if broken_start <= window_start and broken_end >= window_end
    }

    def interval_fully_inside_broken(meter_name, start_dt, end_dt):
        if pd.isna(start_dt) or pd.isna(end_dt):
            return False
        for broken_meter, broken_start, broken_end in broken_intervals:
            if (
                broken_meter == meter_name
                and start_dt >= broken_start
                and end_dt <= broken_end
            ):
                return True
        return False

    if not unresolved_candidates.empty:
        unresolved_candidates["meter_name"] = unresolved_candidates["meter_name"].apply(_normalize_text)

        unresolved_issue_type = unresolved_candidates.get(
            "issue_type",
            pd.Series("", index=unresolved_candidates.index)
        )

        unresolved_candidates = unresolved_candidates[
            ~(
                unresolved_issue_type.fillna("").astype(str).str.strip().str.lower().eq("all_nan")
                & unresolved_candidates["meter_name"].isin(fully_broken_meters)
            )
        ].copy()

        unresolved_candidates = unresolved_candidates[
            ~unresolved_candidates.apply(
                lambda row: interval_fully_inside_broken(
                    row["meter_name"],
                    pd.to_datetime(row["start_datetime"], errors="coerce"),
                    pd.to_datetime(row["end_datetime"], errors="coerce"),
                ),
                axis=1,
            )
        ].copy()

    issue_df = suggest_meter_issues(data)
    if issue_df.empty:
        issue_df = pd.DataFrame(columns=[
            "meter_name",
            "issue_type",
            "start_datetime",
            "end_datetime",
            "description",
            "suggestion",
        ])

    issue_df = issue_df.copy()

    if "details" in issue_df.columns and "description" not in issue_df.columns:
        issue_df = issue_df.rename(columns={"details": "description"})

    if "description" not in issue_df.columns:
        issue_df["description"] = ""

    if "issue_type/status" in issue_df.columns and "issue_type" not in issue_df.columns:
        issue_df = issue_df.rename(columns={"issue_type/status": "issue_type"})

    if "issue_type" not in issue_df.columns:
        issue_df["issue_type"] = ""

    issue_df["meter_name"] = issue_df["meter_name"].apply(_normalize_text)
    issue_df["solution"] = ""
    issue_df["approved"] = 0

    if df_bad_meters is not None and not df_bad_meters.empty:
        meta = df_bad_meters.copy()
        meta["meter_name"] = meta["meter_name"].apply(_normalize_text)
        meta = meta[["meter_name", "r2"]]
        issue_df = issue_df.merge(meta, on="meter_name", how="left")
    else:
        issue_df["r2"] = np.nan

    for col in candidate_cols:
        if col not in unresolved_candidates.columns:
            unresolved_candidates[col] = np.nan if col not in {"solution", "description", "suggestion"} else ""

    for col in candidate_cols:
        if col not in issue_df.columns:
            issue_df[col] = np.nan if col not in {"solution", "description", "suggestion"} else ""

    unresolved_candidates = unresolved_candidates.reindex(columns=candidate_cols)
    issue_df = issue_df.reindex(columns=candidate_cols)

    if not issue_df.empty:
        issue_type_series = issue_df.get(
            "issue_type",
            pd.Series("", index=issue_df.index)
        )

        issue_df = issue_df[
            ~(
                issue_type_series.fillna("").astype(str).str.strip().str.lower().eq("all_nan")
                & issue_df["meter_name"].isin(fully_broken_meters)
            )
        ].copy()

        issue_df = issue_df[
            ~issue_df.apply(
                lambda row: interval_fully_inside_broken(
                    row["meter_name"],
                    row["start_datetime"],
                    row["end_datetime"],
                ),
                axis=1,
            )
        ].copy()

    summary_rows = pd.DataFrame(columns=candidate_cols)

    special_meter_summary_export = pd.DataFrame(columns=["meter_name", "r2", "info"])

    if df_bad_meters is not None and not df_bad_meters.empty:
        summary_df = df_bad_meters.copy()
        summary_df["meter_name"] = summary_df["meter_name"].apply(_normalize_text)
        summary_df["issue_type"] = summary_df["info"].apply(summary_info_to_issue_type)

        special_meter_summary_export = summary_df[["meter_name", "r2", "info"]].copy()

        fully_broken_all_nan_mask = (
            special_meter_summary_export["meter_name"].isin(fully_broken_meters)
            & summary_df["issue_type"].eq("all_nan")
        )

        special_meter_summary_export.loc[
            fully_broken_all_nan_mask,
            "info"
        ] = "all NaN — fully broken during analysis window"

        summary_row_list = []
        for row in summary_df.itertuples(index=False):
            meter_name = row.meter_name
            issue_type = row.issue_type
            info_value = str(row.info).strip().lower()
            r2_value = row.r2

            if meter_name in fully_broken_meters and issue_type in {"all_nan", "missing_start", "missing_end", "missing_start_and_end"}:
                continue

            if issue_type in {"missing_start", "missing_end", "missing_start_and_end"}:
                summary_row_list.extend(build_edge_missing_rows(meter_name, info_value, r2_value))
            elif issue_type in {"low_r2", "too_few_points"}:
                summary_row_list.append({
                    "meter_name": meter_name,
                    "solution": "",
                    "start_datetime": pd.NaT,
                    "end_datetime": pd.NaT,
                    "issue_type": issue_type,
                    "description": str(row.info),
                    "suggestion": "review_summary",
                    "r2": r2_value,
                    "approved": 0,
                })

        summary_rows = pd.DataFrame(summary_row_list, columns=candidate_cols)
        summary_rows = summary_rows.reindex(columns=candidate_cols)

    combined_candidates = pd.concat(
        [
            unresolved_candidates,
            issue_df,
            summary_rows,
        ],
        ignore_index=True,
        sort=False,
    )

    if not combined_candidates.empty:
        temp = combined_candidates.copy()

        if "issue_type" not in temp.columns:
            temp["issue_type"] = ""

        temp["meter_name"] = temp["meter_name"].fillna("").astype(str).str.strip().str.lower()
        temp["issue_type"] = temp["issue_type"].fillna("").astype(str).str.strip().str.lower()
        temp["description"] = temp["description"].fillna("").astype(str).str.strip().str.lower()
        temp["start_datetime"] = pd.to_datetime(temp["start_datetime"], errors="coerce")
        temp["end_datetime"] = pd.to_datetime(temp["end_datetime"], errors="coerce")

        temp["_key"] = (
            temp["meter_name"] + "||"
            + temp["issue_type"] + "||"
            + temp["description"] + "||"
            + temp["start_datetime"].astype(str) + "||"
            + temp["end_datetime"].astype(str)
        )

        combined_candidates["_key"] = temp["_key"]
        combined_candidates = combined_candidates.drop_duplicates(subset="_key", keep="first").drop(columns="_key")

        # stuck runs skip review: they go straight to the master sheet, not the candidate file
        is_stuck = temp.loc[combined_candidates.index, "issue_type"].eq("stuck_run")
        stuck_rows = combined_candidates[is_stuck].copy()
        combined_candidates = combined_candidates[~is_stuck]

        if meter_corrections_file is not None and not stuck_rows.empty:
            stuck_rows["solution"] = stuck_solution
            add_stuck_runs_to_master_sheet(stuck_rows, meter_corrections_file)

        # rows overlapping an open-ended broken row (purple on the review PDF) may mean
        # the meter is not actually broken: flag them for the remove_broken solution
        open_broken = broken_df[broken_df["end_date"].isna()]
        for idx, row in combined_candidates.iterrows():
            cand_start = pd.to_datetime(row["start_datetime"], errors="coerce")
            cand_end = pd.to_datetime(row["end_datetime"], errors="coerce")
            if pd.isna(cand_start) or pd.isna(cand_end):
                continue
            meter_open = open_broken[open_broken["meter_name"].eq(row["meter_name"])]
            if (meter_open["start_date"].isna() | (meter_open["start_date"] <= cand_end)).any():
                combined_candidates.loc[idx, "suggestion"] = "review_remove_broken"

        combined_candidates = combined_candidates.sort_values(
            by=["meter_name", "start_datetime", "issue_type"],
            na_position="last",
        ).reset_index(drop=True)

    output_dir = os.path.dirname(str(candidate_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    combined_candidates.to_csv(candidate_file, index=False)

    restart_output_file = os.path.splitext(candidate_file)[0] + "_restarts.csv"
    if df_restarts is not None and not df_restarts.empty:
        df_restarts.to_csv(restart_output_file, index=False)
        print(f"Restart file saved to {restart_output_file}")
    elif os.path.exists(restart_output_file):
        os.remove(restart_output_file)

    print(f"Candidate review file saved to {candidate_file}")

    if df_bad_meters is not None and not special_meter_summary_export.empty:
        print("\nSpecial meter summary:")
        print(special_meter_summary_export.to_string(index=False))

    return combined_candidates


###################################################################
###################################################################


def get_review_plot_legend_handles(
    include_candidates=True,
    include_removed=False,
):
    """Return consistent legend entries for endpoint and maintenance PDFs."""
    handles = [
        Line2D(
            [0],
            [0],
            color="#1f77b4",
            linewidth=1.4,
            label="Raw meter reading",
        ),
        Patch(
            facecolor="red",
            edgecolor="red",
            alpha=0.20,
            label="Broken meter interval",
        ),
    ]

    if include_candidates:
        handles.extend([
            Patch(
                facecolor="blue",
                edgecolor="blue",
                alpha=0.20,
                label="Candidate issue interval",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Restart / drop event",
            ),
        ])

    if include_removed:
        handles.append(
            Line2D(
                [0],
                [0],
                color="orange",
                marker="o",
                linestyle="None",
                markersize=6,
                label="Removed raw reading",
            )
        )

    return handles


REPORT_RANGE_LINE_STYLE = dict(color="green", linestyle=":", linewidth=1.6)


def _report_range_legend_handle():
    return Line2D([0], [0], label="Report range start / end", **REPORT_RANGE_LINE_STYLE)


def _draw_report_range(ax, report_start=None, report_end=None):
    """Vertical lines at the report range edges (e.g. fiscal year) when the plot shows a wider data range."""
    for edge in (report_start, report_end):
        if edge is not None:
            ax.axvline(pd.to_datetime(edge), alpha=0.9, **REPORT_RANGE_LINE_STYLE)


def _add_pdf_legend_page(
    pdf,
    title,
    handles,
    description="",
    notes=None,
):
    """
    Add a standalone legend page as the first page of a multipage PDF.

    Parameters
    ----------
    pdf : PdfPages
        Open PdfPages object.
    title : str
        Main title displayed at the top of the page.
    handles : list
        Matplotlib Line2D/Patch objects used for the legend.
    description : str
        Short explanation displayed below the title.
    notes : list[str] or None
        Additional explanatory notes displayed below the legend.
    """
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")

    fig.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
        y=0.91,
    )

    if description:
        fig.text(
            0.50,
            0.82,
            description,
            ha="center",
            va="top",
            fontsize=11,
            wrap=True,
        )

    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.50, 0.55),
        title="Plot key",
        fontsize=12,
        title_fontsize=14,
        frameon=True,
        borderpad=1.2,
        labelspacing=1.2,
        handlelength=3,
    )

    if notes:
        note_text = "\n".join(f"• {note}" for note in notes)
        fig.text(
            0.50,
            0.23,
            note_text,
            ha="center",
            va="top",
            fontsize=10,
            linespacing=1.5,
        )

    fig.text(
        0.50,
        0.08,
        "This key applies to every graph on the following pages.",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="0.35",
    )

    pdf.savefig(fig)
    plt.close(fig)


def _format_review_r2_value(r2_val):
    if pd.isna(r2_val):
        return ""
    try:
        r2_float = float(r2_val)
        if r2_float > 0:
            return f"R² = {r2_float:.4f}"
        return f"R² = {int(r2_float)}"
    except (TypeError, ValueError):
        return f"R² = {r2_val}"
    

def _plot_single_meter_review(
    meter_name,
    data,
    broken_df=None,
    candidate_df=None,
    removed_df=None,
    meter_type_map=None,
    zoom_start=None,
    zoom_end=None,
    ylabel="kWh",
    annotation_fields=("issue_type", "r2"),
    title_suffix="",
    footer_note="",
    report_start=None,
    report_end=None,
):
    """
    Draw one review meter graph.

    This is the shared plotting engine used by endpoint review PDFs,
    maintenance review PDFs, and approved change preview PDFs.
    report_start / report_end (optional) draw vertical report range lines.
    """
    if data is None or data.empty:
        raise ValueError("data is empty.")

    meter_name_norm = str(meter_name).strip().lower()
    data_columns_by_name = {
        str(col).strip().lower(): col
        for col in data.columns
    }

    if meter_name_norm not in data_columns_by_name:
        print(f"{meter_name} not found in selected main/submeter data.")
        return None

    data_col = data_columns_by_name[meter_name_norm]
    meter_series = data[data_col].copy()
    meter_series.index = pd.to_datetime(meter_series.index, errors="coerce")
    meter_series = meter_series[~meter_series.index.isna()].sort_index()

    if meter_series.empty:
        print(f"{meter_name} has no valid timestamps to plot.")
        return None

    broken_df = pd.DataFrame() if broken_df is None else broken_df.copy()
    candidate_df = pd.DataFrame() if candidate_df is None else candidate_df.copy()
    removed_df = pd.DataFrame() if removed_df is None else removed_df.copy()

    if "meter_name" in broken_df.columns:
        broken_meter_names = (
            broken_df["meter_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        meter_broken = broken_df[broken_meter_names.eq(meter_name_norm)].copy()
    else:
        meter_broken = pd.DataFrame()

    if "meter_name" in candidate_df.columns:
        candidate_meter_names = (
            candidate_df["meter_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        meter_candidates = candidate_df[
            candidate_meter_names.eq(meter_name_norm)
        ].copy()
    else:
        meter_candidates = pd.DataFrame()

    if "meter_name" in removed_df.columns:
        removed_meter_names = (
            removed_df["meter_name"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        meter_removed = removed_df[
            removed_meter_names.eq(meter_name_norm)
        ].copy()
    else:
        meter_removed = pd.DataFrame()

    plot_start = pd.to_datetime(meter_series.index.min())
    plot_end = pd.to_datetime(meter_series.index.max())

    if zoom_start is not None:
        plot_start = max(plot_start, pd.to_datetime(zoom_start))
    if zoom_end is not None:
        plot_end = min(plot_end, pd.to_datetime(zoom_end))
    if plot_start > plot_end:
        raise ValueError(
            f"Resolved plot window is empty for meter '{meter_name_norm}'."
        )

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(
        meter_series.index,
        meter_series.values,
        color="#1f77b4",
        linewidth=1.2,
    )
    ax.set_xlim(plot_start, plot_end)
    _draw_report_range(ax, report_start, report_end)

    # label meter '[main]' or '[sub]' on plot
    meter_type = ""
    if meter_type_map is not None:
        normalized_type_map = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in meter_type_map.items()
        }
        meter_type = normalized_type_map.get(meter_name_norm, "")

    meter_title = (
        f"{meter_name_norm} [{meter_type}]"
        if meter_type in {"main", "submeter"}
        else meter_name_norm
    )
    if title_suffix:
        meter_title = f"{meter_title}{title_suffix}"

    ax.set_title(meter_title)
    ax.set_xlabel("Datetime")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    broken_start_col = None
    broken_end_col = None
    if {"start_date", "end_date"}.issubset(meter_broken.columns):
        broken_start_col = "start_date"
        broken_end_col = "end_date"
    elif {"start_datetime", "end_datetime"}.issubset(meter_broken.columns):
        broken_start_col = "start_datetime"
        broken_end_col = "end_datetime"

    if broken_start_col is not None:
        for _, row in meter_broken.iterrows():
            raw_start = pd.to_datetime(row[broken_start_col], errors="coerce")
            raw_end = pd.to_datetime(row[broken_end_col], errors="coerce")
            start = plot_start if pd.isna(raw_start) else max(raw_start, plot_start)
            end = plot_end if pd.isna(raw_end) else min(raw_end, plot_end)

            if start <= end:
                ax.axvspan(
                    start,
                    end,
                    facecolor="red",
                    edgecolor="red",
                    alpha=0.20,
                )

                # label the band with its broken dates (open start/end left blank)
                if "broken_dates" in annotation_fields:
                    start_text = "" if pd.isna(raw_start) else f"{raw_start.month}/{raw_start.day}/{raw_start.year}"
                    end_text = "" if pd.isna(raw_end) else f"{raw_end.month}/{raw_end.day}/{raw_end.year}"
                    status = str(row.get("status", "")).strip()
                    label = f"{status} {start_text} – {end_text}".strip()

                    # narrow bands get a vertical label so it doesn't spill over the plot
                    narrow = (end - start) < (plot_end - plot_start) * 0.12
                    ax.text(
                        start + (end - start) / 2 if narrow else start,
                        0.03,
                        label,
                        transform=ax.get_xaxis_transform(),
                        ha="center" if narrow else "left",
                        va="bottom",
                        rotation=90 if narrow else 0,
                        fontsize=8,
                        color="darkred",
                        clip_on=True,
                    )

    if not meter_candidates.empty:
        for _, row in meter_candidates.iterrows():
            issue_type = str(row.get("issue_type", "")).strip().lower()
            start = pd.to_datetime(
                row.get("start_datetime", pd.NaT),
                errors="coerce",
            )
            end = pd.to_datetime(
                row.get("end_datetime", pd.NaT),
                errors="coerce",
            )

            if issue_type == "restart_or_drop":
                if pd.notna(end) and plot_start <= end <= plot_end:
                    ax.axvline(
                        end,
                        color="black",
                        linestyle="--",
                        linewidth=1.2,
                        alpha=0.9,
                    )
            elif pd.notna(start) and pd.notna(end):
                start = max(start, plot_start)
                end = min(end, plot_end)
                if start <= end:
                    ax.axvspan(
                        start,
                        end,
                        facecolor="blue",
                        edgecolor="blue",
                        alpha=0.20,
                    )

    removed_plot = pd.DataFrame()
    if (
        not meter_removed.empty
        and "datetime" in meter_removed.columns
        and "meter_reading" in meter_removed.columns
    ):
        meter_removed["datetime"] = pd.to_datetime(
            meter_removed["datetime"],
            errors="coerce",
        )
        removed_plot = meter_removed[
            meter_removed["datetime"].between(
                plot_start,
                plot_end,
                inclusive="both",
            )
        ].copy()

        if not removed_plot.empty:
            ax.scatter(
                removed_plot["datetime"],
                removed_plot["meter_reading"],
                s=10,
                alpha=0.8,
                marker="o",
                color="orange",
            )

    annotation_lines = []

    if "status" in annotation_fields and "status" in meter_broken.columns:
        status_values = [
            str(value).strip()
            for value in pd.unique(meter_broken["status"].dropna())
            if str(value).strip() != ""
        ]
        if status_values:
            annotation_lines.append(
                "Broken status: " + ", ".join(status_values)
            )

    if (
        "issue_type" in annotation_fields
        and "issue_type" in meter_candidates.columns
    ):
        issue_types = [
            str(value).strip()
            for value in pd.unique(meter_candidates["issue_type"].dropna())
            if str(value).strip() != ""
        ]
        if issue_types:
            label = (
                "Issue type: "
                if "status" in annotation_fields or "broken_dates" in annotation_fields
                else "Issue type: "
            )
            annotation_lines.append(label + ", ".join(issue_types))

    if "r2" in annotation_fields and "r2" in meter_candidates.columns:
        r2_values = [
            _format_review_r2_value(value)
            for value in pd.unique(meter_candidates["r2"].dropna())
            if _format_review_r2_value(value) != ""
        ]
        if r2_values:
            annotation_lines.append(", ".join(r2_values))

    if "removed_count" in annotation_fields and not meter_removed.empty:
        annotation_lines.append(f"Removed raw points: {len(meter_removed)}")

    if annotation_lines:
        ax.text(
            0.02,
            0.97,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.80,
            ),
        )

    if footer_note:
        ax.text(
            0.99,
            0.02,
            footer_note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="gray",
                alpha=0.80,
            ),
        )

    fig.tight_layout()
    return fig


def plot_all_meters_to_pdf(
    data,
    plot_file,
    ylabel="meter_reading",
    line_label="Corrected meter reading",
    meter_type_map=None,
    report_start=None,
    report_end=None,
):
    """
    Plot every meter in the current dataframe to one PDF.

    This function is currently called with data_corrected in
    1.kwh_end_points.ipynb, so each page identifies the line as
    corrected meter data. report_start / report_end (optional) draw
    vertical report range lines.
    """
    if data is None or data.empty:
        print("No data available for all-meter plotting.")
        return 0

    output_dir = os.path.dirname(str(plot_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data = data.copy()
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.sort_index()

    data_start = pd.to_datetime(data.index.min())
    data_end = pd.to_datetime(data.index.max())
    saved_count = 0

    with PdfPages(plot_file) as pdf:
        all_meter_legend_handles = [
            Line2D(
                [0],
                [0],
                color="#1f77b4",
                linewidth=1.4,
                label=line_label,
            ),
        ]
        if report_start is not None or report_end is not None:
            all_meter_legend_handles.append(_report_range_legend_handle())

        _add_pdf_legend_page(
            pdf=pdf,
            title="All Corrected Meter Plots",
            handles=all_meter_legend_handles,
            description=(
                "These plots show meter readings after approved special meter "
                "corrections have been applied."
            ),
            notes=[
                "Each following page contains one meter.",
                "Gaps may represent original missing readings.",
                "Gaps may also represent approved remove or broken meter intervals.",
            ],
        )

        normalized_type_map = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in (meter_type_map or {}).items()
        }

        for meter_name in data.columns:
            fig, ax = plt.subplots(figsize=(14, 3.5))
            ax.plot(
                data.index,
                data[meter_name],
                color="#1f77b4",
                linewidth=1.2,
                label=line_label,
            )
            ax.set_xlim(data_start, data_end)
            _draw_report_range(ax, report_start, report_end)

            meter_name_norm = str(meter_name).strip().lower()
            meter_type = normalized_type_map.get(meter_name_norm, "")
            meter_title = (
                f"{meter_name} [{meter_type}]"
                if meter_type
                else str(meter_name)
            )
            ax.set_title(meter_title)
            ax.set_xlabel("Datetime")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved_count += 1

    print(f"All-meter plots saved to {plot_file}")
    return saved_count



def plot_review_meters_with_overlays(
    data,
    candidate_file,
    broken_meters_file,
    plot_file,
    ylabel="kWh",
    meter_type_map=None,
    report_start=None,
    report_end=None,
):
    """
    Create the kwh_end_points.ipynb review PDF using the shared meter plot function
    with red broken meter overlays and blue candidate overlays.
    report_start / report_end (optional) draw vertical report range lines.
    """
    if data is None or data.empty:
        raise ValueError("data is empty.")

    candidate_df = load_existing_candidates(candidate_file)
    broken_source_df = load_broken_meter_workbook(broken_meters_file)

    candidate_meters = set(candidate_df.get("meter_name", pd.Series(dtype=str)))
    broken_meters = set(broken_source_df.get("meter_name", pd.Series(dtype=str)))
    data_meters = {str(col).strip().lower() for col in data.columns}

    review_meters = sorted(
        (candidate_meters | broken_meters) & data_meters
    )

    if not review_meters:
        print("No review meters found in data.")
        return 0

    output_dir = os.path.dirname(str(plot_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    review_handles = get_review_plot_legend_handles(
        include_candidates=True,
        include_removed=False,
    )
    if report_start is not None or report_end is not None:
        review_handles.append(_report_range_legend_handle())

    saved_count = 0
    with PdfPages(plot_file) as pdf:
        _add_pdf_legend_page(
            pdf=pdf,
            title="Review Meter Plots",
            handles=review_handles,
            description=(
                "These plots show raw meter readings together with broken meter "
                "intervals and unresolved candidate issues."
            ),
            notes=[
                "Red and blue shaded intervals may overlap and appear purple.",
                "Purple on an open-ended red interval (suggestion review_remove_broken): "
                "use solution remove_broken if the meter is not actually broken.",
                "A black dashed line marks the detected restart or drop timestamp.",
                "Each red interval is labeled with its status and dates; a blank start or end means it is open.",
                "Candidate issue type and R² appear in the box at the top left of each plot.",
            ],
        )

        for meter_name in review_meters:
            fig = _plot_single_meter_review(
                meter_name=meter_name,
                data=data,
                broken_df=broken_source_df,
                candidate_df=candidate_df,
                removed_df=None,
                meter_type_map=meter_type_map,
                ylabel=ylabel,
                annotation_fields=("broken_dates", "issue_type", "r2"),
                report_start=report_start,
                report_end=report_end,
            )
            if fig is None:
                continue
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            saved_count += 1

    print(f"Review overlay plots saved to {plot_file}")
    return saved_count


# meter maintenance ploting:
def plot_meter_maintenance_review_pdf(
    data,
    meters_to_review,
    broken_df,
    candidate_df,
    removed_df,
    plot_file,
    meter_type_map=None,
    zoom_start=None,
    zoom_end=None,
    ylabel="kWh",
    show_plots=False,
):
    """
    Create the selected meter maintenance review PDF using the shared meter plot function
    with red broken meter overlays and blue candidate overlays, and also 
    orange data markers for removed meter data.
    """
    meters_to_review = [
        str(meter).strip().lower()
        for meter in (meters_to_review or [])
        if str(meter).strip() != ""
    ]
    if not meters_to_review:
        print("No selected main or submeter review meters were found.")
        return 0

    output_dir = os.path.dirname(str(plot_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    with PdfPages(plot_file) as pdf:
        _add_pdf_legend_page(
            pdf=pdf,
            title="Meter Maintenance Review Plots",
            handles=get_review_plot_legend_handles(
                include_candidates=True,
                include_removed=True,
            ),
            description=(
                "These plots show raw readings for selected main meters and "
                "submeters together with maintenance review overlays."
            ),
            notes=[
                "Every plot title identifies the meter as [main] or [submeter].",
                "Red and blue intervals may overlap and appear purple.",
                "Orange points preserve readings removed by an approved correction.",
            ],
        )

        for meter_name in meters_to_review:
            fig = _plot_single_meter_review(
                meter_name=meter_name,
                data=data,
                broken_df=broken_df,
                candidate_df=candidate_df,
                removed_df=removed_df,
                meter_type_map=meter_type_map,
                zoom_start=zoom_start,
                zoom_end=zoom_end,
                ylabel=ylabel,
                annotation_fields=(
                    "status",
                    "issue_type",
                    "removed_count",
                ),
            )
            if fig is None:
                continue
            pdf.savefig(fig, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close(fig)
            saved_count += 1

    if saved_count:
        print(f"Saved maintenance review plots to {plot_file}")
    return saved_count


def plot_approved_maintenance_preview_pdf(
    data,
    changed_meters,
    reviewed_broken_df,
    plot_file,
    meter_type_map=None,
    zoom_start=None,
    zoom_end=None,
    ylabel="kWh",
    show_plots=False,
):
    """
    Preview approved broken meter changes before overwriting the source CSV.
    """
    changed_meters = [
        str(meter).strip().lower()
        for meter in (changed_meters or [])
        if str(meter).strip() != ""
    ]
    if not changed_meters:
        print("No approved maintenance changes were available to preview.")
        return 0

    output_dir = os.path.dirname(str(plot_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    with PdfPages(plot_file) as pdf:
        _add_pdf_legend_page(
            pdf=pdf,
            title="Approved Maintenance Changes Preview",
            handles=get_review_plot_legend_handles(
                include_candidates=False,
                include_removed=False,
            ),
            description=(
                "These plots preview approved broken meter interval changes "
                "before the source running list is overwritten."
            ),
            notes=[
                "Every plot title identifies the meter as [main] or [submeter].",
                "Red shading shows the proposed post action broken interval.",
                "Candidate and removed data overlays are intentionally omitted.",
            ],
        )

        for meter_name in changed_meters:
            fig = _plot_single_meter_review(
                meter_name=meter_name,
                data=data,
                broken_df=reviewed_broken_df,
                candidate_df=None,
                removed_df=None,
                meter_type_map=meter_type_map,
                zoom_start=zoom_start,
                zoom_end=zoom_end,
                ylabel=ylabel,
                annotation_fields=("status",),
                title_suffix=" — approved broken interval preview",
                footer_note=(
                    "Red = updated broken interval\n"
                    "running_list_broken_meters.csv has not been changed\n"
                    "Candidates and removed data points are not regenerated yet"
                ),
            )
            if fig is None:
                continue
            pdf.savefig(fig, bbox_inches="tight")
            if show_plots:
                plt.show()
            plt.close(fig)
            saved_count += 1

    if saved_count:
        print(
            f"Saved {saved_count} approved change preview plots to {plot_file}."
        )
    return saved_count



###################################################################
# METER MAINTENANCE UPDATE LOG WORKFLOW
###################################################################

BROKEN_UPDATE_LOG_COLUMNS = [
    "meter_name",
    "source_start_date",
    "source_end_date",
    "source_status",
    "source_description",
    "action",
    "new_start_date",
    "new_end_date",
    "new_status",
    "new_description",
    "reason",
    "approved",
]

REQUIRED_BROKEN_METER_COLUMNS = [
    "meter_name",
    "start_date",
    "end_date",
    "status",
    "description",
    "data_source",
    "updated_data",
]


def datetime_to_text(value):
    """
    Convert a scalar datetime-like value to the workflow CSV format.
    Returns an empty string for NaT or invalid values.
    """
    if pd.isna(value) or value == "":
        return ""
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return ""
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_strict_datetime_value(value, field_name):
    """
    Parse one maintenance log datetime using the shared strict formats.
    Returns pd.NaT for empty or invalid values.
    """
    value = "" if pd.isna(value) else str(value).strip()
    if value == "":
        return pd.NaT
    return _parse_strict_date_series(
        pd.Series([value]),
        field_name,
    ).iloc[0]


def _build_update_key(df):
    temp = df.copy()
    for col in [
        "meter_name",
        "source_start_date",
        "source_end_date",
        "source_status",
    ]:
        if col not in temp.columns:
            temp[col] = ""
        temp[col] = temp[col].fillna("").astype(str).str.strip().str.lower()
    return (
        temp["meter_name"] + "||"
        + temp["source_start_date"] + "||"
        + temp["source_end_date"] + "||"
        + temp["source_status"]
    )


def prepare_broken_meter_update_log(
    update_log_file,
    broken_source_df,
    meters_to_review,
    refresh_selected_meters=True,
):
    """
    Create/load the maintenance update log and refresh selected source rows.
    """
    output_dir = os.path.dirname(str(update_log_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(update_log_file):
        update_log_df = pd.read_csv(update_log_file)
        missing_cols = [
            col
            for col in BROKEN_UPDATE_LOG_COLUMNS
            if col not in update_log_df.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Update log file is missing columns: {missing_cols}"
            )
    else:
        update_log_df = pd.DataFrame(columns=BROKEN_UPDATE_LOG_COLUMNS)
        update_log_df.to_csv(update_log_file, index=False)
        print(f"Created new update log: {update_log_file}")

    normalized_meters = [
        _normalize_text(meter)
        for meter in (meters_to_review or [])
        if _normalize_text(meter) != ""
    ]

    # Refresh the update log with the current source rows for the selected meters.
    if refresh_selected_meters and normalized_meters:
        selected_rows = []

        for meter_name in normalized_meters:
            meter_source = broken_source_df[
                broken_source_df["meter_name"].eq(meter_name)
            ].copy()

            if meter_source.empty:
                selected_rows.append({
                    "meter_name": meter_name,
                    "source_start_date": "",
                    "source_end_date": "",
                    "source_status": "",
                    "source_description": "",
                    "action": "",
                    "new_start_date": "",
                    "new_end_date": "",
                    "new_status": "",
                    "new_description": "",
                    "reason": "",
                    "approved": 0,
                })
            else:
                for _, row in meter_source.iterrows():
                    selected_rows.append({
                        "meter_name": meter_name,
                        "source_start_date": datetime_to_text(row["start_date"]),
                        "source_end_date": datetime_to_text(row["end_date"]),
                        "source_status": str(row["status"]).strip(),
                        "source_description": str(row["description"]).strip(),
                        "action": "",
                        "new_start_date": "",
                        "new_end_date": "",
                        "new_status": "",
                        "new_description": "",
                        "reason": "",
                        "approved": 0,
                    })

        selected_log_df = pd.DataFrame(
            selected_rows,
            columns=BROKEN_UPDATE_LOG_COLUMNS,
        )

        update_log_df["_key"] = _build_update_key(update_log_df)
        selected_log_df["_key"] = _build_update_key(selected_log_df)

        selected_keys = set(selected_log_df["_key"])
        preserved_existing = update_log_df[update_log_df["_key"].isin(selected_keys)].copy()
        preserved_other = update_log_df[~update_log_df["_key"].isin(selected_keys)].copy()

        combined_refresh_rows = pd.concat(
            [preserved_existing, selected_log_df],
            ignore_index=True,
            sort=False,
        )
        combined_refresh_rows = combined_refresh_rows.drop_duplicates(
            subset="_key",
            keep="first",
        )
        update_log_df = pd.concat(
            [preserved_other, combined_refresh_rows],
            ignore_index=True,
            sort=False,
        )
        update_log_df = update_log_df.drop(
            columns="_key",
            errors="ignore",
        )

    update_log_df["approved"] = (
        pd.to_numeric(update_log_df["approved"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    update_log_df = update_log_df[BROKEN_UPDATE_LOG_COLUMNS].copy()
    update_log_df.to_csv(update_log_file, index=False)
    return update_log_df


def _match_existing_broken_rows(df, log_row):
    meter_name = _normalize_text(log_row["meter_name"])
    src_start = datetime_to_text(log_row["source_start_date"])
    src_end = datetime_to_text(log_row["source_end_date"])
    src_status = _normalize_text(log_row["source_status"], underscore=True)
    src_desc = _normalize_text(log_row["source_description"])

    mask = (
        df["meter_name"].fillna("").astype(str).str.strip().str.lower().eq(meter_name)
        & df["status"].fillna("").astype(str).str.strip().str.lower().eq(src_status)
        & df["description"].fillna("").astype(str).str.strip().str.lower().eq(src_desc)
    )

    start_text = df["start_date"].apply(datetime_to_text)
    end_text = df["end_date"].apply(datetime_to_text)
    mask = mask & start_text.eq(src_start) & end_text.eq(src_end)
    return df[mask]


def _log_value_is_filled(value):
    return not pd.isna(value) and str(value).strip() != ""


def apply_broken_meter_update_actions(
    broken_source_df,
    update_log_df,
    reviewed_copy_file=None,
    now_ts=None,
):
    """
    Parameters
    broken_source_df : pd.DataFrame
        The current running list of broken meter intervals.
    Apply approved add/update/remove actions to a copy of the running list.

    The source dataframe is never modified in place. When reviewed_copy_file is
    supplied, the complete reviewed dataframe overwrites that copy file.
    """
    approved_actions = {"update", "remove", "add"}

    approved_update_rows = update_log_df[
        pd.to_numeric(update_log_df["approved"], errors="coerce")
        .fillna(0)
        .astype(int)
        .eq(1)
    ].copy()
    approved_update_rows["action"] = (
        approved_update_rows["action"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_actions = sorted([
        action
        for action in approved_update_rows["action"].unique()
        if action not in approved_actions
    ])
    if invalid_actions:
        raise ValueError(
            f"Unsupported action values in approved rows: {invalid_actions}"
        )

    reviewed_broken_df = broken_source_df.copy()
    for col in REQUIRED_BROKEN_METER_COLUMNS:
        if col not in reviewed_broken_df.columns:
            reviewed_broken_df[col] = pd.NaT if "date" in col else ""

    reviewed_broken_df["meter_name"] = reviewed_broken_df["meter_name"].apply(_normalize_text)
    reviewed_broken_df["status"] = reviewed_broken_df["status"].apply(lambda value: _normalize_text(value, underscore=True))
    reviewed_broken_df["description"] = reviewed_broken_df["description"].apply(lambda value: _normalize_text(value, lower=False))
    reviewed_broken_df["data_source"] = reviewed_broken_df["data_source"].apply(_normalize_text)
    reviewed_broken_df["start_date"] = pd.to_datetime(reviewed_broken_df["start_date"], errors="coerce")
    reviewed_broken_df["end_date"] = pd.to_datetime(reviewed_broken_df["end_date"], errors="coerce")
    reviewed_broken_df["updated_data"] = pd.to_datetime(reviewed_broken_df["updated_data"], errors="coerce")

    applied_change_log = []
    if now_ts is None:
        now_ts = pd.Timestamp.now().floor("s")
    else:
        now_ts = pd.to_datetime(now_ts).floor("s")

    for _, log_row in approved_update_rows.iterrows():
        action = log_row["action"]
        meter_name = _normalize_text(log_row["meter_name"])

        if meter_name == "":
            raise ValueError("Approved update log row is missing meter_name.")

        if action in {"update", "remove"}:
            matches = _match_existing_broken_rows(reviewed_broken_df, log_row)

            if len(matches) == 0:
                raise ValueError(
                    f"No matching source row found for action '{action}' "
                    f"on meter '{meter_name}'. Check source_* fields in the "
                    "update log."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"Multiple matching source rows found for action "
                    f"'{action}' on meter '{meter_name}'. Source match must "
                    "be unique."
                )

            match_idx = matches.index[0]

            # if action is "remove", drop the matched row from reviewed_broken_df and log the removal
            if action == "remove":
                removed_row = reviewed_broken_df.loc[match_idx].copy()
                reviewed_broken_df = reviewed_broken_df.drop(index=match_idx).reset_index(drop=True)

                applied_change_log.append({
                    "meter_name": meter_name,
                    "action": "remove",
                    "result": "removed source row",
                    "old_start_date": datetime_to_text(removed_row["start_date"]),
                    "old_end_date": datetime_to_text(removed_row["end_date"]),
                    "old_status": removed_row["status"],
                    "new_start_date": "",
                    "new_end_date": "",
                    "new_status": "",
                })
                continue

            # if action is "update"
            old_row = reviewed_broken_df.loc[match_idx].copy()
            new_start = (
                parse_strict_datetime_value(
                    log_row["new_start_date"],
                    "new_start_date",
                )
                if _log_value_is_filled(log_row["new_start_date"])
                else old_row["start_date"]
            )
            new_end = (
                parse_strict_datetime_value(
                    log_row["new_end_date"],
                    "new_end_date",
                )
                if _log_value_is_filled(log_row["new_end_date"])
                else old_row["end_date"]
            )
            new_status = (
                _normalize_text(
                    log_row["new_status"],
                    underscore=True,
                )
                if _log_value_is_filled(log_row["new_status"])
                else old_row["status"]
            )
            new_desc = (
                _normalize_text(log_row["new_description"], lower=False)
                if _log_value_is_filled(log_row["new_description"])
                else old_row["description"]
            )

            if new_status not in BROKEN_STATUS_VALUES:
                raise ValueError(
                    f"Invalid new_status for update on meter "
                    f"'{meter_name}': {new_status}. Allowed: "
                    f"{sorted(BROKEN_STATUS_VALUES)}"
                )

            reviewed_broken_df.loc[match_idx, "start_date"] = new_start
            reviewed_broken_df.loc[match_idx, "end_date"] = new_end
            reviewed_broken_df.loc[match_idx, "status"] = new_status
            reviewed_broken_df.loc[match_idx, "description"] = new_desc
            reviewed_broken_df.loc[match_idx, "updated_data"] = now_ts

            applied_change_log.append({
                "meter_name": meter_name,
                "action": "update",
                "result": "updated source row",
                "old_start_date": datetime_to_text(old_row["start_date"]),
                "old_end_date": datetime_to_text(old_row["end_date"]),
                "old_status": old_row["status"],
                "new_start_date": datetime_to_text(new_start),
                "new_end_date": datetime_to_text(new_end),
                "new_status": new_status,
            })

        # if action is "add"
        elif action == "add":
            new_start = (
                parse_strict_datetime_value(
                    log_row["new_start_date"],
                    "new_start_date",
                )
                if _log_value_is_filled(log_row["new_start_date"])
                else pd.NaT
            )
            new_end = (
                parse_strict_datetime_value(
                    log_row["new_end_date"],
                    "new_end_date",
                )
                if _log_value_is_filled(log_row["new_end_date"])
                else pd.NaT
            )
            new_status = _normalize_text(
                log_row["new_status"],
                underscore=True,
            )
            new_desc = _normalize_text(log_row["new_description"], lower=False)

            if new_status == "":
                raise ValueError(
                    f"Action 'add' for meter '{meter_name}' requires "
                    "new_status."
                )
            if new_status not in BROKEN_STATUS_VALUES:
                raise ValueError(
                    f"Invalid new_status for add on meter '{meter_name}': "
                    f"{new_status}. Allowed: "
                    f"{sorted(BROKEN_STATUS_VALUES)}"
                )

            duplicate_mask = (
                reviewed_broken_df["meter_name"].eq(meter_name)
                & reviewed_broken_df["status"].eq(new_status)
                & reviewed_broken_df["description"].str.lower().eq(new_desc.lower())
                & reviewed_broken_df["start_date"].apply(datetime_to_text).eq(datetime_to_text(new_start))
                & reviewed_broken_df["end_date"].apply(datetime_to_text).eq(datetime_to_text(new_end))
            )
            if duplicate_mask.any():
                raise ValueError(
                    f"Action 'add' for meter '{meter_name}' would create "
                    "a duplicate row."
                )

            new_row = pd.DataFrame([{
                "meter_name": meter_name,
                "start_date": new_start,
                "end_date": new_end,
                "status": new_status,
                "description": new_desc,
                "data_source": "maintenance_update_log",
                "updated_data": now_ts,
            }])
            reviewed_broken_df = pd.concat([reviewed_broken_df, new_row], ignore_index=True, sort=False)

            applied_change_log.append({
                "meter_name": meter_name,
                "action": "add",
                "result": "added new row",
                "old_start_date": "",
                "old_end_date": "",
                "old_status": "",
                "new_start_date": datetime_to_text(new_start),
                "new_end_date": datetime_to_text(new_end),
                "new_status": new_status,
            })

    if reviewed_broken_df.empty:
        reviewed_broken_df = pd.DataFrame(columns=REQUIRED_BROKEN_METER_COLUMNS)

    reviewed_broken_df = reviewed_broken_df[REQUIRED_BROKEN_METER_COLUMNS].copy()
    reviewed_broken_df = reviewed_broken_df.sort_values(
        by=["meter_name", "start_date", "end_date", "status"],
        na_position="last",
    ).reset_index(drop=True)

    applied_change_log_df = pd.DataFrame(applied_change_log)

    if reviewed_copy_file is not None and str(reviewed_copy_file).strip() != "":
        output_dir = os.path.dirname(str(reviewed_copy_file))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        reviewed_broken_df.to_csv(
            reviewed_copy_file,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
        )
        print(
            f"Reviewed broken meter copy saved to {reviewed_copy_file}"
        )

    return reviewed_broken_df, applied_change_log_df



###################################################################
###################################################################


def export_meter_differences(df, meter_info_file, filename, var="kwh"):
    """
    Prepare export_df with all columns, but when saving to CSV, only output meter_name and difference.
    - 'difference' column is renamed to 'annual_<var>' in CSV.
    - Difference values are rounded to 1 decimal place for CSV output.
    """
    export_df = df.copy().reset_index()
    
    # Merge building info
    meter_info = pd.read_csv(meter_info_file)[['meter_name', 'building_complex_name', 'end_use']]
    meter_info.columns = meter_info.columns.str.strip()
    export_df = export_df.merge(meter_info, on='meter_name', how='left')
    
    # Prepare CSV output: keep meter_name and difference only
    df_csv = export_df[['meter_name', 'difference']].copy()
    df_csv.rename(columns={'difference': f'annual_{var}'}, inplace=True)
    
    # Round to 1 decimal
    df_csv[f'annual_{var}'] = df_csv[f'annual_{var}'].round(1)
    
    # Save to CSV
    df_csv.to_csv(filename, index=False)

    print(f"Annual {var} saved to {filename}")
    
    return export_df


###################################################################
###################################################################

def export_building_differences(export_df, filename, var="kwh"):
    """
    Aggregate main-meter differences per building and save to CSV.
    - export_df: full meter-level DataFrame (from export_meter_differences), must include 'end_use'
    - filename: output CSV path
    - var: variable name for annual consumption (e.g., 'kwh')

    Only meters with end_use == 'main' are summed; submeters are excluded so
    their usage isn't double-counted alongside their building's main meter.

    The CSV contains:
        building_complex_name, annual_<var>, num_main_meters
    """
    # Keep only main meters, then building_complex_name and difference
    main_only = export_df[export_df['end_use'] == 'main']
    df_building = main_only[['building_complex_name', 'difference']].copy()

    # Aggregate sum per building, keep NaN if all are NaN
    df_building_sum = df_building.groupby('building_complex_name', as_index=False).agg(
        difference=('difference', lambda x: x.sum(min_count=1)),  # sum, NaN if all NaN
        num_main_meters=('difference', 'count')  # count of non-NaN main meters
    )

    # Rename column to annual_<var> and round to 1 decimal
    df_building_sum.rename(columns={'difference': f'annual_{var}'}, inplace=True)
    df_building_sum[f'annual_{var}'] = df_building_sum[f'annual_{var}'].round(1)

    # Save to CSV
    df_building_sum.to_csv(filename, index=False)

    print(f"Building annual kwh file saved to {filename}")
    return df_building_sum



###################################################################
###################################################################

def get_calendar_month_periods(index, start_time, end_time):
    """
    Split [start_time, end_time] into consecutive calendar-month periods,
    each resolved to timestamps that actually exist in `index` (same
    resolution rule as resolve_analysis_window).

    The first/last periods are partial when start_time/end_time don't fall
    on a month boundary. Consecutive periods share their boundary timestamp
    (period i's end == period i+1's start), so summing raw differences
    across periods telescopes back to the full-window raw difference.

    Returns a list of (month_label, period_start, period_end) tuples, where
    month_label is the "YYYY-MM" of the period's start.
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    month_starts = pd.date_range(start_time.replace(day=1), end_time, freq="MS")
    boundaries = sorted(set([start_time, end_time]) | set(month_starts))
    boundaries = [b for b in boundaries if start_time <= b <= end_time]

    periods = []
    for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
        resolved_start, resolved_end = resolve_analysis_window(index, seg_start, seg_end)
        if resolved_start == resolved_end:
            continue
        label = resolved_start.strftime("%Y-%m")
        periods.append((label, resolved_start, resolved_end))

    return periods


###################################################################
###################################################################

#
# Interpolation-first differences (scale only when have to)
#
# Each meter's cumulative reading is linearly interpolated (in time) at any
# window boundary that falls inside a data gap, so gaps *between* valid
# readings never need scaling. Scaling is used only for time that no valid
# reading brackets: missing/stuck data at the very start or end of the
# analysis window, and the gap across a meter restart.
#
# Because annual and monthly use the same interpolated readings, a meter's
# monthly values add up to its annual value whenever no scaling is needed.

def _seconds_since(index, ref):
    """Convert a DatetimeIndex to float seconds since ref (resolution-safe)."""
    return np.asarray((index - ref) / pd.Timedelta(seconds=1), dtype=float)


def _meter_interp_segments(series, ref, r2_num, info, meter_restarts, report_lo=None, report_hi=None):
    """
    Split one meter's valid readings into continuous counter segments.

    - Stuck edge runs (R²=-2) are removed first, using the stuck length
      in info (e.g. "stuck 120 pts at start"). The run is counted from the
      report range edge (report_lo / report_hi) when given, else the series edge.
    - Restart meters (R²=-3) are cut at each restart: one segment ends at
      previous_valid_time and the next starts at restart_time, because the
      counter can't be interpolated across a reset.

    Returns a list of (times_s, values) numpy arrays, each with >= 2 readings.
    """
    y = series.copy()

    if r2_num == -2:
        tokens = str(info).split()
        stuck_len = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 0
        if stuck_len > 1:
            # position of the report range edges in the series (whole series if not given)
            lo_pos = 0 if report_lo is None else int(y.index.searchsorted(report_lo, side="left"))
            hi_pos = len(y) if report_hi is None else int(y.index.searchsorted(report_hi, side="right"))
            if "start" in info:
                y.iloc[lo_pos:lo_pos + stuck_len] = np.nan
            elif "end" in info:
                y.iloc[max(hi_pos - stuck_len, 0):hi_pos] = np.nan

    y = y.dropna()
    if len(y) < 2:
        return []

    t = _seconds_since(y.index, ref)
    v = y.to_numpy(dtype=float)

    # Segment bounds (in seconds); a single open-ended segment unless restarts
    bounds = []
    lo = -np.inf
    if meter_restarts is not None and not meter_restarts.empty:
        for r in meter_restarts.sort_values("restart_time").itertuples(index=False):
            prev_valid = (pd.to_datetime(r.previous_valid_time) - ref) / pd.Timedelta(seconds=1)
            restart_t = (pd.to_datetime(r.restart_time) - ref) / pd.Timedelta(seconds=1)
            bounds.append((lo, prev_valid))
            lo = restart_t
    bounds.append((lo, np.inf))

    segments = []
    for seg_lo, seg_hi in bounds:
        mask = (t >= seg_lo) & (t <= seg_hi)
        if mask.sum() >= 2:
            segments.append((t[mask], v[mask]))
    return segments


def _interp_usage_over_window(segments, a, b, max_step):
    """
    Measured + interpolated usage over [a, b] (seconds) from interp segments.

    Returns a dict with:
        usage        : kWh from interpolated readings (no scaling)
        covered_s    : seconds of [a, b] bracketed by valid readings
        interp_s     : seconds of [a, b] that fall inside interior data gaps
        interp_kwh   : share of `usage` that comes from those gaps
        first, last  : (seconds, reading) at the first/last covered point
    """
    usage = covered_s = interp_s = interp_kwh = 0.0
    first = last = None

    for t, v in segments:
        lo, hi = max(a, t[0]), min(b, t[-1])
        if lo >= hi:
            continue

        v_lo, v_hi = np.interp([lo, hi], t, v)
        usage += v_hi - v_lo
        covered_s += hi - lo

        # Interior gaps: consecutive valid readings more than one interval apart
        dt = np.diff(t)
        gap = dt > max_step
        if gap.any():
            g_lo = np.maximum(t[:-1][gap], lo)
            g_hi = np.minimum(t[1:][gap], hi)
            overlap = np.clip(g_hi - g_lo, 0, None)
            interp_s += overlap.sum()
            interp_kwh += (np.diff(v)[gap] * overlap / dt[gap]).sum()

        if first is None:
            first = (lo, v_lo)
        last = (hi, v_hi)

    return {
        "usage": usage, "covered_s": covered_s,
        "interp_s": interp_s, "interp_kwh": interp_kwh,
        "first": first, "last": last,
    }


def _compute_interp_window_differences(
    data,
    windows,
    df_bad_meters,
    df_restarts,
    min_covered_fn,
    r2_threshold=0.9,
    restarts_thres=5,
    min_real_points=None,
):
    """
    Shared engine for compute_meter_differences and
    compute_monthly_meter_differences.

    windows: list of (label, window_start, window_end) timestamps.
    min_covered_fn: window length (s) -> minimum covered seconds required
        before a partly-covered window is scaled up.
    min_real_points: if given, a window with fewer real (non-NaN) readings
        inside it is left NaN, so a window that is mostly interpolated from
        readings outside it isn't reported.

    Segments are built once from the full `data` (not per window), so a
    boundary inside a gap is interpolated using the valid readings on either
    side of it, even when those readings fall in a neighbouring month or
    outside the report range (data range wider than the windows).

    Only restarts inside the report range (first window start to last window
    end) count toward restarts_thres.
    """
    bad_dict = {}
    if df_bad_meters is not None and not df_bad_meters.empty:
        for row in df_bad_meters.itertuples(index=False):
            try:
                r2_num = float(row.r2)
            except Exception:
                r2_num = np.nan
            bad_dict[row.meter_name] = (r2_num, row.info, row.r2)

    ref = data.index[0]
    max_step = 1.5 * np.median(np.diff(_seconds_since(data.index, ref)))

    win_s = [
        (label,
         (pd.to_datetime(ws) - ref) / pd.Timedelta(seconds=1),
         (pd.to_datetime(we) - ref) / pd.Timedelta(seconds=1))
        for label, ws, we in windows
    ]

    # report range = span of all windows
    report_lo = min(pd.to_datetime(ws) for _, ws, _ in windows)
    report_hi = max(pd.to_datetime(we) for _, _, we in windows)

    results = []
    for meter_name in data.columns:
        series = data[meter_name]

        if meter_name in bad_dict:
            r2_num, info_val, r2_val = bad_dict[meter_name]
        else:
            r2_num, info_val, r2_val = np.nan, "", f">= {r2_threshold}"

        meter_restarts = None
        restarts_in_report = 0
        if df_restarts is not None and not df_restarts.empty:
            meter_restarts = df_restarts[df_restarts["meter_name"] == meter_name]
            restart_times = pd.to_datetime(meter_restarts["restart_time"])
            restarts_in_report = int(restart_times.between(report_lo, report_hi).sum())

        # Excluded (left NaN): unreadable R², all NaN, too few points,
        # low R² with no correction rule, too many restarts in the report range
        usable = meter_name not in bad_dict or (
            not np.isnan(r2_num)
            and r2_num in (-1, -2, -3)
            and not (r2_num == -3 and restarts_in_report > restarts_thres)
        )

        segments = (
            _meter_interp_segments(
                series, ref, r2_num, info_val, meter_restarts,
                report_lo=report_lo, report_hi=report_hi,
            )
            if usable else []
        )

        for (label, a, b), (_, ws, we) in zip(win_s, windows):
            st_time = en_time = pd.NaT
            start_val = end_val = np.nan
            raw_diff = scaled_diff = interp_kwh = np.nan
            pct_interp = pct_scaled = np.nan
            row_info = info_val

            window_usable = usable
            if usable and min_real_points is not None:
                real_points = int(series.loc[ws:we].notna().sum())
                if real_points < min_real_points:
                    window_usable = False
                    row_info = (
                        f"{info_val}; " if info_val else ""
                    ) + f"too few real readings in window ({real_points})"

            if window_usable:
                u = _interp_usage_over_window(segments, a, b, max_step)
                window_s = b - a

                if u["covered_s"] > 0:
                    raw_diff = u["usage"]
                    interp_kwh = u["interp_kwh"]
                    pct_interp = u["interp_s"] / window_s * 100
                    st_time = ref + pd.Timedelta(seconds=u["first"][0])
                    en_time = ref + pd.Timedelta(seconds=u["last"][0])
                    start_val, end_val = u["first"][1], u["last"][1]

                    missing_s = window_s - u["covered_s"]
                    if missing_s <= 0:
                        # Fully bracketed by valid readings: no scaling
                        scaled_diff = raw_diff
                        pct_scaled = 0
                    elif u["covered_s"] >= min_covered_fn(window_s):
                        # Scale only the time no reading brackets
                        scaled_diff = raw_diff * window_s / u["covered_s"]
                        pct_scaled = missing_s / window_s * 100

            results.append((
                meter_name, label,
                st_time, start_val,
                en_time, end_val,
                raw_diff, scaled_diff, interp_kwh,
                r2_val, row_info,
                pct_interp, pct_scaled,
            ))

    result_df = pd.DataFrame(
        results,
        columns=[
            "meter_name", "month",
            "start_time", "start",
            "end_time", "end",
            "raw_difference", "difference", "interpolated_kwh",
            "R²", "info",
            "% interpolated", "% scaled",
        ],
    )

    def format_percent(value, info):
        if pd.isna(value):
            return "N/A" if info != "" else ""
        elif value == 0:
            return ""
        else:
            return f"{value:.1f}%"

    for col in ["% interpolated", "% scaled"]:
        result_df[col] = [
            format_percent(val, info)
            for val, info in zip(result_df[col], result_df["info"])
        ]

    return result_df


def compute_meter_differences(
    data,
    start_time,
    end_time,
    df_bad_meters,
    df_restarts,
    valid_len=5*30*96,
    r2_threshold=0.9,
    restarts_thres=5,
):
    """
    Annual (start_time -> end_time) kWh difference per meter, interpolating
    across data gaps and scaling only when the window edges (or a restart
    gap) aren't bracketed by valid readings.

    Scaling requires at least valid_len intervals of covered time; meters
    with less coverage are left NaN ("N/A"). Meters with fewer than valid_len
    real readings inside start_time -> end_time are also left NaN, so a year
    interpolated mostly from readings outside it isn't reported.

    `data` can cover a wider data range than start_time -> end_time; readings
    outside the report range are used to interpolate across its edges.

    raw_difference = measured + interpolated (unscaled);
    difference - raw_difference = the scaled (extrapolated) kWh.
    Returns a DataFrame indexed by meter_name.
    """
    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    step_s = np.median(np.diff(_seconds_since(data.index, data.index[0])))
    min_covered_s = valid_len * step_s

    result_df = _compute_interp_window_differences(
        data,
        [("annual", start_time, end_time)],
        df_bad_meters,
        df_restarts,
        min_covered_fn=lambda window_s: min_covered_s,
        r2_threshold=r2_threshold,
        restarts_thres=restarts_thres,
        min_real_points=valid_len,
    )
    return result_df.drop(columns="month").set_index("meter_name")


def compute_monthly_meter_differences(
    data,
    start_time,
    end_time,
    df_bad_meters,
    df_restarts,
    monthly_min_valid_fraction=0.5,
    r2_threshold=0.9,
    restarts_thres=5,
):
    """
    Calendar month kWh difference per meter, using the same interpolated
    readings as compute_meter_differences, so month boundaries that
    fall in a gap are interpolated instead of left blank or scaled.

    A month is scaled only for time not bracketed by valid readings, and only
    if at least monthly_min_valid_fraction of the month is covered; months
    with less coverage are left NaN ("N/A").

    `data` can cover a wider data range than start_time -> end_time; readings
    outside the report range are used to interpolate across its edges.

    Returns a long format DataFrame with one row per (meter_name, month).
    """
    periods = get_calendar_month_periods(data.index, start_time, end_time)

    return _compute_interp_window_differences(
        data,
        periods,
        df_bad_meters,
        df_restarts,
        min_covered_fn=lambda window_s: monthly_min_valid_fraction * window_s,
        r2_threshold=r2_threshold,
        restarts_thres=restarts_thres,
    )


###################################################################
###################################################################

def export_monthly_meter_differences(monthly_df, meter_info_file, filename, var="kwh"):
    """
    Pivot long-format monthly meter differences (from
    compute_monthly_meter_differences) to one row per meter and one column
    per month, and save to CSV.

    Returns the long-format DataFrame merged with building info (one row
    per meter per month), for use in downstream checks/aggregation.
    """
    export_df = monthly_df.copy()

    meter_info = pd.read_csv(meter_info_file)[['meter_name', 'building_complex_name']]
    meter_info.columns = meter_info.columns.str.strip()
    export_df = export_df.merge(meter_info, on='meter_name', how='left')

    pivot_df = export_df.pivot(index='meter_name', columns='month', values='difference')
    pivot_df = pivot_df.round(1)
    # pivot_df = pivot_df.add_prefix(f"{var}_")
    pivot_df = pivot_df.reset_index()

    pivot_df.to_csv(filename, index=False)

    print(f"Monthly {var} file saved to {filename}")

    return export_df


###################################################################
###################################################################

def export_monthly_scaling_detail(monthly_df, filename):
    """
    Pivot long-format monthly meter differences (from
    compute_monthly_meter_differences) to one row per meter and one column
    per month, with each cell holding that meter's "% scaled" value for
    the month (e.g. "45%", "", or "N/A").
    """
    pivot_df = monthly_df.pivot(index='meter_name', columns='month', values='% scaled')
    pivot_df = pivot_df.rename_axis(columns=None).reset_index()

    pivot_df.to_csv(filename, index=False)

    print(f"Monthly % scaled wide format file saved to {filename}")

    return pivot_df


###################################################################
###################################################################

def export_annual_vs_monthly_check(df_annual, df_monthly_long, filename, var="kwh"):
    """
    Compare each meter's annual difference to the sum of its monthly
    differences and save a review CSV.

    - df_annual: long format meter-level DataFrame from
      export_meter_differences (has 'meter_name' and 'difference')
    - df_monthly_long: long format DataFrame from
      export_monthly_meter_differences (has 'meter_name', 'month',
      'difference')

    Differences between the two are expected when a special meter is
    scaled differently within individual monthly periods vs. over the
    full analysis window; this is a review check, not a replacement for
    the annual calculation.
    """
    annual = df_annual[['meter_name', 'difference']].rename(
        columns={'difference': f'annual_{var}'}
    )

    monthly_sum = df_monthly_long.groupby('meter_name', as_index=False).agg(
        **{f'sum_monthly_{var}': ('difference', lambda x: x.sum(min_count=1))}
    )

    check_df = annual.merge(monthly_sum, on='meter_name', how='outer')
    check_df[f'difference_{var}'] = (
        check_df[f'annual_{var}'] - check_df[f'sum_monthly_{var}']
    ).round(1)
    check_df[f'annual_{var}'] = check_df[f'annual_{var}'].round(1)
    check_df[f'sum_monthly_{var}'] = check_df[f'sum_monthly_{var}'].round(1)

    check_df.to_csv(filename, index=False)

    print(f"Annual vs monthly file saved to {filename}")

    return check_df


###################################################################
######               Function for 2.kw_sum.ipynb              ######
###################################################################

def plot_kw_meters(df, columns=None, figsize_per_plot=(12,3)):
    """
    Plot one column per subplot with full datetime index.
    
    Parameters:
        df: pd.DataFrame with datetime index
        columns: list of columns to plot (default: all)
        figsize_per_plot: tuple, figure size per subplot (width, height)
    """
    if columns is None:
        columns = df.columns.tolist()
    
    num_cols = len(columns)
    
    fig, axes = plt.subplots(num_cols, 1, figsize=(figsize_per_plot[0], figsize_per_plot[1]*num_cols))
    if num_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, columns):
        ax.plot(df.index, df[col], label=col)
        ax.set_ylabel(col)
        ax.legend(loc='upper left')
        ax.grid(True)
        # Use full datetime index, do not limit x-axis
        ax.set_xlim(df.index.min(), df.index.max())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    axes[-1].set_xlabel("Datetime")
    plt.tight_layout()
    plt.show()
    


###################################################################
######       Functions for 4.kwh_processing_all.ipynb        ######
###################################################################

def plot_data(df, var, txt="original", fig_path="./"):
    """
    Plots static grouped line charts (5 meters per page) and saves them as a multi-page PDF.

    Parameters:
    - df (pd.DataFrame): DataFrame with datetime index and one column per meter.
    - var (str): Y-axis label.
    - txt (str): Label to include in output filename.
    - fig_path (str): Directory to save output plots.
    """

    # Ensure datetime is index
    if df.index.name != 'datetime':
        df = df.set_index('datetime')

    os.makedirs(fig_path, exist_ok=True)
    pdf_path = os.path.join(fig_path, f"{var}_{txt}_all.pdf")

    meter_names = df.columns
    lines_per_page = 5
    chunks = [meter_names[i:i + lines_per_page] for i in range(0, len(meter_names), lines_per_page)]

    with PdfPages(pdf_path) as pdf:
        for i, chunk in enumerate(chunks):
            fig, ax = plt.subplots(figsize=(14, 6))

            # Use a large colormap
            cmap = plt.get_cmap('tab10')  # tab10 has 10 distinct colors
            colors = cmap.colors * (lines_per_page // len(cmap.colors) + 1)

            for j, name in enumerate(chunk):
                ax.plot(df.index, df[name], label=name, linewidth=1.5, color=colors[j])

            ax.set_title(f"{var} ({txt}) — Lines {i * lines_per_page + 1}–{i * lines_per_page + len(chunk)}", fontsize=14)
            ax.set_ylabel(var, fontsize=12)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(False)  # No grid

            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, title='Meter Name', title_fontsize=9)
            plt.tight_layout(rect=[0, 0, 0.85, 1])

            pdf.savefig(fig)
            # plt.show()
            plt.close()

    print(f"Multi-page PDF saved to: {pdf_path}")



###################################################################
###################################################################

def clean_and_interpolate(column, flags, df_restarts=None, special_meters_file=None):
    """
    Clean stuck points, apply optional no_interp corrections, and interpolate.
    no_interp windows and restart intervals are set to 0 and are never interpolated
    across: each stretch between them is interpolated on its own, so a 0 is never
    used as an interpolation end point.

    Input Args:
        column: pd.Series (time-indexed meter data)
        flags: pd.Series of same length to track interpolated points
        df_restarts: pd.DataFrame with columns ['meter_name', 'previous_valid_time', 'restart_time']
        special_meters_file: CSV file with columns ['meter_name', 'start_datetime', 'end_datetime', 'solution']

    Output Returns:
        pd.Series: cleaned and interpolated column
    """
    column_cp = column.copy()
    column = column.copy()

    # no_interp windows + restart intervals: set to 0 and not interpolated across
    blocked = pd.Series(False, index=column.index)

    if (
        special_meters_file is not None
        and str(special_meters_file).strip() != ""
        and os.path.exists(special_meters_file)
    ):
        bad_periods = pd.read_csv(special_meters_file, low_memory=False)
        required_cols = {"meter_name", "solution", "start_datetime", "end_datetime"}
        if required_cols.issubset(set(bad_periods.columns)):
            bad_periods["meter_name"] = bad_periods["meter_name"].astype(str).str.strip()
            bad_periods["solution"] = bad_periods["solution"].astype(str).str.strip().str.lower()
            bad_periods["start_datetime"] = pd.to_datetime(bad_periods["start_datetime"])
            bad_periods["end_datetime"] = pd.to_datetime(bad_periods["end_datetime"])

            bad_periods = bad_periods[
                (bad_periods["meter_name"] == column.name) &
                (bad_periods["solution"] == "no_interp")
            ]

            for _, row in bad_periods.iterrows():
                start = row["start_datetime"]
                end = row["end_datetime"]
                mask = (column.index >= start) & (column.index <= end)
                column.loc[mask] = 0
                blocked |= mask

    mask_stuck = (column.duplicated(keep="first")) & (column != 0.0)
    column = column.mask(mask_stuck)

    if df_restarts is not None and not df_restarts.empty:
        restarts_for_col = df_restarts[df_restarts["meter_name"] == column.name]
        for _, r in restarts_for_col.iterrows():
            start = pd.to_datetime(r["previous_valid_time"])
            end = pd.to_datetime(r["restart_time"])
            mask = (column.index >= start) & (column.index <= end)
            column.loc[mask] = 0
            blocked |= mask

    # Interpolate each stretch between blocked points separately
    # (interpolating the whole column would draw a line up from the 0s)
    segment_id = blocked.cumsum()
    open_points = ~blocked
    column.loc[open_points] = (
        column[open_points]
        .groupby(segment_id[open_points])
        .transform(lambda s: s.interpolate(method="linear"))
    )

    changed_mask = (column_cp != column) & column.notna()
    flags[:] = np.where(changed_mask, 1, flags)

    return column



###################################################################
###################################################################

def conditional_round(x):
    """
    Format numbers: 
    - If the value has more than 1 decimal, round to 2 decimals; otherwise, keep as-is
    """
    return round(x, 2) if (x * 10) % 10 != 0 else round(x, 1)  


def reshape_interpolated_data(df, interpolation_flags):
    """
    Reshape the dataframe and merge with interpolation_flags. 
    
    Input Args:
        df: a pandas dataframe
        interpolation_flags: a pandas dataframe with the same shape as df, containing is_interpolated flags of values in df
        
    Output Returns:
        final_df: combined and reshaped dataframe
    """
          
    # Step 1: Join df with interpolation_flags
    combined_df = pd.concat([df, interpolation_flags], axis=1)

    # Step 2: Reset the index (date range) and name it as 'datetime' to keep track
    combined_df = combined_df.reset_index().rename(columns={'index': 'datetime'})

    # Step 3: Reshape the meter readings into one column "meter_reading" and their names into "meter_name"
    meter_columns = df.columns.tolist()  # The original meter reading columns
    flag_columns = interpolation_flags.columns.tolist()  # The corresponding interpolation flag columns

    # Reshape meter readings into one column "meter_reading" and meter names into "meter_name"
    reshaped_df = pd.melt(combined_df, id_vars=['datetime'], 
                          value_vars=meter_columns, 
                          var_name='meter_name', 
                          value_name='meter_reading')

    # Reshape interpolation flags into one column "is_interpolated" and match with reshaped_df
    reshaped_flags = pd.melt(combined_df, id_vars=['datetime'], 
                             value_vars=flag_columns, 
                             var_name='meter_name_flag', 
                             value_name='is_interpolated')

    # Ensure flag names match meter names by removing '_interpolated' suffix
    reshaped_flags['meter_name'] = reshaped_flags['meter_name_flag'].str.replace('_interpolated', '')

    # Drop the extra 'meter_name_flag' column from reshaped_flags
    reshaped_flags = reshaped_flags.drop(columns=['meter_name_flag'])

    # Merge reshaped_df and reshaped_flags on 'datetime' and 'meter_name'
    final_df = pd.merge(reshaped_df, reshaped_flags, on=['datetime', 'meter_name'])

    # Apply conditional rounding to the 'meter_reading' column
    final_df['meter_reading'] = final_df['meter_reading'].apply(conditional_round)  

    # Sort the final dataframe by 'meter_name' and then 'datetime'
    final_df = final_df.sort_values(by=['datetime', 'meter_name']).reset_index(drop=True)
    
    # Return the final reshaped DataFrame
    return final_df


###################################################################
###################################################################

def save_file(data, fname, dname):
    """
    Save a data file to a specific location and filename.
    Automatically overwrites any existing CSV file with the same name.
    """
    if not os.path.exists(dname):
        os.mkdir(dname)
        print(f'Directory {dname} was created.')

    fpath = os.path.join(dname, fname)

    print(f'Writing file: "{fpath}"')
    try:
        data.to_csv(fpath, index=False)
        print(f'File "{fpath}" saved successfully.')
    except Exception as e:
        print(f'Failed to save file: {e}')

    
###################################################################
###################################################################

def calculate_delta_df(df):
    """
    Calculate strict delta_df: current value minus previous value.
    Only valid if both current and previous are not NaN, and current >= previous.
    All NaNs are preserved. No interpolation or past-year filling.

    Args:
        df: pd.DataFrame with time-indexed data (all 0s already replaced by NaN)

    Returns:
        delta_df: pd.DataFrame of deltas
    """
    delta_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

    for col in df.columns:
        prev = df[col].shift(1)
        curr = df[col]

        # If either current or previous is NaN, delta is NaN
        mask_valid = curr.notna() & prev.notna()

        # Calculate delta only for valid points
        delta = pd.Series(np.nan, index=df.index, dtype=float)
        delta.loc[mask_valid] = curr[mask_valid] - prev[mask_valid]

        # If current < previous, set delta to NaN
        delta.loc[delta < 0] = np.nan

        delta_df[col] = delta

    return delta_df


###################################################################
###################################################################

def reshape_delta_df(df, var):
    
    """
    Reshape a merged DataFrame by converting all columns except 'datetime' into two columns: 'meter_name' and 'meter_reading'.

    Parameters:
        df (pd.DataFrame): The merged DataFrame with a 'datetime' column and other meter columns.
    
    Returns:
        pd.DataFrame: Reshaped DataFrame with columns ['datetime', 'meter_reading', 'meter_name'].
    """
    
    # Ensure the 'datetime' column is of datetime type, and print out the datetime range for verification
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Melt the DataFrame to reshape
    ValueName = 'delta_'+var             # New column name for values
    reshaped_delta_df = pd.melt(
        df,
        id_vars=['datetime'],            # Keep 'datetime' column
        var_name='meter_name',           # New column for former column names
        value_name=ValueName             # New column for values
    )
    
    # Sort by 'datetime' and 'meter_name'
    reshaped_delta_df = reshaped_delta_df.sort_values(by=['datetime', 'meter_name']).reset_index(drop=True)

    # Apply conditional rounding to the ValueName column
    reshaped_delta_df[ValueName] = reshaped_delta_df[ValueName].apply(conditional_round)     
    
    # Replace 0 value with NaN missing value
    reshaped_delta_df[ValueName] = reshaped_delta_df[ValueName].replace(0, np.nan)
    
    # Ensure the 'datetime' column remains in the correct datetime format
    reshaped_delta_df['datetime'] = pd.to_datetime(reshaped_delta_df['datetime'])
    
    return reshaped_delta_df