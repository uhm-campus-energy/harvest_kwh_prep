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
from openpyxl import load_workbook
from typing import List, Tuple



# 'broken' is treated like a removal style correction
ALLOWED_CORRECTION_SOLUTIONS = {"broken", "remove", "div100", "no_interp"}
BROKEN_STATUS_VALUES = {"broken", "repaired", "under_renovation", "no_meter"}

###################################################################
###################################################################

def _parse_strict_date_series(series, col_name):
    """
    Parse date columns safely from CSV without ambiguous guessing. 
    (Does not alter csv itself)

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
    # ===== ORIGINAL ZHILING / AURORA CODE (commented out) =====
    #     """
    #     Apply special corrections to selected meters based on an Excel config file.
    #     Correctly handles multiple periods per meter, including overlapping periods.
    #
    #     Returns a copy of the DataFrame.
    #     """
    #     data_corrected = data.copy()
    #     bad_periods = pd.read_excel(special_meters_file)
    #
    #     # Clean up
    #     bad_periods["meter_name"] = bad_periods["meter_name"].astype(str).str.strip()
    #     bad_periods["solution"] = bad_periods["solution"].astype(str).str.strip().str.lower()
    #     bad_periods["start_datetime"] = pd.to_datetime(bad_periods["start_datetime"])
    #     bad_periods["end_datetime"] = pd.to_datetime(bad_periods["end_datetime"])
    #
    #     # Filter to meters that exist in data
    #     bad_periods = bad_periods[bad_periods["meter_name"].isin(data_corrected.columns)]
    #
    #     for _, row in bad_periods.iterrows():
    #         meter = row["meter_name"]
    #         start = row["start_datetime"]
    #         end = row["end_datetime"]
    #         solution = row["solution"]
    #
    #         # mask points in the period
    #         mask = (data_corrected.index >= start) & (data_corrected.index <= end)
    #
    #         if solution == "div100":
    #             data_corrected.loc[mask, meter] = data_corrected.loc[mask, meter] / 100.0
    #         elif solution == "remove":
    #             data_corrected.loc[mask, meter] = np.nan
    #
    #     return data_corrected

    """
    Apply special corrections to selected meters based on an Excel config file.
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

    bad_periods = pd.read_excel(special_meters_file)

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

    Args:
        data (pd.DataFrame): Wide meter dataframe with a DatetimeIndex.
        special_meters_file (str): Excel file with correction rows.
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

    correction_df = pd.read_excel(special_meters_file)

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

        # Restarts / drops
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

def create_special_meters_workbook(
    data,
    output_file,
    df_bad_meters=None,
    df_restarts=None,
    overwrite=True,
):
    """
    Create an Excel workbook of auto-detected candidate meter issues.

    Notes:
    - The first sheet uses the same core columns expected by
      apply_special_meter_corrections: meter_name, solution, start_datetime, end_datetime
    - solution is intentionally left blank so the workbook is safe for review first
    - extra columns are included to help review the candidates
    """
    if output_file is None or str(output_file).strip() == "":
        raise ValueError("output_file must be a non-empty path.")

    output_file = str(output_file)
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    #TODO: remove possibly?
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_file}")

    issue_df = suggest_meter_issues(data)

    if issue_df.empty:
        issue_df = pd.DataFrame(columns=[
            "meter_name", "issue_type", "start_datetime", "end_datetime", "description", "suggestion"
        ])

    issue_df = issue_df.copy()
    issue_df["solution"] = ""

    if df_bad_meters is not None and not df_bad_meters.empty:
        meta = df_bad_meters.rename(columns={"r2": "detected_r2", "info": "detected_info"})
        meta = meta[["meter_name", "detected_r2", "detected_info"]]
        issue_df = issue_df.merge(meta, on="meter_name", how="left")
    else:
        issue_df["detected_r2"] = np.nan
        issue_df["detected_info"] = ""

    ordered_cols = [
        "meter_name",
        "solution",
        "start_datetime",
        "end_datetime",
        "issue_type",
        "description",
        "suggestion",
        "detected_r2",
        "detected_info",
    ]
    for col in ordered_cols:
        if col not in issue_df.columns:
            issue_df[col] = np.nan

    issue_df = issue_df[ordered_cols].sort_values(
        by=["meter_name", "start_datetime", "issue_type"],
        na_position="last"
    ).reset_index(drop=True)

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        issue_df.to_excel(writer, sheet_name="timeframes", index=False)

        if df_bad_meters is not None:
            df_bad_meters.to_excel(writer, sheet_name="special_meter_summary", index=False)

        if df_restarts is not None:
            df_restarts.to_excel(writer, sheet_name="restarts", index=False)

    print(f"Auto-generated special meters workbook saved to {output_file}")
    return issue_df

###################################################################
###################################################################


# ADDED FOR BROKEN-METER MASTER/CANDIDATE WORKFLOW
###################################################################
###################################################################

# shared text normalizer for the broken meter workflow.
def _normalize_text(value, underscore=False):
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    if underscore:
        value = re.sub(r"[\s\-]+", "_", value)
    else:
        value = re.sub(r"\s+", " ", value)

    return value


# loader for the broken meter workbook with simple normalization and day-first dates.
def load_broken_meter_workbook(broken_meters_file, dayfirst=False):
    """
    Load and normalize the running broken meter workbook.

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
        print(f"Broken meter workbook not found: {broken_meters_file}")
        return pd.DataFrame(columns=cols)

    ext = os.path.splitext(str(broken_meters_file))[1].lower()
    if ext in [".xlsx", ".xls", ".xlsm"]:
        df = pd.read_excel(broken_meters_file)
    else:
        df = pd.read_csv(broken_meters_file)

    df = df.copy()
    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]
    df = df.rename(columns={"updated_date": "updated_data"})

    required_cols = {"meter_name", "start_date", "end_date", "status"}
    missing_cols = required_cols.difference(set(df.columns))
    if missing_cols:
        raise ValueError(
            "Broken meter workbook is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    for col in ["description", "data_source", "updated_data"]:
        if col not in df.columns:
            df[col] = ""

    df["meter_name"] = df["meter_name"].apply(_normalize_text)
    df["status"] = df["status"].apply(lambda x: _normalize_text(x, underscore=True))
    # df["status"] = df["status"].replace({"no_meter": "under_renovation"})
    df["description"] = df["description"].apply(_normalize_text)
    df["data_source"] = df["data_source"].apply(_normalize_text)
    df["updated_data"] = pd.to_datetime(df["updated_data"], errors="coerce", format="mixed", dayfirst=True)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce", format="mixed", dayfirst=True)
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce", format="mixed", dayfirst=True)

    df = df[df["meter_name"] != ""].copy()
    df = df[df["status"].isin(BROKEN_STATUS_VALUES)].copy()

    return df.reset_index(drop=True)



# CHANGED: added loader for the candidate workbook used by the review workflow.
def _load_existing_candidates(candidate_file):
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

    xls = pd.ExcelFile(candidate_file)
    sheet_name = "meter_issues" if "meter_issues" in xls.sheet_names else (
        "timeframes" if "timeframes" in xls.sheet_names else xls.sheet_names[0]
    )
    df = pd.read_excel(candidate_file, sheet_name=sheet_name)

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
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
    df["end_datetime"] = pd.to_datetime(df["end_datetime"], errors="coerce")
    df["approved"] = pd.to_numeric(df["approved"], errors="coerce").fillna(0).astype(int)
    df["solution"] = df["solution"].fillna("").astype(str).str.strip().str.lower()
    df["issue_type"] = df["issue_type"].fillna("").astype(str).str.strip().str.lower()

    return df.reset_index(drop=True)


def _load_base_master_corrections(meter_issues_file):
    cols = [
        "meter_name", "solution", "start_datetime", "end_datetime",
        "issue_type/status", "description"
    ]
    if meter_issues_file is None or not os.path.exists(meter_issues_file):
        return pd.DataFrame(columns=cols)

    df = pd.read_excel(meter_issues_file)

    if "issue_type" in df.columns and "issue_type/status" not in df.columns:
        df = df.rename(columns={"issue_type": "issue_type/status"})
    if "status" in df.columns and "issue_type/status" not in df.columns:
        df = df.rename(columns={"status": "issue_type/status"})
    if "details" in df.columns and "description" not in df.columns:
        df = df.rename(columns={"details": "description"})

    for col in cols:
        if col not in df.columns:
            df[col] = "" if col in {"meter_name", "solution", "issue_type/status", "description"} else pd.NaT

    df = df[cols].copy()
    df["meter_name"] = df["meter_name"].apply(_normalize_text)
    df["solution"] = df["solution"].fillna("").astype(str).str.strip().str.lower()
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
    df["end_datetime"] = pd.to_datetime(df["end_datetime"], errors="coerce")
    df["issue_type/status"] = df["issue_type/status"].fillna("").astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()

    return df.reset_index(drop=True)


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


# master sheet sync for official corrections, approved candidates, and broken-meter rows.
def sync_meter_corrections_master_sheet(
    candidate_file,
    broken_meters_file,
    meter_corrections_file,
    window_start,
    window_end,
):
    """
    Build the master corrections file from:
    - approved candidate rows (approved == 1 and solution filled in)
    - broken meter rows from the running broken meter file
    """
    window_start = pd.to_datetime(window_start)
    window_end = pd.to_datetime(window_end)

    master_cols = [
        "meter_name",
        "solution",
        "start_datetime",
        "end_datetime",
        "issue_type/status",
        "description",
    ]
    master_df = pd.DataFrame(columns=master_cols)

    existing_candidates = _load_existing_candidates(candidate_file)
    broken_df = load_broken_meter_workbook(broken_meters_file)

    approved_candidates = existing_candidates[
        (existing_candidates["approved"] == 1) &
        (existing_candidates["solution"] != "")
    ].copy()

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
        true_start = pd.to_datetime(row["start_date"], errors="coerce")
        true_end = pd.to_datetime(row["end_date"], errors="coerce")

        overlap_start = window_start if pd.isna(true_start) else max(true_start, window_start)
        overlap_end = window_end if pd.isna(true_end) else min(true_end, window_end)

        if overlap_start > overlap_end:
            continue

        broken_rows.append({
            "meter_name": row["meter_name"],
            "solution": "remove",
            "start_datetime": true_start,
            "end_datetime": true_end,
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

    if str(meter_corrections_file).lower().endswith(".csv"):
        master_df.to_csv(meter_corrections_file, index=False)
    else:
        with pd.ExcelWriter(meter_corrections_file, engine="openpyxl") as writer:
            master_df.to_excel(writer, sheet_name="master_corrections", index=False)

    print(f"Master correction workbook saved to {meter_corrections_file}")
    return master_df



# candidate workbook updater for unresolved review rows and new detected issues
def update_special_meter_candidates_workbook(
    data,
    candidate_file,
    broken_meters_file,
    window_start,
    window_end,
    df_bad_meters=None,
    df_restarts=None,
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

    existing_candidates = _load_existing_candidates(candidate_file)
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
        combined_candidates = combined_candidates.sort_values(
            by=["meter_name", "start_datetime", "issue_type"],
            na_position="last",
        ).reset_index(drop=True)

    output_dir = os.path.dirname(str(candidate_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(candidate_file, engine="openpyxl") as writer:
        combined_candidates.to_excel(writer, sheet_name="timeframes", index=False)
        if df_bad_meters is not None:
            special_meter_summary_export.to_excel(writer, sheet_name="special_meter_summary", index=False)
        if df_restarts is not None:
            df_restarts.to_excel(writer, sheet_name="restarts", index=False)

    print(f"Candidate review workbook saved to {candidate_file}")
    return combined_candidates


# CHANGED: added all-meter PDF plotting without reloading the raw file.
def plot_all_meters_to_pdf(data, plot_file, ylabel="meter_reading"):
    """
    Plot every meter in the current dataframe to one PDF without reloading the CSV.
    """
    if data is None or data.empty:
        print("No data available for all-meter plotting.")
        return

    output_dir = os.path.dirname(str(plot_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data_start = pd.to_datetime(data.index.min())
    data_end = pd.to_datetime(data.index.max())

    with PdfPages(plot_file) as pdf:
        for meter_name in data.columns:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(data.index, data[meter_name], linewidth=1.2)
            ax.set_xlim(data_start, data_end)
            ax.set_title(meter_name)
            ax.set_xlabel("Datetime")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"All-meter plots saved to {plot_file}")



# review plotting with red broken-meter overlays and blue candidate overlays.
def plot_review_meters_with_overlays(
    data,
    candidate_file,
    broken_meters_file,
    plot_file,
    ylabel="kWh",
    ):
    
    if data is None or data.empty:
        raise ValueError("data is empty.")

    data = data.copy()
    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data.sort_index()

    data_start = pd.to_datetime(data.index.min())
    data_end = pd.to_datetime(data.index.max())

    candidate_df = _load_existing_candidates(candidate_file).copy()

    broken_source_df = load_broken_meter_workbook(broken_meters_file)
    broken_plot_rows = []
    for _, row in broken_source_df.iterrows():
        start = data_start if pd.isna(row["start_date"]) else max(pd.to_datetime(row["start_date"]), data_start)
        end = data_end if pd.isna(row["end_date"]) else min(pd.to_datetime(row["end_date"]), data_end)
        if start <= end:
            broken_plot_rows.append({
                "meter_name": row["meter_name"],
                "start_datetime": start,
                "end_datetime": end,
            })

    broken_df = pd.DataFrame(broken_plot_rows, columns=["meter_name", "start_datetime", "end_datetime"])

    review_meters = sorted(
        (set(candidate_df["meter_name"]) | set(broken_df["meter_name"]))
        & set(data.columns)
    )

    if not review_meters:
        print("No review meters found in data.")
        return

    output_dir = os.path.dirname(str(plot_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def format_r2_value(r2_val):
        if pd.isna(r2_val):
            return ""
        try:
            r2_float = float(r2_val)
            if r2_float > 0:
                return f"R² = {r2_float:.4f}"
            return f"R² = {int(r2_float)}"
        except (TypeError, ValueError):
            return f"R² = {r2_val}"

    with PdfPages(plot_file) as pdf:
        for meter in review_meters:
            plt.figure(figsize=(12, 3))
            plt.plot(data.index, data[meter], linewidth=1.2)
            plt.xlim(data_start, data_end)
            plt.title(meter)
            plt.xlabel("Datetime")
            plt.ylabel(ylabel)
            plt.grid(True, alpha=0.3)

            meter_broken = broken_df[broken_df["meter_name"] == meter]
            for _, row in meter_broken.iterrows():
                plt.axvspan(row["start_datetime"], row["end_datetime"], alpha=0.2, color="red")

            meter_candidates = candidate_df[candidate_df["meter_name"] == meter].copy()
            for _, row in meter_candidates.iterrows():
                if pd.notna(row["start_datetime"]) and pd.notna(row["end_datetime"]):
                    plt.axvspan(row["start_datetime"], row["end_datetime"], alpha=0.2, color="blue")

            issue_types = [
                str(value).strip()
                for value in pd.unique(meter_candidates["issue_type"].dropna())
                if str(value).strip() != ""
            ]
            r2_values = [
                format_r2_value(value)
                for value in pd.unique(meter_candidates["r2"].dropna())
                if format_r2_value(value) != ""
            ]

            annotation_lines = []
            if issue_types:
                annotation_lines.append("Issue type: " + ", ".join(issue_types))
            if r2_values:
                annotation_lines.append(", ".join(r2_values))

            if annotation_lines:
                plt.text(
                    0.05, 0.95,
                    "\n".join(annotation_lines),
                    transform=plt.gca().transAxes,
                    ha="left", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
                )

            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"Review overlay plots saved to {plot_file}")



###################################################################
###################################################################

def plot_special_meters(data, special_meters, plot_file):
    # deprecated because the review overlay PDF now replaces the special-meter PDF.
    # keeping this function as a no-op avoids breaking any older notebook cells.
    """
    Deprecated.

    Use plot_review_meters_with_overlays(...) instead.
    """
    print("plot_special_meters is deprecated. Use plot_review_meters_with_overlays instead.")
    return
    # """
    # Plot time series for special meters and save to a PDF.

    # Args:
    #     data (pd.DataFrame): Time-series data, each column is a meter_name.
    #     special_meters (pd.DataFrame): Has columns ['meter_name', 'r2', 'info'].
    #     plot_file (str): Output PDF file path.
    # """
    # pdf_path = plot_file
    # meters_ordered = special_meters["meter_name"].tolist()

    # skipped_empty = 0

    # with PdfPages(pdf_path) as pdf:
    #     for meter in meters_ordered:
    #         # Skip if meter not in data
    #         if meter not in data.columns:
    #             skipped_empty += 1
    #             continue

    #         row = special_meters[special_meters["meter_name"] == meter].iloc[0]
    #         r2_val, info = row["r2"], row["info"]

    #         # Handle r2 whether it's numeric or string
    #         try:
    #             r2_float = float(r2_val)
    #             if r2_float > 0:
    #                 r2_text = f"R² = {r2_float:.4f}"
    #             else:
    #                 # for 0, -1, -2, -3 we show as integer
    #                 r2_text = f"R² = {int(r2_float)}"
    #         except (TypeError, ValueError):
    #             # fallback if cannot parse as float
    #             r2_text = f"R² = {r2_val}"

    #         plt.figure(figsize=(12, 3))
    #         plt.plot(data.index, data[meter])
    #         plt.xlim(data.index.min(), data.index.max())
    #         plt.title(meter)
    #         plt.xlabel("Datetime")
    #         plt.ylabel("kWh")
    #         plt.grid(True)

    #         plt.text(
    #             0.05, 0.95,
    #             f"{r2_text}\n{info}",
    #             transform=plt.gca().transAxes,
    #             ha="left", va="top",
    #             fontsize=9,
    #             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    #         )

    #         plt.tight_layout()
    #         pdf.savefig()
    #         plt.close()

    # if skipped_empty > 0:
    #     print(f"Skipped plotting for {skipped_empty} meters with no plottable data.")
    # print(f"Special meter plots saved to {pdf_path}")



###################################################################
###################################################################

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
    # ===== ORIGINAL ZHILING / AURORA CODE (commented out) =====
    #     data,
    #     start_time,
    #     end_time,
    #     df_bad_meters,
    #     df_restarts,
    #     valid_len=5*30*96,
    #     r2_threshold=0.9,
    #     restarts_thres=5,
    # ):
    #     """
    #     Compute start/end values, raw difference, scaled difference, R² info, and % scaled for meters.
    #     Returns a DataFrame indexed by meter_name.
    #
    #     Assumptions:
    #         - data has a DatetimeIndex.
    #         - 0→NaN replacements (if needed) are applied before calling this function.
    #         - df_bad_meters has columns: meter_name, r2, info.
    #         - df_restarts has columns: meter_name, previous_valid_time, restart_time.
    #     """
    #
    #     results = []
    #
    #     # Map each meter_name to its (numeric r2, info text, original r2 value)
    #     bad_dict = {}
    #     if df_bad_meters is not None and not df_bad_meters.empty:
    #         for row in df_bad_meters.itertuples(index=False):
    #             meter = row.meter_name
    #             r2_orig = row.r2
    #             info = row.info
    #             try:
    #                 r2_num = float(r2_orig)
    #             except Exception:
    #                 r2_num = np.nan
    #             bad_dict[meter] = (r2_num, info, r2_orig)
    #
    #     start_time = pd.to_datetime(start_time)
    #     end_time = pd.to_datetime(end_time)
    #
    #     # Interval count based on index positions (used for most scaling logic)
    #     start_idx = data.index.get_loc(start_time)
    #     end_idx = data.index.get_loc(end_time)
    #     total_fy_intervals = end_idx - start_idx
    #
    #     # Total duration of the analysis window in seconds (used for restart meters)
    #     fy_seconds = (end_time - start_time).total_seconds()
    #
    #     for meter_name in data.columns:
    #         series = data[meter_name]
    #
    #         # Default values for output fields
    #         start_val = end_val = raw_diff = scaled_diff = np.nan
    #         st_time = en_time = pd.NaT
    #         r2_val = np.nan
    #         info_val = ""
    #         percent_scaled = np.nan
    #
    #         if meter_name not in bad_dict:
    #             # Meters not flagged as "bad": direct start/end difference if both endpoints exist
    #             if start_time in series.index and end_time in series.index:
    #                 start_val = series.loc[start_time]
    #                 end_val = series.loc[end_time]
    #                 raw_diff = scaled_diff = end_val - start_val
    #                 st_time, en_time = start_time, end_time
    #                 percent_scaled = 0
    #             r2_val, info_val = f">= {r2_threshold}", ""
    #
    #         else:
    #             r2_num, info, r2_orig = bad_dict[meter_name]
    #             r2_val = r2_orig
    #             info_val = info
    #
    #             if np.isnan(r2_num):
    #                 # r2 value could not be interpreted numerically
    #                 pass
    #
    #             elif r2_num == 0:
    #                 # All values are NaN
    #                 pass
    #
    #             elif r2_num == -1:
    #                 # Missing start or end: use first and last non-repeated values
    #                 y = series.where(series.ne(series.shift()))
    #                 first_valid = y.first_valid_index()
    #                 last_valid = y.last_valid_index()
    #                 if first_valid is not None and last_valid is not None:
    #                     first_val = y.loc[first_valid]
    #                     last_val = y.loc[last_valid]
    #                     raw_diff = last_val - first_val
    #                     n_intervals = y.index.get_loc(last_valid) - y.index.get_loc(first_valid)
    #                     if n_intervals >= valid_len:
    #                         scaled_diff = raw_diff / n_intervals * total_fy_intervals
    #                         percent_scaled = (total_fy_intervals - n_intervals) / total_fy_intervals * 100
    #                     start_val, end_val = first_val, last_val
    #                     st_time, en_time = first_valid, last_valid
    #
    #             elif r2_num == -2:
    #                 # Stuck at start or end: remove stuck region, then use first/last remaining values
    #                 y = series.copy()
    #                 tokens = str(info).split()
    #                 stuck_len = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 0
    #
    #                 if stuck_len > 1:
    #                     if "start" in info:
    #                         y.iloc[:stuck_len] = np.nan
    #                     elif "end" in info:
    #                         y.iloc[-stuck_len:] = np.nan
    #
    #                 valid = y.dropna()
    #                 if len(valid) >= 2:
    #                     first_idx, last_idx = valid.index[0], valid.index[-1]
    #                     first_val, last_val = valid.iloc[0], valid.iloc[-1]
    #                     raw_diff = last_val - first_val
    #                     n_intervals = y.index.get_loc(last_idx) - y.index.get_loc(first_idx)
    #                     scaled_diff = raw_diff / n_intervals * total_fy_intervals
    #                     percent_scaled = (total_fy_intervals - n_intervals) / total_fy_intervals * 100
    #                     start_val, end_val = first_val, last_val
    #                     st_time, en_time = first_idx, last_idx
    #
    #             elif r2_num == -3:
    #                 # Meters with restart events (cumulative value drops during the year)
    #                 meter_restarts = df_restarts[df_restarts["meter_name"] == meter_name]
    #                 restart_count = len(meter_restarts)
    #
    #                 # If the number of restarts is above the threshold, skip usage reconstruction
    #                 if restart_count > restarts_thres:
    #                     raw_diff = scaled_diff = percent_scaled = np.nan
    #
    #                 else:
    #                     # Sort restart events in chronological order
    #                     meter_restarts = meter_restarts.sort_values("restart_time")
    #
    #                     # Remove consecutive duplicate readings (keeps only the first in each run)
    #                     clean_series = series.mask(series.eq(series.shift()))
    #
    #                     # Restrict data to the analysis window [start_time, end_time]
    #                     meter_series = clean_series.loc[start_time:end_time]
    #
    #                     # Build segments between restart events where cumulative reading increases normally
    #                     periods = []
    #                     prev_time = start_time
    #
    #                     for _, r in meter_restarts.iterrows():
    #                         prev_valid = r["previous_valid_time"]
    #                         restart_t = r["restart_time"]
    #
    #                         # Segment from previous segment start to last valid reading before restart
    #                         if prev_time < prev_valid:
    #                             periods.append((prev_time, prev_valid))
    #
    #                         prev_time = restart_t
    #
    #                     # Final segment from last restart to the end of the analysis window
    #                     if prev_time < end_time:
    #                         periods.append((prev_time, end_time))
    #
    #                     partial_sum = 0.0
    #                     partial_seconds = 0.0
    #
    #                     # Compute usage and elapsed time for all valid segments
    #                     for seg_start, seg_end in periods:
    #                         start_val_seg = meter_series.asof(seg_start)
    #                         end_val_seg = meter_series.asof(seg_end)
    #                         if pd.notna(start_val_seg) and pd.notna(end_val_seg):
    #                             partial_sum += (end_val_seg - start_val_seg)
    #                             partial_seconds += (seg_end - seg_start).total_seconds()
    #
    #                     # Scale partial usage to the full analysis window based on time coverage
    #                     if partial_seconds > 0 and fy_seconds > 0:
    #                         scaled_diff = partial_sum * (fy_seconds / partial_seconds)
    #                         percent_scaled = (fy_seconds - partial_seconds) / fy_seconds * 100
    #
    #                     raw_diff = partial_sum
    #
    #             elif r2_num == -4 or r2_num > 0:
    #                 # Too few points, or positive R² but treated as bad without a correction rule
    #                 raw_diff = scaled_diff = percent_scaled = np.nan
    #
    #         results.append((
    #             meter_name,
    #             st_time, start_val,
    #             en_time, end_val,
    #             raw_diff, scaled_diff,
    #             r2_val, info_val,
    #             percent_scaled
    #         ))
    #
    #     result_df = pd.DataFrame(
    #         results,
    #         columns=[
    #             "meter_name", "start_time", "start",
    #             "end_time", "end",
    #             "raw_difference", "difference",
    #             "R²", "info",
    #             "% scaled"
    #         ]
    #     ).set_index("meter_name")
    #
    #     def format_percent(row):
    #         if pd.isna(row["% scaled"]):
    #             return "N/A" if row["info"] != "" else ""
    #         elif row["% scaled"] == 0:
    #             return ""
    #         else:
    #             return f"{int(row['% scaled'])}%"
    #
    #     result_df["% scaled"] = result_df.apply(format_percent, axis=1)
    #
    #     return result_df

    """
    Compute start/end values, raw difference, scaled difference, R² info, and % scaled for meters.
    Returns a DataFrame indexed by meter_name.

    Assumptions:
        - data has a DatetimeIndex.
        - 0→NaN replacements (if needed) are applied before calling this function.
        - df_bad_meters has columns: meter_name, r2, info.
        - df_restarts has columns: meter_name, previous_valid_time, restart_time.
    """

    results = []

    # Map each meter_name to its (numeric r2, info text, original r2 value)
    bad_dict = {}
    if df_bad_meters is not None and not df_bad_meters.empty:
        for row in df_bad_meters.itertuples(index=False):
            meter = row.meter_name
            r2_orig = row.r2
            info = row.info
            try:
                r2_num = float(r2_orig)
            except Exception:
                r2_num = np.nan
            bad_dict[meter] = (r2_num, info, r2_orig)

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    # Interval count based on index positions (used for most scaling logic)
    start_idx = data.index.get_loc(start_time)
    end_idx = data.index.get_loc(end_time)
    total_fy_intervals = end_idx - start_idx

    # Total duration of the analysis window in seconds (used for restart meters)
    fy_seconds = (end_time - start_time).total_seconds()

    for meter_name in data.columns:
        series = data[meter_name]

        # Default values for output fields
        start_val = end_val = raw_diff = scaled_diff = np.nan
        st_time = en_time = pd.NaT
        r2_val = np.nan
        info_val = ""
        percent_scaled = np.nan

        if meter_name not in bad_dict:
            # Meters not flagged as "bad": direct start/end difference if both endpoints exist
            if start_time in series.index and end_time in series.index:
                start_val = series.loc[start_time]
                end_val = series.loc[end_time]
                raw_diff = scaled_diff = end_val - start_val
                st_time, en_time = start_time, end_time
                percent_scaled = 0
            r2_val, info_val = f">= {r2_threshold}", ""

        else:
            r2_num, info, r2_orig = bad_dict[meter_name]
            r2_val = r2_orig
            info_val = info

            if np.isnan(r2_num):
                # r2 value could not be interpreted numerically
                pass

            elif r2_num == 0:
                # All values are NaN
                pass

            elif r2_num == -1:
                # Missing start or end: use first and last non-repeated values
                y = series.where(series.ne(series.shift()))
                first_valid = y.first_valid_index()
                last_valid = y.last_valid_index()
                if first_valid is not None and last_valid is not None:
                    first_val = y.loc[first_valid]
                    last_val = y.loc[last_valid]
                    raw_diff = last_val - first_val
                    n_intervals = y.index.get_loc(last_valid) - y.index.get_loc(first_valid)
                    if n_intervals >= valid_len:
                        scaled_diff = raw_diff / n_intervals * total_fy_intervals
                        percent_scaled = (total_fy_intervals - n_intervals) / total_fy_intervals * 100
                    start_val, end_val = first_val, last_val
                    st_time, en_time = first_valid, last_valid

            elif r2_num == -2:
                # Stuck at start or end: remove stuck region, then use first/last remaining values
                y = series.copy()
                tokens = str(info).split()
                stuck_len = int(tokens[1]) if len(tokens) > 1 and tokens[1].isdigit() else 0

                if stuck_len > 1:
                    if "start" in info:
                        y.iloc[:stuck_len] = np.nan
                    elif "end" in info:
                        y.iloc[-stuck_len:] = np.nan

                valid = y.dropna()
                if len(valid) >= 2:
                    first_idx, last_idx = valid.index[0], valid.index[-1]
                    first_val, last_val = valid.iloc[0], valid.iloc[-1]
                    raw_diff = last_val - first_val
                    n_intervals = y.index.get_loc(last_idx) - y.index.get_loc(first_idx)
                    scaled_diff = raw_diff / n_intervals * total_fy_intervals
                    percent_scaled = (total_fy_intervals - n_intervals) / total_fy_intervals * 100
                    start_val, end_val = first_val, last_val
                    st_time, en_time = first_idx, last_idx

            elif r2_num == -3:
                # Meters with restart events (cumulative value drops during the year)
                meter_restarts = df_restarts[df_restarts["meter_name"] == meter_name]
                restart_count = len(meter_restarts)

                # If the number of restarts is above the threshold, skip usage reconstruction
                if restart_count > restarts_thres:
                    raw_diff = scaled_diff = percent_scaled = np.nan

                else:
                    # Sort restart events in chronological order
                    meter_restarts = meter_restarts.sort_values("restart_time")

                    # Remove consecutive duplicate readings (keeps only the first in each run)
                    clean_series = series.mask(series.eq(series.shift()))

                    # Restrict data to the analysis window [start_time, end_time]
                    meter_series = clean_series.loc[start_time:end_time]

                    # Build segments between restart events where cumulative reading increases normally
                    periods = []
                    prev_time = start_time

                    for _, r in meter_restarts.iterrows():
                        prev_valid = r["previous_valid_time"]
                        restart_t = r["restart_time"]

                        # Segment from previous segment start to last valid reading before restart
                        if prev_time < prev_valid:
                            periods.append((prev_time, prev_valid))

                        prev_time = restart_t

                    # Final segment from last restart to the end of the analysis window
                    if prev_time < end_time:
                        periods.append((prev_time, end_time))

                    partial_sum = 0.0
                    partial_seconds = 0.0

                    # Compute usage and elapsed time for all valid segments
                    for seg_start, seg_end in periods:
                        start_val_seg = meter_series.asof(seg_start)
                        end_val_seg = meter_series.asof(seg_end)
                        if pd.notna(start_val_seg) and pd.notna(end_val_seg):
                            partial_sum += (end_val_seg - start_val_seg)
                            partial_seconds += (seg_end - seg_start).total_seconds()

                    # Scale partial usage to the full analysis window based on time coverage
                    if partial_seconds > 0 and fy_seconds > 0:
                        scaled_diff = partial_sum * (fy_seconds / partial_seconds)
                        percent_scaled = (fy_seconds - partial_seconds) / fy_seconds * 100

                    raw_diff = partial_sum

            elif r2_num == -4 or r2_num > 0:
                # Too few points, or positive R² but treated as bad without a correction rule
                raw_diff = scaled_diff = percent_scaled = np.nan

        results.append((
            meter_name,
            st_time, start_val,
            en_time, end_val,
            raw_diff, scaled_diff,
            r2_val, info_val,
            percent_scaled
        ))

    result_df = pd.DataFrame(
        results,
        columns=[
            "meter_name", "start_time", "start",
            "end_time", "end",
            "raw_difference", "difference",
            "R²", "info",
            "% scaled"
        ]
    ).set_index("meter_name")

    def format_percent(row):
        if pd.isna(row["% scaled"]):
            return "N/A" if row["info"] != "" else ""
        elif row["% scaled"] == 0:
            return ""
        else:
            return f"{int(row['% scaled'])}%"

    result_df["% scaled"] = result_df.apply(format_percent, axis=1)

    return result_df



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
    meter_info = pd.read_csv(meter_info_file)[['meter_name', 'building_complex_name']]
    meter_info.columns = meter_info.columns.str.strip()
    export_df = export_df.merge(meter_info, on='meter_name', how='left')
    
    # Prepare CSV output: keep meter_name and difference only
    df_csv = export_df[['meter_name', 'difference']].copy()
    df_csv.rename(columns={'difference': f'annual_{var}'}, inplace=True)
    
    # Round to 1 decimal
    df_csv[f'annual_{var}'] = df_csv[f'annual_{var}'].round(1)
    
    # Save to CSV
    df_csv.to_csv(filename, index=False)
    
    return export_df


###################################################################
###################################################################

def export_building_differences(export_df, filename, var="kwh"):
    """
    Aggregate meter differences per building and save to CSV.
    - export_df: full meter-level DataFrame (from export_meter_differences)
    - filename: output CSV path
    - var: variable name for annual consumption (e.g., 'kwh')
    
    The CSV contains:
        building_complex_name, annual_<var>, num_meters
    """
    # Keep only building_complex_name and difference
    df_building = export_df[['building_complex_name', 'difference']].copy()

    # Aggregate sum per building, keep NaN if all are NaN
    df_building_sum = df_building.groupby('building_complex_name', as_index=False).agg(
        difference=('difference', lambda x: x.sum(min_count=1)),  # sum, NaN if all NaN
        num_meters=('difference', 'count')  # count of non-NaN meters
    )

    # Rename column to annual_<var> and round to 1 decimal
    df_building_sum.rename(columns={'difference': f'annual_{var}'}, inplace=True)
    df_building_sum[f'annual_{var}'] = df_building_sum[f'annual_{var}'].round(1)

    # Save to CSV
    df_building_sum.to_csv(filename, index=False)

    return df_building_sum




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

    print(f"✅ Multi-page PDF saved to: {pdf_path}")



###################################################################
###################################################################

def clean_and_interpolate(column, flags, df_restarts=None, special_meters_file=None):
    # ===== ORIGINAL ZHILING / AURORA CODE (commented out) =====
    #     """
    #     Clean stuck points, apply special 'no_interp' corrections, and interpolate.
    #     Restart intervals are set to 0 before interpolation.
    #
    #     Args:
    #         column: pd.Series (time-indexed meter data)
    #         flags: pd.Series of same length to track interpolated points
    #         df_restarts: pd.DataFrame with columns ['meter_name', 'previous_valid_time', 'restart_time']
    #         special_meters_file: Excel file with columns ['meter_name', 'start_datetime', 'end_datetime', 'solution']
    #
    #     Returns:
    #         pd.Series: cleaned and interpolated column
    #     """
    #     column_cp = column.copy()
    #
    #     # STEP 0: Apply 'no_interp' periods from special_meters_file
    #     if special_meters_file is not None:
    #         bad_periods = pd.read_excel(special_meters_file)
    #         bad_periods["meter_name"] = bad_periods["meter_name"].astype(str).str.strip()
    #         bad_periods["solution"] = bad_periods["solution"].astype(str).str.strip().str.lower()
    #         bad_periods["start_datetime"] = pd.to_datetime(bad_periods["start_datetime"])
    #         bad_periods["end_datetime"] = pd.to_datetime(bad_periods["end_datetime"])
    #         # Only keep periods relevant for this column
    #         bad_periods = bad_periods[
    #             (bad_periods["meter_name"] == column.name) &
    #             (bad_periods["solution"] == "no_interp")
    #         ]
    #
    #         for _, row in bad_periods.iterrows():
    #             start = row["start_datetime"]
    #             end = row["end_datetime"]
    #             mask = (column.index >= start) & (column.index <= end)
    #             column.loc[mask] = 0  # replace with 0 for no_interp
    #
    #     # STEP 1: Remove stuck points (consecutive duplicates except 0)
    #     mask_stuck = (column.duplicated(keep='first')) & (column != 0.)
    #     column = column.mask(mask_stuck)
    #
    #     # STEP 2: Replace restart periods with 0
    #     if df_restarts is not None and not df_restarts.empty:
    #         restarts_for_col = df_restarts[df_restarts['meter_name'] == column.name]
    #         for _, r in restarts_for_col.iterrows():
    #             start = r['previous_valid_time']
    #             end = r['restart_time']
    #             column.loc[start:end] = 0
    #
    #     # STEP 3: Interpolate the entire column
    #     column_interpolated = column.interpolate(method='linear')
    #     column[:] = column_interpolated
    #
    #     # STEP 4: Update flags for interpolated values
    #     changed_mask = (column_cp != column) & column.notna()
    #     flags[:] = np.where(changed_mask, 1, flags)
    #
    #     return column

    """
    Clean stuck points, apply optional no_interp corrections, and interpolate.
    Restart intervals are set to 0 before interpolation.
    """
    column_cp = column.copy()
    column = column.copy()

    if (
        special_meters_file is not None
        and str(special_meters_file).strip() != ""
        and os.path.exists(special_meters_file)
    ):
        bad_periods = pd.read_excel(special_meters_file)
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

    mask_stuck = (column.duplicated(keep="first")) & (column != 0.0)
    column = column.mask(mask_stuck)

    if df_restarts is not None and not df_restarts.empty:
        restarts_for_col = df_restarts[df_restarts["meter_name"] == column.name]
        for _, r in restarts_for_col.iterrows():
            start = pd.to_datetime(r["previous_valid_time"])
            end = pd.to_datetime(r["restart_time"])
            column.loc[start:end] = 0

    column_interpolated = column.interpolate(method="linear")
    column[:] = column_interpolated

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