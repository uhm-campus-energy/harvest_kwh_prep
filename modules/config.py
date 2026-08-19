"""Shared configuration for the Harvest kWh notebooks and pipeline scripts.

1.kwh_end_points_TEST_config.ipynb and meter_maintenance_config.ipynb both
read the same raw data file and reference the same meter/broken-meter source
files. Keeping those values here means updating a new data pull once instead
of editing multiple notebooks in lockstep.

Paths resolve relative to this file's location, not the caller's working
directory, so the same config works from a notebook in notebooks/, a script
run from anywhere else in the repo, or a server process with a different
cwd. Set HARVEST_DATA_ROOT to point at a different data directory (e.g. a
server data volume outside the repo checkout) without editing this file.
Set HARVEST_VAR_FILE to pin the raw extract explicitly instead of
auto-discovering it.
"""

import os
import re
from datetime import datetime
from pathlib import Path

# Repo root: modules/config.py -> repo root is one level up
REPO_ROOT = Path(__file__).resolve().parent.parent

# Data root: override with HARVEST_DATA_ROOT (e.g. on a server where raw
# extracts live outside the repo checkout)
DATA_ROOT = Path(os.environ.get("HARVEST_DATA_ROOT", REPO_ROOT / "data"))

# Data Directories (kept as strings with a trailing slash so existing
# `input_dir + "filename.csv"` style concatenation in the notebooks still works)
INPUT_DIR = f"{DATA_ROOT / 'extracts'}/"
OUTPUT_DIR = f"{DATA_ROOT / 'outputs'}/"

# Variable
VAR = "kwh"

# Data frequency
FREQ = "15min"


def _discover_var_file(input_dir: Path, var: str, freq: str) -> Path:
    """Pick the raw extract with the widest date coverage.

    Raw files are named harvest_<var>_<freq>_<YYMMDD>-<YYMMDD>.csv and new
    extracts land with different date ranges (and sometimes overlapping
    ones); the widest start-to-end span is the one that should be used, not
    just the most recently modified file.
    """
    pattern = re.compile(
        rf"^harvest_{re.escape(var)}_{re.escape(freq)}_(\d{{6}})-(\d{{6}})\.csv$"
    )

    candidates = []
    for path in input_dir.glob(f"harvest_{var}_{freq}_*.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        start_str, end_str = match.groups()
        try:
            start = datetime.strptime(start_str, "%y%m%d")
            end = datetime.strptime(end_str, "%y%m%d")
        except ValueError:
            continue
        candidates.append((end - start, path))

    if not candidates:
        raise FileNotFoundError(
            f"No raw extract found in {input_dir} matching "
            f"'harvest_{var}_{freq}_YYMMDD-YYMMDD.csv'. Set the HARVEST_VAR_FILE "
            "environment variable to point at the file explicitly, or check "
            "HARVEST_DATA_ROOT."
        )

    # widest date coverage wins; ties broken by filename for determinism
    candidates.sort(key=lambda pair: (pair[0], pair[1].name))
    return candidates[-1][1]


_var_file_override = os.environ.get("HARVEST_VAR_FILE")
VAR_FILE = (
    _var_file_override
    if _var_file_override
    else str(_discover_var_file(Path(INPUT_DIR), VAR, FREQ))
)

# Shared input/review files
METER_INFO_FILE = f"{INPUT_DIR}meter_info.csv"
METER_ISSUES_CANDIDATES_FILE = f"{INPUT_DIR}special_meter_candidates.csv"
BROKEN_METERS_FILE = f"{INPUT_DIR}running_list_broken_meters.csv"

# Shared generated files
METER_CORRECTIONS_FILE = f"{OUTPUT_DIR}special_meters_corrections_master_sheet.csv"
REMOVED_SPECIAL_METER_DATA_FILE = f"{OUTPUT_DIR}removed_special_meter_data.csv"

# Exclude: Meters of buildings equipped with PV and Student Health
METERS_WITH_PV = [
    'bachman_hall_main',
    'campus_ctr_main',
    'dance_bldg_main',
    'gartley_hall_main',
    'warrior_rec_ctr_main',
]
METERS_EXCLUDED = METERS_WITH_PV + ['student_health_main']  # student_health data is in vitality_v5
