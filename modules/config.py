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
