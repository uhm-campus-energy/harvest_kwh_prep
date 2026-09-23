# Main Branch Notebooks Summary

## meter_maintenance.ipynb
Review/maintenance tool for the broken-meter list, run separately from the main pipeline. Loads raw 15-min readings, filters to main/submeter, builds PDF review plots with broken/candidate/removed overlays, and lets you propose `update`/`remove`/`add` actions in an editable `broken_meter_update_log.csv`. After building the log it unconditionally `raise SystemExit`s, forcing you to set `refresh_update_log_for_selected_meters = False` and rerun before applying anything. Final commit to the source file still requires the explicit `commit_reviewed_changes_to_source = True` switch.

## 1.kwh_end_points.ipynb
Main "Harvest" kWh pipeline for the current schema. Loads raw interval readings, filters to main meters, detects special/unreliable meters via R² linearity, builds a corrections/candidates workflow, then computes annual only (no monthly breakdown yet) kWh via end-minus-start differencing with scaling for edge-missing meters. Has various commented-out/TODO code (building-level export skipped, "no meter_info file yet") and is explicitly mid-refactor.

## 1.kwh_end_points_TEST.ipynb
A more mature version of the same pipeline, for Aurora_v4 data: loads raw readings, filters main+submeter, detects special meters, syncs a master corrections sheet, computes annual differencing with scaling. Stops after annual calculation (no monthly-kWh section).

## 2.kw_sum.ipynb
Computes annual kWh for meters reporting kW (Aurora_v4, FY25) by summing kW/4, then writes into the building master-sheet Excel and merges in HECO annual kWh data.

## 3.sf_report.ipynb
Inserts square-footage sums into the building master-sheet Excel and reports mismatched/missing building IDs.

## 4.kwh_processing_all.ipynb
Legacy full-history (FY22-FY25) Aurora_v4 pipeline that corrects, interpolates, and computes per-interval deltas for export/disaggregation use, rather than annual/monthly summaries.

## 0.extra.ipynb
A small utility notebook: combines two overlapping Harvest raw-data export CSVs into one (keeping the newer rows on overlap, with an audit CSV of what was overlapped/dropped), plus a quick cell to check the min/max datetime in the combined file.
