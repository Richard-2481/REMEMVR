#!/usr/bin/env python3
"""Prepare Scatterplot Data (IRT vs CTT): Merge IRT theta scores and CTT mean scores into single dataset for scatterplot"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step07_prepare_scatterplot_data.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 7: Prepare Scatterplot Data (IRT vs CTT)")
        # Load Input Data

        log("Loading IRT theta scores from RQ 5.5.1...")
        # Load data/step00_irt_theta_from_rq551.csv
        # Expected columns: composite_ID, UID, test, location_type, irt_theta, irt_se, TSVR_hours
        # Expected rows: 800 (100 participants x 4 tests x 2 location types)
        theta_long = pd.read_csv(RQ_DIR / "data" / "step00_irt_theta_from_rq551.csv", encoding='utf-8')
        log(f"IRT theta scores ({len(theta_long)} rows, {len(theta_long.columns)} cols)")

        log("Loading CTT mean scores...")
        # Load data/step01_ctt_scores.csv
        # Expected columns: composite_ID, UID, test, location_type, ctt_mean_score, n_items, TSVR_hours
        # Expected rows: 800 (100 participants x 4 tests x 2 location types)
        ctt_scores = pd.read_csv(RQ_DIR / "data" / "step01_ctt_scores.csv", encoding='utf-8')
        log(f"CTT scores ({len(ctt_scores)} rows, {len(ctt_scores.columns)} cols)")
        # Merge Data

        log("Merging IRT theta and CTT scores on composite_ID + location_type...")
        # Merge on composite_ID and location_type (both datasets have these keys)
        # Inner join ensures only matching records retained (should be all 800)
        merged_data = pd.merge(
            theta_long[['composite_ID', 'UID', 'test', 'location_type', 'irt_theta']],
            ctt_scores[['composite_ID', 'location_type', 'ctt_mean_score']],
            on=['composite_ID', 'location_type'],
            how='inner'
        )
        log(f"Combined dataset ({len(merged_data)} rows, {len(merged_data.columns)} cols)")

        # Check for merge issues (should have same row count as inputs)
        if len(merged_data) != 800:
            log(f"Expected 800 rows after merge, got {len(merged_data)}")
            log(f"IRT input: {len(theta_long)} rows, CTT input: {len(ctt_scores)} rows")
        # Select and Sort Columns
        # These outputs will be used by: plotting pipeline for scatterplot generation

        log("Selecting final columns for scatterplot...")
        # Output columns: UID, test, location_type, irt_theta, ctt_mean_score
        scatterplot_data = merged_data[['UID', 'test', 'location_type', 'irt_theta', 'ctt_mean_score']].copy()

        log("Sorting by location_type, UID, test for consistent ordering...")
        # Sort for reproducible plot ordering
        # Primary: location_type (group source/destination together)
        # Secondary: UID (alphabetical participant order)
        # Tertiary: test (chronological test order)
        scatterplot_data.sort_values(by=['location_type', 'UID', 'test'], inplace=True)
        scatterplot_data.reset_index(drop=True, inplace=True)
        log(f"Final dataset ({len(scatterplot_data)} rows)")
        # Save Output
        # Output will be used by: plotting pipeline for scatterplot visualization

        log("Saving scatterplot data to data/step07_scatterplot_data.csv...")
        # Output: data/step07_scatterplot_data.csv
        # Contains: Merged IRT and CTT scores for scatterplot
        # Columns: UID, test, location_type, irt_theta, ctt_mean_score
        output_path = RQ_DIR / "data" / "step07_scatterplot_data.csv"
        scatterplot_data.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(scatterplot_data)} rows, {len(scatterplot_data.columns)} cols)")
        # Validation
        # Validates: Row count, NaN detection, value ranges, location balance
        # Thresholds: 800 rows, no NaN, irt_theta in [-3,3], ctt_mean_score in [0,1]

        log("Running inline validation checks...")

        # Check 1: Row count
        expected_rows = 800
        if len(scatterplot_data) != expected_rows:
            raise ValueError(f"Expected {expected_rows} rows, got {len(scatterplot_data)}")
        log(f"Row count: {len(scatterplot_data)} rows (expected {expected_rows}) ")

        # Check 2: No NaN in critical columns
        nan_irt = scatterplot_data['irt_theta'].isna().sum()
        nan_ctt = scatterplot_data['ctt_mean_score'].isna().sum()
        if nan_irt > 0 or nan_ctt > 0:
            raise ValueError(f"NaN values detected: irt_theta={nan_irt}, ctt_mean_score={nan_ctt}")
        log(f"No NaN in irt_theta or ctt_mean_score ")

        # Check 3: Location balance
        location_counts = scatterplot_data['location_type'].value_counts()
        log(f"Location type counts: {location_counts.to_dict()}")
        if location_counts.get('source', 0) != 400 or location_counts.get('destination', 0) != 400:
            raise ValueError(f"Expected 400 source + 400 destination, got {location_counts.to_dict()}")
        log(f"Location balance: 400 source, 400 destination ")

        # Check 4: IRT theta range
        irt_min = scatterplot_data['irt_theta'].min()
        irt_max = scatterplot_data['irt_theta'].max()
        log(f"IRT theta range: [{irt_min:.2f}, {irt_max:.2f}]")
        if irt_min < -3 or irt_max > 3:
            log(f"IRT theta outside typical [-3, 3] range")
        else:
            log(f"IRT theta in typical [-3, 3] range ")

        # Check 5: CTT mean score range
        ctt_min = scatterplot_data['ctt_mean_score'].min()
        ctt_max = scatterplot_data['ctt_mean_score'].max()
        log(f"CTT mean score range: [{ctt_min:.3f}, {ctt_max:.3f}]")
        if ctt_min < 0 or ctt_max > 1:
            raise ValueError(f"CTT mean score outside [0, 1] range: [{ctt_min}, {ctt_max}]")
        log(f"CTT mean score in [0, 1] range ")

        # Check 6: UID count
        n_uids = scatterplot_data['UID'].nunique()
        log(f"Unique UIDs: {n_uids}")
        if n_uids != 100:
            log(f"Expected 100 unique UIDs, got {n_uids}")
        else:
            log(f"100 unique UIDs present ")

        # Check 7: Test count
        n_tests = scatterplot_data['test'].nunique()
        test_values = sorted(scatterplot_data['test'].unique())
        log(f"Unique tests: {n_tests} (values: {test_values})")
        if n_tests != 4:
            log(f"Expected 4 unique tests, got {n_tests}")
        else:
            log(f"4 unique tests present ")

        log("Step 7 complete")
        log(f"Generated: {output_path}")
        log(f"Rows: {len(scatterplot_data)}, Columns: {list(scatterplot_data.columns)}")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
