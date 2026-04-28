#!/usr/bin/env python3
"""step00_load_dependency_data: Load IRT theta scores and TSVR mapping from completed RQ 5.5.1 (Source-Destination"""

import sys
from pathlib import Path
import pandas as pd
import yaml
from typing import Dict, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import validation tools
from tools.validation import check_file_exists

# Configuration

# Paths (all relative to project root)
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.2
DEPENDENCY_DIR = PROJECT_ROOT / "results" / "ch5" / "5.5.1"  # RQ 5.5.1
LOG_FILE = RQ_DIR / "logs" / "step00_load_dependency_data.log"

# Dependency files (from RQ 5.5.1)
DEPENDENCY_STATUS = DEPENDENCY_DIR / "status.yaml"
DEPENDENCY_THETA = DEPENDENCY_DIR / "data" / "step03_theta_scores.csv"
DEPENDENCY_TSVR = DEPENDENCY_DIR / "data" / "step00_tsvr_mapping.csv"

# Output file (to RQ 5.5.2 data/ folder)
OUTPUT_MERGED = RQ_DIR / "data" / "step00_theta_from_rq551.csv"

# Expected data dimensions
EXPECTED_ROWS = 400  # 100 UID × 4 test sessions
THETA_COLUMNS = ['composite_ID', 'theta_source', 'theta_destination', 'se_source', 'se_destination']
TSVR_COLUMNS = ['composite_ID', 'UID', 'test', 'TSVR_hours']
OUTPUT_COLUMNS = ['composite_ID', 'UID', 'test', 'theta_source', 'theta_destination', 'se_source', 'se_destination', 'TSVR_hours']

# Validation thresholds
THETA_MIN, THETA_MAX = -3.0, 3.0
SE_MIN, SE_MAX = 0.1, 1.0
TSVR_MIN, TSVR_MAX = 0.0, 360.0  # Extended range: some participants have TSVR > 7 days


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 00: Load Dependency Data from RQ 5.5.1")
        # Check Dependency Completion

        log("Verifying RQ 5.5.1 completion status...")

        # Check status.yaml exists
        if not DEPENDENCY_STATUS.exists():
            raise FileNotFoundError(
                f"RQ 5.5.1 status.yaml not found at {DEPENDENCY_STATUS}\n"
                f"RQ 5.5.1 may not have been created yet. Run RQ 5.5.1 first."
            )

        # Read status.yaml
        with open(DEPENDENCY_STATUS, 'r', encoding='utf-8') as f:
            status_data = yaml.safe_load(f)

        # Check results analysis.status = 'success'
        results analysis_status = status_data.get('results analysis', {}).get('status', 'missing')
        log(f"RQ 5.5.1 results analysis.status = '{results analysis_status}'")

        if results analysis_status != 'success':
            raise ValueError(
                f"RQ 5.5.1 not complete (results analysis.status = '{results analysis_status}', expected 'success')\n"
                f"RQ 5.5.2 depends on RQ 5.5.1 completion. Complete RQ 5.5.1 first."
            )

        log("RQ 5.5.1 completed successfully")
        # Check Input Files Exist

        log("Verifying dependency files exist...")

        # Check theta scores file
        theta_check = check_file_exists(str(DEPENDENCY_THETA), min_size_bytes=1000)
        if not theta_check['valid']:
            raise FileNotFoundError(
                f"Theta scores file not found or too small\n"
                f"Expected: {DEPENDENCY_THETA}\n"
                f"Message: {theta_check['message']}"
            )
        log(f"Theta scores file exists ({theta_check['size_bytes']} bytes)")

        # Check TSVR mapping file
        tsvr_check = check_file_exists(str(DEPENDENCY_TSVR), min_size_bytes=1000)
        if not tsvr_check['valid']:
            raise FileNotFoundError(
                f"TSVR mapping file not found or too small\n"
                f"Expected: {DEPENDENCY_TSVR}\n"
                f"Message: {tsvr_check['message']}"
            )
        log(f"TSVR mapping file exists ({tsvr_check['size_bytes']} bytes)")
        # Load Input Data

        log("Loading theta scores from RQ 5.5.1...")
        df_theta = pd.read_csv(DEPENDENCY_THETA, encoding='utf-8')
        log(f"{DEPENDENCY_THETA.name} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        log(f"{list(df_theta.columns)}")

        log("Loading TSVR mapping from RQ 5.5.1...")
        df_tsvr = pd.read_csv(DEPENDENCY_TSVR, encoding='utf-8')
        log(f"{DEPENDENCY_TSVR.name} ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")
        log(f"{list(df_tsvr.columns)}")
        # Validate Column Structure

        log("Checking column structures...")

        # Check theta columns
        missing_theta_cols = [col for col in THETA_COLUMNS if col not in df_theta.columns]
        if missing_theta_cols:
            raise ValueError(
                f"Theta scores file missing expected columns: {missing_theta_cols}\n"
                f"Expected: {THETA_COLUMNS}\n"
                f"Actual: {list(df_theta.columns)}"
            )
        log(f"Theta file has all expected columns")

        # Check TSVR columns
        missing_tsvr_cols = [col for col in TSVR_COLUMNS if col not in df_tsvr.columns]
        if missing_tsvr_cols:
            raise ValueError(
                f"TSVR mapping file missing expected columns: {missing_tsvr_cols}\n"
                f"Expected: {TSVR_COLUMNS}\n"
                f"Actual: {list(df_tsvr.columns)}"
            )
        log(f"TSVR file has all expected columns")
        # Validate Row Counts

        log("Checking row counts...")

        if len(df_theta) != EXPECTED_ROWS:
            raise ValueError(
                f"Theta scores file has unexpected row count: {len(df_theta)} (expected {EXPECTED_ROWS})\n"
                f"RQ 5.5.1 may have incomplete data."
            )
        log(f"Theta file has {EXPECTED_ROWS} rows")

        if len(df_tsvr) != EXPECTED_ROWS:
            raise ValueError(
                f"TSVR mapping file has unexpected row count: {len(df_tsvr)} (expected {EXPECTED_ROWS})\n"
                f"RQ 5.5.1 may have incomplete data."
            )
        log(f"TSVR file has {EXPECTED_ROWS} rows")
        # Merge Theta Scores with TSVR Mapping
        # Merge key: composite_ID (unique UID_test identifier, e.g., 'A010_1')

        log("Merging theta scores with TSVR mapping on composite_ID...")

        df_merged = df_theta.merge(
            df_tsvr,
            on='composite_ID',
            how='left',
            validate='one_to_one'  # Enforce 1:1 relationship (catch duplicates)
        )

        log(f"{len(df_merged)} rows after merge")

        # Check for unmatched rows (NaN in TSVR_hours indicates no match)
        n_unmatched = df_merged['TSVR_hours'].isna().sum()
        if n_unmatched > 0:
            raise ValueError(
                f"Merge incomplete: {n_unmatched} theta rows have no TSVR match\n"
                f"This indicates composite_ID mismatch between files."
            )
        log(f"All {EXPECTED_ROWS} theta rows successfully matched with TSVR")

        # Reorder columns to match expected output
        df_merged = df_mergedlog(f"Reordered to {OUTPUT_COLUMNS}")
        # Validate Merged Data Ranges

        log("Checking data ranges...")

        # Validate theta_source range
        theta_source_min, theta_source_max = df_merged['theta_source'].min(), df_merged['theta_source'].max()
        if theta_source_min < THETA_MIN or theta_source_max > THETA_MAX:
            raise ValueError(
                f"theta_source out of expected range [{THETA_MIN}, {THETA_MAX}]\n"
                f"Actual range: [{theta_source_min:.3f}, {theta_source_max:.3f}]"
            )
        log(f"theta_source in [{theta_source_min:.3f}, {theta_source_max:.3f}]")

        # Validate theta_destination range
        theta_dest_min, theta_dest_max = df_merged['theta_destination'].min(), df_merged['theta_destination'].max()
        if theta_dest_min < THETA_MIN or theta_dest_max > THETA_MAX:
            raise ValueError(
                f"theta_destination out of expected range [{THETA_MIN}, {THETA_MAX}]\n"
                f"Actual range: [{theta_dest_min:.3f}, {theta_dest_max:.3f}]"
            )
        log(f"theta_destination in [{theta_dest_min:.3f}, {theta_dest_max:.3f}]")

        # Validate se_source range
        se_source_min, se_source_max = df_merged['se_source'].min(), df_merged['se_source'].max()
        if se_source_min < SE_MIN or se_source_max > SE_MAX:
            raise ValueError(
                f"se_source out of expected range [{SE_MIN}, {SE_MAX}]\n"
                f"Actual range: [{se_source_min:.3f}, {se_source_max:.3f}]"
            )
        log(f"se_source in [{se_source_min:.3f}, {se_source_max:.3f}]")

        # Validate se_destination range
        se_dest_min, se_dest_max = df_merged['se_destination'].min(), df_merged['se_destination'].max()
        if se_dest_min < SE_MIN or se_dest_max > SE_MAX:
            raise ValueError(
                f"se_destination out of expected range [{SE_MIN}, {SE_MAX}]\n"
                f"Actual range: [{se_dest_min:.3f}, {se_dest_max:.3f}]"
            )
        log(f"se_destination in [{se_dest_min:.3f}, {se_dest_max:.3f}]")

        # Validate TSVR_hours range
        tsvr_min, tsvr_max = df_merged['TSVR_hours'].min(), df_merged['TSVR_hours'].max()
        if tsvr_min < TSVR_MIN or tsvr_max > TSVR_MAX:
            raise ValueError(
                f"TSVR_hours out of expected range [{TSVR_MIN}, {TSVR_MAX}]\n"
                f"Actual range: [{tsvr_min:.3f}, {tsvr_max:.3f}]"
            )
        log(f"TSVR_hours in [{tsvr_min:.3f}, {tsvr_max:.3f}]")
        # Save Merged Data
        # Output: data/step00_theta_from_rq551.csv (400 rows, 8 columns)

        log(f"Saving merged data to {OUTPUT_MERGED.name}...")
        df_merged.to_csv(OUTPUT_MERGED, index=False, encoding='utf-8')
        log(f"{OUTPUT_MERGED.name} ({len(df_merged)} rows, {len(df_merged.columns)} cols)")
        # Final Validation (File Exists Check)

        log("Verifying output file exists...")
        output_check = check_file_exists(str(OUTPUT_MERGED), min_size_bytes=10000)
        if not output_check['valid']:
            raise FileNotFoundError(
                f"Output file write failed or file too small\n"
                f"Message: {output_check['message']}"
            )
        log(f"Output file exists ({output_check['size_bytes']} bytes)")
        # SUCCESS

        log("Step 00 complete")
        log(f"Loaded {EXPECTED_ROWS} theta scores from RQ 5.5.1")
        log(f"Merged with TSVR mapping (0 unmatched rows)")
        log(f"Output: {OUTPUT_MERGED.relative_to(PROJECT_ROOT)}")
        log(f"Run step01_create_piecewise_time_variables.py")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
