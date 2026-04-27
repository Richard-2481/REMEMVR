#!/usr/bin/env python3
"""
Step 00: Load Theta Scores from RQ 5.3.1
RQ 5.3.3 - Paradigm Consolidation Window

Purpose: Load paradigm-specific theta scores from RQ 5.3.1 and verify
data structure for piecewise LMM analysis.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
RQ_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RQ_DIR.parents[2]

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOG_FILE = RQ_DIR / "logs" / "step00_load_theta_from_rq531.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Load and validate theta scores from RQ 5.3.1."""
    logger.info("=" * 60)
    logger.info("Step 00: Load Theta Scores from RQ 5.3.1")
    logger.info("=" * 60)

    # Define paths
    source_file = PROJECT_ROOT / "results" / "ch5" / "5.3.1" / "data" / "step04_lmm_input.csv"
    output_file = RQ_DIR / "data" / "step00_theta_from_rq531.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # --- Validation 1: File exists ---
    logger.info(f"Checking source file: {source_file}")
    if not source_file.exists():
        logger.error(f"CRITICAL: Source file not found: {source_file}")
        logger.error("RQ 5.3.1 dependency not met - file not found")
        sys.exit(1)
    logger.info("VALIDATION - PASS: Source file exists")

    # --- Load data ---
    logger.info("Loading data...")
    df = pd.read_csv(source_file)
    logger.info(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")

    # --- Validation 2: Required columns ---
    # Actual columns in file: composite_ID, UID, test, TSVR_hours, TSVR_hours_sq, TSVR_hours_log, paradigm, theta
    required_cols = ["UID", "test", "TSVR_hours", "paradigm", "theta"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"CRITICAL: Missing required columns: {missing_cols}")
        sys.exit(1)
    logger.info(f"VALIDATION - PASS: Required columns present: {required_cols}")

    # --- Validation 3: Row count ---
    expected_rows = 1200  # 100 participants x 4 tests x 3 paradigms
    if len(df) != expected_rows:
        logger.error(f"CRITICAL: Expected {expected_rows} rows, found {len(df)}")
        sys.exit(1)
    logger.info(f"VALIDATION - PASS: Row count = {len(df)} (expected {expected_rows})")

    # --- Validation 4: Paradigm levels ---
    # Actual values: free_recall, cued_recall, recognition
    paradigm_levels = df["paradigm"].unique()
    expected_paradigms = {"free_recall", "cued_recall", "recognition"}
    if set(paradigm_levels) != expected_paradigms:
        logger.error(f"CRITICAL: Expected paradigms {expected_paradigms}, found {set(paradigm_levels)}")
        sys.exit(1)
    logger.info(f"VALIDATION - PASS: Paradigms present: {sorted(paradigm_levels)}")

    # Map paradigms to standard codes for downstream analysis
    paradigm_map = {
        "free_recall": "IFR",
        "cued_recall": "ICR",
        "recognition": "IRE"
    }
    df["paradigm_code"] = df["paradigm"].map(paradigm_map)
    logger.info(f"Paradigm codes mapped: {paradigm_map}")

    # --- Validation 5: Test levels ---
    test_levels = sorted(df["test"].unique())
    expected_tests = [1, 2, 3, 4]
    if test_levels != expected_tests:
        logger.error(f"CRITICAL: Expected tests {expected_tests}, found {test_levels}")
        sys.exit(1)
    logger.info(f"VALIDATION - PASS: Tests present: {test_levels}")

    # Map tests to standard codes
    test_map = {1: "T1", 2: "T2", 3: "T3", 4: "T4"}
    df["test_code"] = df["test"].map(test_map)
    logger.info(f"Test codes mapped: {test_map}")

    # --- Validation 6: No missing theta ---
    theta_nan_count = df["theta"].isna().sum()
    if theta_nan_count > 0:
        logger.error(f"CRITICAL: {theta_nan_count} missing theta values")
        sys.exit(1)
    logger.info("VALIDATION - PASS: No missing theta values")

    # --- Validation 7: Theta range ---
    theta_min = df["theta"].min()
    theta_max = df["theta"].max()
    logger.info(f"Theta range: [{theta_min:.4f}, {theta_max:.4f}]")
    if theta_min < -5 or theta_max > 5:
        logger.warning(f"WARNING: Theta values outside typical IRT range [-3, 3]")
    else:
        logger.info("VALIDATION - PASS: Theta in reasonable IRT range")

    # --- Validation 8: TSVR_hours range ---
    tsvr_min = df["TSVR_hours"].min()
    tsvr_max = df["TSVR_hours"].max()
    logger.info(f"TSVR_hours range: [{tsvr_min:.2f}, {tsvr_max:.2f}]")

    # --- Add timestamp ---
    df["loaded_timestamp"] = datetime.now().isoformat()

    # --- Prepare output columns ---
    # Keep original columns plus new mapped codes
    output_cols = [
        "UID", "test", "test_code", "paradigm", "paradigm_code",
        "theta", "TSVR_hours", "loaded_timestamp"
    ]
    df_output = df[output_cols].copy()

    # --- Save output ---
    df_output.to_csv(output_file, index=False)
    logger.info(f"Output saved: {output_file}")
    logger.info(f"Output shape: {df_output.shape}")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("STEP 00 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Data loaded successfully: {len(df_output)} rows")
    logger.info(f"Paradigms present: IFR, ICR, IRE")
    logger.info(f"Tests present: T1, T2, T3, T4")

    # Log participant count
    n_participants = df_output["UID"].nunique()
    logger.info(f"Participants: {n_participants}")

    return df_output


if __name__ == "__main__":
    main()
