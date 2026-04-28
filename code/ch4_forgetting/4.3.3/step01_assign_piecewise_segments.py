#!/usr/bin/env python3
"""
Step 01: Assign Temporal Segments and Compute Days_within
RQ 5.3.3 - Paradigm Consolidation Window

Purpose: Create piecewise data structure by assigning temporal segments
(Early vs Late) and computing Days_within (time recentered within each
segment to start at 0).
"""

import sys
import logging
from pathlib import Path

import pandas as pd

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
RQ_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RQ_DIR.parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import assign_piecewise_segments
from tools.validation import validate_dataframe_structure

# Setup logging
LOG_FILE = RQ_DIR / "logs" / "step01_assign_piecewise_segments.log"
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
    """Assign temporal segments and compute Days_within."""
    logger.info("=" * 60)
    logger.info("Step 01: Assign Piecewise Segments")
    logger.info("=" * 60)

    # Define paths
    input_file = RQ_DIR / "data" / "step00_theta_from_rq531.csv"
    output_file = RQ_DIR / "data" / "step01_piecewise_lmm_input.csv"

    # --- Load data ---
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"Data loaded: {len(df)} rows")

    # --- Apply piecewise segment assignment ---
    logger.info("Applying piecewise segment assignment...")
    logger.info("  Early segment: TSVR_hours <= 24 hours (consolidation window)")
    logger.info("  Late segment: TSVR_hours > 24 hours (decay period)")

    piecewise_data = assign_piecewise_segments(
        df=df,
        tsvr_col='TSVR_hours',
        early_cutoff_hours=24.0
    )

    logger.info(f"Segment assignment complete")

    # --- Validation: Check segment assignment ---
    segment_counts = piecewise_data['Segment'].value_counts()
    logger.info(f"Segment counts: {dict(segment_counts)}")
    if 'Early' not in segment_counts or 'Late' not in segment_counts:
        logger.error("CRITICAL: Missing segment level")
        sys.exit(1)

    logger.info(f"Segment assignment complete: {segment_counts['Early']} Early, {segment_counts['Late']} Late")

    # --- Validation: Days_within computed correctly ---
    early_min = piecewise_data[piecewise_data['Segment'] == 'Early']['Days_within'].min()
    late_min = piecewise_data[piecewise_data['Segment'] == 'Late']['Days_within'].min()

    logger.info(f"Days_within - Early segment min: {early_min:.4f}")
    logger.info(f"Days_within - Late segment min: {late_min:.4f}")

    if late_min < 0:
        logger.error(f"CRITICAL: Negative Days_within in Late segment: {late_min}")
        sys.exit(1)

    logger.info(f"Days_within computed: min={min(early_min, late_min):.4f} for both segments (approx)")

    # --- Validation: Use validation tool ---
    expected_cols = [
        "UID", "test", "test_code", "paradigm", "paradigm_code",
        "theta", "TSVR_hours", "loaded_timestamp", "Segment", "Days_within"
    ]

    validation_result = validate_dataframe_structure(
        df=piecewise_data,
        expected_rows=1200,
        expected_columns=expected_cols
    )

    if not validation_result['valid']:
        logger.error(f"CRITICAL: Validation failed: {validation_result['message']}")
        sys.exit(1)

    logger.info("VALIDATION - PASS: Dataframe structure validated")

    # --- Additional validation: Balanced design ---
    # Each segment-paradigm should have 200 rows (100 participants x 2 tests)
    for segment in ['Early', 'Late']:
        for paradigm in ['IFR', 'ICR', 'IRE']:
            count = len(piecewise_data[
                (piecewise_data['Segment'] == segment) &
                (piecewise_data['paradigm_code'] == paradigm)
            ])
            logger.info(f"  {segment}-{paradigm}: {count} rows")
            if count != 200:
                logger.warning(f"WARNING: Expected 200 rows for {segment}-{paradigm}, got {count}")

    # --- Log summary statistics ---
    logger.info("\nDays_within summary by Segment:")
    for segment in ['Early', 'Late']:
        seg_data = piecewise_data[piecewise_data['Segment'] == segment]['Days_within']
        logger.info(f"  {segment}: min={seg_data.min():.3f}, max={seg_data.max():.3f}, mean={seg_data.mean():.3f}")

    # --- Save output ---
    piecewise_data.to_csv(output_file, index=False)
    logger.info(f"\nOutput saved: {output_file}")
    logger.info(f"Output shape: {piecewise_data.shape}")

    # --- Summary ---
    logger.info("=" * 60)
    logger.info("STEP 01 COMPLETE")
    logger.info("=" * 60)

    return piecewise_data


if __name__ == "__main__":
    main()
