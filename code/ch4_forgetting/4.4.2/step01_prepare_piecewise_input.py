#!/usr/bin/env python3
"""Prepare Piecewise LMM Input: Reshape theta scores to long format, merge TSVR timestamps, create piecewise"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq6
LOG_FILE = RQ_DIR / "logs" / "step01_prepare_piecewise_input.log"
DFDATA_PATH = PROJECT_ROOT / "data" / "cache" / "dfData.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def assign_piecewise_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign piecewise segments and compute Days_within.

    Segments:
      - Early: TSVR_hours in [0, 24] hours (Days 0-1, consolidation phase)
      - Late: TSVR_hours > 24 hours (Days 1+, decay phase)

    Days_within (centered at segment start):
      - Early: Days_within = TSVR_hours / 24
      - Late: Days_within = (TSVR_hours - 24) / 24

    Args:
        df: DataFrame with TSVR_hours column

    Returns:
        DataFrame with Segment and Days_within columns added
    """
    df = df.copy()

    # Assign Segment (use simple threshold instead of pd.cut to handle all values)
    df['Segment'] = np.where(
        df['TSVR_hours'] <= 24,
        'Early',
        'Late'
    )

    # Compute Days_within (centered at segment start)
    df['Days_within'] = np.where(
        df['Segment'] == 'Early',
        df['TSVR_hours'] / 24,  # Early: 0-1 days
        (df['TSVR_hours'] - 24) / 24  # Late: 0-6 days (within Late segment)
    )

    return df

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 1: Prepare Piecewise LMM Input")
        # Load Theta Scores

        log("Loading theta scores from Step 0...")

        input_path = RQ_DIR / "data" / "step00_theta_scores_from_rq5.csv"
        df_theta_wide = pd.read_csv(input_path, encoding='utf-8')

        log(f"{input_path.name} ({len(df_theta_wide)} rows, {len(df_theta_wide.columns)} cols)")
        # Reshape Wide to Long
        # Operation: Melt theta_common/congruent/incongruent into (Congruence, theta, SE)
        # Output: 1200 rows (400 composite_IDs x 3 congruence types)

        log("Reshaping wide to long format...")

        # Melt theta columns
        df_theta_long = df_theta_wide.melt(
            id_vars=['composite_ID'],
            value_vars=['theta_common', 'theta_congruent', 'theta_incongruent'],
            var_name='congruence_type',
            value_name='theta'
        )

        # Melt SE columns
        df_se_long = df_theta_wide.melt(
            id_vars=['composite_ID'],
            value_vars=['se_common', 'se_congruent', 'se_incongruent'],
            var_name='se_type',
            value_name='SE'
        )

        # Combine theta and SE
        df_long = df_theta_long.copy()
        df_long['SE'] = df_se_long['SE']

        # Clean up congruence type labels
        df_long['Congruence'] = df_long['congruence_type'].str.replace('theta_', '').str.capitalize()
        df_long = df_long.drop(columns=['congruence_type'])

        log(f"Long format created: {len(df_long)} rows")
        # Parse composite_ID
        # Operation: Extract UID and test from composite_ID (format: {UID}_{test})

        log("Extracting UID and test from composite_ID...")

        df_long[['UID', 'test']] = df_long['composite_ID'].str.split('_', expand=True)
        df_long['test'] = df_long['test'].astype(int)

        log(f"UID and test extracted")
        # Merge TSVR Hours
        # Operation: Load TSVR from dfData.csv and merge

        log("Loading TSVR timestamps from dfData.csv...")

        # Load dfData.csv with UID, TEST, TSVR columns
        df_master = pd.read_csv(DFDATA_PATH, encoding='utf-8')
        log(f"dfData.csv ({len(df_master)} rows)")

        # Create composite_ID in dfData: UID_TEST
        df_master['composite_ID'] = df_master['UID'].astype(str) + '_' + df_master['TEST'].astype(str)

        # Select TSVR mapping columns
        df_tsvr = df_master[['composite_ID', 'TSVR']].rename(columns={'TSVR': 'TSVR_hours'})

        log(f"Extracted TSVR for {len(df_tsvr)} composite_IDs")

        # Merge TSVR into long format
        df_long = df_long.merge(df_tsvr, on='composite_ID', how='left')

        # Check merge completeness
        if df_long['TSVR_hours'].isna().any():
            n_missing = df_long['TSVR_hours'].isna().sum()
            raise ValueError(
                f"TSVR merge incomplete: {n_missing} rows have missing TSVR_hours"
            )

        log(f"Merged TSVR_hours for all {len(df_long)} rows")
        # Assign Piecewise Segments
        # Operation: Create Segment (Early/Late) and Days_within variables
        # Early: TSVR in [0, 24] hours; Late: TSVR in (24, 168] hours

        log("Assigning piecewise segments...")

        df_piecewise = assign_piecewise_segments(df_long)

        # Verify no segment overlap
        if df_piecewise['Segment'].isna().any():
            n_missing = df_piecewise['Segment'].isna().sum()
            raise ValueError(
                f"Segment assignment incomplete: {n_missing} rows have missing Segment"
            )

        # Count observations per segment
        segment_counts = df_piecewise['Segment'].value_counts()
        log(f"Segment distribution:")
        for segment, count in segment_counts.items():
            log(f"  - {segment}: {count} rows")
        # Apply Treatment Coding
        # Set reference levels: Congruence='Common', Segment='Early'

        log("Applying treatment coding...")

        # Convert to categorical with explicit reference levels
        df_piecewise['Congruence'] = pd.Categorical(
            df_piecewise['Congruence'],
            categories=['Common', 'Congruent', 'Incongruent'],
            ordered=False
        )

        df_piecewise['Segment'] = pd.Categorical(
            df_piecewise['Segment'],
            categories=['Early', 'Late'],
            ordered=False
        )

        log(f"Treatment coding applied (Congruence ref='Common', Segment ref='Early')")
        # Validate and Save

        log("Validating piecewise input structure...")

        # Check row count
        expected_rows = 1200
        if len(df_piecewise) != expected_rows:
            raise ValueError(
                f"Row count incorrect: expected {expected_rows}, found {len(df_piecewise)}"
            )
        log(f"Row count correct (1200)")

        # Check column count
        expected_cols = ['UID', 'test', 'composite_ID', 'Congruence', 'theta', 'SE',
                         'TSVR_hours', 'Segment', 'Days_within']
        if len(df_piecewise.columns) != len(expected_cols):
            raise ValueError(
                f"Column count incorrect: expected {len(expected_cols)}, "
                f"found {len(df_piecewise.columns)}"
            )
        log(f"Column count correct (9)")

        # Check Days_within ranges
        early_days = df_piecewise[df_piecewise['Segment'] == 'Early']['Days_within']
        late_days = df_piecewise[df_piecewise['Segment'] == 'Late']['Days_within']

        if early_days.min() < 0 or early_days.max() > 1:
            raise ValueError(
                f"Early Days_within range invalid: [{early_days.min():.3f}, {early_days.max():.3f}], "
                f"expected [0, 1]"
            )

        if late_days.min() < 0:
            raise ValueError(
                f"Late Days_within minimum invalid: {late_days.min():.3f}, expected >= 0"
            )

        log(f"Days_within ranges valid (Early: [{early_days.min():.3f}, {early_days.max():.3f}], Late: [{late_days.min():.3f}, {late_days.max():.3f}])")

        # Check no missing data
        critical_cols = ['theta', 'SE', 'TSVR_hours', 'Segment', 'Days_within']
        if df_piecewise[critical_cols].isna().any().any():
            n_missing = df_piecewise[critical_cols].isna().sum().sum()
            raise ValueError(
                f"Missing data detected: {n_missing} NaN values in critical columns"
            )
        log(f"No missing data")

        # Reorder columns for clarity
        df_piecewise = df_piecewise[expected_cols]

        # Save output
        output_path = RQ_DIR / "data" / "step01_lmm_input_piecewise.csv"
        df_piecewise.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path.name} ({len(df_piecewise)} rows, {len(df_piecewise.columns)} cols)")

        log("Step 1 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
