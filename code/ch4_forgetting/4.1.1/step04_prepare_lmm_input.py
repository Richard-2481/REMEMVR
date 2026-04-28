"""
RQ 5.1.1 - Step 04: Prepare LMM Input Data

PURPOSE:
Merge IRT theta scores with continuous TSVR time variable for LMM trajectory analysis.

INPUT:
- data/step03_theta_scores.csv (400 rows, theta from IRT Pass 2)
- results/ch5/5.2.1/data/step00_tsvr_mapping.csv (TSVR time variable)

OUTPUT:
- data/step04_lmm_input.csv (400 rows × 6 columns)

CRITICAL:
Uses CONTINUOUS TSVR_hours as time variable (NOT nominal Days or sessions).
All time transformations (log, sqrt, powers) are created automatically by
the model_selection tool in Step 05.

DESIGN PHILOSOPHY:
Minimal data preparation - only merge theta with TSVR. Let downstream tools
handle transformations to ensure consistency with tool requirements.

Date: 2025-12-08
RQ: ch5/5.1.1
Step: 04
"""

# IMPORTS

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.1
LOG_FILE = RQ_DIR / "logs" / "step04_prepare_lmm_input.log"

# LOGGING

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# MAIN ANALYSIS

if __name__ == "__main__":
    try:
        log("Step 04: Prepare LMM Input Data")
        log("=" * 80)
        # Load Theta Scores from IRT Pass 2

        log("Loading theta scores from IRT Pass 2...")
        theta_path = RQ_DIR / "data" / "step03_theta_scores.csv"

        if not theta_path.exists():
            raise FileNotFoundError(
                f"Theta scores missing: {theta_path}\n"
                "Run step03_irt_calibration_pass2.py first"
            )

        theta_data = pd.read_csv(theta_path, encoding='utf-8')
        log(f"  ✓ Loaded {theta_path.name}")
        log(f"    Rows: {len(theta_data)}")
        log(f"    Columns: {theta_data.columns.tolist()}")

        # Verify expected columns
        required_cols = ['UID', 'test', 'Theta_All']
        missing = [col for col in required_cols if col not in theta_data.columns]
        if missing:
            raise ValueError(f"Missing required columns in theta data: {missing}")

        # Create composite_ID (UID_test format)
        theta_data['composite_ID'] = theta_data['UID'] + '_' + theta_data['test'].astype(str)
        log(f"  ✓ Created composite_ID from UID and test")

        # Rename for clarity
        theta_data = theta_data.rename(columns={
            'Theta_All': 'theta'
        })
        log(f"  ✓ Renamed: Theta_All → theta")

        # Add SE placeholder (fixed SE=0.3 for all, typical IRT uncertainty)
        theta_data['SE'] = 0.3
        log(f"  ✓ Added SE=0.3 (fixed standard error)")
        # Load TSVR Time Variable

        log("Loading TSVR time variable...")
        tsvr_path = PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step00_tsvr_mapping.csv"

        if not tsvr_path.exists():
            raise FileNotFoundError(
                f"TSVR mapping missing: {tsvr_path}\n"
                "This is DERIVED data from RQ 5.2.1 Step 00"
            )

        tsvr_data = pd.read_csv(tsvr_path, encoding='utf-8')
        log(f"  ✓ Loaded {tsvr_path.name}")
        log(f"    Rows: {len(tsvr_data)}")
        log(f"    Columns: {tsvr_data.columns.tolist()}")

        # Verify TSVR column
        if 'TSVR_hours' not in tsvr_data.columns:
            raise ValueError(
                f"TSVR_hours column missing from {tsvr_path}\n"
                f"Found columns: {tsvr_data.columns.tolist()}"
            )

        # Check TSVR is continuous
        tsvr_unique = tsvr_data['TSVR_hours'].nunique()
        log(f"    TSVR unique values: {tsvr_unique}")
        if tsvr_unique < 10:
            raise ValueError(
                f"TSVR_hours has only {tsvr_unique} unique values - appears categorical.\n"
                "Expected continuous time variable (hours since encoding)."
            )

        log(f"    TSVR range: [{tsvr_data['TSVR_hours'].min():.2f}, "
            f"{tsvr_data['TSVR_hours'].max():.2f}] hours")
        # Merge Theta with TSVR

        log("Merging theta scores with TSVR on composite_ID...")

        lmm_input = theta_data.merge(
            tsvr_data[['composite_ID', 'TSVR_hours']],
            on='composite_ID',
            how='left'
        )

        # Verify merge success
        if lmm_input['TSVR_hours'].isna().any():
            n_missing = lmm_input['TSVR_hours'].isna().sum()
            raise ValueError(
                f"Merge failed: {n_missing} composite_IDs missing TSVR values.\n"
                "All theta observations must have matching TSVR."
            )

        log(f"  ✓ Merge successful: {len(lmm_input)} rows")
        log(f"  ✓ UIDs: {lmm_input['UID'].nunique()} unique")
        log(f"  ✓ Tests: {lmm_input['test'].nunique()} unique")
        # Validate Output Data

        log("Checking output data quality...")

        # Expected structure: 100 participants × 4 tests = 400 rows
        expected_rows = 400
        expected_participants = 100
        expected_tests_per_participant = 4

        if len(lmm_input) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows, got {len(lmm_input)}.\n"
                "Should be 100 participants × 4 tests."
            )
        log(f"  ✓ Row count: {len(lmm_input)} (expected {expected_rows})")

        n_participants = lmm_input['UID'].nunique()
        if n_participants != expected_participants:
            raise ValueError(
                f"Expected {expected_participants} participants, got {n_participants}"
            )
        log(f"  ✓ Participants: {n_participants} (expected {expected_participants})")

        # Check each participant has exactly 4 observations
        tests_per_uid = lmm_input.groupby('UID').size()
        if not (tests_per_uid == expected_tests_per_participant).all():
            unbalanced = tests_per_uid[tests_per_uid != expected_tests_per_participant]
            raise ValueError(
                f"{len(unbalanced)} participants have != {expected_tests_per_participant} tests:\n"
                f"{unbalanced.to_dict()}"
            )
        log(f"  ✓ Each participant has {expected_tests_per_participant} tests (balanced design)")

        # Check no missing values
        if lmm_input.isna().any().any():
            missing_summary = lmm_input.isna().sum()
            missing_cols = missing_summary[missing_summary > 0]
            raise ValueError(
                f"Missing values detected:\n{missing_cols.to_dict()}"
            )
        log("  ✓ No missing values")

        # Check theta range (typical IRT: -4 to +4)
        theta_min, theta_max = lmm_input['theta'].min(), lmm_input['theta'].max()
        log(f"  ✓ Theta range: [{theta_min:.3f}, {theta_max:.3f}]")
        if theta_min < -6 or theta_max > 6:
            raise ValueError(
                f"Theta range [{theta_min:.3f}, {theta_max:.3f}] outside typical IRT bounds [-6, +6]"
            )

        # Check SE range (typical: 0.2 to 1.5)
        se_min, se_max = lmm_input['SE'].min(), lmm_input['SE'].max()
        log(f"  ✓ SE range: [{se_min:.3f}, {se_max:.3f}]")
        if se_min < 0.05 or se_max > 3.0:
            raise ValueError(
                f"SE range [{se_min:.3f}, {se_max:.3f}] outside typical bounds [0.05, 3.0]"
            )
        # Reorder Columns and Save

        log("Saving LMM input data...")

        # Column order: identifiers, outcome, time
        output_columns = ['composite_ID', 'UID', 'test', 'theta', 'SE', 'TSVR_hours']
        lmm_input = lmm_input[output_columns]

        output_path = RQ_DIR / "data" / "step04_lmm_input.csv"
        lmm_input.to_csv(output_path, index=False, encoding='utf-8')

        log(f"  ✓ Saved to {output_path.name}")
        log(f"    Rows: {len(lmm_input)}")
        log(f"    Columns: {len(lmm_input.columns)}")
        log(f"    Column names: {lmm_input.columns.tolist()}")
        log(f"    File size: {output_path.stat().st_size:,} bytes")
        # SUMMARY

        log("=" * 80)
        log("Step 04 Complete")
        log(f"  Output: {output_path.name} ({len(lmm_input)} rows × {len(lmm_input.columns)} cols)")
        log(f"  Time variable: TSVR_hours (continuous, {tsvr_unique} unique values)")
        log(f"  Outcome: theta (IRT ability estimates)")
        log(f"  Ready for: Step 05 (kitchen_sink model selection)")
        log("=" * 80)

    except Exception as e:
        log(f"\nStep 04 Failed: {e}")
        import traceback
        log(traceback.format_exc())
        raise
