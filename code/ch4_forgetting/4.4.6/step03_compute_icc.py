#!/usr/bin/env python3
"""
Step 03: Compute ICC Estimates

PURPOSE:
Compute three types of Intraclass Correlation Coefficients (ICC) for each congruence
level to quantify between-person vs within-person variance in intercepts and slopes.

EXPECTED INPUTS:
- data/step02_variance_components.csv: Variance components (15 rows: 5 components x 3 congruence)

EXPECTED OUTPUTS:
- data/step03_icc_estimates.csv: ICC estimates (9 rows: 3 ICC types x 3 congruence)
- data/step03_icc_summary.txt: Text report with ICC interpretations

VALIDATION CRITERIA:
- All ICC values in [0, 1]
- Exactly 9 ICC estimates (3 types x 3 congruence)
- No NaN values
- Magnitude classifications valid
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import compute_icc_from_variance_components
from tools.validation import validate_icc_bounds

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_compute_icc.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("[START] Step 03: Compute ICC Estimates")

        # =====================================================================
        # STEP 1: Load Variance Components
        # =====================================================================
        log("[LOAD] Loading variance components...")

        variance_file = RQ_DIR / "data" / "step02_variance_components.csv"
        df_variance = pd.read_csv(variance_file, encoding='utf-8')

        log(f"[LOADED] {variance_file.name} ({len(df_variance)} rows)")

        # =====================================================================
        # STEP 2: Compute ICC Estimates for Each Congruence Level
        # =====================================================================
        log("[ANALYSIS] Computing ICC estimates for each congruence level...")

        all_icc = []

        for congruence in sorted(df_variance['congruence'].unique()):
            log(f"\n[ICC] Computing ICC for {congruence}...")

            # Filter for this congruence
            df_cong = df_variance[df_variance['congruence'] == congruence].copy()

            # Rename components to match what ICC function expects
            # var_intercept -> Intercept
            # var_slope -> TSVR_hours
            # var_residual -> Residual
            component_map = {
                'var_intercept': 'Intercept',
                'var_slope': 'TSVR_hours',
                'var_residual': 'Residual',
                'cov_int_slope': 'Intercept:TSVR_hours'
            }
            df_cong['component'] = df_cong['component'].map(component_map).fillna(df_cong['component'])

            # Rename 'value' to 'variance' (expected by ICC function)
            df_cong = df_cong.rename(columns={'value': 'variance'})

            # Call ICC function
            df_icc_cong = compute_icc_from_variance_components(
                df_cong[['component', 'variance']],
                slope_name='TSVR_hours',
                time_point=144.0  # Day 6 (144 hours)
            )

            # Add congruence column
            df_icc_cong['congruence'] = congruence

            # Rename columns to match expected output format
            if 'icc_value' in df_icc_cong.columns:
                df_icc_cong = df_icc_cong.rename(columns={'icc_value': 'value'})
            if 'interpretation' in df_icc_cong.columns:
                df_icc_cong = df_icc_cong.rename(columns={'interpretation': 'magnitude'})

            all_icc.append(df_icc_cong)

            log(f"  Computed {len(df_icc_cong)} ICC estimates for {congruence}")

        # Combine all ICC estimates
        df_icc = pd.concat(all_icc, ignore_index=True)

        # Reorder columns
        df_icc = df_icc[['congruence', 'icc_type', 'value', 'magnitude']]

        log(f"\n[COMPUTED] {len(df_icc)} total ICC estimates")

        # Display ICC estimates
        log("\n[ICC ESTIMATES]")
        for _, row in df_icc.iterrows():
            log(f"  {row['congruence']:12s} | {row['icc_type']:20s} | {row['value']:.4f} ({row['magnitude']})")

        # =====================================================================
        # STEP 3: Save ICC Estimates
        # =====================================================================
        log("\n[SAVE] Saving ICC estimates...")

        icc_output = RQ_DIR / "data" / "step03_icc_estimates.csv"
        df_icc.to_csv(icc_output, index=False, encoding='utf-8')

        log(f"[SAVED] {icc_output.name} ({len(df_icc)} rows)")

        # =====================================================================
        # STEP 4: Create ICC Summary Report
        # =====================================================================
        log("[REPORT] Creating ICC interpretation report...")

        report_path = RQ_DIR / "data" / "step03_icc_summary.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("INTRACLASS CORRELATION COEFFICIENT (ICC) ESTIMATES\n")
            f.write("Congruence-Stratified Variance Decomposition\n")
            f.write("=" * 80 + "\n\n")

            f.write("ICC INTERPRETATION GUIDE:\n")
            f.write("  Low:         ICC < 0.20 (most variance within-person)\n")
            f.write("  Moderate:    0.20 <= ICC < 0.40\n")
            f.write("  Substantial: ICC >= 0.40 (most variance between-person)\n\n")

            f.write("=" * 80 + "\n\n")

            # Group by congruence
            for congruence in sorted(df_icc['congruence'].unique()):
                df_cong = df_icc[df_icc['congruence'] == congruence]

                f.write(f"CONGRUENCE: {congruence}\n")
                f.write("-" * 80 + "\n")

                for _, row in df_cong.iterrows():
                    f.write(f"  {row['icc_type']:25s}: {row['value']:.4f} ({row['magnitude']})\n")

                # Interpretation
                icc_intercept = df_cong[df_cong['icc_type'] == 'intercept']['value'].values[0]
                icc_slope_simple = df_cong[df_cong['icc_type'] == 'slope_simple']['value'].values[0]

                f.write("\n  INTERPRETATION:\n")
                f.write(f"    - Baseline stability: {icc_intercept:.1%} of variance between-person\n")
                f.write(f"    - Forgetting rate stability: {icc_slope_simple:.1%} of slope variance between-person\n")

                if icc_intercept >= 0.40:
                    f.write("    - SUBSTANTIAL between-person differences in initial memory\n")
                elif icc_intercept >= 0.20:
                    f.write("    - MODERATE between-person differences in initial memory\n")
                else:
                    f.write("    - LOW between-person differences in initial memory\n")

                if icc_slope_simple >= 0.40:
                    f.write("    - SUBSTANTIAL between-person differences in forgetting rates\n")
                elif icc_slope_simple >= 0.20:
                    f.write("    - MODERATE between-person differences in forgetting rates\n")
                else:
                    f.write("    - LOW between-person differences in forgetting rates\n")

                f.write("\n")

        log(f"[SAVED] {report_path.name}")

        # =====================================================================
        # STEP 5: Validate ICC Bounds
        # =====================================================================
        log("\n[VALIDATION] Validating ICC bounds...")

        validation = validate_icc_bounds(df_icc, icc_col='value')

        if validation['valid']:
            log("[PASS] All ICC values within valid bounds [0, 1]")
        else:
            log(f"[FAIL] ICC validation failed: {validation['message']}")
            raise ValueError(validation['message'])

        # Additional validation: check row count
        if len(df_icc) != 9:
            raise ValueError(f"Expected 9 ICC estimates (3 types x 3 congruence), got {len(df_icc)}")

        log("[PASS] ICC estimate count validated (9 estimates)")

        # Check for NaN
        if df_icc['value'].isna().any():
            raise ValueError("Found NaN values in ICC estimates")

        log("[PASS] No NaN values in ICC estimates")

        log("\n[SUCCESS] Step 03 complete - ICC estimates computed and validated")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
