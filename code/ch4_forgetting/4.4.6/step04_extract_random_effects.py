#!/usr/bin/env python3
"""
Step 04: Extract Random Effects

PURPOSE:
Extract individual-level random intercepts and slopes for all 100 participants,
separately for each congruence level, to enable intercept-slope correlation testing
and distribution visualization.

EXPECTED INPUTS:
- data/step02_fitted_model_common.pkl: Common congruence model
- data/step02_fitted_model_congruent.pkl: Congruent congruence model
- data/step02_fitted_model_incongruent.pkl: Incongruent congruence model

EXPECTED OUTPUTS:
- data/step04_random_effects.csv: Random effects (300 rows: 100 UID x 3 congruence)
- data/step04_random_slopes_descriptives.txt: Descriptive statistics report

VALIDATION CRITERIA:
- Exactly 300 rows (100 UID x 3 congruence)
- No NaN in Total_Intercept or Total_Slope
- Each UID appears exactly 3 times
- All 3 congruence levels present
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_extract_random_effects.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("[START] Step 04: Extract Random Effects")

        # =====================================================================
        # STEP 1: Load Fitted Models (using MixedLMResults.load())
        # =====================================================================
        log("[LOAD] Loading fitted LMM models...")

        model_files = {
            'Common': RQ_DIR / "data" / "step02_fitted_model_common.pkl",
            'Congruent': RQ_DIR / "data" / "step02_fitted_model_congruent.pkl",
            'Incongruent': RQ_DIR / "data" / "step02_fitted_model_incongruent.pkl"
        }

        models = {}
        for congruence, model_path in model_files.items():
            # Use MixedLMResults.load() to avoid pickle issues
            models[congruence] = MixedLMResults.load(str(model_path))
            log(f"[LOADED] {model_path.name}")

        # =====================================================================
        # STEP 2: Extract Random Effects for Each Congruence Level
        # =====================================================================
        log("\n[ANALYSIS] Extracting random effects from each model...")

        all_random_effects = []

        for congruence, model in models.items():
            log(f"\n[EXTRACT] Extracting random effects for {congruence}...")

            # Access random effects from model
            # model.random_effects is a dict: {UID: Series with 'Group' (intercept) and 'TSVR_hours' (slope)}
            random_effects = model.random_effects

            # Convert to list of dicts
            re_data = []
            for uid, effects in random_effects.items():
                re_data.append({
                    'UID': uid,
                    'congruence': congruence,
                    'Total_Intercept': effects['Group'],  # Random intercept
                    'Total_Slope': effects['TSVR_hours']  # Random slope
                })

            df_re = pd.DataFrame(re_data)

            all_random_effects.append(df_re)

            log(f"[EXTRACTED] {len(df_re)} participants for {congruence}")
            log(f"  Intercept: mean={df_re['Total_Intercept'].mean():.4f}, std={df_re['Total_Intercept'].std():.4f}")
            log(f"  Slope: mean={df_re['Total_Slope'].mean():.4f}, std={df_re['Total_Slope'].std():.4f}")

        # =====================================================================
        # STEP 3: Combine Random Effects
        # =====================================================================
        log("\n[COMBINE] Combining random effects across congruence levels...")

        df_random_effects = pd.concat(all_random_effects, ignore_index=True)

        log(f"[COMBINED] {len(df_random_effects)} total rows ({df_random_effects['UID'].nunique()} unique participants x {df_random_effects['congruence'].nunique()} congruence levels)")

        # =====================================================================
        # STEP 4: Save Random Effects
        # =====================================================================
        log("[SAVE] Saving random effects...")

        output_file = RQ_DIR / "data" / "step04_random_effects.csv"
        df_random_effects.to_csv(output_file, index=False, encoding='utf-8')

        log(f"[SAVED] {output_file.name} ({len(df_random_effects)} rows)")

        # =====================================================================
        # STEP 5: Create Descriptive Statistics Report
        # =====================================================================
        log("[REPORT] Creating descriptive statistics report...")

        report_path = RQ_DIR / "data" / "step04_random_slopes_descriptives.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RANDOM EFFECTS DESCRIPTIVE STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total observations: {len(df_random_effects)}\n")
            f.write(f"Unique participants: {df_random_effects['UID'].nunique()}\n")
            f.write(f"Congruence levels: {df_random_effects['congruence'].nunique()}\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS (All Congruence Levels)\n")
            f.write("-" * 80 + "\n")

            f.write("\nRandom Intercepts:\n")
            f.write(f"  Mean:   {df_random_effects['Total_Intercept'].mean():8.4f}\n")
            f.write(f"  Std:    {df_random_effects['Total_Intercept'].std():8.4f}\n")
            f.write(f"  Min:    {df_random_effects['Total_Intercept'].min():8.4f}\n")
            f.write(f"  25%:    {df_random_effects['Total_Intercept'].quantile(0.25):8.4f}\n")
            f.write(f"  Median: {df_random_effects['Total_Intercept'].median():8.4f}\n")
            f.write(f"  75%:    {df_random_effects['Total_Intercept'].quantile(0.75):8.4f}\n")
            f.write(f"  Max:    {df_random_effects['Total_Intercept'].max():8.4f}\n")

            f.write("\nRandom Slopes:\n")
            f.write(f"  Mean:   {df_random_effects['Total_Slope'].mean():8.4f}\n")
            f.write(f"  Std:    {df_random_effects['Total_Slope'].std():8.4f}\n")
            f.write(f"  Min:    {df_random_effects['Total_Slope'].min():8.4f}\n")
            f.write(f"  25%:    {df_random_effects['Total_Slope'].quantile(0.25):8.4f}\n")
            f.write(f"  Median: {df_random_effects['Total_Slope'].median():8.4f}\n")
            f.write(f"  75%:    {df_random_effects['Total_Slope'].quantile(0.75):8.4f}\n")
            f.write(f"  Max:    {df_random_effects['Total_Slope'].max():8.4f}\n")

            f.write("\n")

            # By congruence level
            for congruence in sorted(df_random_effects['congruence'].unique()):
                df_cong = df_random_effects[df_random_effects['congruence'] == congruence]

                f.write(f"\nCONGRUENCE: {congruence}\n")
                f.write("-" * 80 + "\n")

                f.write(f"N participants: {len(df_cong)}\n\n")

                f.write("Random Intercepts:\n")
                f.write(f"  Mean:   {df_cong['Total_Intercept'].mean():8.4f}\n")
                f.write(f"  Std:    {df_cong['Total_Intercept'].std():8.4f}\n")
                f.write(f"  Min:    {df_cong['Total_Intercept'].min():8.4f}\n")
                f.write(f"  Median: {df_cong['Total_Intercept'].median():8.4f}\n")
                f.write(f"  Max:    {df_cong['Total_Intercept'].max():8.4f}\n")

                f.write("\nRandom Slopes:\n")
                f.write(f"  Mean:   {df_cong['Total_Slope'].mean():8.4f}\n")
                f.write(f"  Std:    {df_cong['Total_Slope'].std():8.4f}\n")
                f.write(f"  Min:    {df_cong['Total_Slope'].min():8.4f}\n")
                f.write(f"  Median: {df_cong['Total_Slope'].median():8.4f}\n")
                f.write(f"  Max:    {df_cong['Total_Slope'].max():8.4f}\n")

        log(f"[SAVED] {report_path.name}")

        # =====================================================================
        # STEP 6: Validate Random Effects Structure
        # =====================================================================
        log("\n[VALIDATION] Validating random effects structure...")

        validation = validate_dataframe_structure(
            df_random_effects,
            expected_rows=300,
            expected_columns=['UID', 'congruence', 'Total_Intercept', 'Total_Slope']
        )

        if validation['valid']:
            log("[PASS] Random effects structure validated")
        else:
            log(f"[FAIL] Structure validation failed: {validation['message']}")
            raise ValueError(validation['message'])

        # Additional validation: Check each UID appears 3 times
        uid_counts = df_random_effects['UID'].value_counts()
        if not all(uid_counts == 3):
            incorrect_uids = uid_counts[uid_counts != 3]
            raise ValueError(f"Found {len(incorrect_uids)} UIDs that don't appear exactly 3 times")

        log("[PASS] Each UID appears exactly 3 times (once per congruence)")

        # Check for NaN
        if df_random_effects[['Total_Intercept', 'Total_Slope']].isna().any().any():
            raise ValueError("Found NaN values in random effects")

        log("[PASS] No NaN values in random effects")

        # Check all congruence levels present
        if len(df_random_effects['congruence'].unique()) != 3:
            raise ValueError(f"Expected 3 congruence levels, found {len(df_random_effects['congruence'].unique())}")

        log("[PASS] All 3 congruence levels present")

        log("\n[SUCCESS] Step 04 complete - Random effects extracted and validated")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
