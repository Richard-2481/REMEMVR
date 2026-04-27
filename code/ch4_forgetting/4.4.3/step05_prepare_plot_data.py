#!/usr/bin/env python3
"""
===============================================================================
RQ 5.4.3 - Step 05: Prepare Age Effects Plot Data by Tertiles
===============================================================================

PURPOSE:
    Create age tertiles (Young/Middle/Older) and compute marginal means for
    age effects visualization. Generates 36 rows: 3 tertiles x 3 congruence x 4 tests.

INPUTS:
    - data/step01_lmm_input.csv (1200 rows, long format)

OUTPUTS:
    - data/step05_age_effects_plot_data.csv (36 rows for visualization)

VALIDATION CRITERIA:
    - Exactly 36 rows (3 age tertiles x 3 congruence x 4 tests)
    - All factorial combinations present
    - age_tertile contains only {Young, Middle, Older}
    - congruence contains only {Common, Congruent, Incongruent}
    - test contains only {T1, T2, T3, T4}
    - CI_upper > CI_lower for all rows

===============================================================================
"""

import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = PROJECT_ROOT / "results" / "ch5" / "5.4.3"
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
LOG_FILE = LOG_DIR / "step05_prepare_plot_data.log"

# Create directories
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def log(self, message: str):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()

    def close(self):
        self.log_file.close()

logger = Logger(LOG_FILE)
log = logger.log

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================
def main():
    log("[START] Step 05: Prepare Age Effects Plot Data by Tertiles")
    log("")

    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    log("[STEP 1] Load Data")
    log("-" * 70)

    lmm_input = pd.read_csv(DATA_DIR / "step01_lmm_input.csv", encoding='utf-8')
    log(f"[LOADED] LMM input: {len(lmm_input)} rows")
    log(f"[INFO] Columns: {list(lmm_input.columns)}")
    log("")

    # -------------------------------------------------------------------------
    # STEP 2: Create Age Tertiles
    # -------------------------------------------------------------------------
    log("[STEP 2] Create Age Tertiles")
    log("-" * 70)

    # Get unique participants with their ages
    unique_participants = lmm_input.groupby('UID')['Age'].first().reset_index()
    log(f"[INFO] Unique participants: {len(unique_participants)}")

    # Compute tertile cutpoints
    tertile_33 = unique_participants['Age'].quantile(0.333)
    tertile_67 = unique_participants['Age'].quantile(0.667)

    log(f"[INFO] Tertile cutpoints:")
    log(f"  33rd percentile (Young/Middle boundary): {tertile_33:.1f} years")
    log(f"  67th percentile (Middle/Older boundary): {tertile_67:.1f} years")
    log("")

    # Assign tertiles
    def assign_tertile(age):
        if age <= tertile_33:
            return 'Young'
        elif age <= tertile_67:
            return 'Middle'
        else:
            return 'Older'

    unique_participants['age_tertile'] = unique_participants['Age'].apply(assign_tertile)

    # Count per tertile
    tertile_counts = unique_participants['age_tertile'].value_counts()
    log("[INFO] Participants per tertile:")
    for tertile in ['Young', 'Middle', 'Older']:
        count = tertile_counts.get(tertile, 0)
        age_range = unique_participants[unique_participants['age_tertile'] == tertile]['Age']
        log(f"  {tertile}: N={count} (Age range: {age_range.min():.0f}-{age_range.max():.0f})")
    log("")

    # Merge tertile assignment back to main data
    lmm_input = lmm_input.merge(
        unique_participants[['UID', 'age_tertile']],
        on='UID',
        how='left'
    )
    log(f"[MERGED] Age tertiles assigned to all {len(lmm_input)} observations")
    log("")

    # -------------------------------------------------------------------------
    # STEP 3: Compute Marginal Means
    # -------------------------------------------------------------------------
    log("[STEP 3] Compute Marginal Means")
    log("-" * 70)

    # Convert test to string for consistent labeling
    lmm_input['test_str'] = 'T' + lmm_input['test'].astype(str)

    # Group by age_tertile x congruence x test
    plot_data = []

    for age_tertile in ['Young', 'Middle', 'Older']:
        for congruence in ['Common', 'Congruent', 'Incongruent']:
            for test in ['T1', 'T2', 'T3', 'T4']:
                # Filter data for this cell
                mask = (
                    (lmm_input['age_tertile'] == age_tertile) &
                    (lmm_input['congruence'] == congruence) &
                    (lmm_input['test_str'] == test)
                )
                cell_data = lmm_input[mask]

                if len(cell_data) == 0:
                    log(f"[WARNING] No data for {age_tertile} x {congruence} x {test}")
                    continue

                # Compute statistics
                mean_theta = cell_data['theta'].mean()
                se_theta = cell_data['theta'].std() / np.sqrt(len(cell_data))
                mean_tsvr = cell_data['TSVR_hours'].mean()
                n = len(cell_data)

                # 95% CI
                t_crit = stats.t.ppf(0.975, df=n-1) if n > 1 else 1.96
                ci_lower = mean_theta - t_crit * se_theta
                ci_upper = mean_theta + t_crit * se_theta

                plot_data.append({
                    'age_tertile': age_tertile,
                    'congruence': congruence,
                    'test': test,
                    'TSVR_hours': mean_tsvr,
                    'mean_theta_observed': mean_theta,
                    'mean_theta_predicted': mean_theta,  # Using observed as predicted (no separate model prediction)
                    'CI_lower': ci_lower,
                    'CI_upper': ci_upper,
                    'N': n
                })

    plot_df = pd.DataFrame(plot_data)
    log(f"[COMPUTED] Marginal means: {len(plot_df)} cells")
    log(f"[INFO] Expected: 36 cells (3 tertiles x 3 congruence x 4 tests)")
    log("")

    # -------------------------------------------------------------------------
    # STEP 4: Validate Plot Data
    # -------------------------------------------------------------------------
    log("[STEP 4] Validate Plot Data")
    log("-" * 70)

    # Check row count
    if len(plot_df) != 36:
        log(f"[FAIL] Expected 36 rows, found {len(plot_df)}")
        return False
    log("[PASS] Row count: 36")

    # Check age_tertile levels
    tertile_levels = set(plot_df['age_tertile'].unique())
    expected_tertiles = {'Young', 'Middle', 'Older'}
    if tertile_levels != expected_tertiles:
        log(f"[FAIL] Unexpected age_tertile levels: {tertile_levels}")
        return False
    log(f"[PASS] age_tertile levels: {sorted(tertile_levels)}")

    # Check congruence levels
    congruence_levels = set(plot_df['congruence'].unique())
    expected_congruence = {'Common', 'Congruent', 'Incongruent'}
    if congruence_levels != expected_congruence:
        log(f"[FAIL] Unexpected congruence levels: {congruence_levels}")
        return False
    log(f"[PASS] congruence levels: {sorted(congruence_levels)}")

    # Check test levels
    test_levels = set(plot_df['test'].unique())
    expected_tests = {'T1', 'T2', 'T3', 'T4'}
    if test_levels != expected_tests:
        log(f"[FAIL] Unexpected test levels: {test_levels}")
        return False
    log(f"[PASS] test levels: {sorted(test_levels)}")

    # Check CI validity
    if not all(plot_df['CI_upper'] > plot_df['CI_lower']):
        log("[FAIL] CI_upper not greater than CI_lower for all rows")
        return False
    log("[PASS] CI_upper > CI_lower for all rows")

    # Check for NaN
    if plot_df.isna().any().any():
        nan_cols = plot_df.columns[plot_df.isna().any()].tolist()
        log(f"[FAIL] NaN values found in columns: {nan_cols}")
        return False
    log("[PASS] No NaN values")
    log("")

    # -------------------------------------------------------------------------
    # STEP 5: Report Summary Statistics
    # -------------------------------------------------------------------------
    log("[STEP 5] Report Summary Statistics")
    log("-" * 70)

    # Mean theta by tertile
    log("[INFO] Mean Theta by Age Tertile (collapsed across congruence and test):")
    for tertile in ['Young', 'Middle', 'Older']:
        subset = plot_df[plot_df['age_tertile'] == tertile]
        mean_theta = subset['mean_theta_observed'].mean()
        log(f"  {tertile}: {mean_theta:.4f}")
    log("")

    # Mean theta by congruence
    log("[INFO] Mean Theta by Congruence (collapsed across tertile and test):")
    for cong in ['Common', 'Congruent', 'Incongruent']:
        subset = plot_df[plot_df['congruence'] == cong]
        mean_theta = subset['mean_theta_observed'].mean()
        log(f"  {cong}: {mean_theta:.4f}")
    log("")

    # Time variable range
    log("[INFO] TSVR_hours range by test:")
    for test in ['T1', 'T2', 'T3', 'T4']:
        subset = plot_df[plot_df['test'] == test]
        mean_tsvr = subset['TSVR_hours'].mean()
        log(f"  {test}: {mean_tsvr:.2f} hours")
    log("")

    # -------------------------------------------------------------------------
    # STEP 6: Save Output
    # -------------------------------------------------------------------------
    log("[STEP 6] Save Output")
    log("-" * 70)

    output_path = DATA_DIR / "step05_age_effects_plot_data.csv"
    plot_df.to_csv(output_path, index=False, encoding='utf-8')
    log(f"[SAVED] {output_path}")
    log(f"  {len(plot_df)} rows, {len(plot_df.columns)} columns")
    log("")

    log("[SUCCESS] Step 05 complete - Plot data prepared")

    return True

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    try:
        success = main()
        logger.close()
        sys.exit(0 if success else 1)
    except Exception as e:
        log(f"[ERROR] Unexpected error: {e}")
        log(traceback.format_exc())
        logger.close()
        sys.exit(1)
