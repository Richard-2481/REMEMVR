#!/usr/bin/env python3
"""
===============================================================================
RQ 5.4.3 - Step 03: Extract 3-Way Interaction Terms with Dual P-Values
===============================================================================

PURPOSE:
    Extract the 4 three-way interaction terms (Age_c x Congruence x Time) from
    fitted model and apply Bonferroni correction for multiple comparisons.
    Report dual p-values per Decision D068.

INPUTS:
    - data/step02_fixed_effects.csv (18 fixed effects terms)

OUTPUTS:
    - data/step03_interaction_terms.csv (4 rows with dual p-values)

VALIDATION CRITERIA:
    - Exactly 4 interaction terms extracted
    - BOTH p_uncorrected AND p_bonferroni columns present (Decision D068)
    - Bonferroni correction applied correctly (p_bonferroni = min(p * 2, 1.0))
    - significant_bonferroni threshold correct (p_bonferroni < 0.025)

===============================================================================
"""

import sys
import traceback
from pathlib import Path
import numpy as np
import pandas as pd

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = PROJECT_ROOT / "results" / "ch5" / "5.4.3"
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
LOG_FILE = LOG_DIR / "step03_extract_interactions.log"

# Create directories
LOG_DIR.mkdir(parents=True, exist_ok=True)

# LOGGING SETUP
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

# MAIN PROCESSING
def main():
    log("Step 03: Extract 3-Way Interaction Terms with Dual P-Values")
    log("")

    log("[STEP 1] Load Fixed Effects")
    log("-" * 70)

    fixed_effects = pd.read_csv(DATA_DIR / "step02_fixed_effects.csv", encoding='utf-8')
    log(f"Fixed effects: {len(fixed_effects)} rows")
    log(f"Columns: {list(fixed_effects.columns)}")
    log(f"Terms: {list(fixed_effects['term'])}")
    log("")

    log("[STEP 2] Identify 3-Way Interaction Terms")
    log("-" * 70)

    # The 4 three-way interaction terms we need (UPDATED for Recip+Log):
    required_terms = [
        'Age_c:Congruent:recip_TSVR',
        'Age_c:Congruent:log_TSVR',
        'Age_c:Incongruent:recip_TSVR',
        'Age_c:Incongruent:log_TSVR'
    ]

    log(f"Required 3-way interaction terms (Recip+Log two-process):")
    for term in required_terms:
        log(f"  - {term}")
    log("")

    # Filter for 3-way interaction terms
    # These terms contain Age_c, a congruence level, and a time variable
    interaction_terms = fixed_effects[fixed_effects['term'].isin(required_terms)].copy()

    log(f"Found {len(interaction_terms)} matching terms")

    if len(interaction_terms) != 4:
        log(f"Expected 4 terms, found {len(interaction_terms)}")
        missing = set(required_terms) - set(interaction_terms['term'])
        if missing:
            log(f"Missing terms: {missing}")
        return False

    log(f"All 4 three-way interaction terms found")
    log("")

    log("[STEP 3] Apply Bonferroni Correction")
    log("-" * 70)

    # Bonferroni correction for 2 time terms (TSVR_hours and log_TSVR)
    # alpha = 0.05 / 2 = 0.025
    bonferroni_factor = 2
    alpha_bonferroni = 0.025

    log(f"Bonferroni correction for {bonferroni_factor} time terms")
    log(f"Corrected alpha = 0.05 / {bonferroni_factor} = {alpha_bonferroni}")
    log("")

    # Rename p column to p_uncorrected and apply correction
    interaction_terms = interaction_terms.rename(columns={'p': 'p_uncorrected'})
    interaction_terms['p_bonferroni'] = interaction_terms['p_uncorrected'].apply(
        lambda p: min(p * bonferroni_factor, 1.0)
    )
    interaction_terms['significant_bonferroni'] = interaction_terms['p_bonferroni'] < alpha_bonferroni

    log(f"Bonferroni correction: p_bonferroni = min(p_uncorrected * {bonferroni_factor}, 1.0)")
    log(f"Significance threshold: p_bonferroni < {alpha_bonferroni}")
    log("")

    log("[STEP 4] Validate Dual P-Values")
    log("-" * 70)

    # Check both columns present
    if 'p_uncorrected' not in interaction_terms.columns:
        log("p_uncorrected column missing (Decision D068 violation)")
        return False
    if 'p_bonferroni' not in interaction_terms.columns:
        log("p_bonferroni column missing (Decision D068 violation)")
        return False
    log("Dual p-values present (p_uncorrected + p_bonferroni)")

    # Verify correction applied correctly
    for idx, row in interaction_terms.iterrows():
        expected_bonf = min(row['p_uncorrected'] * bonferroni_factor, 1.0)
        if abs(row['p_bonferroni'] - expected_bonf) > 1e-10:
            log(f"Bonferroni correction incorrect for {row['term']}")
            return False
    log("Bonferroni correction verified correct")

    # Check no NaN values
    if interaction_terms.isna().any().any():
        nan_cols = interaction_terms.columns[interaction_terms.isna().any()].tolist()
        log(f"NaN values found in columns: {nan_cols}")
        return False
    log("No NaN values in any column")
    log("")

    log("[STEP 5] Report Results")
    log("-" * 70)

    log("3-Way Interaction Terms (Decision D068 Dual P-Values):")
    log("")
    log(f"{'Term':<40} {'Coef':>10} {'SE':>8} {'z':>8} {'p_uncor':>10} {'p_bonf':>10} {'Sig':>5}")
    log("-" * 95)

    for _, row in interaction_terms.iterrows():
        sig_marker = "*" if row['significant_bonferroni'] else ""
        log(f"{row['term']:<40} {row['coef']:>10.4f} {row['se']:>8.4f} {row['z']:>8.2f} {row['p_uncorrected']:>10.4f} {row['p_bonferroni']:>10.4f} {sig_marker:>5}")

    log("")

    # Summary of significance
    n_significant = interaction_terms['significant_bonferroni'].sum()
    log(f"Significant 3-way interactions: {n_significant} / 4")

    if n_significant == 0:
        log("NULL RESULT: No significant Age x Congruence x Time interactions")
        log("Age effects on forgetting rate do NOT differ by schema congruence level")
    else:
        log(f"{n_significant} significant Age x Congruence x Time interaction(s)")
        sig_terms = interaction_terms[interaction_terms['significant_bonferroni']]['term'].tolist()
        log(f"Age effects differ for: {sig_terms}")
    log("")

    log("[STEP 6] Save Output")
    log("-" * 70)

    # Reorder columns for output
    output_columns = ['term', 'coef', 'se', 'z', 'p_uncorrected', 'p_bonferroni', 'significant_bonferroni']
    output = interaction_terms[output_columns]

    output_path = DATA_DIR / "step03_interaction_terms.csv"
    output.to_csv(output_path, index=False, encoding='utf-8')
    log(f"{output_path}")
    log(f"  {len(output)} rows, {len(output.columns)} columns")
    log("")

    log("Step 03 complete - 3-way interaction terms extracted with dual p-values")

    return True

# ENTRY POINT
if __name__ == "__main__":
    try:
        success = main()
        logger.close()
        sys.exit(0 if success else 1)
    except Exception as e:
        log(f"Unexpected error: {e}")
        log(traceback.format_exc())
        logger.close()
        sys.exit(1)
