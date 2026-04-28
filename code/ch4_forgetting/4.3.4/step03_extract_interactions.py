#!/usr/bin/env python3
"""Extract 3-Way Interaction Terms: Extract the 4 three-way interaction terms (Age_c x paradigm x Time) from the fitted"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_extract_interactions.log"

# Logging Function

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Extract 3-Way Interaction Terms")
        # Load Fixed Effects from Step 2
        log("Loading fixed effects from Step 2...")
        fe_path = RQ_DIR / "data" / "step02_fixed_effects.csv"

        if not fe_path.exists():
            raise FileNotFoundError(f"Fixed effects file not found: {fe_path}")

        df_fe = pd.read_csv(fe_path, encoding='utf-8')
        log(f"{len(df_fe)} fixed effects from {fe_path}")

        # Log all terms
        log("All fixed effect terms:")
        for term in df_fe['term'].values:
            log(f"  {term}")
        # Filter for 3-Way Interaction Terms
        log("Identifying 3-way interaction terms...")

        # The 3-way interactions contain ALL three:
        # - Age_c
        # - Time (TSVR_hours or log_TSVR)
        # - paradigm
        # And at least 2 colons (A:B:C format)

        def is_3way_interaction(term):
            has_age = 'Age_c' in term
            has_time = 'TSVR_hours' in term or 'log_TSVR' in term
            has_paradigm = 'paradigm' in term
            n_colons = term.count(':')
            return has_age and has_time and has_paradigm and n_colons >= 2

        interaction_mask = df_fe['term'].apply(is_3way_interaction)
        interaction_terms = df_fe[interaction_mask].copy()

        log(f"Found {len(interaction_terms)} three-way interaction terms")

        # Log found terms
        for term in interaction_terms['term'].values:
            log(f"  Found: {term}")

        # Verify we found 4 terms
        if len(interaction_terms) != 4:
            log(f"Expected 4 three-way interaction terms, found {len(interaction_terms)}")
        # Rename Columns and Apply Bonferroni Correction
        log("Applying Bonferroni correction for 2 time transformations...")

        # Rename p_value to p_uncorrected for clarity
        interaction_terms = interaction_terms.rename(columns={'p_value': 'p_uncorrected'})

        # Bonferroni correction for 2 time transformations:
        # - Family-wise alpha = 0.025
        # - Correcting for 2 tests: TSVR_hours terms and log_TSVR terms
        # - p_bonferroni = min(p_uncorrected × 2, 1.0)

        bonferroni_multiplier = 2  # 2 time transformations
        bonferroni_alpha = 0.025  # Family-wise alpha from concept.md

        interaction_terms['p_bonferroni'] = interaction_terms['p_uncorrected'].apply(
            lambda p: min(p * bonferroni_multiplier, 1.0)
        )
        interaction_terms['significant_bonferroni'] = interaction_terms['p_bonferroni'] < bonferroni_alpha

        log(f"Bonferroni multiplier: {bonferroni_multiplier}")
        log(f"Family-wise alpha: {bonferroni_alpha}")

        # Log results
        log("3-Way Interaction Terms:")
        for _, row in interaction_terms.iterrows():
            sig_marker = " *" if row['significant_bonferroni'] else ""
            log(f"  {row['term']}")
            log(f"    coef={row['coefficient']:.6f}, SE={row['SE']:.6f}, z={row['z']:.3f}")
            log(f"    p_uncorrected={row['p_uncorrected']:.4f}, p_bonferroni={row['p_bonferroni']:.4f}{sig_marker}")
        # Save Interaction Terms
        log("Saving interaction terms...")

        output_path = RQ_DIR / "data" / "step03_interaction_terms.csv"

        # Select and order columns per 4_analysis.yaml specification
        output_cols = ['term', 'coefficient', 'SE', 'z', 'p_uncorrected', 'p_bonferroni', 'significant_bonferroni']
        interaction_terms = interaction_terms[output_cols]

        interaction_terms.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")
        log(f"Saved {len(interaction_terms)} interaction terms with {len(output_cols)} columns")
        # Validation
        log("Checking output format...")

        # Check row count
        if len(interaction_terms) == 4:
            log("Row count: 4 three-way interaction terms")
        else:
            log(f"Row count: {len(interaction_terms)} (expected 4)")

        # Check column count
        if len(interaction_terms.columns) == 7:
            log("Column count: 7 columns")
        else:
            log(f"Column count: {len(interaction_terms.columns)} (expected 7)")

        # Check dual p-values present (Decision D068)
        if 'p_uncorrected' in interaction_terms.columns and 'p_bonferroni' in interaction_terms.columns:
            log("Dual p-values present (Decision D068 compliance)")
        else:
            log("Missing p-value columns")

        # Check for NaN values
        nan_count = interaction_terms.isna().sum().sum()
        if nan_count == 0:
            log("No NaN values")
        else:
            log(f"{nan_count} NaN values detected")

        # Check significance
        n_sig = interaction_terms['significant_bonferroni'].sum()
        log(f"Significant at Bonferroni alpha=0.025: {n_sig} of {len(interaction_terms)} terms")
        # Final Summary
        log("Step 03 complete")
        log("")
        log(f"  Extracted: {len(interaction_terms)} three-way interaction terms")
        log(f"  Bonferroni correction: × {bonferroni_multiplier}")
        log(f"  Significant (p_bonferroni < 0.025): {n_sig}")
        log(f"  Output: data/step03_interaction_terms.csv")
        log(f"  Ready for Step 4 (compute age effects and post-hoc contrasts)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        traceback.print_exc()
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        sys.exit(1)
