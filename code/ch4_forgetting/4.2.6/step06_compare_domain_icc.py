#!/usr/bin/env python3
"""
RQ 5.2.6 Step 06: Compare ICC Across Domains

Purpose:
    Rank domains by ICC_slope_conditional magnitude to characterize domain-specific
    variance patterns. Compare to theoretical prediction.

Input:
    - data/step03_icc_estimates.csv (6 rows: 3 ICC types x 2 domains)

Output:
    - data/step06_domain_icc_comparison.csv (2 rows: one per domain)

Theoretical Prediction (from 1_concept.md):
    Original: ICC_When >= ICC_Where > ICC_What (hippocampal aging effects)
    Updated (When excluded): ICC_Where > ICC_What

Note: With When domain excluded due to floor effect, comparison is limited to
What vs Where domains only.

Author: Claude (g_code agent)
Date: 2025-12-03
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Set up logging
log_path = Path("results/ch5/5.2.6/logs/step06_compare_domain_icc.log")
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Compare ICC estimates across domains."""

    logger.info("=" * 60)
    logger.info("RQ 5.2.6 Step 06: Compare ICC Across Domains")
    logger.info("=" * 60)

    # ---------------------------------------------------------------------
    # 1. Load ICC estimates from Step 03
    # ---------------------------------------------------------------------
    input_path = Path("results/ch5/5.2.6/data/step03_icc_estimates.csv")

    if not input_path.exists():
        logger.error(f"EXPECTATIONS ERROR: ICC estimates file not found: {input_path}")
        logger.error("Step 03 must complete before Step 06")
        sys.exit(1)

    icc_df = pd.read_csv(input_path)
    logger.info(f"Loaded ICC estimates: {len(icc_df)} rows from {input_path}")

    # ---------------------------------------------------------------------
    # 2. Filter to slope_conditional ICC (primary comparison)
    # ---------------------------------------------------------------------
    slope_cond = icc_df[icc_df['icc_type'] == 'slope_conditional'].copy()
    logger.info(f"\nFiltered to slope_conditional: {len(slope_cond)} rows")

    if len(slope_cond) == 0:
        # Fallback to slope_simple if conditional not available
        slope_cond = icc_df[icc_df['icc_type'] == 'slope_simple'].copy()
        icc_type_used = 'slope_simple'
        logger.warning("No slope_conditional found, using slope_simple")
    else:
        icc_type_used = 'slope_conditional'

    logger.info(f"Using ICC type: {icc_type_used}")

    # ---------------------------------------------------------------------
    # 3. Rank domains by ICC value
    # ---------------------------------------------------------------------
    slope_cond = slope_cond.sort_values('icc_value', ascending=False).reset_index(drop=True)
    slope_cond['rank'] = slope_cond.index + 1

    logger.info("\nDomain ICC ranking (descending):")
    for _, row in slope_cond.iterrows():
        logger.info(f"  Rank {int(row['rank'])}: {row['domain']} (ICC={row['icc_value']:.4f})")

    # ---------------------------------------------------------------------
    # 4. Check threshold (ICC >= 0.40 = Substantial)
    # ---------------------------------------------------------------------
    slope_cond['meets_threshold'] = slope_cond['icc_value'] >= 0.40

    n_substantial = slope_cond['meets_threshold'].sum()
    logger.info(f"\nDomains meeting threshold (ICC >= 0.40): {n_substantial} domains")

    # ---------------------------------------------------------------------
    # 5. Compare to theoretical prediction
    # ---------------------------------------------------------------------
    # Original prediction (with When): ICC_When >= ICC_Where > ICC_What
    # Updated prediction (When excluded): ICC_Where > ICC_What
    # This is based on hippocampal-dependent Where memory showing more individual differences

    logger.info("\n" + "=" * 60)
    logger.info("THEORETICAL PREDICTION CHECK")
    logger.info("=" * 60)
    logger.info("Prediction: ICC_Where > ICC_What")
    logger.info("Rationale: Where memory (hippocampal-dependent) may show greater")
    logger.info("           individual differences than What memory (perirhinal-dependent)")

    what_icc = slope_cond[slope_cond['domain'] == 'What']['icc_value'].values[0]
    where_icc = slope_cond[slope_cond['domain'] == 'Where']['icc_value'].values[0]

    if where_icc > what_icc:
        prediction_match = "Matches"
        logger.info(f"\nResult: MATCHES prediction")
        logger.info(f"  Where ICC ({where_icc:.4f}) > What ICC ({what_icc:.4f})")
    else:
        prediction_match = "Deviates"
        logger.info(f"\nResult: DEVIATES from prediction")
        logger.info(f"  Where ICC ({where_icc:.4f}) <= What ICC ({what_icc:.4f})")

    # Set theoretical prediction for all rows
    # For What, check if it's lower than Where
    # For Where, check if it's higher than What
    def get_prediction(row):
        if row['domain'] == 'Where':
            return "Matches" if where_icc > what_icc else "Deviates"
        else:  # What
            return "Matches" if where_icc > what_icc else "Deviates"

    slope_cond['theoretical_prediction'] = slope_cond.apply(get_prediction, axis=1)

    # ---------------------------------------------------------------------
    # 6. Prepare output DataFrame
    # ---------------------------------------------------------------------
    # Rename columns to match expected output format
    comparison_df = slope_cond[['domain', 'icc_value', 'interpretation', 'rank', 'meets_threshold', 'theoretical_prediction']].copy()
    comparison_df = comparison_df.rename(columns={'icc_value': 'icc_slope_conditional'})

    # Sort by rank for consistent output
    comparison_df = comparison_df.sort_values('rank').reset_index(drop=True)

    # ---------------------------------------------------------------------
    # 7. Validation
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Checking comparison results")
    logger.info("=" * 60)

    # Check row count
    if len(comparison_df) != 2:  # 2 domains (What, Where)
        logger.error(f"Expected 2 rows (2 domains), got {len(comparison_df)}")
        sys.exit(1)
    logger.info(f"  Row count: {len(comparison_df)}: PASS")

    # Check ranks unique
    ranks = comparison_df['rank'].unique()
    if len(ranks) != len(comparison_df):
        logger.warning(f"Ranks not unique: {list(comparison_df['rank'])}")
    else:
        logger.info(f"  Ranks unique: PASS")

    # Check ICC bounds
    for _, row in comparison_df.iterrows():
        if row['icc_slope_conditional'] < 0 or row['icc_slope_conditional'] > 1:
            logger.error(f"ICC out of bounds: {row['domain']} = {row['icc_slope_conditional']}")
            sys.exit(1)
    logger.info(f"  All ICC in [0, 1]: PASS")

    # Check valid interpretation values
    valid_interpretations = {'Low', 'Moderate', 'Substantial'}
    for _, row in comparison_df.iterrows():
        if row['interpretation'] not in valid_interpretations:
            logger.error(f"Invalid interpretation: {row['interpretation']}")
            sys.exit(1)
    logger.info(f"  All interpretations valid: PASS")

    # ---------------------------------------------------------------------
    # 8. Save comparison results
    # ---------------------------------------------------------------------
    output_path = Path("results/ch5/5.2.6/data/step06_domain_icc_comparison.csv")
    comparison_df.to_csv(output_path, index=False)
    logger.info(f"\nSaved comparison results to: {output_path}")

    # ---------------------------------------------------------------------
    # 9. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("DOMAIN ICC COMPARISON SUMMARY")
    logger.info("=" * 60)

    logger.info(f"\n{'Domain':<10} {'ICC_slope_cond':<15} {'Interpretation':<15} {'Rank':<6} {'Threshold':<10}")
    logger.info("-" * 60)
    for _, row in comparison_df.iterrows():
        logger.info(f"{row['domain']:<10} {row['icc_slope_conditional']:<15.4f} {row['interpretation']:<15} {int(row['rank']):<6} {str(row['meets_threshold']):<10}")

    logger.info(f"\nTheoretical prediction (ICC_Where > ICC_What): {prediction_match}")
    logger.info(f"ICC difference (Where - What): {where_icc - what_icc:.4f}")

    # Interpret the comparison
    logger.info("\n" + "-" * 60)
    logger.info("INTERPRETATION:")
    logger.info("-" * 60)
    if where_icc > what_icc:
        logger.info("Where domain shows HIGHER between-person variance than What domain.")
        logger.info("This suggests Where memory (spatial) has more stable individual differences")
        logger.info("than What memory (object identity), consistent with hippocampal aging effects.")
    else:
        logger.info("What domain shows HIGHER between-person variance than Where domain.")
        logger.info("This is unexpected based on dual-process theory predictions.")
        logger.info("May suggest familiarity-based What memory has unexpectedly high trait stability.")

    # ---------------------------------------------------------------------
    # 10. Summary
    # ---------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("STEP 06 COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Compared ICC estimates across {len(comparison_df)} domains")
    logger.info(f"Theoretical prediction: {prediction_match}")
    logger.info("Ready for Step 07: Prepare Domain ICC Barplot Data")


if __name__ == "__main__":
    main()
