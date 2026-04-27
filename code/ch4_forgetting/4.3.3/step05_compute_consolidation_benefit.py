#!/usr/bin/env python3
"""
Step 05: Compute Consolidation Benefit Index
RQ 5.3.3 - Paradigm Consolidation Window

Purpose: Compute consolidation benefit index (Late slope - Early slope)
for each paradigm, rank paradigms by benefit magnitude, and interpret
pattern relative to hypothesis.
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
RQ_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = RQ_DIR.parents[2]

# Setup logging
LOG_FILE = RQ_DIR / "logs" / "step05_compute_consolidation_benefit.log"
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
    """Compute consolidation benefit index per paradigm."""
    logger.info("=" * 60)
    logger.info("Step 05: Compute Consolidation Benefit Index")
    logger.info("=" * 60)

    # Define paths
    slopes_file = RQ_DIR / "data" / "step03_segment_paradigm_slopes.csv"
    output_file = RQ_DIR / "data" / "step05_consolidation_benefit.csv"

    # --- Load slopes ---
    logger.info(f"Loading slopes from: {slopes_file}")
    slopes_df = pd.read_csv(slopes_file)

    # --- Compute consolidation benefit for each paradigm ---
    logger.info("\nComputing consolidation benefit indices...")

    benefit_data = []

    for paradigm in ['IFR', 'ICR', 'IRE']:
        early_slope = slopes_df[
            (slopes_df['Segment'] == 'Early') &
            (slopes_df['paradigm'] == paradigm)
        ]['slope'].values[0]

        late_slope = slopes_df[
            (slopes_df['Segment'] == 'Late') &
            (slopes_df['paradigm'] == paradigm)
        ]['slope'].values[0]

        # Consolidation benefit = Late slope - Early slope
        # Positive benefit means less steep forgetting in Late (memory stabilized)
        # Negative benefit means steeper forgetting in Late (unexpected pattern)
        consolidation_benefit = late_slope - early_slope

        benefit_data.append({
            'paradigm': paradigm,
            'Early_slope': early_slope,
            'Late_slope': late_slope,
            'consolidation_benefit': consolidation_benefit
        })

    benefit_df = pd.DataFrame(benefit_data)

    # --- Rank paradigms by benefit magnitude ---
    # Rank 1 = largest positive benefit (best consolidation)
    # Note: Using absolute value for ranking, but sign matters for interpretation
    benefit_df['rank'] = benefit_df['consolidation_benefit'].rank(ascending=False).astype(int)

    # Sort by rank for display
    benefit_df = benefit_df.sort_values('rank').reset_index(drop=True)

    # --- Compare to hypothesis ---
    # Expected pattern: IFR > ICR > IRE (Free Recall shows greatest consolidation)
    expected_ranking = ['IFR', 'ICR', 'IRE']
    actual_ranking = benefit_df['paradigm'].tolist()

    if actual_ranking == expected_ranking:
        pattern_match = "MATCHES"
        interpretation = "Pattern matches hypothesis: IFR > ICR > IRE"
    else:
        pattern_match = "CONTRADICTS"
        interpretation = f"Pattern contradicts hypothesis: {' > '.join(actual_ranking)} (expected IFR > ICR > IRE)"

    benefit_df['interpretation'] = interpretation

    # --- Display results ---
    logger.info("\n" + "=" * 60)
    logger.info("CONSOLIDATION BENEFIT INDICES")
    logger.info("=" * 60)
    logger.info(f"\n{benefit_df.to_string(index=False)}")

    # --- Detailed interpretation ---
    logger.info("\n" + "=" * 60)
    logger.info("DETAILED INTERPRETATION")
    logger.info("=" * 60)

    logger.info("\nConsolidation benefit = Late slope - Early slope")
    logger.info("Positive value = slower forgetting in Late vs Early (memory stabilized)")
    logger.info("Negative value = faster forgetting in Late vs Early (unexpected)")

    for _, row in benefit_df.iterrows():
        logger.info(f"\n{row['paradigm']} (Rank {row['rank']}):")
        logger.info(f"  Early slope: {row['Early_slope']:.4f} theta/day")
        logger.info(f"  Late slope:  {row['Late_slope']:.4f} theta/day")
        logger.info(f"  Benefit:     {row['consolidation_benefit']:.4f} theta/day")

        if row['consolidation_benefit'] > 0:
            logger.info(f"  → Forgetting slowed by {abs(row['consolidation_benefit']):.4f} theta/day in Late segment")
        else:
            logger.info(f"  → Forgetting accelerated by {abs(row['consolidation_benefit']):.4f} theta/day in Late segment")

    logger.info(f"\nRanking comparison: {actual_ranking}")
    logger.info(f"Expected ranking:   {expected_ranking}")
    logger.info(f"Hypothesis comparison: {pattern_match} expected pattern")

    # --- Validation ---
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION CHECKS")
    logger.info("=" * 60)

    # Check row count
    if len(benefit_df) != 3:
        logger.error(f"CRITICAL: Expected 3 paradigms, got {len(benefit_df)}")
        sys.exit(1)
    logger.info("Consolidation benefit computed for 3 paradigms")

    # Check all paradigms present
    expected_paradigms = {'IFR', 'ICR', 'IRE'}
    actual_paradigms = set(benefit_df['paradigm'])
    if actual_paradigms != expected_paradigms:
        logger.error(f"CRITICAL: Missing paradigms: {expected_paradigms - actual_paradigms}")
        sys.exit(1)

    # Check ranks
    expected_ranks = {1, 2, 3}
    actual_ranks = set(benefit_df['rank'])
    if actual_ranks != expected_ranks:
        logger.error(f"CRITICAL: Ranks not {expected_ranks}, got {actual_ranks}")
        sys.exit(1)

    # Check arithmetic
    for _, row in benefit_df.iterrows():
        expected = row['Late_slope'] - row['Early_slope']
        actual = row['consolidation_benefit']
        if not np.isclose(expected, actual, rtol=1e-6):
            logger.error(f"CRITICAL: Arithmetic error for {row['paradigm']}")
            sys.exit(1)

    # Check for NaN
    if benefit_df['consolidation_benefit'].isna().any():
        logger.error("CRITICAL: NaN in consolidation_benefit")
        sys.exit(1)

    logger.info(f"Ranking: {' > '.join(actual_ranking)}")
    logger.info(f"Hypothesis comparison: {pattern_match} expected pattern (IFR > ICR > IRE)")

    # --- Save output ---
    benefit_df.to_csv(output_file, index=False)
    logger.info(f"\nOutput saved: {output_file}")

    # --- Summary ---
    logger.info("\n" + "=" * 60)
    logger.info("STEP 05 COMPLETE")
    logger.info("=" * 60)

    return benefit_df


if __name__ == "__main__":
    main()
