"""
Step 03: Extract 3-Way Interaction Terms
RQ 5.5.3 - Age Effects on Source-Destination Memory

Purpose: Extract 3-way Age_c x LocationType x Time interaction terms with Bonferroni
         correction (Decision D068 dual p-value reporting). Primary null hypothesis test.

Input:
- data/step02_fixed_effects.csv (all 12 fixed effects)

Output:
- data/step03_interaction_terms.csv (2 rows: 3-way interaction terms with dual p-values)

Log: logs/step03_extract_interactions.log
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / "step03_extract_interactions.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("[START] Step 03: Extract 3-Way Interaction Terms")

    # -------------------------------------------------------------------------
    # 1. Load fixed effects
    # -------------------------------------------------------------------------
    logger.info("[LOAD] Loading fixed effects from Step 02...")

    fixed_effects = pd.read_csv(DATA_DIR / "step02_fixed_effects.csv")
    logger.info(f"[LOADED] {len(fixed_effects)} fixed effects")

    # Display all fixed effects
    logger.info("[INFO] All fixed effects:")
    for _, row in fixed_effects.iterrows():
        logger.info(f"  {row['term']}: coef={row['coef']:.6f}, p={row['p']:.6f}")

    # -------------------------------------------------------------------------
    # 2. Extract 3-way interaction terms
    # -------------------------------------------------------------------------
    logger.info("[EXTRACT] Extracting 3-way interaction terms...")

    # Define interaction terms to extract (Age_c x LocationType x Time)
    # Note: statsmodels uses T.Source because Destination is reference (alphabetical)
    interaction_patterns = [
        'TSVR_hours:Age_c:LocationType',
        'log_TSVR:Age_c:LocationType'
    ]

    interactions = []
    for pattern in interaction_patterns:
        # Find matching term (may have [T.Source] or different order)
        matches = fixed_effects[fixed_effects['term'].str.contains(pattern, regex=False) |
                               fixed_effects['term'].str.contains('Age_c', regex=False) &
                               fixed_effects['term'].str.contains('LocationType', regex=False) &
                               fixed_effects['term'].str.contains(pattern.split(':')[0], regex=False)]

        if len(matches) == 0:
            # Try more flexible matching
            if 'TSVR_hours' in pattern:
                matches = fixed_effects[fixed_effects['term'].str.contains('TSVR_hours') &
                                       fixed_effects['term'].str.contains('Age_c') &
                                       fixed_effects['term'].str.contains('LocationType')]
            else:
                matches = fixed_effects[fixed_effects['term'].str.contains('log_TSVR') &
                                       fixed_effects['term'].str.contains('Age_c') &
                                       fixed_effects['term'].str.contains('LocationType')]

        if len(matches) > 0:
            row = matches.iloc[0]
            interactions.append({
                'term': row['term'],
                'coef': row['coef'],
                'se': row['se'],
                'z': row['z'],
                'p_uncorrected': row['p'],
                'ci_lower': row['ci_lower'],
                'ci_upper': row['ci_upper']
            })
            logger.info(f"[FOUND] {row['term']}: coef={row['coef']:.6f}, p={row['p']:.6f}")
        else:
            logger.info(f"[WARNING] No match found for pattern: {pattern}")

    if len(interactions) != 2:
        raise ValueError(f"Expected 2 interaction terms, found {len(interactions)}")

    # -------------------------------------------------------------------------
    # 3. Apply Bonferroni correction (Decision D068)
    # -------------------------------------------------------------------------
    logger.info("[CORRECT] Applying Bonferroni correction (2 tests, alpha=0.025)...")

    n_tests = 2
    alpha_corrected = 0.05 / n_tests  # 0.025

    for i, interaction in enumerate(interactions):
        # Bonferroni: multiply p-value by number of tests, cap at 1.0
        p_bonf = min(1.0, interaction['p_uncorrected'] * n_tests)
        interaction['p_bonferroni'] = p_bonf
        interaction['significant_at_0025'] = p_bonf < alpha_corrected

        logger.info(f"[BONFERRONI] {interaction['term']}")
        logger.info(f"  p_uncorrected: {interaction['p_uncorrected']:.6f}")
        logger.info(f"  p_bonferroni: {p_bonf:.6f}")
        logger.info(f"  significant at 0.025: {interaction['significant_at_0025']}")

    # -------------------------------------------------------------------------
    # 4. Create output DataFrame
    # -------------------------------------------------------------------------
    interaction_df = pd.DataFrame(interactions)

    # Reorder columns
    interaction_df = interaction_df[['term', 'coef', 'se', 'z', 'p_uncorrected',
                                     'p_bonferroni', 'ci_lower', 'ci_upper',
                                     'significant_at_0025']]

    # -------------------------------------------------------------------------
    # 5. Validation
    # -------------------------------------------------------------------------
    logger.info("[VALIDATION] Running validation checks...")

    all_pass = True

    # Check 1: 2 rows
    if len(interaction_df) == 2:
        logger.info("[PASS] 2 interaction terms present")
    else:
        logger.info(f"[FAIL] Expected 2 terms, found {len(interaction_df)}")
        all_pass = False

    # Check 2: Both p-value columns present (Decision D068)
    if 'p_uncorrected' in interaction_df.columns and 'p_bonferroni' in interaction_df.columns:
        logger.info("[PASS] Dual p-values present (Decision D068)")
    else:
        logger.info("[FAIL] Missing p-value columns")
        all_pass = False

    # Check 3: p_bonferroni >= p_uncorrected
    if all(interaction_df['p_bonferroni'] >= interaction_df['p_uncorrected']):
        logger.info("[PASS] Bonferroni correction applied correctly")
    else:
        logger.info("[FAIL] Bonferroni correction error")
        all_pass = False

    # Check 4: p-values in [0, 1]
    if all(0 <= p <= 1 for p in interaction_df['p_uncorrected']) and \
       all(0 <= p <= 1 for p in interaction_df['p_bonferroni']):
        logger.info("[PASS] All p-values in [0, 1]")
    else:
        logger.info("[FAIL] Invalid p-values")
        all_pass = False

    if not all_pass:
        raise ValueError("Validation failed - see above for details")

    # -------------------------------------------------------------------------
    # 6. Save output
    # -------------------------------------------------------------------------
    logger.info("[SAVE] Saving interaction terms...")

    output_path = DATA_DIR / "step03_interaction_terms.csv"
    interaction_df.to_csv(output_path, index=False)
    logger.info(f"[SAVED] {output_path.name} ({len(interaction_df)} rows)")

    # -------------------------------------------------------------------------
    # 7. Summary
    # -------------------------------------------------------------------------
    logger.info("[SUMMARY] Primary Hypothesis Test Results:")
    logger.info("  Null Hypothesis: Age does NOT moderate source-destination forgetting")
    logger.info("  3-way interactions (Age_c x LocationType x Time):")

    for _, row in interaction_df.iterrows():
        sig_marker = "*" if row['significant_at_0025'] else ""
        logger.info(f"    {row['term']}: beta={row['coef']:.4f}, z={row['z']:.2f}, "
                   f"p={row['p_bonferroni']:.4f}{sig_marker}")

    any_significant = interaction_df['significant_at_0025'].any()
    if any_significant:
        logger.info("[RESULT] PRIMARY HYPOTHESIS REJECTED - Age moderates source-destination effect")
    else:
        logger.info("[RESULT] PRIMARY HYPOTHESIS SUPPORTED (NULL) - Age does NOT moderate effect")

    logger.info("[SUCCESS] Step 03 complete - Interaction terms extracted")


if __name__ == "__main__":
    main()
