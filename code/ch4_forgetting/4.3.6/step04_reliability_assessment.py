#!/usr/bin/env python3
"""
Step 04: Reliability Assessment with Cronbach's Alpha

RQ 5.3.6 - Purified CTT Effects (Paradigms)

PURPOSE:
Compute Cronbach's alpha for Full CTT (all 24 items per paradigm) and
Purified CTT (retained items only) with bootstrap 95% confidence intervals.

EXPECTED INPUTS:
- data/cache/dfData.csv: Raw item-level responses (UID, test, TQ_* columns)
- data/step01_item_mapping.csv: Item retention status per paradigm
- results/ch5/5.3.1/data/step00_irt_input.csv: Full item list (72 VR items)

EXPECTED OUTPUTS:
- data/step04_reliability_assessment.csv:
  Columns: paradigm, n_items_full, n_items_purified, alpha_full,
           alpha_full_CI_lower, alpha_full_CI_upper, alpha_purified,
           alpha_purified_CI_lower, alpha_purified_CI_upper,
           alpha_purified_SB_adjusted, delta_alpha
  Expected rows: 3 (IFR, ICR, IRE)

VALIDATION CRITERIA:
- alpha values in [0, 1]
- CI_lower <= alpha <= CI_upper
- n_items_purified <= n_items_full
- All 3 paradigms present

METHODOLOGY:
- Cronbach's alpha: (K/(K-1)) * (1 - sum(item_variances) / total_variance)
- Bootstrap: 1000 iterations, sample participants with replacement
- 95% CI: 2.5th and 97.5th percentiles of bootstrap distribution
- Spearman-Brown prophecy: alpha_SB = (n*alpha) / (1 + (n-1)*alpha)
  where n = n_items_full / n_items_purified

IMPLEMENTATION NOTES:
- Uses tools.analysis_ctt.compute_cronbachs_alpha for bootstrap CIs
- Full item list extracted from RQ 5.3.1 IRT input file
- Reduced bootstrap to 1000 iterations for speed (vs 10000 default)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compute_cronbachs_alpha

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_reliability_assessment.log"

# Bootstrap iterations (reduced from 10000 for speed)
N_BOOTSTRAP = 1000

# Logging

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)


# Helper Functions

def compute_spearman_brown_adjustment(alpha: float, n_items_full: int, n_items_purified: int) -> float:
    """
    Compute Spearman-Brown prophecy formula to adjust purified alpha.

    Extrapolates purified alpha (computed on fewer items) to estimate
    reliability if test had full item count.

    Formula: alpha_SB = (n * alpha) / (1 + (n - 1) * alpha)
    where n = n_items_full / n_items_purified

    Args:
        alpha: Cronbach's alpha for purified item set
        n_items_full: Number of items in full set (e.g., 24)
        n_items_purified: Number of items in purified set (e.g., 12)

    Returns:
        Spearman-Brown adjusted alpha
    """
    n = n_items_full / n_items_purified
    alpha_sb = (n * alpha) / (1 + (n - 1) * alpha)
    return alpha_sb


def extract_paradigm_items_from_data(
    df_data: pd.DataFrame,
    item_list: List[str]
) -> pd.DataFrame:
    """
    Extract item-level responses for specified items.

    Args:
        df_data: Raw data with columns [UID, test, TQ_*]
        item_list: List of item column names to extract

    Returns:
        DataFrame with only item columns (participants × items)
    """
    # Filter to only the requested item columns that exist in data
    available_items = [item for item in item_list if item in df_data.columns]

    if len(available_items) != len(item_list):
        missing = set(item_list) - set(available_items)
        log(f"{len(missing)} items not found in data: {missing}")

    return df_data[available_items].copy()


# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Reliability Assessment")
        # Load Input Data

        log("Loading raw item-level data...")
        raw_data_path = RQ_DIR / "data" / "cache" / "dfData.csv"
        df_raw = pd.read_csv(raw_data_path)
        log(f"Raw data: {len(df_raw)} rows, {len(df_raw.columns)} columns")

        log("Loading item mapping...")
        item_mapping_path = RQ_DIR / "data" / "step01_item_mapping.csv"
        df_item_mapping = pd.read_csv(item_mapping_path)
        log(f"Item mapping: {len(df_item_mapping)} items")

        log("Loading full item list from RQ 5.3.1...")
        full_item_list_path = Path(PROJECT_ROOT) / "results" / "ch5" / "5.3.1" / "data" / "step00_irt_input.csv"
        df_full_items = pd.read_csv(full_item_list_path, nrows=0)  # Read only header
        full_item_columns = [col for col in df_full_items.columns if col.startswith('TQ_')]
        log(f"Full item list: {len(full_item_columns)} items")
        # Organize Items by Paradigm (Full vs Purified)

        log("Organizing items by paradigm...")

        paradigms = ['IFR', 'ICR', 'IRE']
        paradigm_items = {}

        for paradigm in paradigms:
            # Full item set: all items for this paradigm from full item list
            full_items = [item for item in full_item_columns if f'-{paradigm.lower()}-' in item.lower() or f'_{paradigm}_' in item]

            # Alternative: extract from item name pattern TQ_{paradigm}-*
            if len(full_items) == 0:
                # Try alternative pattern matching
                full_items = [item for item in full_item_columns if item.startswith(f'TQ_{paradigm}')]

            # Purified item set: retained items only from item mapping
            purified_items = df_item_mapping[
                (df_item_mapping['paradigm'] == paradigm) &
                (df_item_mapping['retained'] == True)
            ]['item_name'].tolist()

            paradigm_items[paradigm] = {
                'full': full_items,
                'purified': purified_items
            }

            log(f"{paradigm}: {len(full_items)} full items, {len(purified_items)} purified items")
        # Compute Cronbach's Alpha for Each Paradigm

        log("Computing Cronbach's alpha with bootstrap CIs...")

        results = []

        for paradigm in paradigms:
            log(f"\nProcessing {paradigm}...")

            full_items = paradigm_items[paradigm]['full']
            purified_items = paradigm_items[paradigm]['purified']

            n_items_full = len(full_items)
            n_items_purified = len(purified_items)

            # ----------------------------------------------------------------
            # Full CTT Alpha
            # ----------------------------------------------------------------
            log(f"Computing alpha_full for {paradigm} ({n_items_full} items)...")

            df_full_items = extract_paradigm_items_from_data(df_raw, full_items)

            full_alpha_result = compute_cronbachs_alpha(
                data=df_full_items,
                n_bootstrap=N_BOOTSTRAP
            )

            alpha_full = full_alpha_result['alpha']
            alpha_full_ci_lower = full_alpha_result['ci_lower']
            alpha_full_ci_upper = full_alpha_result['ci_upper']

            log(f"alpha_full = {alpha_full:.3f} [{alpha_full_ci_lower:.3f}, {alpha_full_ci_upper:.3f}]")

            # ----------------------------------------------------------------
            # Purified CTT Alpha
            # ----------------------------------------------------------------
            log(f"Computing alpha_purified for {paradigm} ({n_items_purified} items)...")

            df_purified_items = extract_paradigm_items_from_data(df_raw, purified_items)

            purified_alpha_result = compute_cronbachs_alpha(
                data=df_purified_items,
                n_bootstrap=N_BOOTSTRAP
            )

            alpha_purified = purified_alpha_result['alpha']
            alpha_purified_ci_lower = purified_alpha_result['ci_lower']
            alpha_purified_ci_upper = purified_alpha_result['ci_upper']

            log(f"alpha_purified = {alpha_purified:.3f} [{alpha_purified_ci_lower:.3f}, {alpha_purified_ci_upper:.3f}]")

            # ----------------------------------------------------------------
            # Spearman-Brown Adjustment
            # ----------------------------------------------------------------
            log(f"[SPEARMAN-BROWN] Adjusting purified alpha to full item count...")

            alpha_purified_sb = compute_spearman_brown_adjustment(
                alpha=alpha_purified,
                n_items_full=n_items_full,
                n_items_purified=n_items_purified
            )

            log(f"[SPEARMAN-BROWN] alpha_purified_SB_adjusted = {alpha_purified_sb:.3f}")

            # ----------------------------------------------------------------
            # Delta Alpha
            # ----------------------------------------------------------------
            delta_alpha = alpha_purified - alpha_full
            log(f"delta_alpha = {delta_alpha:.3f} (purified - full)")

            # ----------------------------------------------------------------
            # Store Results
            # ----------------------------------------------------------------
            results.append({
                'paradigm': paradigm,
                'n_items_full': n_items_full,
                'n_items_purified': n_items_purified,
                'alpha_full': alpha_full,
                'alpha_full_CI_lower': alpha_full_ci_lower,
                'alpha_full_CI_upper': alpha_full_ci_upper,
                'alpha_purified': alpha_purified,
                'alpha_purified_CI_lower': alpha_purified_ci_lower,
                'alpha_purified_CI_upper': alpha_purified_ci_upper,
                'alpha_purified_SB_adjusted': alpha_purified_sb,
                'delta_alpha': delta_alpha
            })
        # Save Results

        log("\nSaving reliability assessment results...")

        df_results = pd.DataFrame(results)
        output_path = RQ_DIR / "data" / "step04_reliability_assessment.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path}")
        log(f"{len(df_results)} rows (3 paradigms)")
        # Validation

        log("\nValidating results...")

        validation_passed = True

        # Check all 3 paradigms present
        if set(df_results['paradigm'].tolist()) != {'IFR', 'ICR', 'IRE'}:
            log("FAIL: Not all paradigms present")
            validation_passed = False
        else:
            log("PASS: All 3 paradigms present")

        # Check alpha values in [0, 1]
        for col in ['alpha_full', 'alpha_purified', 'alpha_purified_SB_adjusted']:
            if not df_results[col].between(0, 1).all():
                log(f"FAIL: {col} values outside [0, 1]")
                validation_passed = False
            else:
                log(f"PASS: {col} in [0, 1]")

        # Check CI_lower <= alpha <= CI_upper
        for paradigm in paradigms:
            row = df_results[df_results['paradigm'] == paradigm].iloc[0]

            # Full CTT
            if not (row['alpha_full_CI_lower'] <= row['alpha_full'] <= row['alpha_full_CI_upper']):
                log(f"FAIL: {paradigm} alpha_full not within CI")
                validation_passed = False

            # Purified CTT
            if not (row['alpha_purified_CI_lower'] <= row['alpha_purified'] <= row['alpha_purified_CI_upper']):
                log(f"FAIL: {paradigm} alpha_purified not within CI")
                validation_passed = False

        if validation_passed:
            log("PASS: All CIs contain alpha values")

        # Check n_items_purified <= n_items_full
        if not (df_results['n_items_purified'] <= df_results['n_items_full']).all():
            log("FAIL: n_items_purified > n_items_full for some paradigms")
            validation_passed = False
        else:
            log("PASS: n_items_purified <= n_items_full")

        # No NaN values
        if df_results.isnull().any().any():
            log("FAIL: NaN values found in results")
            validation_passed = False
        else:
            log("PASS: No NaN values")

        if validation_passed:
            log("\nStep 04 complete - All validations passed")
            sys.exit(0)
        else:
            log("\nStep 04 validation failed")
            sys.exit(1)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
