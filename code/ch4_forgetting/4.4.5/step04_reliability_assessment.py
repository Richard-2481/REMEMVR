#!/usr/bin/env python3
"""
Step 04: Cronbach's Alpha Reliability Assessment

Assesses internal consistency reliability (Cronbach's alpha) for Full and Purified CTT scores
per congruence level with bootstrap 95% CIs.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_reliability_assessment.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def cronbach_alpha(item_scores):
    """
    Compute Cronbach's alpha from item scores matrix.

    Args:
        item_scores: DataFrame or array, rows = observations, columns = items

    Returns:
        float: Cronbach's alpha
    """
    item_scores = np.array(item_scores)
    n_items = item_scores.shape[1]

    # Variance of each item
    item_vars = item_scores.var(axis=0, ddof=1)

    # Variance of total score
    total_var = item_scores.sum(axis=1).var(ddof=1)

    # Cronbach's alpha formula
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)

    return alpha

def bootstrap_alpha(item_scores, n_bootstrap=1000, random_seed=42):
    """
    Compute bootstrap 95% CI for Cronbach's alpha.

    Args:
        item_scores: DataFrame or array, rows = observations, columns = items
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (lower_ci, upper_ci)
    """
    np.random.seed(random_seed)
    item_scores = np.array(item_scores)
    n_obs = item_scores.shape[0]

    alphas = []
    for _ in range(n_bootstrap):
        # Resample observations with replacement
        indices = np.random.choice(n_obs, size=n_obs, replace=True)
        boot_sample = item_scores[indices, :]
        alphas.append(cronbach_alpha(boot_sample))

    # Compute 95% CI using percentile method
    lower_ci = np.percentile(alphas, 2.5)
    upper_ci = np.percentile(alphas, 97.5)

    return lower_ci, upper_ci

if __name__ == "__main__":
    try:
        log("[START] Step 04: Cronbach's Alpha Reliability Assessment")

        # Load wide-format data
        data_path = PROJECT_ROOT / "data" / "cache" / "dfData.csv"
        log(f"[LOAD] Reading {data_path}")
        df_data = pd.read_csv(data_path, encoding='utf-8')
        log(f"[LOADED] {len(df_data)} rows, {len(df_data.columns)} columns")

        # Load full item list
        full_item_list_path = RQ_DIR / "data" / "step00_full_item_list.csv"
        log(f"[LOAD] Reading {full_item_list_path}")
        full_item_list = pd.read_csv(full_item_list_path, encoding='utf-8')

        # Load purified item list
        purified_items_path = PROJECT_ROOT / "results" / "ch5" / "5.4.1" / "data" / "step02_purified_items.csv"
        log(f"[LOAD] Reading {purified_items_path}")
        purified_items = pd.read_csv(purified_items_path, encoding='utf-8')

        # Create dimension mappings
        full_dimension_map = dict(zip(full_item_list['item_code'], full_item_list['dimension']))
        purified_dimension_map = dict(zip(purified_items['item_name'], purified_items['dimension']))

        # Compute alpha for each dimension
        results = []

        for dimension in ['common', 'congruent', 'incongruent']:
            log(f"[ANALYSIS] Processing {dimension.capitalize()} dimension")

            # Get items for this dimension - Full set
            full_items = [col for col in df_data.columns
                         if col.startswith('TQ_') and full_dimension_map.get(col) == dimension]

            # Get items for this dimension - Purified set
            purified_items_list = [col for col in df_data.columns
                                  if col.startswith('TQ_') and purified_dimension_map.get(col) == dimension]

            log(f"  Full: {len(full_items)} items")
            log(f"  Purified: {len(purified_items_list)} items")

            # Compute alpha for Full set
            full_item_scores = df_data[full_items]
            alpha_full = cronbach_alpha(full_item_scores)
            alpha_full_ci_lower, alpha_full_ci_upper = bootstrap_alpha(full_item_scores, n_bootstrap=1000)

            log(f"  Full alpha: {alpha_full:.3f} [{alpha_full_ci_lower:.3f}, {alpha_full_ci_upper:.3f}]")

            # Compute alpha for Purified set
            purified_item_scores = df_data[purified_items_list]
            alpha_purified = cronbach_alpha(purified_item_scores)
            alpha_purified_ci_lower, alpha_purified_ci_upper = bootstrap_alpha(purified_item_scores, n_bootstrap=1000)

            log(f"  Purified alpha: {alpha_purified:.3f} [{alpha_purified_ci_lower:.3f}, {alpha_purified_ci_upper:.3f}]")

            # Delta alpha
            delta_alpha = alpha_purified - alpha_full
            log(f"  Delta alpha (Purified - Full): {delta_alpha:+.3f}")

            results.append({
                'dimension': dimension.capitalize(),
                'alpha_full': alpha_full,
                'alpha_full_CI_lower': alpha_full_ci_lower,
                'alpha_full_CI_upper': alpha_full_ci_upper,
                'alpha_purified': alpha_purified,
                'alpha_purified_CI_lower': alpha_purified_ci_lower,
                'alpha_purified_CI_upper': alpha_purified_ci_upper,
                'delta_alpha': delta_alpha
            })

        # Create DataFrame
        reliability_assessment = pd.DataFrame(results)

        # Validation: Check alpha values in [0, 1]
        log("[VALIDATION] Checking alpha values in [0, 1]")
        alpha_cols = ['alpha_full', 'alpha_purified']
        for col in alpha_cols:
            min_val = reliability_assessment[col].min()
            max_val = reliability_assessment[col].max()
            if min_val < 0.0 or max_val > 1.0:
                raise ValueError(f"{col} out of range: [{min_val:.3f}, {max_val:.3f}]")
        log("[PASS] All alpha values in valid range")

        # Validation: Check CI contains estimate
        log("[VALIDATION] Checking CI_lower < alpha < CI_upper")
        for idx, row in reliability_assessment.iterrows():
            if not (row['alpha_full_CI_lower'] < row['alpha_full'] < row['alpha_full_CI_upper']):
                raise ValueError(f"Full alpha CI does not contain estimate for {row['dimension']}")
            if not (row['alpha_purified_CI_lower'] < row['alpha_purified'] < row['alpha_purified_CI_upper']):
                raise ValueError(f"Purified alpha CI does not contain estimate for {row['dimension']}")
        log("[PASS] All CIs contain estimates")

        # Save results
        output_path = RQ_DIR / "data" / "step04_reliability_assessment.csv"
        log(f"[SAVE] Writing {output_path}")
        reliability_assessment.to_csv(output_path, index=False, encoding='utf-8')
        log(f"[SAVED] {len(reliability_assessment)} rows, {len(reliability_assessment.columns)} columns")

        log("[SUCCESS] Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
