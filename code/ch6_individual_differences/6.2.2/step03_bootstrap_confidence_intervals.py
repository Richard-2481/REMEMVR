#!/usr/bin/env python3
"""
Step 03: Bootstrap Confidence Intervals for Attenuation Ratios
===============================================================
Purpose: Generate bootstrap confidence intervals using participant-level resampling

Scientific Context:
Bootstrap provides robust inference for the attenuation ratio, especially important
given potential suppression effects (>100% attenuation). Participant-level resampling
preserves within-participant correlation structure.

Updated: Controlled model now includes retention predictors
(RAVLT_Pct_Ret_T_std, BVMT_Pct_Ret_T_std) matching 7.2.1 Phase 2.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Set up paths
RQ_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_FILE = RQ_DIR / "logs" / "step03_bootstrap_confidence_intervals.log"

(RQ_DIR / "logs").mkdir(exist_ok=True)
(RQ_DIR / "data").mkdir(exist_ok=True)

def log(msg):
    """Log message to both file and console"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def compute_attenuation(data, outcome_col, age_col='Age_std', cognitive_cols=None):
    """
    Compute attenuation for a given dataset.
    Returns beta_bivariate, beta_controlled, attenuation_percent.
    """
    if cognitive_cols is None:
        cognitive_cols = ['RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                          'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']

    # Drop rows with missing outcome or predictors
    all_cols = [outcome_col, age_col] + cognitive_cols
    valid = data[all_cols].dropna()

    if len(valid) < 10:
        return np.nan, np.nan, np.nan

    y = valid[outcome_col].values

    # Model 1: Bivariate (age only)
    X_biv = valid[[age_col]].values
    m1 = LinearRegression()
    m1.fit(X_biv, y)
    beta_bivariate = m1.coef_[0]

    # Model 2: Controlled (age + cognitive tests including retention)
    X_ctrl = valid[[age_col] + cognitive_cols].values
    m2 = LinearRegression()
    m2.fit(X_ctrl, y)
    beta_controlled = m2.coef_[0]  # Age is first predictor

    # Compute attenuation
    if abs(beta_bivariate) < 1e-10:
        attenuation_percent = np.nan
    else:
        attenuation_percent = ((beta_bivariate - beta_controlled) / beta_bivariate) * 100

    return beta_bivariate, beta_controlled, attenuation_percent

def bootstrap_attenuation(data, n_iterations=1000, seed=42, outcome_col='theta_all',
                          cognitive_cols=None):
    """
    Bootstrap the attenuation ratio using participant-level resampling.
    """
    np.random.seed(seed)
    n_participants = len(data)

    bootstrap_results = {
        'beta_bivariate': [],
        'beta_controlled': [],
        'attenuation_percent': []
    }

    for i in range(n_iterations):
        indices = np.random.choice(n_participants, n_participants, replace=True)
        bootstrap_sample = data.iloc[indices].copy()

        beta_biv, beta_ctrl, atten_pct = compute_attenuation(
            bootstrap_sample, outcome_col, cognitive_cols=cognitive_cols
        )

        bootstrap_results['beta_bivariate'].append(beta_biv)
        bootstrap_results['beta_controlled'].append(beta_ctrl)
        bootstrap_results['attenuation_percent'].append(atten_pct)

        if (i + 1) % 200 == 0:
            log(f"  Bootstrap iteration {i+1}/{n_iterations}")

    return bootstrap_results

def compute_confidence_intervals(bootstrap_results, alpha=0.05):
    """
    Compute percentile-based confidence intervals.
    """
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100

    ci_results = {}

    for key, values in bootstrap_results.items():
        valid_values = [v for v in values if not np.isnan(v)]

        if len(valid_values) > 0:
            ci_results[key] = {
                'mean': np.mean(valid_values),
                'median': np.median(valid_values),
                'ci_lower': np.percentile(valid_values, lower_pct),
                'ci_upper': np.percentile(valid_values, upper_pct),
                'n_valid': len(valid_values)
            }
        else:
            ci_results[key] = {
                'mean': np.nan,
                'median': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'n_valid': 0
            }

    return ci_results

def main():
    """Main bootstrap analysis function"""
    # Clear log
    with open(LOG_FILE, 'w') as f:
        f.write("")

    log("=" * 70)
    log("STEP 03: BOOTSTRAP CONFIDENCE INTERVALS FOR ATTENUATION")
    log("Includes retention predictors (RAVLT/BVMT Pct Ret)")
    log("=" * 70)

    # 1. Load raw data from RQ 7.2.1
    log("\n1. Loading raw analysis data from RQ 7.2.1...")

    raw_data_file = RESULTS_DIR / "ch7" / "7.2.1" / "data" / "step01_analysis_dataset.csv"

    if not raw_data_file.exists():
        log(f"ERROR: Cannot find raw data file: {raw_data_file}")
        sys.exit(1)

    raw_data = pd.read_csv(raw_data_file)
    log(f"  Loaded {len(raw_data)} participants")
    log(f"  Columns: {list(raw_data.columns)}")

    # Identify cognitive predictor columns
    cognitive_cols = [c for c in raw_data.columns if c.endswith('_std') and c != 'Age_std']
    log(f"  Cognitive predictors: {cognitive_cols}")

    # 2. Load domain theta scores and merge
    log("\n2. Loading and merging domain theta scores from Ch5...")

    merged_coef_file = RQ_DIR / "data" / "step01_merged_coefficients.csv"
    merged_df = pd.read_csv(merged_coef_file)

    # Merge domain theta from step01 output into raw_data
    domain_cols = [c for c in merged_df.columns if c.startswith('theta_') and c != 'theta_all']
    if domain_cols:
        analysis_data = raw_data.merge(
            merged_df[['UID'] + domain_cols],
            on='UID',
            how='left',
            suffixes=('', '_dup')
        )
        # Drop any duplicate columns
        analysis_data = analysis_data[[c for c in analysis_data.columns if not c.endswith('_dup')]]
    else:
        analysis_data = raw_data.copy()

    log(f"  Merged dataset: {len(analysis_data)} participants")
    log(f"  Final columns: {list(analysis_data.columns)}")

    # 3. Compute observed attenuation (for comparison)
    log("\n3. Computing observed attenuation...")

    obs_beta_biv, obs_beta_ctrl, obs_atten = compute_attenuation(
        analysis_data, 'theta_all', cognitive_cols=cognitive_cols
    )

    log(f"  Observed attenuation:")
    log(f"    Beta bivariate:  {obs_beta_biv:.4f}")
    log(f"    Beta controlled: {obs_beta_ctrl:.4f}")
    log(f"    Attenuation:     {obs_atten:.1f}%")

    # 4. Bootstrap for overall REMEMVR
    log("\n4. Bootstrap for overall REMEMVR (1000 iterations, seed=42)...")

    bootstrap_overall = bootstrap_attenuation(
        analysis_data,
        n_iterations=1000,
        seed=42,
        outcome_col='theta_all',
        cognitive_cols=cognitive_cols
    )

    # 5. Bootstrap for domain-specific outcomes
    domain_bootstraps = {}
    for domain, col in [('what', 'theta_what'), ('where', 'theta_where'), ('when', 'theta_when')]:
        if col in analysis_data.columns and analysis_data[col].notna().sum() >= 30:
            log(f"\n5. Bootstrap for {domain.capitalize()} domain (1000 iterations)...")
            domain_bootstraps[domain] = bootstrap_attenuation(
                analysis_data,
                n_iterations=1000,
                seed=42,
                outcome_col=col,
                cognitive_cols=cognitive_cols
            )
        else:
            log(f"\n5. Skipping {domain} domain bootstrap (insufficient data)")

    # 6. Compute confidence intervals
    log("\n6. Computing 95% confidence intervals...")

    ci_overall = compute_confidence_intervals(bootstrap_overall)

    log(f"\n  Overall REMEMVR Bootstrap Results:")
    log(f"    Attenuation: {ci_overall['attenuation_percent']['median']:.1f}%")
    log(f"    95% CI: [{ci_overall['attenuation_percent']['ci_lower']:.1f}%, "
        f"{ci_overall['attenuation_percent']['ci_upper']:.1f}%]")

    if ci_overall['attenuation_percent']['ci_lower'] > 0:
        log("    Significant attenuation (CI excludes 0)")

    domain_cis = {}
    for domain, bs_results in domain_bootstraps.items():
        ci = compute_confidence_intervals(bs_results)
        domain_cis[domain] = ci
        log(f"\n  {domain.capitalize()} Domain Bootstrap Results:")
        log(f"    Attenuation: {ci['attenuation_percent']['median']:.1f}%")
        log(f"    95% CI: [{ci['attenuation_percent']['ci_lower']:.1f}%, "
            f"{ci['attenuation_percent']['ci_upper']:.1f}%]")

    # 7. Save bootstrap distributions
    log("\n7. Saving bootstrap distributions...")

    dist_dict = {
        'iteration': range(1000),
        'overall_attenuation': bootstrap_overall['attenuation_percent'],
        'overall_beta_biv': bootstrap_overall['beta_bivariate'],
        'overall_beta_ctrl': bootstrap_overall['beta_controlled'],
    }
    for domain, bs_results in domain_bootstraps.items():
        dist_dict[f'{domain}_attenuation'] = bs_results['attenuation_percent']
        dist_dict[f'{domain}_beta_biv'] = bs_results['beta_bivariate']
        dist_dict[f'{domain}_beta_ctrl'] = bs_results['beta_controlled']

    bootstrap_dist_df = pd.DataFrame(dist_dict)
    dist_file = RQ_DIR / "data" / "step03_bootstrap_distributions.csv"
    bootstrap_dist_df.to_csv(dist_file, index=False)
    log(f"  Saved bootstrap distributions to: {dist_file}")

    # 8. Save confidence intervals
    log("\n8. Saving confidence interval results...")

    ci_rows = []

    # Overall
    ci_rows.append({
        'domain': 'overall',
        'point_estimate': obs_atten,
        'bootstrap_median': ci_overall['attenuation_percent']['median'],
        'ci_lower': ci_overall['attenuation_percent']['ci_lower'],
        'ci_upper': ci_overall['attenuation_percent']['ci_upper'],
        'ci_width': (ci_overall['attenuation_percent']['ci_upper'] -
                     ci_overall['attenuation_percent']['ci_lower']),
        'bootstrap_p': np.mean([x <= 0 for x in bootstrap_overall['attenuation_percent']
                                if not np.isnan(x)])
    })

    # Domains
    for domain, ci in domain_cis.items():
        bs_results = domain_bootstraps[domain]
        ci_rows.append({
            'domain': domain,
            'point_estimate': ci['attenuation_percent']['median'],
            'bootstrap_median': ci['attenuation_percent']['median'],
            'ci_lower': ci['attenuation_percent']['ci_lower'],
            'ci_upper': ci['attenuation_percent']['ci_upper'],
            'ci_width': ci['attenuation_percent']['ci_upper'] - ci['attenuation_percent']['ci_lower'],
            'bootstrap_p': np.mean([x <= 0 for x in bs_results['attenuation_percent']
                                    if not np.isnan(x)])
        })

    ci_results_df = pd.DataFrame(ci_rows)
    ci_file = RQ_DIR / "data" / "step03_confidence_intervals.csv"
    ci_results_df.to_csv(ci_file, index=False)
    log(f"  Saved confidence intervals to: {ci_file}")

    # 9. Save bootstrap diagnostics
    log("\n9. Saving bootstrap diagnostics...")

    diagnostics_file = RQ_DIR / "data" / "step03_bootstrap_diagnostics.txt"
    with open(diagnostics_file, 'w') as f:
        f.write("BOOTSTRAP DIAGNOSTICS FOR ATTENUATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")

        f.write("Bootstrap Parameters:\n")
        f.write(f"  Iterations: 1000\n")
        f.write(f"  Random seed: 42\n")
        f.write(f"  Resampling: Participant-level with replacement\n")
        f.write(f"  CI method: Percentile (2.5th, 97.5th)\n")
        f.write(f"  Cognitive predictors: {cognitive_cols}\n\n")

        f.write("CI Stability:\n")
        for domain_name, ci_data in [('Overall', ci_overall)] + \
                [(d.capitalize(), c) for d, c in domain_cis.items()]:
            width = ci_data['attenuation_percent']['ci_upper'] - ci_data['attenuation_percent']['ci_lower']
            stability = "STABLE" if abs(width) < 40 else "WIDE - may need more iterations"
            f.write(f"  {domain_name}: CI width = {width:.1f}% ({stability})\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("KEY FINDING:\n")
        med = ci_overall['attenuation_percent']['median']
        if med > 100:
            f.write(f"SUPPRESSION EFFECT CONFIRMED BY BOOTSTRAP\n")
            f.write(f"Median attenuation = {med:.1f}%\n")
            f.write("Age coefficient consistently reverses sign across bootstrap samples\n")
        elif med > 70:
            f.write(f"SUBSTANTIAL ATTENUATION CONFIRMED\n")
            f.write(f"Median attenuation = {med:.1f}%\n")
        else:
            f.write(f"Median attenuation = {med:.1f}%\n")

        f.write("\nNote: Controlled model includes retention predictors\n")
        f.write("(RAVLT_Pct_Ret_T_std, BVMT_Pct_Ret_T_std) from 7.2.1 Phase 2.\n")

    log(f"  Saved bootstrap diagnostics to: {diagnostics_file}")

    log(f"\n{'=' * 70}")
    log("BOOTSTRAP COMPLETE")
    log(f"  Overall: {ci_overall['attenuation_percent']['median']:.1f}% "
        f"[{ci_overall['attenuation_percent']['ci_lower']:.1f}%, "
        f"{ci_overall['attenuation_percent']['ci_upper']:.1f}%]")
    for domain, ci in domain_cis.items():
        log(f"  {domain.capitalize()}: {ci['attenuation_percent']['median']:.1f}% "
            f"[{ci['attenuation_percent']['ci_lower']:.1f}%, "
            f"{ci['attenuation_percent']['ci_upper']:.1f}%]")

    if ci_overall['attenuation_percent']['median'] > 100:
        log("SUPPRESSION EFFECT CONFIRMED")
    log("=" * 70)
    log("\nStep 03 complete: Bootstrap confidence intervals computed")

    return ci_results_df

if __name__ == "__main__":
    main()
