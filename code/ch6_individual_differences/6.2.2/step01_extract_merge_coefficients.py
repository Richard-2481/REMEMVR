#!/usr/bin/env python3
"""
Step 01: Extract and Merge Coefficients
RQ 7.2.2: Cognitive test attenuation of age effects

Dynamically reads age coefficients from RQ 7.2.1 outputs (not hardcoded).
Computes domain-specific attenuation by fitting OLS regressions using 7.2.1
analysis dataset merged with Ch5 domain theta scores.

Updated: Includes retention predictors (RAVLT_Pct_Ret_T, BVMT_Pct_Ret_T)
from 7.2.1's updated analysis dataset.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Define paths
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.2.2
RESULTS_DIR = PROJECT_ROOT / "results"

LOG_FILE = RQ_DIR / "logs" / "step01_extract_merge_coefficients.log"

def log(msg):
    """Print and save log messages."""
    print(msg, flush=True)
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")

def fit_age_models(data, outcome_col, age_col='Age_std',
                   cognitive_cols=None):
    """
    Fit bivariate (age-only) and controlled (age + cognitive) OLS models.

    Returns dict with beta_bivariate, beta_controlled, se, p-values.
    """
    if cognitive_cols is None:
        cognitive_cols = ['RAVLT_T_std', 'BVMT_T_std', 'RPM_T_std',
                          'RAVLT_Pct_Ret_T_std', 'BVMT_Pct_Ret_T_std']

    # Drop rows with missing outcome
    valid = data[[outcome_col, age_col] + cognitive_cols].dropna()
    y = valid[outcome_col]

    # Model 1: Age only
    X1 = sm.add_constant(valid[[age_col]])
    m1 = sm.OLS(y, X1).fit()
    beta_biv = m1.params[age_col]
    se_biv = m1.bse[age_col]
    p_biv = m1.pvalues[age_col]
    r2_biv = m1.rsquared

    # Model 2: Age + cognitive tests
    X2 = sm.add_constant(valid[[age_col] + cognitive_cols])
    m2 = sm.OLS(y, X2).fit()
    beta_ctrl = m2.params[age_col]
    se_ctrl = m2.bse[age_col]
    p_ctrl = m2.pvalues[age_col]
    r2_ctrl = m2.rsquared

    return {
        'beta_bivariate': beta_biv,
        'se_bivariate': se_biv,
        'p_bivariate': p_biv,
        'r2_bivariate': r2_biv,
        'beta_controlled': beta_ctrl,
        'se_controlled': se_ctrl,
        'p_controlled': p_ctrl,
        'r2_controlled': r2_ctrl,
        'n': len(valid),
        'n_predictors_controlled': len(cognitive_cols) + 1  # age + cognitive
    }

def main():
    """Main execution."""
    # Clear log file
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("")

    log("=" * 60)
    log("Step 01: Extract and Merge Coefficients")
    log("Reads betas dynamically from RQ 7.2.1 + computes domain models")
    log("Includes retention predictors (RAVLT/BVMT Pct Ret)")
    log("=" * 60)

    # =========================================================================
    # 1. Read overall age coefficients from RQ 7.2.1 mediation output
    # =========================================================================
    log("\n1. Reading age coefficients from RQ 7.2.1 mediation analysis...")

    mediation_path = RESULTS_DIR / "ch7" / "7.2.1" / "data" / "step04_mediation_analysis.csv"
    if not mediation_path.exists():
        log(f"ERROR: Cannot find RQ 7.2.1 mediation results at {mediation_path}")
        return 1

    mediation_df = pd.read_csv(mediation_path)
    log(f"  Loaded mediation results: {mediation_df.shape}")
    log(f"  Columns: {mediation_df.columns.tolist()}")

    # beta_total = bivariate age effect (c path)
    # beta_direct = controlled age effect (c' path)
    beta_age_bivariate = mediation_df['beta_total'].iloc[0]
    beta_age_controlled = mediation_df['beta_direct'].iloc[0]

    log(f"  Beta (age only, total effect):      {beta_age_bivariate:.4f}")
    log(f"  Beta (age + cognitive, direct):      {beta_age_controlled:.4f}")
    log(f"  Mediation effect:                    {mediation_df['mediation_effect'].iloc[0]:.4f}")
    log(f"  Proportion mediated:                 {mediation_df['proportion_mediated'].iloc[0]:.4f}")

    # =========================================================================
    # 2. Load 7.2.1 analysis dataset (has all predictors + theta_all)
    # =========================================================================
    log("\n2. Loading RQ 7.2.1 analysis dataset...")

    analysis_path = RESULTS_DIR / "ch7" / "7.2.1" / "data" / "step01_analysis_dataset.csv"
    if not analysis_path.exists():
        log(f"ERROR: Cannot find analysis dataset at {analysis_path}")
        return 1

    analysis_df = pd.read_csv(analysis_path)
    log(f"  Loaded: {analysis_df.shape}")
    log(f"  Columns: {analysis_df.columns.tolist()}")

    # Identify cognitive predictor columns (standardized)
    cognitive_cols = [c for c in analysis_df.columns if c.endswith('_std') and c != 'Age_std']
    log(f"  Cognitive predictors (standardized): {cognitive_cols}")

    # =========================================================================
    # 3. Verify overall coefficients by re-fitting from raw data
    # =========================================================================
    log("\n3. Verifying overall coefficients by re-fitting from raw data...")

    overall_results = fit_age_models(analysis_df, 'theta_all', 'Age_std', cognitive_cols)

    log(f"  Re-fitted beta_bivariate:  {overall_results['beta_bivariate']:.4f}")
    log(f"  Re-fitted beta_controlled: {overall_results['beta_controlled']:.4f}")
    log(f"  7.2.1 mediation beta_total:  {beta_age_bivariate:.4f}")
    log(f"  7.2.1 mediation beta_direct: {beta_age_controlled:.4f}")

    # Use re-fitted values (they include retention predictors)
    # Note: 7.2.1 mediation may have been computed before retention predictors were added
    beta_age_bivariate_verified = overall_results['beta_bivariate']
    beta_age_controlled_verified = overall_results['beta_controlled']

    log(f"\n  Using verified values (with retention predictors in controlled model):")
    log(f"    beta_bivariate:  {beta_age_bivariate_verified:.4f}")
    log(f"    beta_controlled: {beta_age_controlled_verified:.4f}")

    # =========================================================================
    # 4. Extract domain theta scores from Ch5 and compute domain-specific betas
    # =========================================================================
    log("\n4. Extracting theta scores from Ch5...")

    # OVERALL theta from 5.1.1
    ch5_overall_path = RESULTS_DIR / "ch5" / "5.1.1" / "data" / "step03_theta_scores.csv"
    if ch5_overall_path.exists():
        overall_df = pd.read_csv(ch5_overall_path)
        log(f"  Loaded overall theta from 5.1.1: {overall_df.shape}")
        overall_df['UID'] = overall_df['UID'].str.strip()
        overall_by_uid = overall_df.groupby('UID')['Theta_All'].mean().reset_index()
        overall_by_uid.columns = ['UID', 'theta_all']
    else:
        log(f"  ERROR: Cannot find overall theta at {ch5_overall_path}")
        return 1

    # DOMAIN-SPECIFIC theta from 5.2.1
    ch5_domain_path = RESULTS_DIR / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv"
    if ch5_domain_path.exists():
        domain_df = pd.read_csv(ch5_domain_path)
        log(f"  Loaded domain theta from 5.2.1: {domain_df.shape}")
        log(f"  Domain columns: {domain_df.columns.tolist()}")

        domain_df['UID'] = domain_df['composite_ID'].str.split('_').str[0]

        what_by_uid = domain_df.groupby('UID')['theta_what'].mean().reset_index()
        where_by_uid = domain_df.groupby('UID')['theta_where'].mean().reset_index()
        when_by_uid = domain_df.groupby('UID')['theta_when'].mean().reset_index()

        log(f"  What domain: {len(what_by_uid)} participants")
        log(f"  Where domain: {len(where_by_uid)} participants")
        log(f"  When domain: {len(when_by_uid)} participants")

        when_mean = when_by_uid['theta_when'].mean()
        when_std = when_by_uid['theta_when'].std()
        log(f"  When domain stats: M={when_mean:.3f}, SD={when_std:.3f}")
        if when_mean < -1.0:
            log("  WARNING: When domain shows floor effects (M < -1.0)")
    else:
        log(f"  ERROR: Cannot find domain theta at {ch5_domain_path}")
        return 1

    # =========================================================================
    # 5. Merge domain theta with 7.2.1 analysis dataset for domain regressions
    # =========================================================================
    log("\n5. Merging domain theta with analysis dataset for domain-specific regressions...")

    # Start with 7.2.1 analysis data (has Age_std, cognitive predictors)
    domain_analysis = analysis_df.copy()

    # Merge domain theta scores
    domain_analysis = domain_analysis.merge(what_by_uid, on='UID', how='left')
    domain_analysis = domain_analysis.merge(where_by_uid, on='UID', how='left')
    domain_analysis = domain_analysis.merge(when_by_uid, on='UID', how='left')

    log(f"  Merged dataset: {domain_analysis.shape}")
    has_what = domain_analysis['theta_what'].notna().sum()
    has_where = domain_analysis['theta_where'].notna().sum()
    has_when = domain_analysis['theta_when'].notna().sum()
    log(f"  What: {has_what}, Where: {has_where}, When: {has_when}")

    # Fit domain-specific regressions
    domain_betas = {}
    for domain, col in [('what', 'theta_what'), ('where', 'theta_where'), ('when', 'theta_when')]:
        valid_n = domain_analysis[col].notna().sum()
        if valid_n >= 30:  # Need reasonable N for regression
            log(f"\n  Fitting {domain} domain regression (n={valid_n})...")
            results = fit_age_models(domain_analysis, col, 'Age_std', cognitive_cols)
            domain_betas[domain] = results
            log(f"    beta_bivariate:  {results['beta_bivariate']:.4f} (p={results['p_bivariate']:.4f})")
            log(f"    beta_controlled: {results['beta_controlled']:.4f} (p={results['p_controlled']:.4f})")
        else:
            log(f"\n  Skipping {domain} domain: insufficient data (n={valid_n})")
            domain_betas[domain] = None

    # =========================================================================
    # 6. Merge all data for output
    # =========================================================================
    log("\n6. Creating merged output dataset...")

    merged_df = overall_by_uid.copy()
    merged_df = merged_df.merge(what_by_uid, on='UID', how='left')
    merged_df = merged_df.merge(where_by_uid, on='UID', how='left')
    merged_df = merged_df.merge(when_by_uid, on='UID', how='left')

    # Add overall coefficients (verified with retention predictors)
    merged_df['beta_age_bivariate_all'] = beta_age_bivariate_verified
    merged_df['beta_age_controlled_all'] = beta_age_controlled_verified

    # Add domain-specific coefficients
    for domain in ['what', 'where', 'when']:
        if domain_betas[domain] is not None:
            merged_df[f'beta_age_bivariate_{domain}'] = domain_betas[domain]['beta_bivariate']
            merged_df[f'beta_age_controlled_{domain}'] = domain_betas[domain]['beta_controlled']
        else:
            merged_df[f'beta_age_bivariate_{domain}'] = np.nan
            merged_df[f'beta_age_controlled_{domain}'] = np.nan

    log(f"\n  Merged dataset: {len(merged_df)} rows, {len(merged_df.columns)} cols")
    log(f"  Columns: {list(merged_df.columns)}")

    # =========================================================================
    # 7. Save outputs
    # =========================================================================
    log("\n7. Saving outputs...")

    # Save merged coefficients
    output_file = RQ_DIR / "data" / "step01_merged_coefficients.csv"
    merged_df.to_csv(output_file, index=False)
    log(f"  Saved merged coefficients to: {output_file}")

    # Save coefficient comparison table
    coef_table = []
    # Overall
    coef_table.append({
        'domain': 'overall',
        'n': overall_results['n'],
        'beta_bivariate': overall_results['beta_bivariate'],
        'se_bivariate': overall_results['se_bivariate'],
        'p_bivariate': overall_results['p_bivariate'],
        'r2_bivariate': overall_results['r2_bivariate'],
        'beta_controlled': overall_results['beta_controlled'],
        'se_controlled': overall_results['se_controlled'],
        'p_controlled': overall_results['p_controlled'],
        'r2_controlled': overall_results['r2_controlled'],
        'n_predictors_controlled': overall_results['n_predictors_controlled'],
        'source_bivariate': 'refit from 7.2.1 data',
        'source_controlled': f'refit with {len(cognitive_cols)} cognitive predictors (incl. retention)',
    })
    # Domains
    for domain in ['what', 'where', 'when']:
        if domain_betas[domain] is not None:
            r = domain_betas[domain]
            coef_table.append({
                'domain': domain,
                'n': r['n'],
                'beta_bivariate': r['beta_bivariate'],
                'se_bivariate': r['se_bivariate'],
                'p_bivariate': r['p_bivariate'],
                'r2_bivariate': r['r2_bivariate'],
                'beta_controlled': r['beta_controlled'],
                'se_controlled': r['se_controlled'],
                'p_controlled': r['p_controlled'],
                'r2_controlled': r['r2_controlled'],
                'n_predictors_controlled': r['n_predictors_controlled'],
                'source_bivariate': 'fit from 7.2.1 data + Ch5 domain theta',
                'source_controlled': f'fit with {len(cognitive_cols)} cognitive predictors (incl. retention)',
            })

    coef_df = pd.DataFrame(coef_table)
    coef_file = RQ_DIR / "data" / "step01_coefficient_table.csv"
    coef_df.to_csv(coef_file, index=False)
    log(f"  Saved coefficient table to: {coef_file}")

    # Save data summary
    summary_file = RQ_DIR / "data" / "step01_data_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("DATA SUMMARY FOR RQ 7.2.2 ATTENUATION ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total participants: {len(merged_df)}\n")
        f.write(f"Variables: {len(merged_df.columns)}\n\n")

        f.write("Age Coefficients from RQ 7.2.1 (dynamically extracted):\n")
        f.write(f"  7.2.1 mediation beta_total:  {beta_age_bivariate:.4f}\n")
        f.write(f"  7.2.1 mediation beta_direct: {beta_age_controlled:.4f}\n")
        f.write(f"  Re-fitted beta_bivariate:    {beta_age_bivariate_verified:.4f}\n")
        f.write(f"  Re-fitted beta_controlled:   {beta_age_controlled_verified:.4f}\n\n")

        f.write("Cognitive predictors in controlled model:\n")
        for c in cognitive_cols:
            f.write(f"  - {c}\n")
        f.write(f"\nNOTE: Retention predictors (RAVLT_Pct_Ret_T_std, BVMT_Pct_Ret_T_std)\n")
        f.write(f"are included in the controlled model following 7.2.1 Phase 2 update.\n\n")

        f.write("Domain-specific coefficients:\n")
        for domain in ['what', 'where', 'when']:
            if domain_betas[domain] is not None:
                r = domain_betas[domain]
                f.write(f"  {domain.upper()}: biv={r['beta_bivariate']:.4f}, ctrl={r['beta_controlled']:.4f} (n={r['n']})\n")
            else:
                f.write(f"  {domain.upper()}: insufficient data\n")

    log(f"  Saved data summary to: {summary_file}")

    # Save domain availability report
    domain_report = RQ_DIR / "data" / "step01_domain_availability.csv"
    domain_avail = pd.DataFrame({
        'domain': ['overall', 'what', 'where', 'when'],
        'data_available': [True, has_what >= 30, has_where >= 30, has_when >= 30],
        'n_participants': [len(merged_df), has_what, has_where, has_when],
        'mean_theta': [
            merged_df['theta_all'].mean(),
            merged_df['theta_what'].mean() if has_what > 0 else np.nan,
            merged_df['theta_where'].mean() if has_where > 0 else np.nan,
            merged_df['theta_when'].mean() if has_when > 0 else np.nan,
        ],
        'notes': [
            'From Ch5 5.1.1',
            'From Ch5 5.2.1',
            'From Ch5 5.2.1',
            'From Ch5 5.2.1 - possible floor effects'
        ]
    })
    domain_avail.to_csv(domain_report, index=False)
    log(f"  Saved domain availability to: {domain_report}")

    log("\n" + "=" * 60)
    log("Step 01 COMPLETE")
    log(f"  Overall: biv={beta_age_bivariate_verified:.4f}, ctrl={beta_age_controlled_verified:.4f}")
    for domain in ['what', 'where', 'when']:
        if domain_betas[domain] is not None:
            r = domain_betas[domain]
            log(f"  {domain.capitalize()}: biv={r['beta_bivariate']:.4f}, ctrl={r['beta_controlled']:.4f}")
    log("=" * 60)

    return 0

if __name__ == "__main__":
    sys.exit(main())
