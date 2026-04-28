#!/usr/bin/env python3
"""
RQ 6.5.3 - GEE Validation for High-Confidence Errors

Purpose: Validate LPM results using proper binomial GEE model
Date: 2025-12-30

Background:
- Original analysis (2025-12-12) used Linear Probability Model (LPM)
- LPM has known limitations for binary outcomes (heteroscedasticity, unbounded predictions)
- Summary.md Section 5 flagged GEE re-analysis as HIGH PRIORITY
- This script uses statsmodels GEE with binomial family and logit link

Input: data/step01_hce_flags.csv (7,200 item-responses with HCE flags)
Output:
  - data/step03b_gee_results.csv (fixed effect estimates with p-values)
  - data/step03b_gee_contrasts.csv (post-hoc pairwise comparisons)
  - data/step03b_gee_model_summary.txt (full model output)
  - logs/step03b_gee_validation.log

Model Specification:
- Outcome: HCE_flag (binary: 0/1)
- Fixed effects: Congruence + Time + Congruence×Time
- Working correlation: Exchangeable (repeated measures per participant)
- Family: Binomial with logit link (proper model for binary outcome)
- Reference level: Common items
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'step03b_gee_validation.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run GEE validation for HCE analysis"""

    logger.info("="*80)
    logger.info("RQ 6.5.3 - GEE Validation for High-Confidence Errors")
    logger.info("="*80)

    # Import statsmodels GEE
    try:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable
        import statsmodels.api as sm
        logger.info("✓ Imported statsmodels GEE modules")
    except ImportError as e:
        logger.error(f"Failed to import statsmodels: {e}")
        logger.error("Install with: pip install statsmodels")
        sys.exit(1)

    # Load data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Load HCE Data")
    logger.info("="*80)

    data_file = Path('data/step01_hce_flags.csv')
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        sys.exit(1)

    df = pd.read_csv(data_file)
    logger.info(f"✓ Loaded {len(df):,} item-responses from {data_file}")
    logger.info(f"  Columns: {', '.join(df.columns)}")
    logger.info(f"  Participants: {df['UID'].nunique()}")
    logger.info(f"  HCE events: {df['HCE_flag'].sum():,} ({df['HCE_flag'].mean()*100:.2f}%)")

    # Prepare data for GEE
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Prepare Data for GEE")
    logger.info("="*80)

    # Create numeric time variable (Days: 0, 1, 3, 6)
    test_to_days = {1: 0, 2: 1, 3: 3, 4: 6}
    df['Days'] = df['Test'].map(test_to_days)
    logger.info(f"✓ Created Days variable from Test (0, 1, 3, 6)")

    # Create dummy variables for Congruence (reference: Common)
    df['Congruent_vs_Common'] = (df['Congruence'] == 'Congruent').astype(int)
    df['Incongruent_vs_Common'] = (df['Congruence'] == 'Incongruent').astype(int)
    logger.info(f"✓ Created dummy variables (reference: Common)")
    logger.info(f"  Common: n={sum(df['Congruence']=='Common'):,}")
    logger.info(f"  Congruent: n={sum(df['Congruence']=='Congruent'):,}")
    logger.info(f"  Incongruent: n={sum(df['Congruence']=='Incongruent'):,}")

    # Create interaction terms
    df['Congruent_x_Days'] = df['Congruent_vs_Common'] * df['Days']
    df['Incongruent_x_Days'] = df['Incongruent_vs_Common'] * df['Days']
    logger.info(f"✓ Created interaction terms")

    # Sort by UID for GEE clustering
    df = df.sort_values('UID').reset_index(drop=True)
    logger.info(f"✓ Sorted by UID for GEE clustering")

    # Build design matrix
    X = df[['Congruent_vs_Common', 'Incongruent_vs_Common', 'Days',
            'Congruent_x_Days', 'Incongruent_x_Days']]
    X = sm.add_constant(X)  # Add intercept
    y = df['HCE_flag']
    groups = df['UID']

    logger.info(f"✓ Design matrix: {X.shape[0]} observations × {X.shape[1]} predictors")

    # Fit GEE model
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Fit GEE Model")
    logger.info("="*80)
    logger.info("Model specification:")
    logger.info("  Family: Binomial (logit link)")
    logger.info("  Correlation: Exchangeable")
    logger.info("  Clustering: By participant (UID)")

    try:
        gee_model = GEE(
            endog=y,
            exog=X,
            groups=groups,
            family=Binomial(),
            cov_struct=Exchangeable()
        )

        logger.info("Fitting GEE model...")
        gee_result = gee_model.fit()
        logger.info("✓ GEE model converged successfully")

    except Exception as e:
        logger.error(f"GEE model failed to converge: {e}")
        logger.error("Check data structure and correlation specification")
        sys.exit(1)

    # Extract results
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Extract Fixed Effect Estimates")
    logger.info("="*80)

    results_df = pd.DataFrame({
        'Effect': [
            'Intercept (Common at Day 0)',
            'Congruent vs Common',
            'Incongruent vs Common',
            'Time (Days)',
            'Congruent × Time',
            'Incongruent × Time'
        ],
        'Beta': gee_result.params.values,
        'SE': gee_result.bse.values,
        'z': gee_result.tvalues.values,
        'p_value': gee_result.pvalues.values,
        'CI_lower': gee_result.conf_int()[0].values,
        'CI_upper': gee_result.conf_int()[1].values
    })

    # Add significance flags
    results_df['Sig'] = results_df['p_value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )

    # Save results
    output_file = Path('data/step03b_gee_results.csv')
    results_df.to_csv(output_file, index=False, float_format='%.6f')
    logger.info(f"✓ Saved fixed effect estimates to {output_file}")

    # Print results table
    logger.info("\nFixed Effect Estimates:")
    logger.info("-" * 100)
    for _, row in results_df.iterrows():
        logger.info(f"{row['Effect']:30s} β={row['Beta']:8.4f} SE={row['SE']:7.4f} "
                   f"z={row['z']:7.3f} p={row['p_value']:.6f} {row['Sig']}")

    # Post-hoc contrasts with Bonferroni correction
    logger.info("\n" + "="*80)
    logger.info("STEP 5: Post-Hoc Pairwise Contrasts")
    logger.info("="*80)

    # Extract main effect coefficients for contrasts
    beta_congruent = gee_result.params['Congruent_vs_Common']
    beta_incongruent = gee_result.params['Incongruent_vs_Common']
    se_congruent = gee_result.bse['Congruent_vs_Common']
    se_incongruent = gee_result.bse['Incongruent_vs_Common']

    # Contrast 1: Incongruent vs Common (already in model)
    contrast1_z = gee_result.tvalues['Incongruent_vs_Common']
    contrast1_p = gee_result.pvalues['Incongruent_vs_Common']

    # Contrast 2: Congruent vs Common (already in model)
    contrast2_z = gee_result.tvalues['Congruent_vs_Common']
    contrast2_p = gee_result.pvalues['Congruent_vs_Common']

    # Contrast 3: Incongruent vs Congruent
    # Beta_diff = beta_incongruent - beta_congruent
    # SE_diff = sqrt(SE_incong^2 + SE_cong^2 - 2*Cov(incong, cong))
    # For GEE, we can approximate SE assuming independence (conservative)
    contrast3_beta = beta_incongruent - beta_congruent
    contrast3_se = np.sqrt(se_incongruent**2 + se_congruent**2)  # Conservative estimate
    contrast3_z = contrast3_beta / contrast3_se
    contrast3_p = 2 * (1 - np.abs(contrast3_z))  # Two-tailed (approximate)

    # Note: For exact p-value, we'd use scipy.stats.norm
    from scipy.stats import norm
    contrast3_p = 2 * (1 - norm.cdf(np.abs(contrast3_z)))

    # Create contrasts dataframe
    contrasts_df = pd.DataFrame({
        'Contrast': [
            'Incongruent vs Common',
            'Congruent vs Common',
            'Incongruent vs Congruent'
        ],
        'Estimate': [
            beta_incongruent,
            beta_congruent,
            contrast3_beta
        ],
        'SE': [
            se_incongruent,
            se_congruent,
            contrast3_se
        ],
        'z': [
            contrast1_z,
            contrast2_z,
            contrast3_z
        ],
        'p_uncorrected': [
            contrast1_p,
            contrast2_p,
            contrast3_p
        ]
    })

    # Apply Bonferroni correction (3 contrasts)
    contrasts_df['p_bonferroni'] = contrasts_df['p_uncorrected'] * 3
    contrasts_df['p_bonferroni'] = contrasts_df['p_bonferroni'].clip(upper=1.0)

    # Add significance flags
    contrasts_df['Sig_uncorr'] = contrasts_df['p_uncorrected'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )
    contrasts_df['Sig_bonf'] = contrasts_df['p_bonferroni'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )

    # Save contrasts
    contrasts_file = Path('data/step03b_gee_contrasts.csv')
    contrasts_df.to_csv(contrasts_file, index=False, float_format='%.6f')
    logger.info(f"✓ Saved post-hoc contrasts to {contrasts_file}")

    # Print contrasts table
    logger.info("\nPairwise Contrasts (Bonferroni correction for 3 tests):")
    logger.info("-" * 110)
    logger.info(f"{'Contrast':30s} {'Estimate':>10s} {'SE':>8s} {'z':>8s} "
               f"{'p_uncorr':>10s} {'p_bonf':>10s} {'Sig':>5s}")
    logger.info("-" * 110)
    for _, row in contrasts_df.iterrows():
        logger.info(f"{row['Contrast']:30s} {row['Estimate']:10.4f} {row['SE']:8.4f} "
                   f"{row['z']:8.3f} {row['p_uncorrected']:10.6f} {row['p_bonferroni']:10.6f} "
                   f"{row['Sig_bonf']:>5s}")

    # Save full model summary
    logger.info("\n" + "="*80)
    logger.info("STEP 6: Save Full Model Summary")
    logger.info("="*80)

    summary_file = Path('data/step03b_gee_model_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("RQ 6.5.3 - GEE Validation for High-Confidence Errors\n")
        f.write("="*80 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Sample Size: {len(df):,} item-responses from {df['UID'].nunique()} participants\n")
        f.write(f"HCE Events: {df['HCE_flag'].sum():,} ({df['HCE_flag'].mean()*100:.2f}%)\n")
        f.write("\n")
        f.write(str(gee_result.summary()))
        f.write("\n\n")
        f.write("Post-Hoc Contrasts (Bonferroni Corrected):\n")
        f.write("-" * 80 + "\n")
        f.write(contrasts_df.to_string(index=False))

    logger.info(f"✓ Saved full model summary to {summary_file}")

    # Compare to LPM results
    logger.info("\n" + "="*80)
    logger.info("STEP 7: Compare GEE vs LPM Results")
    logger.info("="*80)

    # Load original LPM results
    lpm_file = Path('data/step03_congruence_hce_test.csv')
    if lpm_file.exists():
        lpm_df = pd.read_csv(lpm_file)
        logger.info("LPM Results (from original analysis):")
        logger.info(lpm_df.to_string(index=False))

        # Key comparison: Incongruent vs Common
        lpm_incong_p = 0.043  # From summary.md
        gee_incong_p = contrasts_df.loc[
            contrasts_df['Contrast'] == 'Incongruent vs Common', 'p_uncorrected'
        ].values[0]

        logger.info("\nKey Comparison (Incongruent vs Common):")
        logger.info(f"  LPM: p_uncorrected = {lpm_incong_p:.3f}, p_bonferroni = .130")
        logger.info(f"  GEE: p_uncorrected = {gee_incong_p:.6f}, p_bonferroni = "
                   f"{contrasts_df.loc[contrasts_df['Contrast']=='Incongruent vs Common', 'p_bonferroni'].values[0]:.6f}")

        if gee_incong_p < 0.05 and contrasts_df.loc[contrasts_df['Contrast']=='Incongruent vs Common', 'p_bonferroni'].values[0] < 0.05:
            logger.info("  → GEE reveals SIGNIFICANT effect (NULL overturned)")
        elif gee_incong_p < 0.05 and contrasts_df.loc[contrasts_df['Contrast']=='Incongruent vs Common', 'p_bonferroni'].values[0] >= 0.05:
            logger.info("  → GEE confirms marginal effect (does not survive Bonferroni)")
        else:
            logger.info("  → GEE confirms NULL result (consistent with LPM)")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY: GEE Validation Complete")
    logger.info("="*80)

    logger.info(f"✓ GEE model converged successfully")
    logger.info(f"✓ Fixed effects: {len(results_df)} parameters estimated")
    logger.info(f"✓ Post-hoc contrasts: 3 pairwise comparisons with Bonferroni correction")

    # Determine outcome
    any_significant = (contrasts_df['p_bonferroni'] < 0.05).any()
    if any_significant:
        sig_contrasts = contrasts_df[contrasts_df['p_bonferroni'] < 0.05]['Contrast'].tolist()
        logger.info(f"✓ SIGNIFICANT contrasts (p_bonf < 0.05): {', '.join(sig_contrasts)}")
    else:
        logger.info(f"✓ NULL RESULT confirmed: No contrasts significant after Bonferroni correction")

    logger.info("\nFiles created:")
    logger.info(f"  1. {output_file} (fixed effect estimates)")
    logger.info(f"  2. {contrasts_file} (post-hoc contrasts)")
    logger.info(f"  3. {summary_file} (full model summary)")
    logger.info(f"  4. {log_file} (analysis log)")

    logger.info("\n" + "="*80)
    logger.info("GEE VALIDATION COMPLETE")
    logger.info("="*80)

if __name__ == '__main__':
    main()
