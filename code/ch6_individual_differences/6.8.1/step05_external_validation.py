#!/usr/bin/env python3
"""External Validation: Validate profile distinctions using external cognitive tests (age, NART, RPM,"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.1
LOG_FILE = RQ_DIR / "logs" / "step05_external_validation.log"
OUTPUT_DIR = RQ_DIR / "data"

# Inputs
INPUT_PROFILES = OUTPUT_DIR / 'step03_optimal_profiles.csv'
INPUT_COGNITIVE = PROJECT_ROOT / 'data' / 'dfnonvr.csv'

# Outputs
OUTPUT_VALIDATION = OUTPUT_DIR / 'step05_external_validation.csv'
OUTPUT_POSTHOC = OUTPUT_DIR / 'step05_external_validation_posthoc.csv'

# Validation Variables (exact column names from DATA_DICTIONARY.md)
EXTERNAL_VALIDATORS = [
    'age',
    'nart-score',
    'rpm-score',
    'RAVLT_Total',  # Computed from ravlt-trial-1-score through ravlt-trial-5-score
    'BVMT_Total',   # Computed from bvmt-trial-1-score through bvmt-trial-3-score
    'RAVLT_Pct_Ret',  # DR / max(trials 1-5) * 100
    'BVMT_Pct_Ret'    # Pre-computed from dfnonvr.csv
]

# Statistical Parameters (Decision D068)
ALPHA_UNCORRECTED = 0.05
ALPHA_BONFERRONI = 0.05 / 7  # 7 validators
FDR_METHOD = 'fdr_bh'  # Benjamini-Hochberg

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Helper Functions

def compute_eta_squared(f_stat, df_between, df_within):
    """
    Compute eta-squared effect size for ANOVA.

    Formula: eta^2 = SS_between / SS_total
             = df_between * F / (df_between * F + df_within)

    Interpretation:
    - eta^2 = 0.01: Small effect
    - eta^2 = 0.06: Medium effect
    - eta^2 = 0.14: Large effect
    """
    eta_squared = (df_between * f_stat) / (df_between * f_stat + df_within)
    return eta_squared

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: External Validation")
        # Load Profile Assignments

        log("\nLoading profile assignments...")
        log(f"{INPUT_PROFILES}")

        df_profiles = pd.read_csv(INPUT_PROFILES)
        log(f"{len(df_profiles)} participants with profile assignments")
        # Load Cognitive Test Data

        log(f"\nLoading cognitive test data from dfnonvr.csv...")
        log(f"{INPUT_COGNITIVE}")

        df_cognitive = pd.read_csv(INPUT_COGNITIVE)
        log(f"{len(df_cognitive)} rows")
        log(f"First 20 columns: {df_cognitive.columns.tolist()[:20]}")

        # RAVLT Ceiling Fix: if trial N == 0 and trial N-1 >= 14, set trial N = 15
        # Known affected: A064, A070, A077, A103 (7 total fixes)
        log("[CEILING FIX] Applying RAVLT ceiling fix...")
        ravlt_trial_cols_fix = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
        fixes_applied = 0
        for idx in df_cognitive.index:
            for i in range(1, 5):
                current_col = ravlt_trial_cols_fix[i]
                prev_col = ravlt_trial_cols_fix[i - 1]
                if df_cognitive.at[idx, current_col] == 0 and df_cognitive.at[idx, prev_col] >= 14:
                    uid = df_cognitive.at[idx, 'UID']
                    df_cognitive.at[idx, current_col] = 15
                    fixes_applied += 1
                    log(f"[CEILING FIX] {uid}: {current_col} 0 -> 15 (prev trial = {df_cognitive.at[idx, prev_col]})")
        log(f"[CEILING FIX] Total fixes applied: {fixes_applied}")

        # Scientific Mantra Checkpoint
        log("\nCognitive data validation")

        # Check required columns exist
        required_cols = ['UID', 'age', 'nart-score', 'rpm-score']
        missing_cols = [col for col in required_cols if col not in df_cognitive.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns in dfnonvr.csv: {missing_cols}")

        log(f"Required base columns present")
        # Compute Composite Scores (RAVLT Total, BVMT Total)

        log("\nComputing composite scores...")

        # RAVLT Total: Sum trials 1-5 ONLY (exclude distraction trial)
        log("Computing RAVLT Total (trials 1-5)...")

        ravlt_trial_cols = []
        for i in range(1, 6):
            col = f'ravlt-trial-{i}-score'
            if col in df_cognitive.columns:
                ravlt_trial_cols.append(col)
            else:
                log(f"RAVLT trial column not found: {col}")

        if len(ravlt_trial_cols) != 5:
            raise ValueError(f"Expected 5 RAVLT trial columns, found {len(ravlt_trial_cols)}")

        df_cognitive['RAVLT_Total'] = df_cognitive[ravlt_trial_cols].sum(axis=1)
        log(f"Computed RAVLT Total from {len(ravlt_trial_cols)} trials")
        log(f"RAVLT_Total range: [{df_cognitive['RAVLT_Total'].min()}, {df_cognitive['RAVLT_Total'].max()}]")

        # BVMT Total: Sum trials 1-3
        log("Computing BVMT Total (trials 1-3)...")

        bvmt_trial_cols = []
        for i in range(1, 4):
            col = f'bvmt-trial-{i}-score'
            if col in df_cognitive.columns:
                bvmt_trial_cols.append(col)
            else:
                log(f"BVMT trial column not found: {col}")

        if len(bvmt_trial_cols) != 3:
            raise ValueError(f"Expected 3 BVMT trial columns, found {len(bvmt_trial_cols)}")

        df_cognitive['BVMT_Total'] = df_cognitive[bvmt_trial_cols].sum(axis=1)
        log(f"Computed BVMT Total from {len(bvmt_trial_cols)} trials")
        log(f"BVMT_Total range: [{df_cognitive['BVMT_Total'].min()}, {df_cognitive['BVMT_Total'].max()}]")

        # RAVLT Percent Retention: DR / max(trials 1-5 after ceiling fix) * 100
        dr_col = 'ravlt-delayed-recall-score'
        ravlt_max_trial = df_cognitive[ravlt_trial_cols].max(axis=1)
        df_cognitive['RAVLT_Pct_Ret'] = np.where(
            ravlt_max_trial > 0,
            (df_cognitive[dr_col] / ravlt_max_trial) * 100,
            np.nan
        )
        log(f"RAVLT_Pct_Ret range: [{df_cognitive['RAVLT_Pct_Ret'].min():.1f}, {df_cognitive['RAVLT_Pct_Ret'].max():.1f}]")

        # BVMT Percent Retained: pre-computed column from dfnonvr.csv
        df_cognitive['BVMT_Pct_Ret'] = df_cognitive['bvmt-percent-retained']
        log(f"BVMT_Pct_Ret range: [{df_cognitive['BVMT_Pct_Ret'].min():.1f}, {df_cognitive['BVMT_Pct_Ret'].max():.1f}]")

        # Scientific Mantra Checkpoint
        log("\nComposite score validation")
        log(f"RAVLT and BVMT totals and percent retention computed")
        # Merge Profiles with Cognitive Data

        log("\nMerging profiles with cognitive test data...")

        df_merged = pd.merge(df_profiles, df_cognitive, on='UID', how='inner')
        log(f"{len(df_merged)} participants (expected: {len(df_profiles)})")

        if len(df_merged) != len(df_profiles):
            log(f"Merge resulted in {len(df_profiles) - len(df_merged)} missing participants")

        # Check data completeness for external validators
        log("\nData completeness for external validators:")
        for var in EXTERNAL_VALIDATORS:
            if var not in df_merged.columns:
                raise ValueError(f"Validator '{var}' not found in merged data")

            missing_count = df_merged[var].isnull().sum()
            if missing_count > 0:
                log(f"{var}: {missing_count} missing values ({100*missing_count/len(df_merged):.1f}%)")
            else:
                log(f"[OK] {var}: No missing values")

        # Scientific Mantra Checkpoint
        log("\nMerge validation")
        log(f"Profiles merged with cognitive data")
        # Perform One-Way ANOVAs

        log(f"\nPerforming one-way ANOVAs for external validation...")
        log(f"Alpha (uncorrected): {ALPHA_UNCORRECTED}")
        log(f"Alpha (Bonferroni): {ALPHA_BONFERRONI} (0.05/{len(EXTERNAL_VALIDATORS)})")

        anova_results = []

        for var in EXTERNAL_VALIDATORS:
            log(f"\nTesting: {var}")

            # Remove missing values
            df_var = df_merged[['Profile', var]].dropna()
            log(f"N={len(df_var)} (after removing missing values)")

            # Group data by profile
            groups = [group[var].values for name, group in df_var.groupby('Profile')]
            n_profiles = len(groups)

            log(f"{n_profiles} profiles")

            # Check for single profile (K=1) - ANOVA requires at least 2 groups
            if n_profiles == 1:
                log(f"Only 1 profile - ANOVA requires at least 2 groups")
                log(f"Assigning NA values for single-profile solution")
                f_stat = np.nan
                p_uncorrected = 1.0  # No difference when all in one group
                df_between = 0
                df_within = len(df_var) - 1
                eta_squared = 0.0  # No variance between groups
            else:
                # One-way ANOVA (for K>=2)
                f_stat, p_uncorrected = stats.f_oneway(*groups)
                # Degrees of freedom
                df_between = n_profiles - 1
                df_within = len(df_var) - n_profiles
                # Effect size (eta-squared)
                eta_squared = compute_eta_squared(f_stat, df_between, df_within)

            log(f"F({df_between}, {df_within}) = {f_stat if not np.isnan(f_stat) else 'NA'}, p = {p_uncorrected:.4f}, eta^2 = {eta_squared:.4f}")

            anova_results.append({
                'Variable': var,
                'F_stat': f_stat,
                'df_between': df_between,
                'df_within': df_within,
                'p_uncorrected': p_uncorrected,
                'eta_squared': eta_squared,
                'Test_Type': 'One-Way ANOVA'
            })

        # Scientific Mantra Checkpoint
        log("\nANOVA execution validation")
        log(f"{len(anova_results)} ANOVAs performed")
        # Apply Multiple Comparison Corrections (Decision D068)

        log("\nApplying multiple comparison corrections...")

        # Extract p-values
        p_values = [r['p_uncorrected'] for r in anova_results]

        # Bonferroni correction
        p_bonferroni = [p * len(p_values) for p in p_values]
        p_bonferroni = [min(p, 1.0) for p in p_bonferroni]  # Cap at 1.0

        # FDR correction (Benjamini-Hochberg)
        reject_fdr, p_fdr, _, _ = multipletests(p_values, alpha=ALPHA_UNCORRECTED, method=FDR_METHOD)

        # Add corrections to results
        for i, result in enumerate(anova_results):
            result['p_bonferroni'] = p_bonferroni[i]
            result['p_fdr'] = p_fdr[i]

        log("Corrections applied:")
        log(f"  Bonferroni alpha: {ALPHA_BONFERRONI}")
        log(f"  FDR method: {FDR_METHOD}")

        # Log significance results (Decision D068: report BOTH uncorrected and corrected)
        log("\nResults summary (Decision D068 dual p-values):")
        for result in anova_results:
            sig_uncorrected = "sig" if result['p_uncorrected'] < ALPHA_UNCORRECTED else "ns"
            sig_bonferroni = "sig" if result['p_bonferroni'] < ALPHA_BONFERRONI else "ns"

            log(f"{result['Variable']:15} p_uncorrected={result['p_uncorrected']:.4f} ({sig_uncorrected}), "
                f"p_bonferroni={result['p_bonferroni']:.4f} ({sig_bonferroni}), "
                f"eta^2={result['eta_squared']:.4f}")

        # Scientific Mantra Checkpoint
        log("\nMultiple comparison correction validation")
        log(f"Dual p-values computed (Decision D068 compliance)")
        # Save ANOVA Results

        log(f"\nSaving ANOVA results to {OUTPUT_VALIDATION}")

        df_anova = pd.DataFrame(anova_results)
        df_anova = df_anova[['Variable', 'F_stat', 'p_uncorrected', 'p_bonferroni', 'p_fdr', 'eta_squared', 'Test_Type']]

        df_anova.to_csv(OUTPUT_VALIDATION, index=False, encoding='utf-8')

        log(f"{len(df_anova)} rows")
        log(f"\n[ANOVA TABLE]")
        log(f"{df_anova.to_string(index=False)}")
        # Post-Hoc Tukey HSD (for significant ANOVAs)

        log("\n[POST-HOC] Performing Tukey HSD for significant ANOVAs...")

        # Identify significant results (use Bonferroni-corrected threshold)
        sig_results = [r for r in anova_results if r['p_bonferroni'] < ALPHA_BONFERRONI]

        log(f"{len(sig_results)} significant ANOVAs (Bonferroni-corrected)")

        posthoc_results = []

        for result in sig_results:
            var = result['Variable']
            log(f"\nPost-hoc for: {var}")

            # Prepare data for Tukey HSD
            df_var = df_merged[['Profile', var]].dropna()

            # Perform Tukey HSD
            tukey = pairwise_tukeyhsd(
                endog=df_var[var],
                groups=df_var['Profile'],
                alpha=ALPHA_UNCORRECTED
            )

            # Extract results
            tukey_summary = tukey.summary()

            log(f"{var} pairwise comparisons:")
            log(f"{tukey_summary}")

            # Parse Tukey results (convert to DataFrame)
            tukey_data = tukey.summary().data[1:]  # Skip header row

            for row in tukey_data:
                posthoc_results.append({
                    'Variable': var,
                    'Comparison': f"{row[0]} vs {row[1]}",
                    'Mean_Diff': float(row[2]),
                    'SE': float(row[3]),  # Standard error
                    'p_tukey': float(row[5])  # Adjusted p-value
                })

        # Save post-hoc results
        if posthoc_results:
            log(f"\nSaving post-hoc results to {OUTPUT_POSTHOC}")

            df_posthoc = pd.DataFrame(posthoc_results)
            df_posthoc.to_csv(OUTPUT_POSTHOC, index=False, encoding='utf-8')

            log(f"{len(df_posthoc)} pairwise comparisons")
        else:
            log(f"\nNo significant ANOVAs - no post-hoc tests performed")
            log(f"Creating empty post-hoc results file")

            # Create empty DataFrame with expected columns
            df_posthoc = pd.DataFrame(columns=['Variable', 'Comparison', 'Mean_Diff', 'SE', 'p_tukey'])
            df_posthoc.to_csv(OUTPUT_POSTHOC, index=False, encoding='utf-8')

        # Scientific Mantra Checkpoint
        log("\nPost-hoc validation")
        log(f"Post-hoc tests completed for significant results")
        # VALIDATION: Final Checks

        log("\nFinal output validation...")

        # Check 1: All validators tested
        if len(anova_results) != len(EXTERNAL_VALIDATORS):
            raise ValueError(f"Expected {len(EXTERNAL_VALIDATORS)} ANOVAs, performed {len(anova_results)}")
        log(f"All {len(EXTERNAL_VALIDATORS)} validators tested")

        # Check 2: P-values in valid range
        for result in anova_results:
            if not (0 <= result['p_uncorrected'] <= 1):
                raise ValueError(f"Invalid p_uncorrected for {result['Variable']}: {result['p_uncorrected']}")
            if not (0 <= result['p_bonferroni'] <= 1):
                raise ValueError(f"Invalid p_bonferroni for {result['Variable']}: {result['p_bonferroni']}")
        log(f"P-values in valid range [0, 1]")

        # Check 3: Effect sizes non-negative
        for result in anova_results:
            if result['eta_squared'] < 0:
                raise ValueError(f"Negative eta-squared for {result['Variable']}: {result['eta_squared']}")
        log(f"Effect sizes valid")

        log("\nStep 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
