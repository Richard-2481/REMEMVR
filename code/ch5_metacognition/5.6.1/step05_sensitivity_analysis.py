#!/usr/bin/env python3
"""Sensitivity Analysis (Model Specification Robustness): Verify robustness of primary finding (HCE rate decreases over time) across"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for LMM
import statsmodels.formula.api as smf
from scipy import stats

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_sensitivity_analysis.log"

# Logging Function

def log(msg, flush=True):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg, flush=flush)

# Main Analysis

if __name__ == "__main__":
    try:
        # Clear old log
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w') as f:
            f.write("")

        log("Step 05: Sensitivity Analysis")
        log("=" * 70)
        # LOAD DATA
        log("Loading HCE rates from Step 01...")
        input_path = RQ_DIR / "data" / "step01_hce_rates.csv"
        df_hce = pd.read_csv(input_path)
        log(f"{len(df_hce)} rows, {df_hce['UID'].nunique()} participants")

        # Create Days variable
        df_hce['Days'] = df_hce['TSVR'] / 24.0
        df_hce['Days_sq'] = df_hce['Days'] ** 2
        log(f"Days range: [{df_hce['Days'].min():.2f}, {df_hce['Days'].max():.2f}]")

        # Store results
        results = []
        # MODEL A: Full Model (Reference - Random Intercepts + Slopes)
        log("\n" + "=" * 70)
        log("[MODEL A] Full Model: HCE_rate ~ Days + (Days | UID)")
        log("=" * 70)

        model_a = smf.mixedlm(
            formula='HCE_rate ~ Days',
            data=df_hce,
            groups=df_hce['UID'],
            re_formula='~Days'  # Random intercepts AND random slopes
        )
        result_a = model_a.fit(method='powell', reml=True)

        log(f"[MODEL A] Converged: {result_a.converged}")
        log(f"[MODEL A] Days β = {result_a.params['Days']:.6f}")
        log(f"[MODEL A] Days SE = {result_a.bse['Days']:.6f}")
        log(f"[MODEL A] Days p = {result_a.pvalues['Days']:.6f}")
        log(f"[MODEL A] Random slope var = {result_a.cov_re.iloc[1,1]:.6f}")

        results.append({
            'model': 'A_Full_RandomSlopes',
            'formula': 'HCE_rate ~ Days + (Days | UID)',
            'days_coef': result_a.params['Days'],
            'days_se': result_a.bse['Days'],
            'days_p': result_a.pvalues['Days'],
            'converged': result_a.converged,
            'n_obs': result_a.nobs,
            'llf': result_a.llf,
            'note': 'REFERENCE MODEL'
        })
        # MODEL B: Reduced Model (Random Intercepts Only)
        log("\n" + "=" * 70)
        log("[MODEL B] Reduced Model: HCE_rate ~ Days + (1 | UID)")
        log("=" * 70)

        model_b = smf.mixedlm(
            formula='HCE_rate ~ Days',
            data=df_hce,
            groups=df_hce['UID'],
            re_formula='~1'  # Random intercepts ONLY (no random slopes)
        )
        result_b = model_b.fit(method='powell', reml=True)

        log(f"[MODEL B] Converged: {result_b.converged}")
        log(f"[MODEL B] Days β = {result_b.params['Days']:.6f}")
        log(f"[MODEL B] Days SE = {result_b.bse['Days']:.6f}")
        log(f"[MODEL B] Days p = {result_b.pvalues['Days']:.6f}")

        results.append({
            'model': 'B_Reduced_InterceptOnly',
            'formula': 'HCE_rate ~ Days + (1 | UID)',
            'days_coef': result_b.params['Days'],
            'days_se': result_b.bse['Days'],
            'days_p': result_b.pvalues['Days'],
            'converged': result_b.converged,
            'n_obs': result_b.nobs,
            'llf': result_b.llf,
            'note': 'Tests if random slopes necessary'
        })

        # Compare Model A vs B (LRT for random slopes)
        lrt_ab = -2 * (result_b.llf - result_a.llf)
        df_lrt_ab = 2  # Random slope variance + covariance
        p_lrt_ab = 1 - stats.chi2.cdf(lrt_ab, df_lrt_ab)
        log(f"[LRT A vs B] Chi² = {lrt_ab:.4f}, df=2, p = {p_lrt_ab:.6f}")
        if p_lrt_ab < 0.05:
            log("[LRT A vs B] Random slopes SIGNIFICANT - keep Model A")
        else:
            log("[LRT A vs B] Random slopes NOT significant - Model B adequate")
        # MODEL C: Quadratic Time Effect
        log("\n" + "=" * 70)
        log("[MODEL C] Quadratic Model: HCE_rate ~ Days + Days² + (Days | UID)")
        log("=" * 70)

        model_c = smf.mixedlm(
            formula='HCE_rate ~ Days + Days_sq',
            data=df_hce,
            groups=df_hce['UID'],
            re_formula='~Days'
        )
        result_c = model_c.fit(method='powell', reml=True)

        log(f"[MODEL C] Converged: {result_c.converged}")
        log(f"[MODEL C] Days β = {result_c.params['Days']:.6f}")
        log(f"[MODEL C] Days² β = {result_c.params['Days_sq']:.6f}")
        log(f"[MODEL C] Days p = {result_c.pvalues['Days']:.6f}")
        log(f"[MODEL C] Days² p = {result_c.pvalues['Days_sq']:.6f}")

        results.append({
            'model': 'C_Quadratic',
            'formula': 'HCE_rate ~ Days + Days² + (Days | UID)',
            'days_coef': result_c.params['Days'],
            'days_se': result_c.bse['Days'],
            'days_p': result_c.pvalues['Days'],
            'converged': result_c.converged,
            'n_obs': result_c.nobs,
            'llf': result_c.llf,
            'note': f'Days² β={result_c.params["Days_sq"]:.6f}, p={result_c.pvalues["Days_sq"]:.4f}'
        })
        # MODEL D: Exclude Late-Tested Participants
        log("\n" + "=" * 70)
        log("[MODEL D] Exclude Late: HCE_rate ~ Days + (Days | UID), Days ≤ 7.5")
        log("=" * 70)

        df_early = df_hce[df_hce['Days'] <= 7.5].copy()
        n_excluded = len(df_hce) - len(df_early)
        log(f"[MODEL D] Excluding {n_excluded} observations with Days > 7.5")
        log(f"[MODEL D] Remaining: {len(df_early)} observations")

        if len(df_early) >= 300:  # Need sufficient data
            model_d = smf.mixedlm(
                formula='HCE_rate ~ Days',
                data=df_early,
                groups=df_early['UID'],
                re_formula='~Days'
            )
            result_d = model_d.fit(method='powell', reml=True)

            log(f"[MODEL D] Converged: {result_d.converged}")
            log(f"[MODEL D] Days β = {result_d.params['Days']:.6f}")
            log(f"[MODEL D] Days SE = {result_d.bse['Days']:.6f}")
            log(f"[MODEL D] Days p = {result_d.pvalues['Days']:.6f}")

            results.append({
                'model': 'D_ExcludeLate',
                'formula': 'HCE_rate ~ Days + (Days | UID), Days ≤ 7.5',
                'days_coef': result_d.params['Days'],
                'days_se': result_d.bse['Days'],
                'days_p': result_d.pvalues['Days'],
                'converged': result_d.converged,
                'n_obs': result_d.nobs,
                'llf': result_d.llf,
                'note': f'Excluded {n_excluded} late observations'
            })
        else:
            log(f"[MODEL D] Skipped - insufficient data after exclusion")
        # COMPARE RESULTS
        log("\n" + "=" * 70)
        log("Days Coefficient Across Models")
        log("=" * 70)

        df_results = pd.DataFrame(results)

        # Reference coefficient from Model A
        ref_coef = results[0]['days_coef']

        for _, row in df_results.iterrows():
            pct_diff = ((row['days_coef'] - ref_coef) / abs(ref_coef)) * 100
            sig_label = "***" if row['days_p'] < 0.001 else ("**" if row['days_p'] < 0.01 else ("*" if row['days_p'] < 0.05 else ""))
            log(f"  {row['model']:30s}: β={row['days_coef']:.6f} (SE={row['days_se']:.6f}) p={row['days_p']:.4f}{sig_label} [{pct_diff:+.1f}% vs ref]")
        # ROBUSTNESS ASSESSMENT
        log("\n" + "=" * 70)
        log("Assessment")
        log("=" * 70)

        # Check 1: All models show significant negative effect?
        all_significant = all(row['days_p'] < 0.05 for row in results)
        all_negative = all(row['days_coef'] < 0 for row in results)

        log(f"  All models significant (p < 0.05): {all_significant}")
        log(f"  All models negative coefficient: {all_negative}")

        # Check 2: Coefficients within 50% of reference?
        max_deviation = max(abs((row['days_coef'] - ref_coef) / ref_coef) * 100 for row in results)
        within_tolerance = max_deviation < 50

        log(f"  Max deviation from reference: {max_deviation:.1f}%")
        log(f"  Within 50% tolerance: {within_tolerance}")

        # Overall robustness
        robust = all_significant and all_negative and within_tolerance

        if robust:
            log("\n✓ PASS - Primary finding robust across all specifications")
        else:
            log("\n⚠ WARNING - Sensitivity concerns detected")
        # SAVE RESULTS
        output_path = RQ_DIR / "data" / "step05_sensitivity_results.csv"
        df_results.to_csv(output_path, index=False)
        log(f"\nstep05_sensitivity_results.csv")

        # Summary for validation.md
        log("\n" + "=" * 70)
        log("For validation.md")
        log("=" * 70)
        log(f"Sensitivity Analysis: {'PASS' if robust else 'WARNING'}")
        log(f"Models tested: {len(results)}")
        log(f"Reference (Model A): β={ref_coef:.6f}, p={results[0]['days_p']:.6f}")
        log(f"All models significant: {all_significant}")
        log(f"All coefficients negative: {all_negative}")
        log(f"Maximum deviation: {max_deviation:.1f}%")
        log(f"Random slopes significant (LRT): p={p_lrt_ab:.4f}")

        log("\nStep 05 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
