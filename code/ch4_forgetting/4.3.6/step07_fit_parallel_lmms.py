#!/usr/bin/env python3
# =============================================================================
# STEP 07: Fit Parallel LMMs and Compare AIC
# =============================================================================
"""
Step ID: step07
Step Name: Fit Parallel LMMs and Compare AIC
RQ: 5.3.6 - Purified CTT Effects (Paradigms)
Generated: 2025-12-04

PURPOSE:
Fit 9 parallel LMMs (3 paradigms x 3 measurement types) using random intercepts
only. Compare AIC values to determine which measurement type provides best model fit.

EXPECTED INPUTS:
- data/step06_standardized_scores.csv
  Columns: ['composite_ID', 'UID', 'test', 'TSVR_hours', 'z_theta_IFR', 'z_theta_ICR',
            'z_theta_IRE', 'z_CTT_full_IFR', 'z_CTT_full_ICR', 'z_CTT_full_IRE',
            'z_CTT_purified_IFR', 'z_CTT_purified_ICR', 'z_CTT_purified_IRE']
  Format: Z-standardized scores for parallel LMM fitting
  Expected rows: 400 (100 participants x 4 tests)

EXPECTED OUTPUTS:
- data/step07_lmm_model_comparison.csv
  Columns: ['paradigm', 'AIC_IRT', 'AIC_full', 'AIC_purified', 'delta_AIC_full_purified',
            'delta_AIC_purified_IRT', 'coef_intercept_IRT', 'coef_slope_IRT',
            'coef_intercept_full', 'coef_slope_full', 'coef_intercept_purified',
            'coef_slope_purified', 'slope_correlation_full_purified',
            'convergence_flag_IRT', 'convergence_flag_full', 'convergence_flag_purified',
            'random_structure']
  Format: Model comparison results
  Expected rows: 3 (one per paradigm)

- data/step07_lmm_convergence_report.txt
  Format: Plain text summary of convergence status and model comparison

VALIDATION CRITERIA:
- All convergence_flag_* = TRUE
- AIC values positive and finite
- All 3 paradigms present
- No NaN in coefficients

IMPLEMENTATION NOTES:
- Uses statsmodels.formula.api.mixedlm for LMM fitting
- Random structure: intercept only (1|UID) for convergence stability
- REML=False for valid AIC comparison across models
- Computes delta_AIC for model comparisons (positive = first model worse)
"""
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import statsmodels.formula.api as smf

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07_fit_parallel_lmms.log"

# =============================================================================
# Logging Function
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 07: Fit Parallel LMMs and Compare AIC")

        # =========================================================================
        # STEP 1: Load Standardized Scores
        # =========================================================================
        log("[LOAD] Loading standardized scores...")
        input_path = RQ_DIR / "data" / "step06_standardized_scores.csv"
        df = pd.read_csv(input_path)
        log(f"[LOADED] {len(df)} rows, {len(df.columns)} columns")

        # Validate expected columns
        required_cols = ['composite_ID', 'UID', 'test', 'TSVR_hours',
                        'z_theta_IFR', 'z_theta_ICR', 'z_theta_IRE',
                        'z_CTT_full_IFR', 'z_CTT_full_ICR', 'z_CTT_full_IRE',
                        'z_CTT_purified_IFR', 'z_CTT_purified_ICR', 'z_CTT_purified_IRE']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        log(f"[VALIDATION] All required columns present")

        # =========================================================================
        # STEP 2: Fit Parallel LMMs for Each Paradigm
        # =========================================================================
        log("[ANALYSIS] Fitting 9 parallel LMMs (3 paradigms x 3 measurement types)...")

        paradigms = ['IFR', 'ICR', 'IRE']
        measurement_types = ['IRT', 'full', 'purified']
        results = []

        for paradigm in paradigms:
            log(f"[FIT] Paradigm: {paradigm}")

            paradigm_results = {
                'paradigm': paradigm,
                'random_structure': 'intercept_only'
            }

            # Fit models for each measurement type
            for mtype in measurement_types:
                if mtype == 'IRT':
                    dv = f'z_theta_{paradigm}'
                else:
                    dv = f'z_CTT_{mtype}_{paradigm}'

                log(f"  [FIT] {mtype.upper()} model: {dv} ~ TSVR_hours + (1|UID)")

                # Fit LMM with random intercepts only
                try:
                    model = smf.mixedlm(f"{dv} ~ TSVR_hours", data=df, groups=df["UID"])
                    result = model.fit(reml=False)

                    # Extract results
                    aic = result.aic
                    converged = result.converged
                    intercept = result.fe_params['Intercept']
                    slope = result.fe_params['TSVR_hours']

                    # Store results
                    paradigm_results[f'AIC_{mtype}'] = aic
                    paradigm_results[f'convergence_flag_{mtype}'] = converged
                    paradigm_results[f'coef_intercept_{mtype}'] = intercept
                    paradigm_results[f'coef_slope_{mtype}'] = slope

                    log(f"    [RESULT] AIC={aic:.2f}, converged={converged}, "
                        f"intercept={intercept:.4f}, slope={slope:.4f}")

                except Exception as e:
                    log(f"    [ERROR] Model fitting failed: {e}")
                    paradigm_results[f'AIC_{mtype}'] = np.nan
                    paradigm_results[f'convergence_flag_{mtype}'] = False
                    paradigm_results[f'coef_intercept_{mtype}'] = np.nan
                    paradigm_results[f'coef_slope_{mtype}'] = np.nan

            results.append(paradigm_results)

        # Create results DataFrame
        df_results = pd.DataFrame(results)

        # =========================================================================
        # STEP 3: Compute Model Comparisons
        # =========================================================================
        log("[COMPARISON] Computing AIC differences...")

        # Compute delta AICs
        # Positive delta = first model worse (higher AIC)
        df_results['delta_AIC_full_purified'] = (
            df_results['AIC_full'] - df_results['AIC_purified']
        )
        df_results['delta_AIC_purified_IRT'] = (
            df_results['AIC_purified'] - df_results['AIC_IRT']
        )

        log("[COMPARISON] AIC differences computed:")
        for _, row in df_results.iterrows():
            log(f"  {row['paradigm']}: delta(Full-Purified)={row['delta_AIC_full_purified']:.2f}, "
                f"delta(Purified-IRT)={row['delta_AIC_purified_IRT']:.2f}")

        # =========================================================================
        # STEP 4: Compute Slope Correlation Between Full and Purified CTT
        # =========================================================================
        log("[COMPARISON] Computing slope correlation between Full and Purified CTT...")

        # Extract slopes for full and purified across paradigms
        slopes_full = df_results['coef_slope_full'].values
        slopes_purified = df_results['coef_slope_purified'].values

        # Compute Pearson correlation
        slope_corr = np.corrcoef(slopes_full, slopes_purified)[0, 1]
        df_results['slope_correlation_full_purified'] = slope_corr

        log(f"[COMPARISON] Slope correlation (Full vs Purified): r={slope_corr:.4f}")

        # =========================================================================
        # STEP 5: Save Model Comparison Results
        # =========================================================================
        log("[SAVE] Saving model comparison results...")

        # Reorder columns for clarity
        col_order = [
            'paradigm',
            'AIC_IRT', 'AIC_full', 'AIC_purified',
            'delta_AIC_full_purified', 'delta_AIC_purified_IRT',
            'coef_intercept_IRT', 'coef_slope_IRT',
            'coef_intercept_full', 'coef_slope_full',
            'coef_intercept_purified', 'coef_slope_purified',
            'slope_correlation_full_purified',
            'convergence_flag_IRT', 'convergence_flag_full', 'convergence_flag_purified',
            'random_structure'
        ]
        df_results = df_results[col_order]

        output_path = RQ_DIR / "data" / "step07_lmm_model_comparison.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"[SAVED] {output_path} ({len(df_results)} rows)")

        # =========================================================================
        # STEP 6: Generate Convergence Report
        # =========================================================================
        log("[REPORT] Generating convergence report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("LMM MODEL COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(f"Total models fitted: 9 (3 paradigms x 3 measurement types)")
        report_lines.append(f"Random structure: Random intercepts only (1|UID)")
        report_lines.append(f"REML: False (ML estimation for AIC comparison)")
        report_lines.append("")

        # Convergence summary
        report_lines.append("-" * 80)
        report_lines.append("CONVERGENCE SUMMARY")
        report_lines.append("-" * 80)
        all_converged = True
        for _, row in df_results.iterrows():
            paradigm = row['paradigm']
            conv_irt = row['convergence_flag_IRT']
            conv_full = row['convergence_flag_full']
            conv_purified = row['convergence_flag_purified']

            report_lines.append(f"{paradigm}:")
            report_lines.append(f"  IRT:      {'CONVERGED' if conv_irt else 'FAILED'}")
            report_lines.append(f"  Full:     {'CONVERGED' if conv_full else 'FAILED'}")
            report_lines.append(f"  Purified: {'CONVERGED' if conv_purified else 'FAILED'}")
            report_lines.append("")

            if not (conv_irt and conv_full and conv_purified):
                all_converged = False

        if all_converged:
            report_lines.append("[PASS] All models converged successfully")
        else:
            report_lines.append("[WARNING] Some models failed to converge")
        report_lines.append("")

        # AIC comparison
        report_lines.append("-" * 80)
        report_lines.append("AIC COMPARISON")
        report_lines.append("-" * 80)
        report_lines.append("Lower AIC = better model fit")
        report_lines.append("Positive delta_AIC = first model worse (higher AIC)")
        report_lines.append("")

        for _, row in df_results.iterrows():
            paradigm = row['paradigm']
            aic_irt = row['AIC_IRT']
            aic_full = row['AIC_full']
            aic_purified = row['AIC_purified']
            delta_fp = row['delta_AIC_full_purified']
            delta_pi = row['delta_AIC_purified_IRT']

            report_lines.append(f"{paradigm}:")
            report_lines.append(f"  AIC_IRT:      {aic_irt:8.2f}")
            report_lines.append(f"  AIC_Full:     {aic_full:8.2f}")
            report_lines.append(f"  AIC_Purified: {aic_purified:8.2f}")
            report_lines.append(f"  delta(Full-Purified): {delta_fp:+7.2f} "
                              f"({'Purified better' if delta_fp > 0 else 'Full better'})")
            report_lines.append(f"  delta(Purified-IRT):  {delta_pi:+7.2f} "
                              f"({'IRT better' if delta_pi > 0 else 'Purified better'})")
            report_lines.append("")

        # Fixed effects
        report_lines.append("-" * 80)
        report_lines.append("FIXED EFFECTS COEFFICIENTS")
        report_lines.append("-" * 80)
        report_lines.append("Model: z_score ~ TSVR_hours + (1|UID)")
        report_lines.append("")

        for _, row in df_results.iterrows():
            paradigm = row['paradigm']
            report_lines.append(f"{paradigm}:")
            report_lines.append(f"  IRT:")
            report_lines.append(f"    Intercept:  {row['coef_intercept_IRT']:7.4f}")
            report_lines.append(f"    TSVR_hours: {row['coef_slope_IRT']:7.4f}")
            report_lines.append(f"  Full CTT:")
            report_lines.append(f"    Intercept:  {row['coef_intercept_full']:7.4f}")
            report_lines.append(f"    TSVR_hours: {row['coef_slope_full']:7.4f}")
            report_lines.append(f"  Purified CTT:")
            report_lines.append(f"    Intercept:  {row['coef_intercept_purified']:7.4f}")
            report_lines.append(f"    TSVR_hours: {row['coef_slope_purified']:7.4f}")
            report_lines.append("")

        # Slope correlation
        report_lines.append("-" * 80)
        report_lines.append("COEFFICIENT AGREEMENT")
        report_lines.append("-" * 80)
        report_lines.append(f"Slope correlation (Full vs Purified CTT): r={slope_corr:.4f}")
        report_lines.append("")
        report_lines.append("Interpretation:")
        if slope_corr > 0.9:
            report_lines.append("  Very high agreement - both CTT types show similar decay patterns")
        elif slope_corr > 0.7:
            report_lines.append("  High agreement - both CTT types show similar decay patterns")
        elif slope_corr > 0.5:
            report_lines.append("  Moderate agreement - some differences in decay patterns")
        else:
            report_lines.append("  Low agreement - different decay patterns between CTT types")
        report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Save report
        report_path = RQ_DIR / "data" / "step07_lmm_convergence_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        log(f"[SAVED] {report_path}")

        # Print report to log
        log("")
        for line in report_lines:
            log(line)

        # =========================================================================
        # STEP 7: Validation
        # =========================================================================
        log("[VALIDATION] Validating results...")

        # Check convergence
        if not all_converged:
            log("[WARNING] Not all models converged - review convergence report")
        else:
            log("[PASS] All models converged")

        # Check AIC values
        aic_cols = ['AIC_IRT', 'AIC_full', 'AIC_purified']
        for col in aic_cols:
            if df_results[col].isna().any():
                raise ValueError(f"NaN values found in {col}")
            if (df_results[col] <= 0).any():
                raise ValueError(f"Non-positive AIC values found in {col}")
        log("[PASS] All AIC values positive and finite")

        # Check paradigms
        if set(df_results['paradigm']) != {'IFR', 'ICR', 'IRE'}:
            raise ValueError("Not all paradigms present")
        log("[PASS] All 3 paradigms present")

        # Check coefficients
        coef_cols = [c for c in df_results.columns if c.startswith('coef_')]
        for col in coef_cols:
            if df_results[col].isna().any():
                raise ValueError(f"NaN values found in {col}")
        log("[PASS] No NaN values in coefficients")

        log("[SUCCESS] Step 07 complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
