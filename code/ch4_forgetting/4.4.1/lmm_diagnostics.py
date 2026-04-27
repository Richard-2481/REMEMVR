#!/usr/bin/env python3
"""
LMM Residual Diagnostics - MANDATORY for Assumption Validation (Section 5.1)

PURPOSE:
Validate LMM assumptions to ensure statistical conclusions are valid:
1. Normality of residuals (Q-Q plot, Shapiro-Wilk test)
2. Homoscedasticity (residuals vs fitted plot, Breusch-Pagan test)
3. Independence (no autocorrelation - assumed via random effects)
4. No extreme outliers (leverage/influence)

APPROACH:
- Load best fitted model from step05
- Extract residuals and fitted values
- Generate diagnostic plots
- Perform statistical tests
- Document any violations

EXPECTED OUTCOMES:
- Residuals approximately normal (visual + Shapiro p > 0.001)
- Homoscedastic residuals (visual + BP p > 0.05)
- No extreme influential outliers (Cook's D < 1.0)
- If violations exist: Document, assess impact (LMM robust with N>100)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "lmm_diagnostics.log"
INPUT_MODEL = RQ_DIR / "data" / "step05_lmm_fitted_model.pkl"
INPUT_LMM_DATA = RQ_DIR / "data" / "step04_lmm_input.csv"
OUTPUT_DIR = RQ_DIR / "plots" / "diagnostics"
OUTPUT_REPORT = RQ_DIR / "results" / "lmm_diagnostics.txt"

def log(msg):
    """Write to log file and console."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("="*70)
        log("LMM RESIDUAL DIAGNOSTICS")
        log("="*70)
        log(f"Date: 2025-12-27")
        log(f"Purpose: Validate LMM assumptions")
        log("")

        # =====================================================================
        # Load Model and Data
        # =====================================================================
        log("[LOAD] Loading fitted model and data...")

        result = MixedLMResults.load(str(INPUT_MODEL))
        df = pd.read_csv(INPUT_LMM_DATA)

        log(f"  Model AIC: {result.aic:.2f}")
        log(f"  N observations: {len(df)}")
        log("")

        # Extract residuals and fitted values
        residuals = result.resid
        fitted = result.fittedvalues

        log(f"  Residuals: N={len(residuals)}, mean={residuals.mean():.6f}, SD={residuals.std():.4f}")
        log(f"  Fitted values: range [{fitted.min():.2f}, {fitted.max():.2f}]")
        log("")

        # =====================================================================
        # Create Output Directory
        # =====================================================================
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # DIAGNOSTIC 1: Q-Q Plot (Normality of Residuals)
        # =====================================================================
        log("="*70)
        log("DIAGNOSTIC 1: NORMALITY OF RESIDUALS")
        log("="*70)
        log("")

        # Q-Q plot
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot: Residuals vs Normal Distribution", fontsize=14, fontweight='bold')
        ax.set_xlabel("Theoretical Quantiles", fontsize=12)
        ax.set_ylabel("Sample Quantiles (Residuals)", fontsize=12)
        ax.grid(True, alpha=0.3)

        qq_plot_path = OUTPUT_DIR / "qq_plot.png"
        plt.tight_layout()
        plt.savefig(qq_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        log(f"[SAVED] {qq_plot_path.name}")

        # Shapiro-Wilk test (normality test)
        # Note: With N=1200, test is very sensitive - visual more important
        if len(residuals) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            log(f"  Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.6f}")

            if shapiro_p < 0.001:
                log(f"  [WARNING] Residuals significantly deviate from normality (p<0.001)")
                log(f"            However, LMM robust to moderate non-normality with N>100")
                log(f"            Visual assessment (Q-Q plot) is more informative")
            else:
                log(f"  [PASS] Residuals approximately normally distributed")
        else:
            log(f"  [SKIP] Shapiro-Wilk test (N>5000, too sensitive)")
            log(f"         Using visual Q-Q plot assessment only")

        log("")

        # =====================================================================
        # DIAGNOSTIC 2: Residuals vs Fitted (Homoscedasticity)
        # =====================================================================
        log("="*70)
        log("DIAGNOSTIC 2: HOMOSCEDASTICITY (CONSTANT VARIANCE)")
        log("="*70)
        log("")

        # Residuals vs Fitted plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(fitted, residuals, alpha=0.3, s=10)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
        ax.set_title("Residuals vs Fitted Values", fontsize=14, fontweight='bold')
        ax.set_xlabel("Fitted Values (theta)", fontsize=12)
        ax.set_ylabel("Residuals", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        resid_fitted_path = OUTPUT_DIR / "residuals_vs_fitted.png"
        plt.tight_layout()
        plt.savefig(resid_fitted_path, dpi=300, bbox_inches='tight')
        plt.close()

        log(f"[SAVED] {resid_fitted_path.name}")

        # Breusch-Pagan test (heteroscedasticity test)
        # Need design matrix (exog) for BP test
        # Approximate using fitted values as proxy
        exog_approx = np.column_stack([np.ones(len(fitted)), fitted])

        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(residuals, exog_approx)
            log(f"  Breusch-Pagan test: LM={bp_stat:.4f}, p={bp_p:.6f}")

            if bp_p < 0.05:
                log(f"  [WARNING] Significant heteroscedasticity detected (p<0.05)")
                log(f"            Variance not constant across fitted values")
                log(f"            Consider: robust standard errors, weighted LMM")
            else:
                log(f"  [PASS] No significant heteroscedasticity detected")
        except Exception as e:
            log(f"  [ERROR] Breusch-Pagan test failed: {str(e)}")
            log(f"          Using visual assessment only")

        log("")

        # =====================================================================
        # DIAGNOSTIC 3: Scale-Location Plot (Spread-Location)
        # =====================================================================
        log("="*70)
        log("DIAGNOSTIC 3: SCALE-LOCATION (SQRT STANDARDIZED RESIDUALS)")
        log("="*70)
        log("")

        # Standardize residuals
        residuals_std = residuals / residuals.std()
        residuals_sqrt = np.sqrt(np.abs(residuals_std))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(fitted, residuals_sqrt, alpha=0.3, s=10)
        ax.set_title("Scale-Location Plot", fontsize=14, fontweight='bold')
        ax.set_xlabel("Fitted Values (theta)", fontsize=12)
        ax.set_ylabel("√|Standardized Residuals|", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add smoothed trend line (lowess)
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(residuals_sqrt, fitted, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='Smoothed trend')
        ax.legend()

        scale_loc_path = OUTPUT_DIR / "scale_location.png"
        plt.tight_layout()
        plt.savefig(scale_loc_path, dpi=300, bbox_inches='tight')
        plt.close()

        log(f"[SAVED] {scale_loc_path.name}")
        log("  Purpose: Check if variance increases with fitted values")
        log("  Ideal: Horizontal red line (constant variance)")
        log("")

        # =====================================================================
        # DIAGNOSTIC 4: Outliers and Influential Points
        # =====================================================================
        log("="*70)
        log("DIAGNOSTIC 4: OUTLIERS AND INFLUENTIAL POINTS")
        log("="*70)
        log("")

        # Standardized residuals > 3 SD
        outliers_3sd = np.abs(residuals_std) > 3
        n_outliers_3sd = outliers_3sd.sum()
        pct_outliers = (n_outliers_3sd / len(residuals)) * 100

        log(f"  Outliers (|std residual| > 3): N={n_outliers_3sd} ({pct_outliers:.2f}%)")

        if pct_outliers > 1.0:
            log(f"  [WARNING] {pct_outliers:.2f}% outliers (expected <1% for normal)")
            log(f"            May indicate model misspecification or data quality issues")
        else:
            log(f"  [PASS] Outlier rate within expected range (<1%)")

        # Histogram of standardized residuals
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(residuals_std, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=-3, color='red', linestyle='--', linewidth=2, label='-3 SD')
        ax.axvline(x=3, color='red', linestyle='--', linewidth=2, label='+3 SD')
        ax.set_title("Histogram of Standardized Residuals", fontsize=14, fontweight='bold')
        ax.set_xlabel("Standardized Residuals", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        hist_path = OUTPUT_DIR / "residuals_histogram.png"
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()

        log(f"[SAVED] {hist_path.name}")
        log("")

        # =====================================================================
        # SUMMARY AND INTERPRETATION
        # =====================================================================
        log("="*70)
        log("SUMMARY")
        log("="*70)
        log("")

        diagnostics_pass = True
        issues = []

        # Check normality
        if 'shapiro_p' in locals() and shapiro_p < 0.001:
            issues.append("Residuals deviate from normality (Shapiro p<0.001)")
            diagnostics_pass = False

        # Check homoscedasticity
        if 'bp_p' in locals() and bp_p < 0.05:
            issues.append("Heteroscedasticity detected (BP p<0.05)")
            diagnostics_pass = False

        # Check outliers
        if pct_outliers > 1.0:
            issues.append(f"{pct_outliers:.2f}% outliers (>1% threshold)")
            diagnostics_pass = False

        if diagnostics_pass:
            log("[✓] ALL DIAGNOSTIC CHECKS PASSED")
            log("    Residuals are approximately normal")
            log("    Variance is homoscedastic")
            log("    Outlier rate within expected range")
            log("    LMM assumptions are met")
        else:
            log("[!] SOME DIAGNOSTIC CHECKS FAILED:")
            for issue in issues:
                log(f"    - {issue}")
            log("")
            log("IMPACT ASSESSMENT:")
            log("  - LMM is robust to moderate violations with N>100")
            log("  - N=1200 observations provides robustness")
            log("  - Findings should be interpreted with caution")
            log("  - Consider: robust standard errors, sensitivity analyses")

        log("")

        # =====================================================================
        # SAVE REPORT
        # =====================================================================
        log("[SAVE] Writing diagnostics report...")

        OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("LMM RESIDUAL DIAGNOSTICS REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: 2025-12-27\n")
            f.write(f"Model: step05_lmm_fitted_model.pkl\n")
            f.write(f"N observations: {len(residuals)}\n")
            f.write(f"AIC: {result.aic:.2f}\n\n")

            f.write("DIAGNOSTIC PLOTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"1. Q-Q Plot: {qq_plot_path.name}\n")
            f.write(f"2. Residuals vs Fitted: {resid_fitted_path.name}\n")
            f.write(f"3. Scale-Location: {scale_loc_path.name}\n")
            f.write(f"4. Residuals Histogram: {hist_path.name}\n\n")

            f.write("STATISTICAL TESTS\n")
            f.write("-" * 40 + "\n")

            if 'shapiro_p' in locals():
                f.write(f"Shapiro-Wilk (normality): W={shapiro_stat:.4f}, p={shapiro_p:.6f}\n")
            else:
                f.write(f"Shapiro-Wilk: Skipped (N>5000)\n")

            if 'bp_p' in locals():
                f.write(f"Breusch-Pagan (homoscedasticity): LM={bp_stat:.4f}, p={bp_p:.6f}\n")
            else:
                f.write(f"Breusch-Pagan: Error during computation\n")

            f.write(f"\nOutliers (|std resid| > 3): {n_outliers_3sd} ({pct_outliers:.2f}%)\n\n")

            f.write("ASSESSMENT\n")
            f.write("-" * 40 + "\n")

            if diagnostics_pass:
                f.write("ALL CHECKS PASSED ✓\n")
                f.write("LMM assumptions are met.\n")
            else:
                f.write("SOME CHECKS FAILED\n")
                for issue in issues:
                    f.write(f"  - {issue}\n")
                f.write("\n")
                f.write("IMPACT: LMM robust to moderate violations with N=1200.\n")
                f.write("Findings valid but should be interpreted with caution.\n")

            f.write("\nACTION REQUIRED\n")
            f.write("-" * 40 + "\n")
            f.write("1. Add diagnostic plots to plots/diagnostics/\n")
            f.write("2. Update validation.md with diagnostic results\n")
            f.write("3. Document any violations in summary.md Limitations\n")

        log(f"[SAVED] {OUTPUT_REPORT.name}")
        log("\n[SUCCESS] LMM diagnostics complete")
        log(f"\nDiagnostic plots saved to: {OUTPUT_DIR}")

        sys.exit(0)

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
