#!/usr/bin/env python3
"""Test Intercept-Slope Correlation and Visualize Distribution: Test hypothesis that baseline memory ability (random intercepts) and forgetting"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any
import traceback
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import test_intercept_slope_correlation_d068

from tools.validation import validate_correlation_test_d068

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.1.4
LOG_FILE = RQ_DIR / "logs" / "step05_correlation_test.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Test Intercept-Slope Correlation and Visualize Distribution")
        # Load Input Data

        log("Loading random effects from Step 4...")
        random_effects_path = RQ_DIR / "data" / "step04_random_effects.csv"

        if not random_effects_path.exists():
            raise FileNotFoundError(f"Input file missing: {random_effects_path}")

        random_effects = pd.read_csv(random_effects_path, encoding='utf-8')
        log(f"{random_effects_path.name} ({len(random_effects)} rows, {len(random_effects.columns)} cols)")

        # Validate expected columns
        required_cols = ['UID', 'random_intercept', 'random_slope']
        missing_cols = [col for col in required_cols if col not in random_effects.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        log(f"Columns present: {random_effects.columns.tolist()}")
        log(f"Random intercepts: mean={random_effects['random_intercept'].mean():.4f}, SD={random_effects['random_intercept'].std():.4f}")
        log(f"Random slopes: mean={random_effects['random_slope'].mean():.4f}, SD={random_effects['random_slope'].std():.4f}")
        # Run Analysis Tool (Correlation Test with Decision D068)

        log("Running test_intercept_slope_correlation_d068...")
        log("Parameters: family_alpha=0.05, n_tests=15 (Chapter 5 family size)")

        correlation_result = test_intercept_slope_correlation_d068(
            random_effects_df=random_effects,
            family_alpha=0.05,  # Significance threshold
            n_tests=15,  # Chapter 5 family size for Bonferroni correction
            intercept_col='random_intercept',  # Column name for random intercepts
            slope_col='random_slope'  # Column name for random slopes
        )

        log("Correlation test complete")
        log(f"Correlation r = {correlation_result['r']:.4f}")
        log(f"p_uncorrected = {correlation_result['p_uncorrected']:.6f}")
        log(f"p_bonferroni = {correlation_result['p_bonferroni']:.6f}")
        log(f"Significant (uncorrected): {correlation_result['significant_uncorrected']}")
        log(f"Significant (Bonferroni): {correlation_result['significant_bonferroni']}")
        log(f"Interpretation: {correlation_result['interpretation']}")
        # Save Analysis Outputs
        # These outputs will be used by: rq_inspect (validation), results analysis (interpretation)

        log("Saving correlation test results...")

        # 3a. Save correlation results as CSV (wide format for validation)
        correlation_csv_path = RQ_DIR / "data" / "step05_intercept_slope_correlation.csv"
        correlation_df = pd.DataFrame([{
            'r': correlation_result['r'],
            'p_uncorrected': correlation_result['p_uncorrected'],
            'p_bonferroni': correlation_result['p_bonferroni'],
            'df': float(len(random_effects) - 2),  # N-2 for Pearson
            'alpha_corrected': 0.05 / 15,  # 0.05/15 = 0.0033
            'significant_uncorrected': correlation_result['significant_uncorrected'],
            'significant_bonferroni': correlation_result['significant_bonferroni']
        }])
        correlation_df.to_csv(correlation_csv_path, index=False, encoding='utf-8')
        log(f"{correlation_csv_path.name} (1 row, {len(correlation_df.columns)} columns)")

        # 3b. Save interpretation as plain text
        interpretation_path = RQ_DIR / "results" / "step05_correlation_interpretation.txt"
        with open(interpretation_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("INTERCEPT-SLOPE CORRELATION TEST RESULTS (RQ 5.13 Step 5)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Correlation: r = {correlation_result['r']:.4f}\n")
            f.write(f"Direction: {'Negative' if correlation_result['r'] < 0 else 'Positive'}\n")
            f.write(f"Magnitude: {correlation_result['interpretation']}\n\n")
            f.write("-"*80 + "\n")
            f.write("SIGNIFICANCE TESTS (Decision D068 Dual P-Value Reporting)\n")
            f.write("-"*80 + "\n\n")
            f.write(f"Uncorrected p-value: p = {correlation_result['p_uncorrected']:.6f}\n")
            f.write(f"  Significant at alpha=0.05? {'YES' if correlation_result['significant_uncorrected'] else 'NO'}\n\n")
            f.write(f"Bonferroni-corrected p-value: p = {correlation_result['p_bonferroni']:.6f}\n")
            f.write(f"  Family size: 15 tests (Chapter 5)\n")
            f.write(f"  Corrected alpha: 0.05/15 = 0.0033\n")
            f.write(f"  Significant at corrected alpha? {'YES' if correlation_result['significant_bonferroni'] else 'NO'}\n\n")
            f.write("-"*80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("-"*80 + "\n\n")
            if correlation_result['r'] < 0:
                f.write("Negative correlation suggests that participants with higher baseline\n")
                f.write("memory (positive random intercepts) tend to have slower forgetting rates\n")
                f.write("(negative random slopes, meaning less steep decline).\n\n")
            else:
                f.write("Positive correlation suggests that participants with higher baseline\n")
                f.write("memory (positive random intercepts) tend to have faster forgetting rates\n")
                f.write("(positive random slopes, meaning steeper decline).\n\n")

            if correlation_result['significant_bonferroni']:
                f.write("The correlation is statistically significant even after Bonferroni\n")
                f.write("correction, indicating a robust relationship between baseline ability\n")
                f.write("and forgetting rate that is unlikely due to chance.\n")
            elif correlation_result['significant_uncorrected']:
                f.write("The correlation is statistically significant without correction but\n")
                f.write("NOT significant after Bonferroni correction, suggesting the relationship\n")
                f.write("may not be robust to multiple testing concerns.\n")
            else:
                f.write("The correlation is NOT statistically significant, suggesting no strong\n")
                f.write("relationship between baseline ability and forgetting rate in this sample.\n")

        log(f"{interpretation_path.name}")
        # Create Visualizations
        # Create histogram and Q-Q plot for random slopes distribution

        log("Creating random slopes distribution visualizations...")

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300

        # 4a. Histogram with normal overlay
        fig_hist, ax_hist = plt.subplots(figsize=(8, 6))

        slopes = random_effects['random_slope'].values

        # Plot histogram
        n, bins, patches = ax_hist.hist(slopes, bins=20, density=True, alpha=0.7,
                                         color='steelblue', edgecolor='black', label='Observed')

        # Overlay normal distribution
        mu, sigma = slopes.mean(), slopes.std()
        x = np.linspace(slopes.min(), slopes.max(), 100)
        normal_curve = stats.norm.pdf(x, mu, sigma)
        ax_hist.plot(x, normal_curve, 'r-', linewidth=2, label=f'Normal (mean={mu:.4f}, SD={sigma:.4f})')

        # Add mean reference line
        ax_hist.axvline(mu, color='black', linestyle='--', linewidth=1.5, label=f'Mean = {mu:.4f}')

        ax_hist.set_xlabel('Random Slope (Forgetting Rate)', fontsize=12)
        ax_hist.set_ylabel('Density', fontsize=12)
        ax_hist.set_title('Distribution of Random Slopes\n(Individual Differences in Forgetting Rate)', fontsize=14)
        ax_hist.legend(fontsize=10)
        ax_hist.grid(True, alpha=0.3)

        histogram_path = RQ_DIR / "plots" / "step05_random_slopes_histogram.png"
        fig_hist.tight_layout()
        fig_hist.savefig(histogram_path, dpi=300, bbox_inches='tight')
        plt.close(fig_hist)
        log(f"{histogram_path.name}")

        # 4b. Q-Q plot
        fig_qq, ax_qq = plt.subplots(figsize=(8, 6))

        stats.probplot(slopes, dist="norm", plot=ax_qq)
        ax_qq.set_title('Q-Q Plot: Random Slopes vs Normal Distribution', fontsize=14)
        ax_qq.set_xlabel('Theoretical Quantiles (Normal Distribution)', fontsize=12)
        ax_qq.set_ylabel('Observed Quantiles (Random Slopes)', fontsize=12)
        ax_qq.grid(True, alpha=0.3)

        qqplot_path = RQ_DIR / "plots" / "step05_random_slopes_qqplot.png"
        fig_qq.tight_layout()
        fig_qq.savefig(qqplot_path, dpi=300, bbox_inches='tight')
        plt.close(fig_qq)
        log(f"{qqplot_path.name}")

        log("Visualization complete")
        # Run Validation Tool
        # Validates: Decision D068 dual p-value reporting compliance
        # Checks: p_uncorrected and p_bonferroni columns present, values in valid ranges

        log("Running validate_correlation_test_d068...")

        validation_result = validate_correlation_test_d068(
            correlation_df=correlation_df,
            required_cols=None  # Uses default D068 spec
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS - {validation_result['message']}")
            log(f"D068 compliant: {validation_result['d068_compliant']}")
        else:
            log(f"FAIL - {validation_result['message']}")
            if validation_result.get('missing_cols'):
                log(f"Missing columns: {validation_result['missing_cols']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        log("Step 5 complete")
        log("")
        log("Summary:")
        log(f"  Correlation: r = {correlation_result['r']:.4f}")
        log(f"  Uncorrected p = {correlation_result['p_uncorrected']:.6f} ({'sig' if correlation_result['significant_uncorrected'] else 'ns'})")
        log(f"  Bonferroni p = {correlation_result['p_bonferroni']:.6f} ({'sig' if correlation_result['significant_bonferroni'] else 'ns'})")
        log(f"  Interpretation: {correlation_result['interpretation']}")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
