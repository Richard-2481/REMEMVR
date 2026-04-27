#!/usr/bin/env python3
"""
Step 05: Test Intercept-Slope Correlation and Create Diagnostic Plots

PURPOSE:
Test the correlation between random intercepts and random slopes within each
congruence level using Pearson correlation. Apply Bonferroni correction per
Decision D068 (dual p-values). Create diagnostic plots (histograms + Q-Q plots)
to assess random slope distribution normality.

EXPECTED INPUTS:
- data/step04_random_effects.csv: Random effects (300 rows: 100 UID x 3 congruence)

EXPECTED OUTPUTS:
- data/step05_intercept_slope_correlation.csv: Correlation statistics with D068 dual p-values
- data/step05_correlation_interpretation.txt: Text report with interpretations
- data/step05_random_slopes_histogram_common.png: Histogram for Common congruence
- data/step05_random_slopes_histogram_congruent.png: Histogram for Congruent congruence
- data/step05_random_slopes_histogram_incongruent.png: Histogram for Incongruent congruence
- data/step05_random_slopes_qqplot_common.png: Q-Q plot for Common congruence
- data/step05_random_slopes_qqplot_congruent.png: Q-Q plot for Congruent congruence
- data/step05_random_slopes_qqplot_incongruent.png: Q-Q plot for Incongruent congruence

VALIDATION CRITERIA:
- BOTH p_uncorrected AND p_bonferroni present (Decision D068)
- Correlation coefficient r in [-1, 1]
- All 3 congruence levels present
- All plot files exist and > 10KB
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_lmm import test_intercept_slope_correlation_d068
from tools.validation import validate_correlation_test_d068, check_file_exists

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_test_correlations_plot_diagnostics.log"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def create_histogram_with_normal_overlay(data, title, output_path):
    """Create histogram with normal distribution overlay."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot histogram
    n, bins, patches = ax.hist(data, bins=20, density=True, alpha=0.7,
                               color='steelblue', edgecolor='black')

    # Overlay normal distribution
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2,
           label=f'Normal(μ={mu:.3f}, σ={sigma:.3f})')

    ax.set_xlabel('Random Slope (Forgetting Rate)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_qq_plot(data, title, output_path):
    """Create Q-Q plot to assess normality."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Compute theoretical quantiles
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist="norm")

    # Plot Q-Q plot
    ax.scatter(osm, osr, alpha=0.6, s=30, color='steelblue')

    # Add reference line
    ax.plot(osm, slope * osm + intercept, 'r-', linewidth=2,
           label=f'Reference line (R²={r**2:.3f})')

    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Sample Quantiles', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        log("[START] Step 05: Test Correlations and Create Diagnostic Plots")

        # =====================================================================
        # STEP 1: Load Random Effects
        # =====================================================================
        log("[LOAD] Loading random effects...")

        random_effects_file = RQ_DIR / "data" / "step04_random_effects.csv"
        df_re = pd.read_csv(random_effects_file, encoding='utf-8')

        log(f"[LOADED] {random_effects_file.name} ({len(df_re)} rows)")

        # =====================================================================
        # STEP 2: Test Intercept-Slope Correlation (Decision D068)
        # =====================================================================
        log("\n[ANALYSIS] Testing intercept-slope correlations with D068 dual p-values...")

        # Call catalogued tool with D068 compliance for each congruence level
        all_correlations = []

        for congruence in sorted(df_re['congruence'].unique()):
            log(f"  Testing {congruence}...")

            # Filter for this congruence
            df_cong = df_re[df_re['congruence'] == congruence]

            # Test correlation
            corr_result = test_intercept_slope_correlation_d068(
                df_cong,
                family_alpha=0.05,
                n_tests=3,  # 3 congruence levels
                intercept_col='Total_Intercept',
                slope_col='Total_Slope'
            )

            # Add congruence to result
            corr_result['congruence'] = congruence

            # Add CI bounds if not present (required by validation)
            if 'CI_lower' not in corr_result or 'CI_upper' not in corr_result:
                # Compute Fisher's z CI
                import numpy as np
                from scipy import stats as sp_stats
                
                r = corr_result['r']
                n = len(df_cong)
                z = np.arctanh(r)
                se = 1 / np.sqrt(n - 3)
                ci_z = sp_stats.norm.ppf([0.025, 0.975]) * se + z
                ci_r = np.tanh(ci_z)
                
                corr_result['CI_lower'] = ci_r[0]
                corr_result['CI_upper'] = ci_r[1]

            all_correlations.append(corr_result)

        # Convert to DataFrame
        df_corr = pd.DataFrame(all_correlations)

        log(f"\n[COMPUTED] {len(df_corr)} correlation tests")

        # Display results
        log("\n[CORRELATION RESULTS]")
        for _, row in df_corr.iterrows():
            log(f"  {row['congruence']:12s} | r={row['r']:7.4f} | p_raw={row['p_uncorrected']:.4f} | p_bonf={row['p_bonferroni']:.4f}")

        # =====================================================================
        # STEP 3: Save Correlation Results
        # =====================================================================
        log("\n[SAVE] Saving correlation results...")

        corr_output = RQ_DIR / "data" / "step05_intercept_slope_correlation.csv"
        df_corr.to_csv(corr_output, index=False, encoding='utf-8')

        log(f"[SAVED] {corr_output.name}")

        # =====================================================================
        # STEP 4: Create Correlation Interpretation Report
        # =====================================================================
        log("[REPORT] Creating correlation interpretation report...")

        report_path = RQ_DIR / "data" / "step05_correlation_interpretation.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("INTERCEPT-SLOPE CORRELATION ANALYSIS\n")
            f.write("Decision D068: Dual P-Value Reporting\n")
            f.write("=" * 80 + "\n\n")

            f.write("BONFERRONI CORRECTION:\n")
            f.write("  Family-wise alpha: 0.05\n")
            f.write("  Number of tests: 3 (one per congruence level)\n")
            f.write("  Corrected alpha: 0.0167 (0.05 / 3)\n\n")

            f.write("=" * 80 + "\n\n")

            for _, row in df_corr.iterrows():
                f.write(f"CONGRUENCE: {row['congruence']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Pearson r:          {row['r']:7.4f}\n")
                f.write(f"  95% CI:             [{row['CI_lower']:7.4f}, {row['CI_upper']:7.4f}]\n")
                f.write(f"  p-value (raw):      {row['p_uncorrected']:.4f}\n")
                f.write(f"  p-value (Bonf):     {row['p_bonferroni']:.4f}\n")

                # Interpret direction
                if row['r'] > 0:
                    f.write("\n  DIRECTION: Positive correlation (higher baseline -> slower forgetting)\n")
                elif row['r'] < 0:
                    f.write("\n  DIRECTION: Negative correlation (higher baseline -> faster forgetting)\n")
                else:
                    f.write("\n  DIRECTION: No correlation\n")

                # Interpret significance
                if row['p_bonferroni'] < 0.05:
                    f.write("  SIGNIFICANCE: Statistically significant after Bonferroni correction\n")
                elif row['p_uncorrected'] < 0.05:
                    f.write("  SIGNIFICANCE: Significant (raw p) but NOT after Bonferroni correction\n")
                else:
                    f.write("  SIGNIFICANCE: Not statistically significant\n")

                f.write("\n")

        log(f"[SAVED] {report_path.name}")

        # =====================================================================
        # STEP 5: Create Histograms with Normal Overlay
        # =====================================================================
        log("\n[PLOT] Creating histograms with normal distribution overlay...")

        congruence_levels = df_re['congruence'].unique()

        for congruence in congruence_levels:
            log(f"  Creating histogram for {congruence}...")

            df_subset = df_re[df_re['congruence'] == congruence]
            slopes = df_subset['Total_Slope'].values

            output_path = RQ_DIR / "data" / f"step05_random_slopes_histogram_{congruence.lower()}.png"

            create_histogram_with_normal_overlay(
                slopes,
                f"Random Slopes Distribution - {congruence} Congruence",
                output_path
            )

            log(f"  [SAVED] {output_path.name}")

        # =====================================================================
        # STEP 6: Create Q-Q Plots
        # =====================================================================
        log("\n[PLOT] Creating Q-Q plots for normality assessment...")

        for congruence in congruence_levels:
            log(f"  Creating Q-Q plot for {congruence}...")

            df_subset = df_re[df_re['congruence'] == congruence]
            slopes = df_subset['Total_Slope'].values

            output_path = RQ_DIR / "data" / f"step05_random_slopes_qqplot_{congruence.lower()}.png"

            create_qq_plot(
                slopes,
                f"Q-Q Plot - {congruence} Congruence",
                output_path
            )

            log(f"  [SAVED] {output_path.name}")

        # =====================================================================
        # STEP 7: Validate Correlation Test Results (Decision D068)
        # =====================================================================
        log("\n[VALIDATION] Validating correlation test results (D068 compliance)...")

        validation = validate_correlation_test_d068(
            df_corr,
            required_cols=['r', 'p_uncorrected', 'p_bonferroni', 'CI_lower', 'CI_upper']
        )

        if validation['valid']:
            log("[PASS] Correlation test validated (D068 compliant)")
        else:
            log(f"[FAIL] Correlation validation failed: {validation['message']}")
            raise ValueError(validation['message'])

        # =====================================================================
        # STEP 8: Validate Plot Files Exist
        # =====================================================================
        log("\n[VALIDATION] Validating plot files...")

        expected_plots = []
        for congruence in ['common', 'congruent', 'incongruent']:
            expected_plots.append(RQ_DIR / "data" / f"step05_random_slopes_histogram_{congruence}.png")
            expected_plots.append(RQ_DIR / "data" / f"step05_random_slopes_qqplot_{congruence}.png")

        missing_plots = []
        for plot_path in expected_plots:
            if not plot_path.exists():
                missing_plots.append(plot_path.name)
            elif plot_path.stat().st_size < 10000:
                missing_plots.append(f"{plot_path.name} (too small)")

        if missing_plots:
            raise FileNotFoundError(f"Missing or invalid plot files: {', '.join(missing_plots)}")

        log(f"[PASS] All {len(expected_plots)} plot files validated (exist and > 10KB)")

        log("\n[SUCCESS] Step 05 complete - Correlations tested and diagnostic plots created")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        log("[TRACEBACK] Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
