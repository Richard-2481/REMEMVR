#!/usr/bin/env python3
"""Validate Statistical Assumptions: Comprehensive assumption validation: normality tests (Shapiro-Wilk), outlier"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import statsmodels for Cook's distance
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.9.1
LOG_FILE = RQ_DIR / "logs" / "step07_validate_assumptions.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 7: Validate statistical assumptions")
        # Load Input Data

        log("Loading input data...")

        df_rates = pd.read_csv(RQ_DIR / "data" / "step02_individual_decline_rates.csv", encoding='utf-8')
        log(f"step02_individual_decline_rates.csv ({len(df_rates)} rows)")

        df_ttest = pd.read_csv(RQ_DIR / "data" / "step03_paired_ttest_results.csv", encoding='utf-8')
        log(f"step03_paired_ttest_results.csv ({len(df_ttest)} rows)")

        df_ratio = pd.read_csv(RQ_DIR / "data" / "step06_decline_ratio.csv", encoding='utf-8')
        log(f"step06_decline_ratio.csv ({len(df_ratio)} rows)")

        # Extract arrays
        difference = df_rates['difference'].values
        valid_ratios = df_rates[~df_rates['no_decline_flag']]['ratio'].values
        # Normality Check - Difference

        log("Testing normality of paired differences...")

        # Shapiro-Wilk test
        shapiro_diff = stats.shapiro(difference)
        W_diff = shapiro_diff.statistic
        p_diff = shapiro_diff.pvalue

        log(f"Difference: W = {W_diff:.4f}, p = {p_diff:.6f}")

        if p_diff > 0.05:
            interpretation_diff = 'Normal'
            log("Normality accepted (p > 0.05) - parametric CI valid")
        else:
            interpretation_diff = 'Non-normal'
            log("Normality rejected (p < 0.05) - use bootstrap CI as primary")

        # Q-Q plot data
        (osm, osr), (slope, intercept, r) = stats.probplot(difference, dist='norm')
        qqplot_diff_data = pd.DataFrame({
            'theoretical_quantiles': osm,
            'sample_quantiles': osr,
            'distribution': 'difference'
        })
        # Normality Check - Ratio

        log("Testing normality of ratio distribution...")

        # Shapiro-Wilk test
        shapiro_ratio = stats.shapiro(valid_ratios)
        W_ratio = shapiro_ratio.statistic
        p_ratio = shapiro_ratio.pvalue

        log(f"Ratio: W = {W_ratio:.4f}, p = {p_ratio:.6f}")

        # Skewness and kurtosis
        skew_ratio = stats.skew(valid_ratios)
        kurt_ratio = stats.kurtosis(valid_ratios)

        log(f"Ratio skewness: {skew_ratio:.4f}")
        log(f"Ratio kurtosis: {kurt_ratio:.4f}")

        if p_ratio > 0.05:
            interpretation_ratio = 'Normal'
            log("Ratio normality accepted")
        else:
            interpretation_ratio = 'Non-normal'
            log("Ratio normality rejected")

        # Check if log transformation needed
        log_transform_needed = abs(skew_ratio) > 2
        if log_transform_needed:
            log(f"Severe skewness (|skew| = {abs(skew_ratio):.2f} > 2) - log transformation recommended")

        # Q-Q plot data for ratio
        (osm_r, osr_r), (slope_r, intercept_r, r_r) = stats.probplot(valid_ratios, dist='norm')
        qqplot_ratio_data = pd.DataFrame({
            'theoretical_quantiles': osm_r,
            'sample_quantiles': osr_r,
            'distribution': 'ratio'
        })

        # Combine Q-Q plot data
        qqplot_data = pd.concat([qqplot_diff_data, qqplot_ratio_data], ignore_index=True)
        # Outlier Detection

        log("Detecting outliers using IQR method...")

        # IQR method for difference
        q1 = np.percentile(difference, 25)
        q3 = np.percentile(difference, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (difference < lower_bound) | (difference > upper_bound)
        outlier_uids = df_rates.loc[outlier_mask, 'UID'].values
        outlier_diffs = difference[outlier_mask]

        log(f"IQR bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
        log(f"Outliers detected: {len(outlier_uids)}")

        # Cook's distance (if statsmodels available)
        if STATSMODELS_AVAILABLE:
            log("Computing Cook's distance...")

            # Fit intercept-only model: difference ~ 1
            X = np.ones((len(difference), 1))  # Intercept only
            y = difference
            model = sm.OLS(y, X).fit()
            influence = model.get_influence()
            cooks_d = influence.cooks_distance[0]

            # Threshold: 4/N
            cooks_threshold = 4 / len(difference)
            influential_mask = cooks_d > cooks_threshold
            influential_count = np.sum(influential_mask)

            log(f"Threshold: {cooks_threshold:.4f}")
            log(f"Influential cases: {influential_count}")

            # Combine outlier info
            outlier_detection = []
            for i, uid in enumerate(df_rates['UID']):
                if outlier_mask[i] or influential_mask[i]:
                    outlier_type = []
                    if outlier_mask[i]:
                        outlier_type.append('IQR')
                    if influential_mask[i]:
                        outlier_type.append('Cook')
                    outlier_detection.append({
                        'UID': uid,
                        'difference': difference[i],
                        'outlier_type': '+'.join(outlier_type),
                        'cook_d': cooks_d[i]
                    })
        else:
            log("statsmodels not available, skipping Cook's distance")
            # IQR only
            outlier_detection = []
            for i, uid in enumerate(outlier_uids):
                idx = df_rates[df_rates['UID'] == uid].index[0]
                outlier_detection.append({
                    'UID': uid,
                    'difference': outlier_diffs[list(outlier_uids).index(uid)],
                    'outlier_type': 'IQR',
                    'cook_d': np.nan
                })

        df_outliers = pd.DataFrame(outlier_detection) if outlier_detection else pd.DataFrame(columns=['UID', 'difference', 'outlier_type', 'cook_d'])
        # Sensitivity Analyses

        log("Running sensitivity analyses...")

        # Wilcoxon test (non-parametric alternative)
        log("Non-parametric paired test...")
        conf_rate = df_rates['conf_rate'].values
        acc_rate = df_rates['acc_rate'].values

        wilcoxon_result = stats.wilcoxon(conf_rate, acc_rate, alternative='greater')
        wilcoxon_statistic = wilcoxon_result.statistic
        wilcoxon_p = wilcoxon_result.pvalue

        log(f"statistic = {wilcoxon_statistic:.2f}, p = {wilcoxon_p:.6f}")

        # Paired t-test excluding outliers
        log("Re-running paired t-test without outliers...")
        non_outlier_mask = ~outlier_mask
        conf_rate_no_out = conf_rate[non_outlier_mask]
        acc_rate_no_out = acc_rate[non_outlier_mask]

        if len(conf_rate_no_out) > 0:
            ttest_no_outliers = stats.ttest_rel(conf_rate_no_out, acc_rate_no_out)
            ttest_no_out_t = ttest_no_outliers.statistic
            ttest_no_out_p = ttest_no_outliers.pvalue / 2 if ttest_no_outliers.statistic > 0 else 1 - (ttest_no_outliers.pvalue / 2)
            log(f"t = {ttest_no_out_t:.4f}, p = {ttest_no_out_p:.6f} (N={len(conf_rate_no_out)})")
        else:
            ttest_no_out_t = np.nan
            ttest_no_out_p = np.nan
            log("No data after outlier exclusion")

        # Log transformation test (if needed)
        if log_transform_needed:
            log("Testing log-transformed ratio...")
            # Ensure all ratios are positive
            if np.all(valid_ratios > 0):
                log_ratios = np.log(valid_ratios)
                # One-sample t-test on log(ratio) vs 0
                log_ratio_test = stats.ttest_1samp(log_ratios, popmean=0)
                log_ratio_t = log_ratio_test.statistic
                log_ratio_p = log_ratio_test.pvalue / 2 if log_ratio_test.statistic > 0 else 1 - (log_ratio_test.pvalue / 2)
                log(f"t = {log_ratio_t:.4f}, p = {log_ratio_p:.6f}")
            else:
                log("Cannot log-transform (negative ratios)")
                log_ratio_t = np.nan
                log_ratio_p = np.nan
        else:
            log_ratio_t = np.nan
            log_ratio_p = np.nan
        # Generate Histogram Data

        log("Generating histogram bin data...")

        # Histogram for difference
        hist_diff, bins_diff = np.histogram(difference, bins=20)
        bin_centers_diff = (bins_diff[:-1] + bins_diff[1:]) / 2
        hist_diff_data = pd.DataFrame({
            'bin_center': bin_centers_diff,
            'count': hist_diff,
            'distribution': 'difference'
        })

        # Histogram for ratio
        hist_ratio, bins_ratio = np.histogram(valid_ratios, bins=20)
        bin_centers_ratio = (bins_ratio[:-1] + bins_ratio[1:]) / 2
        hist_ratio_data = pd.DataFrame({
            'bin_center': bin_centers_ratio,
            'count': hist_ratio,
            'distribution': 'ratio'
        })

        hist_data = pd.concat([hist_diff_data, hist_ratio_data], ignore_index=True)
        # Save Analysis Outputs

        log("Saving analysis outputs...")

        # Normality checks
        normality_results = pd.DataFrame({
            'test_name': ['difference', 'ratio'],
            'W_statistic': [W_diff, W_ratio],
            'p_value': [p_diff, p_ratio],
            'interpretation': [interpretation_diff, interpretation_ratio]
        })
        output_path_norm = RQ_DIR / "data" / "step07_normality_checks.csv"
        normality_results.to_csv(output_path_norm, index=False, encoding='utf-8')
        log(f"{output_path_norm.name} ({len(normality_results)} rows)")

        # Outlier detection
        output_path_outliers = RQ_DIR / "data" / "step07_outlier_detection.csv"
        df_outliers.to_csv(output_path_outliers, index=False, encoding='utf-8')
        log(f"{output_path_outliers.name} ({len(df_outliers)} rows)")

        # Ratio diagnostics
        ratio_diagnostics = pd.DataFrame({
            'skewness': [skew_ratio],
            'kurtosis': [kurt_ratio],
            'shapiro_W': [W_ratio],
            'shapiro_p': [p_ratio],
            'log_transform_needed': [log_transform_needed]
        })
        output_path_ratio_diag = RQ_DIR / "data" / "step07_ratio_diagnostics.csv"
        ratio_diagnostics.to_csv(output_path_ratio_diag, index=False, encoding='utf-8')
        log(f"{output_path_ratio_diag.name} ({len(ratio_diagnostics)} rows)")

        # Sensitivity analyses
        sensitivity_results = pd.DataFrame({
            'wilcoxon_statistic': [wilcoxon_statistic],
            'wilcoxon_p': [wilcoxon_p],
            'ttest_no_outliers_t': [ttest_no_out_t],
            'ttest_no_outliers_p': [ttest_no_out_p],
            'log_ratio_t': [log_ratio_t],
            'log_ratio_p': [log_ratio_p]
        })
        output_path_sens = RQ_DIR / "data" / "step07_sensitivity_analyses.csv"
        sensitivity_results.to_csv(output_path_sens, index=False, encoding='utf-8')
        log(f"{output_path_sens.name} ({len(sensitivity_results)} rows)")

        # Q-Q plot data
        output_path_qq = RQ_DIR / "data" / "step07_qqplot_data.csv"
        qqplot_data.to_csv(output_path_qq, index=False, encoding='utf-8')
        log(f"{output_path_qq.name} ({len(qqplot_data)} rows)")

        # Histogram data
        output_path_hist = RQ_DIR / "data" / "step07_histogram_data.csv"
        hist_data.to_csv(output_path_hist, index=False, encoding='utf-8')
        log(f"{output_path_hist.name} ({len(hist_data)} rows)")
        # Validation

        log("Running inline validation...")

        # Check normality file
        if len(normality_results) != 2:
            log(f"Expected 2 rows in normality file, got {len(normality_results)}")

        for interp in normality_results['interpretation']:
            if interp not in ['Normal', 'Non-normal']:
                log(f"Invalid interpretation: {interp}")

        # Check p-values in range
        for p in [p_diff, p_ratio, wilcoxon_p]:
            if not (0 <= p <= 1):
                log(f"p-value outside [0,1]: {p}")

        # Check Shapiro W in range
        for w in [W_diff, W_ratio]:
            if not (0 <= w <= 1):
                log(f"Shapiro W outside [0,1]: {w}")

        log(f"Shapiro-Wilk (difference): W={W_diff:.4f}, p={p_diff:.6f}")
        log(f"Shapiro-Wilk (ratio): W={W_ratio:.4f}, p={p_ratio:.6f}")
        log(f"Outliers detected: {len(df_outliers)}")
        log(f"Ratio skewness: {skew_ratio:.4f}")
        log("Sensitivity analyses: Wilcoxon, outlier exclusion, log transform")

        log("Step 7 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
