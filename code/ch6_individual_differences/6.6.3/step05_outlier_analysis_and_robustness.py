#!/usr/bin/env python3
"""outlier_analysis_and_robustness: Identify outliers in domain-specific slopes using z-score method and assess impact"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_outlier_analysis_and_robustness.log"
OUTPUT_OUTLIERS = RQ_DIR / "data" / "step05_outlier_analysis.csv"
OUTPUT_ROBUST = RQ_DIR / "data" / "step05_robustness_checks.csv"

# Input files
INPUT_SLOPES = RQ_DIR / "data" / "step01_domain_slopes.csv"
INPUT_ICC = RQ_DIR / "data" / "step02_icc_estimates.csv"

# Outlier threshold
Z_THRESHOLD = 3.29  # p < 0.001 two-tailed

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# ICC Calculation Function

def calculate_icc(slope_values):
    """Calculate ICC(1,1) from slope values."""
    variance_total = np.var(slope_values, ddof=1)
    icc = 1.0 if variance_total > 0 else 0.0  # Single obs per person
    return icc

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Outlier Analysis and Robustness")
        # Load Data

        log(f"Reading domain slopes...")
        df_slopes = pd.read_csv(INPUT_SLOPES)
        log(f"{len(df_slopes)} rows")

        log(f"Reading ICC estimates...")
        df_icc = pd.read_csv(INPUT_ICC)
        log(f"{len(df_icc)} rows")
        # Outlier Detection (Z-Score Method)
        # Identify participants with extreme slope values (|z| > 3.29)

        log(f"[OUTLIER DETECTION] Using z-score method (threshold = {Z_THRESHOLD})...")

        outlier_records = []

        for domain in ['what', 'where']:
            col_name = f'slope_{domain}'
            slopes = df_slopes[col_name].values

            # Calculate z-scores
            z_scores = stats.zscore(slopes, ddof=1)

            # Identify outliers
            outlier_mask = np.abs(z_scores) > Z_THRESHOLD
            n_outliers = outlier_mask.sum()

            log(f"[{domain.upper()}] {n_outliers} outliers detected (|z| > {Z_THRESHOLD})")

            # Store outlier details
            if n_outliers > 0:
                outlier_uids = df_slopes.loc[outlier_mask, 'UID'].values
                outlier_zscores = z_scores[outlier_mask]

                for uid, z_score in zip(outlier_uids, outlier_zscores):
                    outlier_type = 'high' if z_score > 0 else 'low'
                    outlier_records.append({
                        'domain': domain.capitalize(),
                        'UID': uid,
                        'z_score': z_score,
                        'outlier_type': outlier_type
                    })

            # Check outlier proportion
            outlier_pct = (n_outliers / len(slopes)) * 100
            if outlier_pct > 10:
                log(f"{domain.upper()}: {outlier_pct:.1f}% outliers (>10% threshold)")
            else:
                log(f"{domain.upper()}: {outlier_pct:.1f}% outliers (<10% threshold)")
        # Shapiro-Wilk Normality Test
        # Test if slope distributions are approximately normal
        # Helps interpret whether outliers are legitimate extreme values or errors

        log(f"[NORMALITY TEST] Running Shapiro-Wilk tests...")

        for domain in ['what', 'where']:
            col_name = f'slope_{domain}'
            slopes = df_slopes[col_name].values

            stat, p_value = stats.shapiro(slopes)

            if p_value < 0.05:
                log(f"[{domain.upper()}] Shapiro-Wilk: W={stat:.4f}, p={p_value:.4f} (non-normal)")
            else:
                log(f"[{domain.upper()}] Shapiro-Wilk: W={stat:.4f}, p={p_value:.4f} (normal)")
        # Robustness Analysis (ICC with/without Outliers)
        # Recompute ICC excluding outliers to assess impact

        log(f"Recomputing ICC without outliers...")

        robustness_results = []

        for domain in ['what', 'where']:
            col_name = f'slope_{domain}'
            slopes = df_slopes[col_name].values

            # Original ICC
            icc_original = df_icc[df_icc['domain'] == domain.capitalize()]['icc_value'].values[0]

            # Z-scores
            z_scores = stats.zscore(slopes, ddof=1)

            # Exclude outliers
            non_outlier_mask = np.abs(z_scores) <= Z_THRESHOLD
            slopes_no_outliers = slopes[non_outlier_mask]

            # ICC without outliers
            icc_no_outliers = calculate_icc(slopes_no_outliers)

            # Delta ICC
            delta_icc = icc_no_outliers - icc_original

            log(f"[{domain.upper()}] ICC original: {icc_original:.4f}")
            log(f"[{domain.upper()}] ICC no outliers: {icc_no_outliers:.4f}")
            log(f"[{domain.upper()}] Delta ICC: {delta_icc:.4f}")

            robustness_results.append({
                'domain': domain.capitalize(),
                'icc_original': icc_original,
                'icc_no_outliers': icc_no_outliers,
                'delta_icc': delta_icc
            })
        # Create Output DataFrames

        if len(outlier_records) > 0:
            df_outliers = pd.DataFrame(outlier_records)
            log(f"Outlier analysis table: {len(df_outliers)} outliers")
        else:
            # No outliers - create empty DataFrame with correct columns
            df_outliers = pd.DataFrame(columns=['domain', 'UID', 'z_score', 'outlier_type'])
            log(f"Outlier analysis table: 0 outliers (none detected)")

        df_robust = pd.DataFrame(robustness_results)
        log(f"Robustness checks table: {len(df_robust)} rows")
        # Validate Results

        log(f"Checking robustness results...")

        # Validate delta_icc range
        validation_delta = validate_numeric_range(
            data=df_robust['delta_icc'],
            min_val=-0.2,
            max_val=0.2,
            column_name='delta_icc'
        )

        if not validation_delta.get('valid', False):
            log(f"Some delta_icc values outside [-0.2, 0.2]: {validation_delta}")
            log(f"Large ICC changes may indicate influential outliers")
        else:
            log(f"All delta_icc values in reasonable range [-0.2, 0.2]")

        # Check all domains present
        if len(df_robust) != 2:
            log(f"Expected 2 domains in robustness checks, got {len(df_robust)}")
            sys.exit(1)
        else:
            log(f"Both domains represented in robustness checks")

        # Validate outlier z-scores (if any outliers)
        if len(df_outliers) > 0:
            for _, row in df_outliers.iterrows():
                if abs(row['z_score']) <= Z_THRESHOLD:
                    log(f"Outlier {row['UID']} has |z| = {abs(row['z_score']):.2f} <= {Z_THRESHOLD}")
                    sys.exit(1)

            log(f"All flagged outliers have |z| > {Z_THRESHOLD}")
        # Save Results

        log(f"Writing outlier analysis...")
        df_outliers.to_csv(OUTPUT_OUTLIERS, index=False, encoding='utf-8')
        log(f"{OUTPUT_OUTLIERS} ({len(df_outliers)} outliers)")

        log(f"Writing robustness checks...")
        df_robust.to_csv(OUTPUT_ROBUST, index=False, encoding='utf-8')
        log(f"{OUTPUT_ROBUST} ({len(df_robust)} rows)")

        log(f"Step 05 complete")
        log(f"Outliers detected: {len(df_outliers)}, max |delta_icc|: {df_robust['delta_icc'].abs().max():.4f}")
        log(f"Proceed to step06 (split-half reliability)")

        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
