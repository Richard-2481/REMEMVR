#!/usr/bin/env python3
"""compute_discrepancy_scores: Calculate discrepancy scores (REMEMVR_z - RAVLT_z) and examine distribution."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import analysis and validation tools
from tools.analysis_extensions import compute_discrepancy_scores
from tools.validation import validate_numeric_range

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step02_compute_discrepancy_scores.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Compute Discrepancy Scores")
        # Load Standardized Scores
        log("Loading standardized z-scores from step 1...")

        input_file = RQ_DIR / "data" / "step01_standardized_scores.csv"
        df = pd.read_csv(input_file)

        log(f"{len(df)} participants with standardized scores")

        # Verify required columns
        required_cols = ['UID', 'REMEMVR_z', 'RAVLT_z', 'RAVLT_Pct_Ret_z']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        # Compute Discrepancy Scores
        # Formula: Discrepancy = REMEMVR_z - RAVLT_z
        # Positive = Better on VR than RAVLT (VR-favored)
        # Negative = Better on RAVLT than VR (RAVLT-favored)

        log("Computing discrepancy scores (REMEMVR_z - RAVLT_z)...")

        # Use catalogued function for RAVLT Total discrepancy
        discrepancy_df = compute_discrepancy_scores(
            traditional_scores=df['RAVLT_z'],
            vr_scores=df['REMEMVR_z']
        )

        # Compute both discrepancy metrics manually to ensure correct formula
        # RAVLT Total discrepancy (original)
        df['Discrepancy'] = df['REMEMVR_z'] - df['RAVLT_z']
        # RAVLT Percent Retention discrepancy (new)
        df['Discrepancy_PctRet'] = df['REMEMVR_z'] - df['RAVLT_Pct_Ret_z']

        log(f"Discrepancy scores for {len(df)} participants")

        # Verify calculations
        mean_discrepancy = df['Discrepancy'].mean()
        sd_discrepancy = df['Discrepancy'].std()
        log(f"Discrepancy (Total): M={mean_discrepancy:.4f}, SD={sd_discrepancy:.4f}")

        mean_discrepancy_pr = df['Discrepancy_PctRet'].mean()
        sd_discrepancy_pr = df['Discrepancy_PctRet'].std()
        log(f"Discrepancy (Pct Ret): M={mean_discrepancy_pr:.4f}, SD={sd_discrepancy_pr:.4f}")
        # Describe Distribution (both metrics)
        log("Computing distribution statistics...")

        discrepancy_metrics = {
            'RAVLT_Total': 'Discrepancy',
            'RAVLT_Pct_Ret': 'Discrepancy_PctRet'
        }

        all_ci_results = []

        for metric_label, disc_col in discrepancy_metrics.items():
            log(f"--- {metric_label} ---")

            # Basic descriptive statistics
            descriptives = {
                'mean': df[disc_col].mean(),
                'sd': df[disc_col].std(),
                'median': df[disc_col].median(),
                'q1': df[disc_col].quantile(0.25),
                'q3': df[disc_col].quantile(0.75),
                'iqr': df[disc_col].quantile(0.75) - df[disc_col].quantile(0.25),
                'min': df[disc_col].min(),
                'max': df[disc_col].max(),
                'range': df[disc_col].max() - df[disc_col].min(),
                'skewness': stats.skew(df[disc_col]),
                'kurtosis': stats.kurtosis(df[disc_col])
            }

            # Normality test (Shapiro-Wilk)
            shapiro_stat, shapiro_p = stats.shapiro(df[disc_col])
            descriptives['shapiro_W'] = shapiro_stat
            descriptives['shapiro_p'] = shapiro_p

            # Count by direction
            vr_favored = (df[disc_col] > 0).sum()
            ravlt_favored = (df[disc_col] < 0).sum()
            concordant = (df[disc_col] == 0).sum()

            log(f"VR-favored ({disc_col} > 0): {vr_favored} ({vr_favored/len(df)*100:.1f}%)")
            log(f"RAVLT-favored ({disc_col} < 0): {ravlt_favored} ({ravlt_favored/len(df)*100:.1f}%)")
            log(f"Concordant ({disc_col} = 0): {concordant}")
            log(f"Shapiro-Wilk normality test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

            # Bootstrap 95% CIs
            n_iterations = 1000
            np.random.seed(42)

            boot_stats = {stat: [] for stat in ['mean', 'sd', 'median', 'iqr', 'skewness', 'kurtosis']}

            for i in range(n_iterations):
                boot_indices = np.random.choice(len(df), size=len(df), replace=True)
                boot_sample = df.iloc[boot_indices][disc_col]

                boot_stats['mean'].append(boot_sample.mean())
                boot_stats['sd'].append(boot_sample.std())
                boot_stats['median'].append(boot_sample.median())
                boot_stats['iqr'].append(boot_sample.quantile(0.75) - boot_sample.quantile(0.25))
                boot_stats['skewness'].append(stats.skew(boot_sample))
                boot_stats['kurtosis'].append(stats.kurtosis(boot_sample))

            # Calculate percentile CIs
            for stat_name in ['mean', 'sd', 'median', 'iqr', 'min', 'max', 'range', 'skewness', 'kurtosis', 'shapiro_W', 'shapiro_p']:
                value = descriptives[stat_name]

                if stat_name in boot_stats:
                    ci_lower, ci_upper = np.percentile(boot_stats[stat_name], [2.5, 97.5])
                else:
                    ci_lower, ci_upper = np.nan, np.nan

                all_ci_results.append({
                    'metric': metric_label,
                    'statistic': stat_name,
                    'value': value,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                })

            log(f"{metric_label} Mean 95% CI: [{[r for r in all_ci_results if r['metric']==metric_label and r['statistic']=='mean'][0]['ci_lower']:.4f}, {[r for r in all_ci_results if r['metric']==metric_label and r['statistic']=='mean'][0]['ci_upper']:.4f}]")

        descriptives_df = pd.DataFrame(all_ci_results)
        # Save Outputs
        log("Saving discrepancy scores and descriptives...")

        output_scores = RQ_DIR / "data" / "step02_discrepancy_scores.csv"
        df[['UID', 'REMEMVR_z', 'RAVLT_z', 'RAVLT_Pct_Ret_z', 'Discrepancy', 'Discrepancy_PctRet']].to_csv(output_scores, index=False, encoding='utf-8')
        log(f"{output_scores.name} ({len(df)} rows)")

        output_descriptives = RQ_DIR / "data" / "step02_discrepancy_descriptives.csv"
        descriptives_df.to_csv(output_descriptives, index=False, encoding='utf-8')
        log(f"{output_descriptives.name} ({len(descriptives_df)} rows)")
        # Validate Discrepancy Scores
        log("Running validate_numeric_range...")

        for disc_col, z_col, label in [
            ('Discrepancy', 'RAVLT_z', 'RAVLT Total'),
            ('Discrepancy_PctRet', 'RAVLT_Pct_Ret_z', 'RAVLT Pct Ret')
        ]:
            validation_result = validate_numeric_range(
                data=df[disc_col],
                min_val=-4.0,
                max_val=4.0,
                column_name=disc_col
            )

            if validation_result.get('valid', False):
                log(f"{label} discrepancy scores within reasonable range [-4, 4]")
            else:
                log(f"{label} validation warnings: {validation_result}")

            # Check for missing values
            n_missing = df[disc_col].isna().sum()
            if n_missing > 0:
                log(f"{n_missing} missing {label} discrepancy values")
                raise ValueError(f"Missing {label} discrepancy values detected")
            else:
                log(f"No missing {label} discrepancy values")

            # Verify calculation (REMEMVR_z - z_col)
            calculated_discrepancy = df['REMEMVR_z'] - df[z_col]
            if not np.allclose(df[disc_col], calculated_discrepancy):
                log(f"{label} discrepancy calculation mismatch")
                raise ValueError(f"{disc_col} != REMEMVR_z - {z_col}")
            else:
                log(f"{label} discrepancy calculation verified (REMEMVR_z - {z_col})")

        log("Step 02 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
