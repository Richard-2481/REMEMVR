#!/usr/bin/env python3
"""prepare_analysis_dataset: Merge RAVLT and theta data into analysis-ready dataset with quality checks."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# RQ directory
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_prepare_analysis_dataset.log"
OUTPUT_DATASET = RQ_DIR / "data" / "step03_analysis_dataset.csv"
OUTPUT_CORR = RQ_DIR / "data" / "step03_correlation_matrix.csv"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Prepare Analysis Dataset")
        # Load RAVLT Scores
        log("Loading RAVLT scores...")

        ravlt_path = RQ_DIR / "data" / "step01_ravlt_scores.csv"
        ravlt_df = pd.read_csv(ravlt_path)

        log(f"{ravlt_path.name} ({len(ravlt_df)} participants, {len(ravlt_df.columns)} columns)")
        # Load Theta Scores
        log("Loading theta scores...")

        theta_path = RQ_DIR / "data" / "step02_theta_scores.csv"
        theta_df = pd.read_csv(theta_path)

        log(f"{theta_path.name} ({len(theta_df)} participants, {len(theta_df.columns)} columns)")
        # Merge Datasets
        log("Merging RAVLT and theta data on UID...")

        # Inner join: only participants with both RAVLT and theta data
        merged_df = pd.merge(ravlt_df, theta_df, on='UID', how='inner')

        log(f"{len(merged_df)} participants after merge")

        if len(merged_df) < 100:
            log(f"Expected 100 participants, found {len(merged_df)} after merge")
            log(f"RAVLT: {len(ravlt_df)} participants, Theta: {len(theta_df)} participants")
        # Select Analysis Columns
        # Select only columns needed for regression:
        # - UID (identifier)
        # - Predictors: Total_z, Learning_z, LearningSlope_z, Recognition_z (standardized RAVLT)
        # - Outcome: theta_all (REMEMVR omnibus ability)
        log("Selecting analysis columns...")

        analysis_cols = ['UID', 'Total_z', 'Learning_z', 'LearningSlope_z', 'Forgetting_z', 'Recognition_z', 'PctRet_z', 'theta_all']
        analysis_df = merged_df[analysis_cols].copy()

        log(f"{len(analysis_cols)} columns for analysis")
        # Quality Checks
        log("Running quality checks...")

        # Check for missing data
        n_missing = analysis_df.isnull().sum().sum()
        if n_missing > 0:
            log(f"{n_missing} missing values detected")
            log(f"Missing by column:\n{analysis_df.isnull().sum()[analysis_df.isnull().sum() > 0]}")

            # Drop rows with missing values
            before_drop = len(analysis_df)
            analysis_df = analysis_df.dropna()
            after_drop = len(analysis_df)
            log(f"Dropped {before_drop - after_drop} rows with missing values")
        else:
            log("No missing values")

        # Check final sample size
        log(f"Final sample size: N={len(analysis_df)}")
        # Compute Correlation Matrix
        log("Computing predictor intercorrelation matrix...")

        # Select only predictors (not UID or outcome)
        predictor_cols = ['Total_z', 'Learning_z', 'LearningSlope_z', 'Forgetting_z', 'Recognition_z', 'PctRet_z']
        corr_matrix = analysis_df[predictor_cols].corr()

        log("Correlation matrix")

        # Check for high correlations (multicollinearity warning)
        high_corr_threshold = 0.9
        high_corrs = []

        for i in range(len(predictor_cols)):
            for j in range(i+1, len(predictor_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > high_corr_threshold:
                    high_corrs.append((predictor_cols[i], predictor_cols[j], corr_val))

        if high_corrs:
            log(f"High correlations detected (>{high_corr_threshold}):")
            for pred1, pred2, corr_val in high_corrs:
                log(f"  {pred1} <-> {pred2}: r={corr_val:.3f}")
        else:
            log(f"No correlations exceed {high_corr_threshold} threshold")

        # Display correlation matrix
        log("Predictor correlation matrix:")
        for i, row in enumerate(predictor_cols):
            corr_str = "  " + row + ":"
            for j, col in enumerate(predictor_cols):
                corr_str += f" {corr_matrix.iloc[i,j]:6.3f}"
            log(corr_str)
        # Save Outputs
        log("Saving analysis dataset...")

        analysis_df.to_csv(OUTPUT_DATASET, index=False, encoding='utf-8')
        log(f"{OUTPUT_DATASET} ({len(analysis_df)} participants, {len(analysis_df.columns)} columns)")

        log("Saving correlation matrix...")
        corr_matrix.to_csv(OUTPUT_CORR, encoding='utf-8')
        log(f"{OUTPUT_CORR}")
        # Summary Statistics
        log("Descriptive statistics:")
        log(f"  Total_z: M={analysis_df['Total_z'].mean():.2f}, SD={analysis_df['Total_z'].std():.2f}")
        log(f"  Learning_z: M={analysis_df['Learning_z'].mean():.2f}, SD={analysis_df['Learning_z'].std():.2f}")
        log(f"  LearningSlope_z: M={analysis_df['LearningSlope_z'].mean():.2f}, SD={analysis_df['LearningSlope_z'].std():.2f}")
        log(f"  Forgetting_z: M={analysis_df['Forgetting_z'].mean():.2f}, SD={analysis_df['Forgetting_z'].std():.2f}")
        log(f"  Recognition_z: M={analysis_df['Recognition_z'].mean():.2f}, SD={analysis_df['Recognition_z'].std():.2f}")
        log(f"  PctRet_z: M={analysis_df['PctRet_z'].mean():.2f}, SD={analysis_df['PctRet_z'].std():.2f}")
        log(f"  theta_all: M={analysis_df['theta_all'].mean():.2f}, SD={analysis_df['theta_all'].std():.2f}")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
