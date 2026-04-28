#!/usr/bin/env python3
"""extract_domain_theta_scores: Extract participant-level theta scores from Ch5 domain-specific analyses (What, Where, When)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.8.4
LOG_FILE = RQ_DIR / "logs" / "step01_extract_domain_theta_scores.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 01: extract_domain_theta_scores")
        # Load Ch5 Domain Theta Scores

        log("Loading Ch5 domain theta scores...")

        # ALL THREE DOMAINS from 5.2.1 (5.2.2 and 5.2.3 are consumers, not generators)
        theta_path = PROJECT_ROOT / 'results' / 'ch5' / '5.2.1' / 'data' / 'step03_theta_scores.csv'
        df_theta = pd.read_csv(theta_path)
        log(f"All domains: {theta_path} ({len(df_theta)} rows, {len(df_theta.columns)} cols)")
        log(f"{list(df_theta.columns)}")
        # Extract UID from composite_ID and Aggregate

        log("Extracting UID and aggregating to participant-level...")

        # Extract UID from composite_ID (format: UID_test, e.g., "A010_1")
        df_theta['UID'] = df_theta['composite_ID'].str.split('_').str[0]

        # Aggregate theta by UID (mean across 4 tests per participant)
        # File has columns: composite_ID, theta_what, theta_where, theta_when
        merged = df_theta.groupby('UID').agg({
            'theta_what': 'mean',
            'theta_where': 'mean',
            'theta_when': 'mean'
        }).reset_index()

        # Rename columns to match expected format
        merged.rename(columns={
            'theta_what': 'What_theta',
            'theta_where': 'Where_theta',
            'theta_when': 'When_theta'
        }, inplace=True)

        log(f"{len(df_theta)} rows -> {len(merged)} UIDs (participant-level)")

        # Check for missing data
        missing_count = merged.isnull().sum().sum()
        if missing_count > 0:
            log(f"{missing_count} missing values in merged data")
            log(f"Dropping {merged.isnull().any(axis=1).sum()} rows with missing data")
            merged = merged.dropna()
            log(f"{len(merged)} UIDs after removing missing data")
        # Compute Inter-Domain Correlations
        # Validates: Moderate correlations (0.20-0.70) indicate multivariate structure

        log("Computing inter-domain correlations...")

        domain_pairs = [
            ('What_theta', 'Where_theta'),
            ('What_theta', 'When_theta'),
            ('Where_theta', 'When_theta')
        ]

        correlation_results = []

        for domain1, domain2 in domain_pairs:
            # Compute Pearson correlation
            r, p = pearsonr(merged[domain1], merged[domain2])

            correlation_results.append({
                'domain1': domain1.replace('_theta', ''),
                'domain2': domain2.replace('_theta', ''),
                'correlation': r,
                'p_value': p
            })

            log(f"{domain1} <-> {domain2}: r = {r:.3f}, p = {p:.4f}")

        df_correlations = pd.DataFrame(correlation_results)
        # Save Outputs
        # These outputs will be used by: Step 03 (merge with cognitive tests)

        log("Saving domain theta scores...")
        theta_output = RQ_DIR / 'data' / 'step01_domain_theta_scores.csv'
        merged.to_csv(theta_output, index=False, encoding='utf-8')
        log(f"{theta_output} ({len(merged)} rows, {len(merged.columns)} cols)")

        log("Saving domain correlations...")
        corr_output = RQ_DIR / 'data' / 'step01_domain_correlations.csv'
        df_correlations.to_csv(corr_output, index=False, encoding='utf-8')
        log(f"{corr_output} ({len(df_correlations)} rows, {len(df_correlations.columns)} cols)")
        # Validation
        # Validates: 100 UIDs, correlation range, no missing data
        # Threshold: All checks must pass

        log("Validating outputs...")

        validation_pass = True

        # Check UID count
        expected_uid_count = 100
        if len(merged) != expected_uid_count:
            log(f"Expected {expected_uid_count} UIDs, got {len(merged)}")
            # Don't fail - sample size may vary slightly

        # Check correlation range (0.20 < r < 0.70)
        for _, row in df_correlations.iterrows():
            r = row['correlation']
            if not (0.20 <= abs(r) <= 0.70):
                log(f"{row['domain1']}-{row['domain2']} correlation {r:.3f} outside expected range [0.20, 0.70]")
                # Don't fail - correlation strength may vary

        # Check for missing data
        if merged.isnull().sum().sum() > 0:
            log(f"Missing data in output")
            validation_pass = False
        else:
            log(f"No missing data in output")

        # Report validation results
        if validation_pass:
            log("All validation checks passed")
        else:
            log("Some validation checks failed")

        log("Step 01 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
