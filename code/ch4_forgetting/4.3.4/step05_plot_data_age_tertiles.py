#!/usr/bin/env python3
"""Prepare Plot Data by Age Tertiles: Create age tertiles (Young/Middle/Older) and aggregate observed theta means by"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.3.4 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step05_plot_data_age_tertiles.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 5: Prepare Plot Data by Age Tertiles")
        # Load Input Data

        log("Loading LMM input data...")
        input_path = RQ_DIR / "data" / "step01_lmm_input.csv"

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(df)} rows, {len(df.columns)} cols)")

        # Validate input structure
        required_cols = ['composite_ID', 'UID', 'test', 'paradigm', 'theta',
                        'Age', 'TSVR_hours']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Input data missing required columns: {missing_cols}")

        # Validate row count
        if len(df) != 1200:
            log(f"Expected 1200 rows, found {len(df)} rows")

        log(f"Data structure: {len(df['UID'].unique())} participants, "
            f"{df['paradigm'].nunique()} paradigms, {df['test'].nunique()} tests")
        # Create Age Tertiles

        log("Creating age tertiles...")

        # Get unique participant ages (one row per UID)
        unique_participants = df[['UID', 'Age']].drop_duplicates()
        log(f"{len(unique_participants)} unique participants")
        log(f"Age range: {unique_participants['Age'].min():.1f} to "
            f"{unique_participants['Age'].max():.1f} years")

        # Create tertiles using qcut (equal-sized bins)
        # This assigns each participant to a tertile based on their age
        unique_participants['age_tertile'] = pd.qcut(
            unique_participants['Age'],
            q=3,
            labels=['Young', 'Middle', 'Older']
        )

        # Merge tertile assignment back to full dataset
        df = df.merge(unique_participants[['UID', 'age_tertile']], on='UID', how='left')

        # Validate tertile assignment
        if df['age_tertile'].isna().any():
            raise ValueError("Age tertile assignment failed - some participants have NaN tertile")

        # Report tertile composition
        tertile_counts = unique_participants['age_tertile'].value_counts().sort_index()
        log(f"Age tertile distribution:")
        for tertile, count in tertile_counts.items():
            log(f"  - {tertile}: {count} participants")

        # Create tertile metadata (cutpoints and ranges)
        tertile_metadata = []
        for i, label in enumerate(['Young', 'Middle', 'Older'], 1):
            tertile_data = unique_participants[unique_participants['age_tertile'] == label]
            tertile_metadata.append({
                'tertile': i,
                'label': label,
                'age_min': tertile_data['Age'].min(),
                'age_max': tertile_data['Age'].max(),
                'N': len(tertile_data)
            })

        df_tertiles = pd.DataFrame(tertile_metadata)
        log(f"Tertile age ranges:")
        for _, row in df_tertiles.iterrows():
            log(f"  - {row['label']}: {row['age_min']:.1f} to {row['age_max']:.1f} years "
                f"(N={row['N']})")
        # Aggregate Statistics by Age Tertile, Paradigm, and Test
        #               age_tertile x paradigm x test combination

        log("Aggregating statistics by age tertile, paradigm, and test...")

        # Group by age_tertile, paradigm, and test
        # Compute: mean theta, SE of theta, N observations, mean TSVR_hours
        grouped = df.groupby(['age_tertile', 'paradigm', 'test']).agg({
            'theta': ['mean', 'sem', 'count'],  # Mean, standard error, N
            'TSVR_hours': 'mean'  # Mean time for plotting
        }).reset_index()

        # Flatten multi-level columns
        grouped.columns = ['age_tertile', 'paradigm', 'test', 'theta_mean',
                          'theta_SE', 'N', 'TSVR_hours_mean']

        # Validate aggregation
        expected_rows = 3 * 3 * 4  # 3 tertiles x 3 paradigms x 4 tests
        if len(grouped) != expected_rows:
            log(f"Expected {expected_rows} rows in aggregated data, "
                f"found {len(grouped)} rows")

        # Check for missing combinations
        missing_combos = []
        for tertile in ['Young', 'Middle', 'Older']:
            for paradigm in df['paradigm'].unique():
                for test in df['test'].unique():
                    if len(grouped[(grouped['age_tertile'] == tertile) &
                                  (grouped['paradigm'] == paradigm) &
                                  (grouped['test'] == test)]) == 0:
                        missing_combos.append(f"{tertile} x {paradigm} x test {test}")

        if missing_combos:
            log(f"Missing combinations: {', '.join(missing_combos)}")

        # Validate SE values
        invalid_se = grouped[grouped['theta_SE'] <= 0]
        if len(invalid_se) > 0:
            log(f"{len(invalid_se)} rows have invalid SE (<=0)")

        # Validate sample sizes
        small_n = grouped[grouped['N'] < 10]
        if len(small_n) > 0:
            log(f"{len(small_n)} groups have N < 10 (may be unreliable):")
            for _, row in small_n.iterrows():
                log(f"  - {row['age_tertile']} x {row['paradigm']} x test {row['test']}: "
                    f"N={row['N']}")

        log(f"Aggregated data: {len(grouped)} rows")
        log(f"Mean sample size per group: {grouped['N'].mean():.1f}")
        log(f"Range of sample sizes: {grouped['N'].min():.0f} to {grouped['N'].max():.0f}")
        # Save Output Files
        # These outputs will be used by plotting step (step06)

        log("Saving plot data...")

        # Save aggregated plot data
        output_path = RQ_DIR / "data" / "step05_plot_data.csv"
        grouped.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(grouped)} rows, {len(grouped.columns)} cols)")

        # Save age tertile metadata
        tertiles_path = RQ_DIR / "data" / "step05_age_tertiles.csv"
        df_tertiles.to_csv(tertiles_path, index=False, encoding='utf-8')
        log(f"{tertiles_path.name} ({len(df_tertiles)} rows, {len(df_tertiles.columns)} cols)")
        # Validation (Inline)
        # Validates: Completeness of factorial design, positive SEs, adequate N

        log("Running completeness checks...")

        validation_passed = True
        validation_messages = []

        # Check 1: All paradigms present
        paradigms_present = set(grouped['paradigm'].unique())
        expected_paradigms = {'IFR', 'ICR', 'IRE'}
        if paradigms_present != expected_paradigms:
            validation_passed = False
            validation_messages.append(
                f"Missing paradigms: {expected_paradigms - paradigms_present}"
            )
        else:
            validation_messages.append("All 3 paradigms present (IFR, ICR, IRE)")

        # Check 2: All age tertiles present
        tertiles_present = set(grouped['age_tertile'].unique())
        expected_tertiles = {'Young', 'Middle', 'Older'}
        if tertiles_present != expected_tertiles:
            validation_passed = False
            validation_messages.append(
                f"Missing age tertiles: {expected_tertiles - tertiles_present}"
            )
        else:
            validation_messages.append("All 3 age tertiles present (Young, Middle, Older)")

        # Check 3: Expected row count
        if len(grouped) == expected_rows:
            validation_messages.append(f"Row count: {len(grouped)} (expected {expected_rows})")
        else:
            validation_passed = False
            validation_messages.append(
                f"Row count: {len(grouped)} (expected {expected_rows})"
            )

        # Check 4: No missing values
        if grouped.isna().sum().sum() == 0:
            validation_messages.append("No missing values in aggregated data")
        else:
            validation_passed = False
            na_counts = grouped.isna().sum()
            validation_messages.append(
                f"Missing values detected: {na_counts[na_counts > 0].to_dict()}"
            )

        # Check 5: Positive standard errors
        if (grouped['theta_SE'] > 0).all():
            validation_messages.append("All standard errors positive")
        else:
            validation_passed = False
            validation_messages.append(
                f"{(grouped['theta_SE'] <= 0).sum()} groups have non-positive SE"
            )

        # Check 6: Adequate sample sizes
        if (grouped['N'] >= 10).all():
            validation_messages.append("All groups have N >= 10")
        else:
            # This is a warning, not a failure
            validation_messages.append(
                f"{(grouped['N'] < 10).sum()} groups have N < 10"
            )

        # Report all validation results
        for msg in validation_messages:
            log(f"{msg}")

        if not validation_passed:
            raise ValueError("Validation failed - see log for details")

        log("Step 5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
