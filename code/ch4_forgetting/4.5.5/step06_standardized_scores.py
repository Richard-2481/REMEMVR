#!/usr/bin/env python3
"""Z-Standardize All Measurements: Standardize all measurements (IRT theta, Full CTT, Purified CTT) to z-scores"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step06_standardized_scores.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 6: Z-Standardize All Measurements")
        # Load Input Data
        #   - Theta scores: 400 rows (100 UID × 4 test), wide format
        #   - Full CTT: 800 rows (100 UID × 4 test × 2 location_type), long format
        #   - Purified CTT: 800 rows (same structure as Full CTT)

        log("Loading IRT theta scores from RQ 5.5.1...")
        theta_path = Path("results/ch5/5.5.1/data/step03_theta_scores.csv")
        theta_df = pd.read_csv(theta_path, encoding='utf-8')
        log(f"{theta_path.name} ({len(theta_df)} rows, {len(theta_df.columns)} cols)")
        log(f"Theta columns: {theta_df.columns.tolist()}")

        log("Loading Full CTT scores from Step 2...")
        full_ctt_path = RQ_DIR / "data" / "step02_ctt_full_scores.csv"
        full_ctt_df = pd.read_csv(full_ctt_path, encoding='utf-8')
        log(f"{full_ctt_path.name} ({len(full_ctt_df)} rows, {len(full_ctt_df.columns)} cols)")

        log("Loading Purified CTT scores from Step 3...")
        purif_ctt_path = RQ_DIR / "data" / "step03_ctt_purified_scores.csv"
        purif_ctt_df = pd.read_csv(purif_ctt_path, encoding='utf-8')
        log(f"{purif_ctt_path.name} ({len(purif_ctt_df)} rows, {len(purif_ctt_df.columns)} cols)")
        # Parse composite_ID and Reshape Theta to Long Format

        log("Parsing composite_ID into UID and test...")
        # composite_ID format: "A010_1" -> UID="A010", test="T1" (converted to match CTT format)
        theta_df[['UID', 'test_num']] = theta_df['composite_ID'].str.split('_', expand=True)
        # Convert test number to T1/T2/T3/T4 format to match CTT scores
        theta_df['test'] = 'T' + theta_df['test_num']
        theta_df.drop(columns=['test_num'], inplace=True)
        log(f"Parsed {len(theta_df)} composite_IDs")
        log(f"Test values: {theta_df['test'].unique().tolist()}")

        log("Reshaping theta to long format...")
        # Create two dataframes: one for source, one for destination
        theta_source = theta_df[['UID', 'test', 'theta_source']].copy()
        theta_source['location_type'] = 'source'
        theta_source.rename(columns={'theta_source': 'theta'}, inplace=True)

        theta_dest = theta_df[['UID', 'test', 'theta_destination']].copy()
        theta_dest['location_type'] = 'destination'
        theta_dest.rename(columns={'theta_destination': 'theta'}, inplace=True)

        # Concatenate
        theta_long = pd.concat([theta_source, theta_dest], ignore_index=True)
        log(f"Theta reshaped to long format ({len(theta_long)} rows)")
        log(f"Theta location_type counts: {theta_long['location_type'].value_counts().to_dict()}")
        # Merge All Measurements

        log("Merging theta with Full CTT...")
        merged_df = theta_long.merge(
            full_ctt_df,
            on=['UID', 'test', 'location_type'],
            how='inner',
            validate='one_to_one'
        )
        log(f"After Full CTT merge: {len(merged_df)} rows")

        log("Merging with Purified CTT...")
        merged_df = merged_df.merge(
            purif_ctt_df,
            on=['UID', 'test', 'location_type'],
            how='inner',
            validate='one_to_one'
        )
        log(f"After Purified CTT merge: {len(merged_df)} rows")

        # Validate merge
        if len(merged_df) != 800:
            log(f"Expected 800 rows after merge, got {len(merged_df)}")
            raise ValueError(f"Merge produced {len(merged_df)} rows, expected 800")

        log(f"Merged columns: {merged_df.columns.tolist()}")
        log(f"Missing values: {merged_df.isnull().sum().to_dict()}")
        # Z-Standardize Within Location Type
        # Formula: z_i = (x_i - mean(x)) / std(x) where mean/std computed within location_type
        # ddof=1: Sample standard deviation (N-1 denominator) for consistency with LMM

        log("Z-standardizing measurements within location_type...")
        log("Using sample std (ddof=1) for consistency with LMM estimation")

        # Standardize theta
        merged_df['irt_z'] = merged_df.groupby('location_type')['theta'].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1)
        )
        log("IRT theta -> irt_z")

        # Standardize Full CTT
        merged_df['ctt_full_z'] = merged_df.groupby('location_type')['ctt_full_score'].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1)
        )
        log("Full CTT -> ctt_full_z")

        # Standardize Purified CTT
        merged_df['ctt_purified_z'] = merged_df.groupby('location_type')['ctt_purified_score'].transform(
            lambda x: (x - x.mean()) / x.std(ddof=1)
        )
        log("Purified CTT -> ctt_purified_z")
        # Validate Standardization
        # Criteria: For each location_type and each z-score column:
        #   - mean ≈ 0 (±0.05 tolerance)
        #   - std ≈ 1 (±0.05 tolerance)
        # Validation ensures standardization was computed correctly

        log("Validating z-score standardization...")

        validation_results = []
        z_cols = ['irt_z', 'ctt_full_z', 'ctt_purified_z']
        location_types = ['source', 'destination']

        all_valid = True
        for loc in location_types:
            loc_data = merged_df[merged_df['location_type'] == loc]
            log(f"\nLocation type: {loc}")

            for col in z_cols:
                mean_val = loc_data[col].mean()
                std_val = loc_data[col].std(ddof=1)

                mean_ok = abs(mean_val) <= 0.05
                std_ok = abs(std_val - 1.0) <= 0.05

                status = "" if (mean_ok and std_ok) else ""
                log(f"  {status} {col}: mean={mean_val:.6f}, std={std_val:.6f}")

                validation_results.append({
                    'location_type': loc,
                    'column': col,
                    'mean': mean_val,
                    'std': std_val,
                    'mean_ok': mean_ok,
                    'std_ok': std_ok,
                    'valid': mean_ok and std_ok
                })

                if not (mean_ok and std_ok):
                    all_valid = False

        if not all_valid:
            log("\nStandardization validation failed")
            log("Some z-scores do not have mean ≈ 0 or std ≈ 1")
            validation_df = pd.DataFrame(validation_results)
            log(f"Validation details:\n{validation_df.to_string()}")
            raise ValueError("Z-standardization validation failed")

        log("\nAll z-scores validated: mean ≈ 0, std ≈ 1")

        # Check for NaN values
        nan_counts = merged_df[z_cols].isnull().sum()
        if nan_counts.any():
            log(f"NaN values found in z-scores: {nan_counts.to_dict()}")
            raise ValueError("NaN values detected in standardized scores")
        log("No NaN values in z-score columns")
        # Save Standardized Scores
        # Output: 800 rows with UID, test, location_type, irt_z, ctt_full_z, ctt_purified_z
        # These z-scores will be used by Step 7 for parallel LMM fitting and AIC comparison

        output_cols = ['UID', 'test', 'location_type', 'irt_z', 'ctt_full_z', 'ctt_purified_z']
        output_df = merged_df[output_cols].copy()

        log(f"Saving standardized scores...")
        output_path = RQ_DIR / "data" / "step06_standardized_scores.csv"
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(output_df)} rows, {len(output_df.columns)} cols)")

        # Final summary
        log("\nStep 6 Complete:")
        log(f"  Input files: 3 (theta, full_ctt, purified_ctt)")
        log(f"  Output rows: {len(output_df)}")
        log(f"  Location types: {output_df['location_type'].nunique()}")
        log(f"  Participants: {output_df['UID'].nunique()}")
        log(f"  Test sessions: {output_df['test'].nunique()}")
        log(f"  Z-score columns: {len(z_cols)}")
        log(f"  Validation: All z-scores have mean ≈ 0, std ≈ 1 per location_type")

        log("Step 6 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
