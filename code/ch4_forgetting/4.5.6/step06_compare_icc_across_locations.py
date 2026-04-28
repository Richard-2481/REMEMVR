#!/usr/bin/env python3
"""Compare ICC Across Locations: Compare ICC_intercept between Source and Destination locations to test whether"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Note: validation done manually (no external validation tool call)

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.6
LOG_FILE = RQ_DIR / "logs" / "step06_compare_icc_across_locations.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Compare ICC Across Locations")
        # Load Input Data

        log("Loading ICC estimates from Step 3...")
        input_path = RQ_DIR / "data" / "step03_icc_estimates.csv"

        if not input_path.exists():
            log(f"Input file not found: {input_path}")
            sys.exit(1)

        icc_estimates = pd.read_csv(input_path, encoding='utf-8')
        log(f"{input_path.name} ({len(icc_estimates)} rows, {len(icc_estimates.columns)} cols)")

        # Check expected structure
        if len(icc_estimates) != 6:
            log(f"Expected 6 rows, got {len(icc_estimates)}")

        expected_cols = ['location', 'icc_type', 'value', 'interpretation']
        if not all(col in icc_estimates.columns for col in expected_cols):
            log(f"Missing expected columns. Expected: {expected_cols}, Got: {list(icc_estimates.columns)}")
            sys.exit(1)
        # Filter by Location

        log("Separating Source and Destination ICC estimates...")

        source_df = icc_estimates[icc_estimates['location'] == 'Source'].copy()
        destination_df = icc_estimates[icc_estimates['location'] == 'Destination'].copy()

        log(f"Source: {len(source_df)} rows, Destination: {len(destination_df)} rows")

        if len(source_df) != 3:
            log(f"Expected 3 Source ICC types, got {len(source_df)}")
        if len(destination_df) != 3:
            log(f"Expected 3 Destination ICC types, got {len(destination_df)}")
        # Merge on ICC Type

        log("Merging Source and Destination on icc_type...")

        # Rename value columns before merge
        source_df = source_df.rename(columns={'value': 'source_value', 'interpretation': 'source_interpretation'})
        destination_df = destination_df.rename(columns={'value': 'destination_value', 'interpretation': 'destination_interpretation'})

        # Merge on icc_type
        comparison_df = source_df[['icc_type', 'source_value', 'source_interpretation']].merge(
            destination_df[['icc_type', 'destination_value', 'destination_interpretation']],
            on='icc_type',
            how='inner'
        )

        log(f"Combined table: {len(comparison_df)} rows")
        # Compute Difference
        # Formula: difference = source_value - destination_value
        # Interpretation:
        #   diff > 0: Source shows higher stability
        #   diff < 0: Destination shows higher stability
        #   diff near 0: Equivalent stability

        log("Computing difference (source_value - destination_value)...")

        comparison_df['difference'] = comparison_df['source_value'] - comparison_df['destination_value']

        log(f"Difference column added")
        # Generate Interpretation String
        # Describe difference magnitude and direction

        log("Generating interpretation strings...")

        def generate_interpretation(row):
            """Generate descriptive interpretation based on difference magnitude."""
            diff = row['difference']
            icc_type = row['icc_type']

            if icc_type == 'ICC_intercept':
                # Primary comparison of interest (baseline stability)
                if diff > 0.10:
                    return "Source shows substantially higher baseline stability than Destination"
                elif diff > 0.05:
                    return "Source shows moderately higher baseline stability than Destination"
                elif diff > 0:
                    return "Source shows slightly higher baseline stability than Destination"
                elif diff > -0.05:
                    return "Source and Destination show approximately equal baseline stability"
                elif diff > -0.10:
                    return "Destination shows moderately higher baseline stability than Source"
                else:
                    return "Destination shows substantially higher baseline stability than Source"
            elif 'slope' in icc_type.lower():
                # Slope ICCs expected near zero (both locations)
                if abs(diff) < 0.01:
                    return "Both locations show near-zero slope variance (expected pattern)"
                else:
                    return f"Difference in slope variance: {diff:.4f} (both near zero)"
            else:
                # Generic interpretation
                if diff > 0:
                    return f"Source higher by {diff:.4f}"
                elif diff < 0:
                    return f"Destination higher by {abs(diff):.4f}"
                else:
                    return "Equivalent"

        comparison_df['interpretation'] = comparison_df.apply(generate_interpretation, axis=1)

        log(f"Interpretation column added")
        # Select Output Columns
        # Output format: icc_type, source_value, destination_value, difference, interpretation

        log("Selecting output columns...")

        output_df = comparison_df[['icc_type', 'source_value', 'destination_value', 'difference', 'interpretation']].copy()

        # Sort by icc_type for consistent output (intercept, slope_conditional, slope_simple)
        output_df = output_df.sort_values('icc_type').reset_index(drop=True)

        log(f"Final output: {len(output_df)} rows, {len(output_df.columns)} cols")
        # Save Output
        # Output: data/step06_location_icc_comparison.csv

        log("Saving ICC comparison table...")
        output_path = RQ_DIR / "data" / "step06_location_icc_comparison.csv"
        output_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path.name} ({len(output_df)} rows, {len(output_df.columns)} cols)")
        # Run Validation
        # Validate: 3 rows, 5 columns, correct structure, value bounds

        log("Checking output structure...")

        # Check row count
        if len(output_df) != 3:
            log(f"FAIL - Expected 3 rows, got {len(output_df)}")
            sys.exit(1)
        log("PASS - Row count: 3")

        # Check column count
        expected_cols = ['icc_type', 'source_value', 'destination_value', 'difference', 'interpretation']
        if list(output_df.columns) != expected_cols:
            log(f"FAIL - Expected columns {expected_cols}, got {list(output_df.columns)}")
            sys.exit(1)
        log("PASS - All expected columns present")

        # Additional value range checks
        log("Checking ICC value bounds...")

        # Check source_value in [0, 1]
        if not all((output_df['source_value'] >= 0) & (output_df['source_value'] <= 1)):
            invalid_rows = output_df[(output_df['source_value'] < 0) | (output_df['source_value'] > 1)]
            log(f"FAIL - source_value out of bounds [0, 1]:")
            log(f"{invalid_rows}")
            sys.exit(1)
        log("PASS - source_value in [0, 1]")

        # Check destination_value in [0, 1]
        if not all((output_df['destination_value'] >= 0) & (output_df['destination_value'] <= 1)):
            invalid_rows = output_df[(output_df['destination_value'] < 0) | (output_df['destination_value'] > 1)]
            log(f"FAIL - destination_value out of bounds [0, 1]:")
            log(f"{invalid_rows}")
            sys.exit(1)
        log("PASS - destination_value in [0, 1]")

        # Check difference correctly computed
        expected_diff = output_df['source_value'] - output_df['destination_value']
        if not all(np.abs(output_df['difference'] - expected_diff) < 1e-10):
            log("FAIL - difference not correctly computed")
            sys.exit(1)
        log("PASS - difference correctly computed")

        # Check for NaN values
        if output_df.isna().any().any():
            log("FAIL - NaN values detected")
            log(f"NaN counts per column:\n{output_df.isna().sum()}")
            sys.exit(1)
        log("PASS - No NaN values")
        # Report Summary

        log("\nICC Comparison Results:")
        log("=" * 60)
        for _, row in output_df.iterrows():
            log(f"{row['icc_type']}:")
            log(f"  Source:      {row['source_value']:.4f}")
            log(f"  Destination: {row['destination_value']:.4f}")
            log(f"  Difference:  {row['difference']:+.4f}")
            log(f"  {row['interpretation']}")
            log("")

        # Highlight ICC_intercept (primary comparison)
        intercept_row = output_df[output_df['icc_type'] == 'ICC_intercept'].iloc[0]
        log("[PRIMARY COMPARISON] ICC_intercept (baseline stability):")
        log(f"  Source:      {intercept_row['source_value']:.4f}")
        log(f"  Destination: {intercept_row['destination_value']:.4f}")
        log(f"  Difference:  {intercept_row['difference']:+.4f}")

        if intercept_row['difference'] > 0:
            log(f"  -> Source shows HIGHER baseline stability (supports hypothesis if destination encoding weaker)")
        elif intercept_row['difference'] < 0:
            log(f"  -> Destination shows HIGHER baseline stability (contradicts hypothesis)")
        else:
            log(f"  -> Equivalent baseline stability (null finding)")

        log("\nStep 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
