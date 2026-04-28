#!/usr/bin/env python3
"""
Step 08: Prepare Plot Data for Visualization

Creates plot source CSVs for two visualizations:
  1. Correlation comparison (r_full vs r_purified per dimension)
  2. AIC comparison (AIC_Full vs AIC_Purified per dimension)

Output format: Long format for grouped bar charts (6 rows per plot).
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step08_prepare_plot_data.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("Step 08: Prepare Plot Data for Visualization")

        # Load correlation analysis results
        corr_path = RQ_DIR / "data" / "step05_correlation_analysis.csv"
        log(f"Reading {corr_path}")
        correlation_analysis = pd.read_csv(corr_path, encoding='utf-8')
        log(f"{len(correlation_analysis)} rows")

        # Load LMM model comparison results
        lmm_path = RQ_DIR / "data" / "step07_lmm_model_comparison.csv"
        log(f"Reading {lmm_path}")
        lmm_model_comparison = pd.read_csv(lmm_path, encoding='utf-8')
        log(f"{len(lmm_model_comparison)} rows")
        # PLOT 1: Correlation Comparison
        log("\n[PLOT 1] Preparing correlation comparison data")

        # Reshape from wide to long format
        corr_data_long = []

        for idx, row in correlation_analysis.iterrows():
            dimension = row['dimension']

            # Full CTT correlation
            corr_data_long.append({
                'dimension': dimension,
                'CTT_type': 'Full',
                'r_value': row['r_full']
            })

            # Purified CTT correlation
            corr_data_long.append({
                'dimension': dimension,
                'CTT_type': 'Purified',
                'r_value': row['r_purified']
            })

        correlation_comparison_data = pd.DataFrame(corr_data_long)
        log(f"{len(correlation_comparison_data)} rows (3 dimensions x 2 CTT types)")

        # Validation: Check expected structure
        if len(correlation_comparison_data) != 6:
            raise ValueError(f"Expected 6 rows, got {len(correlation_comparison_data)}")

        log(f"Dimensions: {correlation_comparison_data['dimension'].unique().tolist()}")
        log(f"CTT types: {correlation_comparison_data['CTT_type'].unique().tolist()}")

        # Save Plot 1 data
        output_path_1 = RQ_DIR / "data" / "step08_correlation_comparison_data.csv"
        log(f"Writing {output_path_1}")
        correlation_comparison_data.to_csv(output_path_1, index=False, encoding='utf-8')
        log(f"{len(correlation_comparison_data)} rows")
        # PLOT 2: AIC Comparison
        log("\n[PLOT 2] Preparing AIC comparison data")

        # Reshape from wide to long format
        aic_data_long = []

        for idx, row in lmm_model_comparison.iterrows():
            dimension = row['dimension']

            # Full CTT AIC
            aic_data_long.append({
                'dimension': dimension,
                'CTT_type': 'Full',
                'AIC': row['AIC_Full']
            })

            # Purified CTT AIC
            aic_data_long.append({
                'dimension': dimension,
                'CTT_type': 'Purified',
                'AIC': row['AIC_Purified']
            })

        aic_comparison_data = pd.DataFrame(aic_data_long)
        log(f"{len(aic_comparison_data)} rows (3 dimensions x 2 CTT types)")

        # Validation: Check expected structure
        if len(aic_comparison_data) != 6:
            raise ValueError(f"Expected 6 rows, got {len(aic_comparison_data)}")

        log(f"Dimensions: {aic_comparison_data['dimension'].unique().tolist()}")
        log(f"CTT types: {aic_comparison_data['CTT_type'].unique().tolist()}")

        # Save Plot 2 data
        output_path_2 = RQ_DIR / "data" / "step08_aic_comparison_data.csv"
        log(f"Writing {output_path_2}")
        aic_comparison_data.to_csv(output_path_2, index=False, encoding='utf-8')
        log(f"{len(aic_comparison_data)} rows")
        # Validation: Check no NaN values
        log("\nChecking for NaN values")

        nan_count_1 = correlation_comparison_data.isna().sum().sum()
        nan_count_2 = aic_comparison_data.isna().sum().sum()

        if nan_count_1 > 0 or nan_count_2 > 0:
            raise ValueError(f"Found NaN values: Plot 1 = {nan_count_1}, Plot 2 = {nan_count_2}")

        log("No NaN values in either plot dataset")
        # Validation: Check complete factorial design (3 dimensions x 2 CTT types)
        log("\nChecking complete factorial design")

        expected_dimensions = ['Common', 'Congruent', 'Incongruent']
        expected_ctt_types = ['Full', 'Purified']

        for df, name in [(correlation_comparison_data, 'correlation'), (aic_comparison_data, 'AIC')]:
            actual_dims = set(df['dimension'].unique())
            actual_types = set(df['CTT_type'].unique())

            if actual_dims != set(expected_dimensions):
                raise ValueError(f"{name} plot: Missing dimensions: {set(expected_dimensions) - actual_dims}")

            if actual_types != set(expected_ctt_types):
                raise ValueError(f"{name} plot: Missing CTT types: {set(expected_ctt_types) - actual_types}")

            # Check each dimension has both CTT types
            for dim in expected_dimensions:
                dim_data = df[df['dimension'] == dim]
                if len(dim_data) != 2:
                    raise ValueError(f"{name} plot: Dimension {dim} has {len(dim_data)} rows (expected 2)")

        log("Complete factorial design verified for both plots")

        log("\nStep 08 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
