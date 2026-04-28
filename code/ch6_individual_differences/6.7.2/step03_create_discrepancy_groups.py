#!/usr/bin/env python3
"""create_discrepancy_groups: Classify participants into three groups based on ±1 SD discrepancy cutoffs:"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_create_discrepancy_groups.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 03: Create Discrepancy Groups")
        # Load Discrepancy Scores
        log("Loading discrepancy scores from step 2...")

        input_file = RQ_DIR / "data" / "step02_discrepancy_scores.csv"
        df = pd.read_csv(input_file)

        log(f"{len(df)} participants with discrepancy scores")
        # STEP 2-6: Process BOTH discrepancy metrics

        discrepancy_metrics = {
            'RAVLT_Total': 'Discrepancy',
            'RAVLT_Pct_Ret': 'Discrepancy_PctRet'
        }

        all_group_descs = []
        all_sensitivity = []

        for metric_label, disc_col in discrepancy_metrics.items():
            group_col = f'Group_{metric_label}'
            log(f"\n=== Processing {metric_label} (column: {disc_col}) ===")

            # Calculate cutoffs
            disc_mean = df[disc_col].mean()
            disc_sd = df[disc_col].std()
            upper_cutoff = disc_mean + 1.0 * disc_sd
            lower_cutoff = disc_mean - 1.0 * disc_sd

            log(f"{metric_label} distribution: M={disc_mean:.4f}, SD={disc_sd:.4f}")
            log(f"Upper cutoff (+1 SD): {upper_cutoff:.4f}")
            log(f"Lower cutoff (-1 SD): {lower_cutoff:.4f}")

            # Assign groups
            def assign_group(discrepancy, uc=upper_cutoff, lc=lower_cutoff):
                if discrepancy > uc:
                    return "VR-favored"
                elif discrepancy < lc:
                    return "RAVLT-favored"
                else:
                    return "Concordant"

            df[group_col] = df[disc_col].apply(assign_group)

            group_counts = df[group_col].value_counts()
            log(f"VR-favored: {group_counts.get('VR-favored', 0)} participants")
            log(f"RAVLT-favored: {group_counts.get('RAVLT-favored', 0)} participants")
            log(f"Concordant: {group_counts.get('Concordant', 0)} participants")

            # Group descriptives
            group_desc = df.groupby(group_col)[disc_col].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).rename(columns={'count': 'n'}).reset_index()
            group_desc = group_desc.rename(columns={group_col: 'Group'})
            group_desc['metric'] = metric_label

            for _, row in group_desc.iterrows():
                log(f"{row['Group']}: n={row['n']}, M={row['mean']:.4f}, SD={row['std']:.4f}, range=[{row['min']:.4f}, {row['max']:.4f}]")

            all_group_descs.append(group_desc)

            # Sensitivity analysis
            for cutoff_sd in [0.75, 1.0, 1.25]:
                alt_upper = disc_mean + cutoff_sd * disc_sd
                alt_lower = disc_mean - cutoff_sd * disc_sd

                alt_groups = df[disc_col].apply(
                    lambda x, au=alt_upper, al=alt_lower: "VR-favored" if x > au
                             else "RAVLT-favored" if x < al
                             else "Concordant"
                )

                alt_group_counts = alt_groups.value_counts()

                all_sensitivity.append({
                    'metric': metric_label,
                    'cutoff_sd': cutoff_sd,
                    'vr_favored_n': alt_group_counts.get('VR-favored', 0),
                    'ravlt_favored_n': alt_group_counts.get('RAVLT-favored', 0),
                    'concordant_n': alt_group_counts.get('Concordant', 0)
                })

                log(f"{metric_label} ±{cutoff_sd} SD: VR={alt_group_counts.get('VR-favored', 0)}, RAVLT={alt_group_counts.get('RAVLT-favored', 0)}, Concordant={alt_group_counts.get('Concordant', 0)}")

            # Check minimum group sizes
            min_group_size = 10
            for group in ['VR-favored', 'RAVLT-favored', 'Concordant']:
                group_n = group_counts.get(group, 0)
                if group_n < min_group_size:
                    log(f"{metric_label} {group} group has only {group_n} participants (< {min_group_size})")
                else:
                    log(f"{metric_label} {group} group has adequate sample size ({group_n} >= {min_group_size})")

        # Combine across metrics
        combined_group_desc = pd.concat(all_group_descs, ignore_index=True)
        sensitivity_df = pd.DataFrame(all_sensitivity)

        # For backward compatibility, keep 'Group' and 'Discrepancy' as the RAVLT_Total version
        df['Group'] = df['Group_RAVLT_Total']
        # Save Outputs
        log("Saving group assignments and descriptives...")

        output_assignments = RQ_DIR / "data" / "step03_group_assignments.csv"
        df[['UID', 'Discrepancy', 'Discrepancy_PctRet', 'Group', 'Group_RAVLT_Total', 'Group_RAVLT_Pct_Ret']].to_csv(output_assignments, index=False, encoding='utf-8')
        log(f"{output_assignments.name} ({len(df)} rows)")

        output_descriptives = RQ_DIR / "data" / "step03_group_descriptives.csv"
        combined_group_desc.to_csv(output_descriptives, index=False, encoding='utf-8')
        log(f"{output_descriptives.name} ({len(combined_group_desc)} rows)")

        output_sensitivity = RQ_DIR / "data" / "step03_sensitivity_analysis.csv"
        sensitivity_df.to_csv(output_sensitivity, index=False, encoding='utf-8')
        log(f"{output_sensitivity.name} ({len(sensitivity_df)} rows)")
        # Validate Group Assignments
        log("Running validate_data_format...")

        validation_result = validate_data_format(
            df=df,
            required_cols=['UID', 'Discrepancy', 'Group']
        )

        if validation_result.get('valid', False):
            log("Group assignment structure valid")
        else:
            log(f"Validation warnings: {validation_result}")

        # Check all participants assigned to exactly one group (both metrics)
        for metric_label in ['RAVLT_Total', 'RAVLT_Pct_Ret']:
            group_col = f'Group_{metric_label}'
            expected_groups = {'VR-favored', 'RAVLT-favored', 'Concordant'}
            actual_groups = set(df[group_col].unique())
            if actual_groups != expected_groups:
                log(f"{metric_label} unexpected group values: {actual_groups} (expected {expected_groups})")
                raise ValueError(f"{metric_label} group values do not match expected")
            else:
                log(f"{metric_label} group values correct: {actual_groups}")

        log(f"All {len(df)} participants assigned to groups for both metrics")

        log("Step 03 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
