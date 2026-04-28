#!/usr/bin/env python3
"""extract_demographic_predictors: Extract demographic variables (Age, Education, VR_Experience, NART, BVMT) from dfnonvr.csv"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_columns

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_extract_demographic_predictors.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

if __name__ == "__main__":
    try:
        log("Step 04: Extract Demographic Predictors")

        # Load group assignments
        log("Loading group assignments...")
        groups_df = pd.read_csv(RQ_DIR / "data" / "step03_group_assignments.csv")
        log(f"{len(groups_df)} participants with group assignments")

        # Load dfnonvr.csv with exact column names
        log("Loading dfnonvr.csv...")
        dfnonvr = pd.read_csv(PROJECT_ROOT / "data" / "dfnonvr.csv")
        log(f"{len(dfnonvr)} participants from dfnonvr.csv")

        # Extract required variables with exact column names from DATA_DICTIONARY.md
        demographic_vars = {
            'Age': 'age',
            'Education': 'education',
            'VR_Experience': 'vr-exposure',
            'NART': 'nart-score',
            'BVMT': 'bvmt-total-recall',
        }

        # Check which columns exist
        available_vars = {}
        missing_vars = []

        for var_name, col_name in demographic_vars.items():
            if col_name in dfnonvr.columns:
                available_vars[var_name] = col_name
            else:
                missing_vars.append(f"{var_name} (expected column: {col_name})")

        if missing_vars:
            log(f"Missing variables in dfnonvr.csv: {missing_vars}")

        # Extract available demographic data
        demo_cols = ['UID'] + list(available_vars.values())
        demo_df = dfnonvr[demo_cols].copy()

        # Rename columns to standard names
        rename_map = {v: k for k, v in available_vars.items()}
        demo_df = demo_df.rename(columns=rename_map)

        # Merge with group assignments (now includes both Group_RAVLT_Total and Group_RAVLT_Pct_Ret)
        log("Merging demographics with group assignments...")
        merged_df = groups_df.merge(demo_df, on='UID', how='left')
        log(f"{len(merged_df)} participants with demographics and groups")
        log(f"Group columns available: {[c for c in merged_df.columns if c.startswith('Group')]}")

        # Check for missing data
        log("Analyzing missing data patterns...")
        missing_report = []
        for var in available_vars.keys():
            if var in merged_df.columns:
                n_missing = merged_df[var].isna().sum()
                percent_missing = (n_missing / len(merged_df)) * 100
                missing_report.append({
                    'variable': var,
                    'n_missing': n_missing,
                    'percent_missing': percent_missing,
                    'pattern': 'random' if n_missing > 0 else 'complete'
                })
                log(f"{var}: {n_missing} missing ({percent_missing:.1f}%)")

        missing_df = pd.DataFrame(missing_report)

        # Exclude variables with >15% missing data
        valid_vars = []
        for var in available_vars.keys():
            if var in merged_df.columns:
                missing_pct = (merged_df[var].isna().sum() / len(merged_df)) * 100
                if missing_pct <= 15.0:
                    valid_vars.append(var)
                else:
                    log(f"Excluding {var} due to {missing_pct:.1f}% missing data")

        log(f"Valid variables for analysis: {valid_vars}")

        # Final demographic data with valid variables only
        # Include both group columns and both discrepancy columns
        group_cols = [c for c in merged_df.columns if c.startswith('Group')]
        disc_cols = [c for c in merged_df.columns if c.startswith('Discrepancy')]
        final_cols = ['UID'] + group_cols + disc_cols + valid_vars
        final_df = merged_df[final_cols].copy()

        # Compute by-group descriptive statistics for both grouping schemes
        log("Computing by-group descriptives...")
        descriptive_results = []

        for group_col_name in ['Group', 'Group_RAVLT_Total', 'Group_RAVLT_Pct_Ret']:
            if group_col_name not in final_df.columns:
                continue
            for var in valid_vars:
                group_stats = final_df.groupby(group_col_name)[var].agg([
                    'count', 'mean', 'std', 'median',
                    lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
                ]).rename(columns={'<lambda_0>': 'q1', '<lambda_1>': 'q3'})
                group_stats['variable'] = var
                group_stats['grouping'] = group_col_name
                group_stats['iqr'] = group_stats['q3'] - group_stats['q1']
                descriptive_results.append(group_stats.reset_index().rename(columns={group_col_name: 'Group'}))

        if descriptive_results:
            descriptives_df = pd.concat(descriptive_results, ignore_index=True)
            descriptives_df = descriptives_df[['grouping', 'Group', 'variable', 'count', 'mean', 'std', 'median', 'q1', 'q3', 'iqr']]
            descriptives_df = descriptives_df.rename(columns={'count': 'n'})
        else:
            descriptives_df = pd.DataFrame(columns=['grouping', 'Group', 'variable', 'n', 'mean', 'std', 'median', 'q1', 'q3', 'iqr'])

        # Save outputs
        log("Saving demographic data and reports...")

        output_data = RQ_DIR / "data" / "step04_demographic_data.csv"
        final_df.to_csv(output_data, index=False, encoding='utf-8')
        log(f"{output_data.name} ({len(final_df)} rows)")

        output_descriptives = RQ_DIR / "data" / "step04_demographic_descriptives.csv"
        descriptives_df.to_csv(output_descriptives, index=False, encoding='utf-8')
        log(f"{output_descriptives.name} ({len(descriptives_df)} rows)")

        output_missing = RQ_DIR / "data" / "step04_missing_data_report.csv"
        missing_df.to_csv(output_missing, index=False, encoding='utf-8')
        log(f"{output_missing.name} ({len(missing_df)} rows)")

        # Validation
        log("Running validate_data_columns...")
        validation_result = validate_data_columns(
            df=final_df,
            required_columns=['UID', 'Group']
        )

        if validation_result.get('valid', False):
            log("Demographic data structure valid")
        else:
            log(f"Validation warnings: {validation_result}")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
