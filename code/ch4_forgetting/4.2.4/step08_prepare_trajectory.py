#!/usr/bin/env python3
"""Prepare Trajectory Comparison Plot Data: Create plot source CSV for trajectory comparison showing IRT vs CTT trajectories"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_plot_data_completeness

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/rq11 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step08_prepare_trajectory.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 8: Prepare Trajectory Comparison Plot Data")
        # Load Input Data

        log("Loading IRT LMM input...")
        irt_lmm_input = pd.read_csv(RQ_DIR / "data" / "step03_irt_lmm_input.csv", encoding='utf-8')
        log(f"step03_irt_lmm_input.csv ({len(irt_lmm_input)} rows, {len(irt_lmm_input.columns)} cols)")

        log("Loading CTT LMM input...")
        ctt_lmm_input = pd.read_csv(RQ_DIR / "data" / "step03_ctt_lmm_input.csv", encoding='utf-8')
        log(f"step03_ctt_lmm_input.csv ({len(ctt_lmm_input)} rows, {len(ctt_lmm_input.columns)} cols)")
        # Aggregate IRT Scores by TSVR_hours + Domain

        log("Aggregating IRT scores by TSVR_hours + domain...")

        irt_agg = irt_lmm_input.groupby(['TSVR_hours', 'domain'])['IRT_score'].agg([
            ('mean_score', 'mean'),
            ('sem', lambda x: x.sem()),  # Standard error of the mean
            ('n', 'count')
        ]).reset_index()

        # Compute 95% CI using SEM * 1.96
        irt_agg['CI_lower'] = irt_agg['mean_score'] - 1.96 * irt_agg['sem']
        irt_agg['CI_upper'] = irt_agg['mean_score'] + 1.96 * irt_agg['sem']

        # Add model identifier
        irt_agg['model'] = 'IRT'

        # Drop SEM column (not needed for output)
        irt_agg = irt_agg[['TSVR_hours', 'domain', 'model', 'mean_score', 'CI_lower', 'CI_upper', 'n']]

        log(f"IRT: {len(irt_agg)} aggregated rows")
        # Aggregate CTT Scores by TSVR_hours + Domain

        log("Aggregating CTT scores by TSVR_hours + domain...")

        ctt_agg = ctt_lmm_input.groupby(['TSVR_hours', 'domain'])['CTT_score'].agg([
            ('mean_score', 'mean'),
            ('sem', lambda x: x.sem()),  # Standard error of the mean
            ('n', 'count')
        ]).reset_index()

        # Compute 95% CI using SEM * 1.96
        ctt_agg['CI_lower'] = ctt_agg['mean_score'] - 1.96 * ctt_agg['sem']
        ctt_agg['CI_upper'] = ctt_agg['mean_score'] + 1.96 * ctt_agg['sem']

        # Add model identifier
        ctt_agg['model'] = 'CTT'

        # Drop SEM column (not needed for output)
        ctt_agg = ctt_agg[['TSVR_hours', 'domain', 'model', 'mean_score', 'CI_lower', 'CI_upper', 'n']]

        log(f"CTT: {len(ctt_agg)} aggregated rows")
        # Stack IRT and CTT Aggregations
        # These outputs will be used by: plotting pipeline for trajectory comparison visualization

        log("Combining IRT and CTT aggregations...")

        trajectory_data = pd.concat([irt_agg, ctt_agg], axis=0, ignore_index=True)

        # Sort by domain, model, TSVR_hours for consistent plotting
        trajectory_data = trajectory_data.sort_values(by=['domain', 'model', 'TSVR_hours']).reset_index(drop=True)

        log(f"Combined trajectory data: {len(trajectory_data)} rows")
        # Save Trajectory Plot Data
        # Output: data/step08_trajectory_data.csv
        # Contains: Aggregated means and CIs for all domain x model x timepoint combinations
        # Columns: TSVR_hours, domain, model, mean_score, CI_lower, CI_upper, n

        output_path = RQ_DIR / "data" / "step08_trajectory_data.csv"
        log(f"Saving trajectory data to {output_path.name}...")

        trajectory_data.to_csv(output_path, index=False, encoding='utf-8')

        log(f"{output_path.name} ({len(trajectory_data)} rows, {len(trajectory_data.columns)} cols)")
        log(f"Unique TSVR_hours values: {sorted(trajectory_data['TSVR_hours'].unique())}")
        log(f"Domains: {sorted(trajectory_data['domain'].unique())}")
        log(f"Models: {sorted(trajectory_data['model'].unique())}")
        # Run Validation Tool
        # Validates: All domains and models present (complete factorial design)
        # Threshold: Both domains (What, Where - When excluded), both models (IRT, CTT)

        log("Running validate_plot_data_completeness (When excluded)...")

        validation_result = validate_plot_data_completeness(
            plot_data=trajectory_data,
            required_domains=['What', 'Where'],  # NO 'When' - excluded
            required_groups=['IRT', 'CTT'],
            domain_col='domain',
            group_col='model'
        )

        # Report validation results
        if validation_result['valid']:
            log(f"PASS - {validation_result['message']}")
        else:
            log(f"FAIL - {validation_result['message']}")
            if 'missing_domains' in validation_result and validation_result['missing_domains']:
                log(f"Missing domains: {validation_result['missing_domains']}")
            if 'missing_groups' in validation_result and validation_result['missing_groups']:
                log(f"Missing models: {validation_result['missing_groups']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Additional sanity check: CI_lower < mean_score < CI_upper
        log("Checking CI bounds bracket mean...")
        ci_violations = trajectory_data[
            (trajectory_data['CI_lower'] > trajectory_data['mean_score']) |
            (trajectory_data['CI_upper'] < trajectory_data['mean_score'])
        ]

        if len(ci_violations) > 0:
            log(f"FAIL - {len(ci_violations)} rows have invalid CI bounds")
            log(f"Violations:\n{ci_violations}")
            raise ValueError("CI bounds do not bracket mean for some rows")
        else:
            log("PASS - All CI bounds bracket means correctly")

        log("Step 8 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
