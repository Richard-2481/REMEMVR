#!/usr/bin/env python3
"""Mixed Model Three-Way Interaction (measure x location x time): Fit single mixed model with measure as a factor to test measure x location x"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import tools
from tools.validation import (
    validate_lmm_convergence,
    validate_lmm_assumptions_comprehensive,
    validate_hypothesis_test_dual_pvalues
)

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step03_mixed_model.log"

# TSVR mapping (normalized test format: 1/2/3/4)
TSVR_MAP = {'1': 1.0, '2': 26.0, '3': 74.0, '4': 148.0}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 3: Mixed Model Three-Way Interaction")
        # STEP 3.1: LOAD AND RESHAPE ACCURACY DATA
        log("\n[STEP 3.1] Load and reshape accuracy data")
        acc_path = PROJECT_ROOT / "results" / "ch5" / "5.5.1" / "data" / "step03_theta_scores.csv"
        df_acc_wide = pd.read_csv(acc_path)
        log(f"Accuracy: {len(df_acc_wide)} rows")

        # Parse composite_ID (format: A010_1, A010_2, etc.)
        df_acc_wide['UID'] = df_acc_wide['composite_ID'].str.split('_').str[0]
        df_acc_wide['test'] = df_acc_wide['composite_ID'].str.split('_').str[1]
        df_acc_wide['TSVR_hours'] = df_acc_wide['test'].map(TSVR_MAP)

        # Reshape to long (source/destination)
        df_acc_source = df_acc_wide[['UID', 'test', 'TSVR_hours', 'theta_source', 'se_source']].copy()
        df_acc_source.columns = ['UID', 'test', 'TSVR_hours', 'theta_accuracy', 'se_accuracy']
        df_acc_source['location'] = 'source'

        df_acc_dest = df_acc_wide[['UID', 'test', 'TSVR_hours', 'theta_destination', 'se_destination']].copy()
        df_acc_dest.columns = ['UID', 'test', 'TSVR_hours', 'theta_accuracy', 'se_accuracy']
        df_acc_dest['location'] = 'destination'

        df_acc_long = pd.concat([df_acc_source, df_acc_dest], ignore_index=True)
        log(f"Accuracy: {len(df_acc_wide)} -> {len(df_acc_long)} rows")
        # STEP 3.2: LOAD AND RESHAPE CONFIDENCE DATA
        log("\n[STEP 3.2] Load and reshape confidence data")
        conf_path = PROJECT_ROOT / "results" / "ch6" / "6.8.1" / "data" / "step03_theta_confidence.csv"
        df_conf_wide = pd.read_csv(conf_path)
        log(f"Confidence: {len(df_conf_wide)} rows")

        # Parse composite_ID (format: A010_T1, A010_T2, etc.)
        df_conf_wide['UID'] = df_conf_wide['composite_ID'].str.split('_').str[0]
        df_conf_wide['test_raw'] = df_conf_wide['composite_ID'].str.split('_').str[1]

        # Normalize test format: T1->1, T2->2, T3->3, T4->4
        df_conf_wide['test'] = df_conf_wide['test_raw'].str.replace('T', '')
        df_conf_wide['TSVR_hours'] = df_conf_wide['test'].map(TSVR_MAP)

        log(f"Confidence test format: T1/T2/T3/T4 -> 1/2/3/4")

        # Reshape to long (harmonize to lowercase column names)
        df_conf_source = df_conf_wide[['UID', 'test', 'TSVR_hours', 'theta_Source']].copy()
        df_conf_source.columns = ['UID', 'test', 'TSVR_hours', 'theta_confidence']
        df_conf_source['location'] = 'source'

        df_conf_dest = df_conf_wide[['UID', 'test', 'TSVR_hours', 'theta_Destination']].copy()
        df_conf_dest.columns = ['UID', 'test', 'TSVR_hours', 'theta_confidence']
        df_conf_dest['location'] = 'destination'

        df_conf_long = pd.concat([df_conf_source, df_conf_dest], ignore_index=True)
        log(f"Confidence: {len(df_conf_wide)} -> {len(df_conf_long)} rows")
        # STEP 3.3: MERGE ACCURACY AND CONFIDENCE ON (UID, test, location)
        log("\n[STEP 3.3] Merge accuracy and confidence on (UID, test, location)")

        # Both now have test in 1/2/3/4 format
        log(f"Accuracy test values: {sorted(df_acc_long['test'].unique())}")
        log(f"Confidence test values: {sorted(df_conf_long['test'].unique())}")

        df_merged = pd.merge(
            df_acc_long,
            df_conf_long,
            on=['UID', 'test', 'location', 'TSVR_hours'],
            how='inner'
        )
        log(f"{len(df_merged)} rows (should be 800)")

        if len(df_merged) != 800:
            log(f"Expected 800 rows, got {len(df_merged)}")
            log(f"Accuracy UIDs: {df_acc_long['UID'].nunique()}")
            log(f"Confidence UIDs: {df_conf_long['UID'].nunique()}")
            raise ValueError(f"Merge failed: expected 800 rows, got {len(df_merged)}")
        # STEP 3.4: STACK MEASURES (WIDE -> LONG BY MEASURE)
        log("\n[STEP 3.4] Stack measures to create measure factor")

        # Create accuracy rows
        df_acc_meas = df_merged[['UID', 'location', 'TSVR_hours', 'test', 'theta_accuracy', 'se_accuracy']].copy()
        df_acc_meas.columns = ['UID', 'location', 'TSVR_hours', 'test', 'theta', 'se']
        df_acc_meas['measure'] = 'accuracy'

        # Create confidence rows
        df_conf_meas = df_merged[['UID', 'location', 'TSVR_hours', 'test', 'theta_confidence']].copy()
        df_conf_meas.columns = ['UID', 'location', 'TSVR_hours', 'test', 'theta']
        df_conf_meas['measure'] = 'confidence'
        df_conf_meas['se'] = np.nan  # Confidence has no SE

        # Stack
        df_long = pd.concat([df_acc_meas, df_conf_meas], ignore_index=True)
        log(f"{len(df_merged)} -> {len(df_long)} rows (should be 1600)")

        log(f"  Unique UIDs: {df_long['UID'].nunique()}")
        log(f"  Unique measures: {df_long['measure'].unique()}")
        log(f"  Unique locations: {df_long['location'].unique()}")
        log(f"  Unique TSVR: {sorted(df_long['TSVR_hours'].unique())}")
        # STEP 3.5: FIT MIXED MODEL (DIRECT STATSMODELS CALL)
        log("\n[STEP 3.5] Fit mixed model: theta ~ measure*location*TSVR_hours")

        lmm_result = smf.mixedlm(
            formula='theta ~ measure * location * TSVR_hours',
            data=df_long,
            groups=df_long['UID'],
            re_formula='~1'
        ).fit(reml=False)

        # Validate convergence
        result = validate_lmm_convergence(lmm_result)
        if not result['converged']:
            raise ValueError(f"Model did not converge: {result['message']}")
        log(f"Mixed model converged successfully")
        # STEP 3.6: EXTRACT FIXED EFFECTS (DECISION D068)
        log("\n[STEP 3.6] Extract fixed effects with dual p-values (Decision D068)")

        # Get fixed effects summary
        # NOTE: lmm_result.summary().tables[1] returns DataFrame directly in modern statsmodels
        fe_summary = lmm_result.summary().tables[1]

        # Check if it's already a DataFrame or needs conversion
        if isinstance(fe_summary, pd.DataFrame):
            df_fixed = fe_summary.reset_index()
        else:
            # Old statsmodels: SimpleTable with .data attribute
            df_fixed = pd.DataFrame(fe_summary.data[1:], columns=fe_summary.data[0])

        # Standardize column names
        df_fixed.columns = ['term', 'beta', 'se', 'z', 'p_uncorrected', 'ci_lower', 'ci_upper']

        # Convert to numeric
        for col in ['beta', 'se', 'z', 'p_uncorrected']:
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors='coerce')

        # Add Bonferroni correction (n_tests=1 for primary hypothesis)
        df_fixed['p_bonferroni'] = df_fixed['p_uncorrected'].apply(lambda p: min(p * 1, 1.0))

        # Add df column (use z as t_stat, df = N - k)
        df_fixed['t_stat'] = df_fixed['z']
        df_fixed['df'] = len(df_long) - len(df_fixed)

        # Reorder columns
        df_fixed = df_fixed[['term', 'beta', 'se', 't_stat', 'df', 'p_uncorrected', 'p_bonferroni']]

        log(f"{len(df_fixed)} fixed effects")

        # Find three-way interaction
        three_way_terms = [term for term in df_fixed['term'] if 'measure' in term and 'location' in term and 'TSVR' in term]
        if three_way_terms:
            three_way = df_fixed[df_fixed['term'] == three_way_terms[0]].iloc[0]
            log(f"[THREE-WAY] {three_way['term']}: p_uncorr={three_way['p_uncorrected']:.4f}, p_bonf={three_way['p_bonferroni']:.4f}")
        else:
            log(f"Three-way interaction term not found")
        # STEP 3.7: VALIDATE D068 COMPLIANCE
        log("\n[STEP 3.7] Validate Decision D068 compliance")

        result = validate_hypothesis_test_dual_pvalues(
            interaction_df=df_fixed.set_index('term'),
            required_terms=three_way_terms,
            alpha_bonferroni=0.05
        )

        if result['d068_compliant']:
            log(f"Decision D068 compliant: both p-values present")
        else:
            log(f"{result['message']}")
        # STEP 3.8: ASSUMPTION DIAGNOSTICS
        log("\n[STEP 3.8] Run comprehensive assumption diagnostics")

        assumption_result = validate_lmm_assumptions_comprehensive(
            lmm_result=lmm_result,
            data=df_long,
            output_dir=RQ_DIR / "data",
            acf_lag1_threshold=0.1,
            alpha=0.05
        )

        if assumption_result['valid']:
            log(f"All assumption checks passed")
        else:
            log(f"Some assumption checks failed: {assumption_result['message']}")

        # Save assumption report
        assumption_path = RQ_DIR / "data" / "step03_mixed_model_assumptions.txt"
        with open(assumption_path, 'w', encoding='utf-8') as f:
            f.write(assumption_result['message'])
        log(f"{assumption_path.name}")
        # STEP 3.9: SAVE OUTPUTS
        log("\n[STEP 3.9] Save outputs")

        # Save fixed effects
        fe_path = RQ_DIR / "data" / "step03_mixed_model_results.csv"
        df_fixed.to_csv(fe_path, index=False, encoding='utf-8')
        log(f"{fe_path.name} ({len(df_fixed)} rows)")

        # Save merged data (for Steps 4, 6, 7, 8)
        data_path = RQ_DIR / "data" / "step03_merged_data_long.csv"
        df_long.to_csv(data_path, index=False, encoding='utf-8')
        log(f"{data_path.name} ({len(df_long)} rows)")

        log("Step 3 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
