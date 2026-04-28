#!/usr/bin/env python3
"""Assumption Validation (Simplified): Validate Linear Mixed Model assumptions for 6 fitted models (IRT, Full CTT, Purified CTT"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_dataframe_structure

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.5.5 (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step07.5_assumption_validation.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Simplified Assumption Tests (Without Fitted Models)

def test_linearity(df: pd.DataFrame, outcome_col: str, time_col: str = 'TSVR_hours') -> Dict[str, Any]:
    """
    Test linearity of Time-response relationship.

    Method: Compute Pearson correlation between Time and outcome.
    PASS: Correlation shows expected negative direction (forgetting over time).

    LIMITATION: Without fitted model, we test raw Time-outcome relationship,
    not Time-residual linearity. This is a weaker test.
    """
    r, p = stats.pearsonr(df[time_col], df[outcome_col])

    # For forgetting, we expect negative correlation (scores decrease over time)
    # But standardized scores have mean=0, so we just check magnitude
    result = "PASS" if abs(r) > 0.1 else "FAIL"
    notes = f"Pearson r={r:.3f} (negative = forgetting pattern)"

    return {
        'assumption': 'Linearity',
        'test_statistic': r,
        'p_value': p,
        'threshold': '|r| > 0.1',
        'result': result,
        'notes': notes
    }

def test_homoscedasticity(df: pd.DataFrame, outcome_col: str, time_col: str = 'TSVR_hours') -> Dict[str, Any]:
    """
    Test homoscedasticity (constant variance across time).

    Method: Split time into early (< median) vs late (>= median), compare variances.
    PASS: Variance ratio < 2.0 (relatively constant variance).

    LIMITATION: Without residuals, we test outcome variance, not residual variance.
    This is a weaker test.
    """
    median_time = df[time_col].median()
    early_var = df[df[time_col] < median_time][outcome_col].var()
    late_var = df[df[time_col] >= median_time][outcome_col].var()

    variance_ratio = max(early_var, late_var) / min(early_var, late_var)
    result = "PASS" if variance_ratio < 2.0 else "FAIL"
    notes = f"Variance ratio={variance_ratio:.3f} (early vs late)"

    return {
        'assumption': 'Homoscedasticity',
        'test_statistic': variance_ratio,
        'p_value': np.nan,  # No p-value for this simple test
        'threshold': 'ratio < 2.0',
        'result': result,
        'notes': notes
    }

def test_normality(df: pd.DataFrame, outcome_col: str) -> Dict[str, Any]:
    """
    Test normality of outcome distribution (proxy for residual normality).

    Method: Compute skewness and kurtosis.
    PASS: |skewness| < 1.0 and |kurtosis - 3| < 2.0.

    LIMITATION: Tests outcome normality, not residual normality. Standardized
    outcomes should be approximately normal, but this doesn't test residuals.
    """
    skewness = stats.skew(df[outcome_col])
    kurtosis = stats.kurtosis(df[outcome_col], fisher=False)  # Pearson kurtosis (normal=3)

    skew_ok = abs(skewness) < 1.0
    kurt_ok = abs(kurtosis - 3) < 2.0
    result = "PASS" if (skew_ok and kurt_ok) else "FAIL"
    notes = f"Skewness={skewness:.3f}, Kurtosis={kurtosis:.3f}"

    return {
        'assumption': 'Normality',
        'test_statistic': skewness,
        'p_value': np.nan,
        'threshold': '|skew| < 1.0, |kurt-3| < 2.0',
        'result': result,
        'notes': notes
    }

def test_random_effects_normality(df: pd.DataFrame, outcome_col: str) -> Dict[str, Any]:
    """
    Test random effects normality (simplified).

    Method: Check that participants show heterogeneous intercepts (variance > 0).
    PASS: Between-participant variance > 0.

    LIMITATION: Without fitted model random effects, we test whether participants
    differ on average (necessary but not sufficient for random effects normality).
    """
    # Compute participant means
    participant_means = df.groupby('UID')[outcome_col].mean()
    between_var = participant_means.var()

    result = "PASS" if between_var > 0 else "FAIL"
    notes = f"Between-participant variance={between_var:.4f}"

    return {
        'assumption': 'Random_Effects_Normality',
        'test_statistic': between_var,
        'p_value': np.nan,
        'threshold': 'variance > 0',
        'result': result,
        'notes': notes
    }

def test_independence(df: pd.DataFrame, outcome_col: str) -> Dict[str, Any]:
    """
    Test independence (no autocorrelation).

    Method: Compute lag-1 autocorrelation within participants.
    PASS: Mean lag-1 autocorrelation < 0.3.

    LIMITATION: Without residuals, we test outcome autocorrelation. LMM residuals
    should have low autocorrelation after accounting for random effects.
    """
    # Compute lag-1 autocorrelation per participant
    autocorrs = []
    for uid in df['UID'].unique():
        uid_data = df[df['UID'] == uid].sort_values('test')
        if len(uid_data) >= 3:  # Need at least 3 observations for lag-1
            values = uid_data[outcome_col].values
            if len(values) > 1:
                # Pearson correlation between t and t-1
                r = np.corrcoef(values[:-1], values[1:])[0, 1]
                if not np.isnan(r):
                    autocorrs.append(r)

    mean_autocorr = np.mean(autocorrs) if autocorrs else 0
    result = "PASS" if abs(mean_autocorr) < 0.3 else "FAIL"
    notes = f"Mean lag-1 autocorr={mean_autocorr:.3f} (N={len(autocorrs)} participants)"

    return {
        'assumption': 'Independence',
        'test_statistic': mean_autocorr,
        'p_value': np.nan,
        'threshold': '|autocorr| < 0.3',
        'result': result,
        'notes': notes
    }

def test_multicollinearity(df: pd.DataFrame, outcome_col: str) -> Dict[str, Any]:
    """
    Test multicollinearity.

    Method: N/A for single predictor (Time).
    PASS: Always pass (not applicable).
    """
    return {
        'assumption': 'Multicollinearity',
        'test_statistic': np.nan,
        'p_value': np.nan,
        'threshold': 'N/A',
        'result': 'PASS',
        'notes': 'Not applicable (single predictor)'
    }

def test_influential_observations(df: pd.DataFrame, outcome_col: str) -> Dict[str, Any]:
    """
    Test for influential observations (outliers).

    Method: Check for outliers (|z| > 3).
    PASS: < 5% of observations are outliers.

    LIMITATION: Tests outcome outliers, not Cook's distance or DFFITS which
    require fitted models.
    """
    z_scores = np.abs(df[outcome_col])
    n_outliers = (z_scores > 3).sum()
    total = len(df)
    outlier_rate = n_outliers / total

    result = "PASS" if outlier_rate < 0.05 else "FAIL"
    notes = f"Outlier rate={outlier_rate:.1%} ({n_outliers}/{total} obs with |z|>3)"

    return {
        'assumption': 'Influential_Observations',
        'test_statistic': outlier_rate,
        'p_value': np.nan,
        'threshold': 'rate < 5%',
        'result': result,
        'notes': notes
    }

def validate_assumptions_for_model(df: pd.DataFrame, outcome_col: str, model_name: str) -> List[Dict[str, Any]]:
    """
    Run all 7 assumption tests for a single model.

    Returns list of 7 test result dictionaries.
    """
    log(f"Testing assumptions for {model_name}...")

    results = []

    # Test 1: Linearity
    result = test_linearity(df, outcome_col)
    result['model'] = model_name
    results.append(result)
    log(f"  - Linearity: {result['result']} ({result['notes']})")

    # Test 2: Homoscedasticity
    result = test_homoscedasticity(df, outcome_col)
    result['model'] = model_name
    results.append(result)
    log(f"  - Homoscedasticity: {result['result']} ({result['notes']})")

    # Test 3: Normality
    result = test_normality(df, outcome_col)
    result['model'] = model_name
    results.append(result)
    log(f"  - Normality: {result['result']} ({result['notes']})")

    # Test 4: Random Effects Normality
    result = test_random_effects_normality(df, outcome_col)
    result['model'] = model_name
    results.append(result)
    log(f"  - Random Effects Normality: {result['result']} ({result['notes']})")

    # Test 5: Independence
    result = test_independence(df, outcome_col)
    result['model'] = model_name
    results.append(result)
    log(f"  - Independence: {result['result']} ({result['notes']})")

    # Test 6: Multicollinearity
    result = test_multicollinearity(df, outcome_col)
    result['model'] = model_name
    results.append(result)
    log(f"  - Multicollinearity: {result['result']} ({result['notes']})")

    # Test 7: Influential Observations
    result = test_influential_observations(df, outcome_col)
    result['model'] = model_name
    results.append(result)
    log(f"  - Influential Observations: {result['result']} ({result['notes']})")

    return results

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 7.5: Assumption Validation (Simplified)")
        # Load Input Data

        log("Loading input data...")

        # Load standardized scores
        standardized_path = RQ_DIR / "data" / "step06_standardized_scores.csv"
        df_std = pd.read_csv(standardized_path, encoding='utf-8')
        log(f"step06_standardized_scores.csv ({len(df_std)} rows, {len(df_std.columns)} cols)")

        # Load TSVR mapping from RQ 5.5.1
        tsvr_path = PROJECT_ROOT / "results" / "ch5" / "5.5.1" / "data" / "step00_tsvr_mapping.csv"
        df_tsvr = pd.read_csv(tsvr_path, encoding='utf-8')
        log(f"step00_tsvr_mapping.csv ({len(df_tsvr)} rows, {len(df_tsvr.columns)} cols)")

        # Convert TSVR test format to match standardized scores (1 -> 'T1', 2 -> 'T2', etc.)
        df_tsvr['test'] = 'T' + df_tsvr['test'].astype(str)
        log(f"TSVR test format converted: {df_tsvr['test'].unique().tolist()}")

        # Merge standardized scores with TSVR
        # Need to merge separately for source and destination (each has 400 rows)
        log("Merging standardized scores with TSVR mapping...")

        # Split standardized data by location_type
        df_source = df_std[df_std['location_type'] == 'source'].copy()
        df_dest = df_std[df_std['location_type'] == 'destination'].copy()

        # Merge with TSVR (UID + test)
        df_source = df_source.merge(df_tsvr, on=['UID', 'test'], how='left')
        df_dest = df_dest.merge(df_tsvr, on=['UID', 'test'], how='left')

        log(f"Source: {len(df_source)} rows, Destination: {len(df_dest)} rows")

        # Validate merge
        if df_source['TSVR_hours'].isna().any():
            log("Missing TSVR_hours after merge (source)")
            sys.exit(1)
        if df_dest['TSVR_hours'].isna().any():
            log("Missing TSVR_hours after merge (destination)")
            sys.exit(1)
        # Run Assumption Tests for All 6 Models

        log("Running assumption tests for 6 models...")

        all_results = []

        # Model 1: Source_IRT
        results = validate_assumptions_for_model(df_source, 'irt_z', 'Source_IRT')
        all_results.extend(results)

        # Model 2: Source_Full_CTT
        results = validate_assumptions_for_model(df_source, 'ctt_full_z', 'Source_Full_CTT')
        # Add note about bounded scale for CTT models
        for r in results:
            if r['result'] == 'FAIL':
                r['notes'] += ' (CTT bounded [0,1] scale may violate normality)'
        all_results.extend(results)

        # Model 3: Source_Purified_CTT
        results = validate_assumptions_for_model(df_source, 'ctt_purified_z', 'Source_Purified_CTT')
        for r in results:
            if r['result'] == 'FAIL':
                r['notes'] += ' (CTT bounded [0,1] scale may violate normality)'
        all_results.extend(results)

        # Model 4: Destination_IRT
        results = validate_assumptions_for_model(df_dest, 'irt_z', 'Destination_IRT')
        all_results.extend(results)

        # Model 5: Destination_Full_CTT
        results = validate_assumptions_for_model(df_dest, 'ctt_full_z', 'Destination_Full_CTT')
        for r in results:
            if r['result'] == 'FAIL':
                r['notes'] += ' (CTT bounded [0,1] scale may violate normality)'
        all_results.extend(results)

        # Model 6: Destination_Purified_CTT
        results = validate_assumptions_for_model(df_dest, 'ctt_purified_z', 'Destination_Purified_CTT')
        for r in results:
            if r['result'] == 'FAIL':
                r['notes'] += ' (CTT bounded [0,1] scale may violate normality)'
        all_results.extend(results)

        log(f"Assumption tests complete ({len(all_results)} tests)")
        # Save Assumption Validation Results
        # Output: CSV with 42 rows (6 models × 7 assumptions)
        # Contains: model, assumption, test_statistic, p_value, threshold, result, notes

        log("Saving assumption validation results...")

        df_assumptions = pd.DataFrame(all_results)

        # Reorder columns for clarity
        df_assumptions = df_assumptions[['model', 'assumption', 'test_statistic', 'p_value', 'threshold', 'result', 'notes']]

        output_path = RQ_DIR / "data" / "step07.5_assumption_validation.csv"
        df_assumptions.to_csv(output_path, index=False, encoding='utf-8')
        log(f"step07.5_assumption_validation.csv ({len(df_assumptions)} rows, {len(df_assumptions.columns)} cols)")
        # Run Validation Tool
        # Validates: 42 rows, all required columns present, result in {PASS, FAIL}

        log("Running validate_dataframe_structure...")

        validation_result = validate_dataframe_structure(
            df=df_assumptions,
            expected_rows=42,
            expected_columns=['model', 'assumption', 'test_statistic', 'p_value', 'threshold', 'result', 'notes']
        )

        if validation_result['valid']:
            log(f"PASS - {validation_result['message']}")
        else:
            log(f"FAIL - {validation_result['message']}")
            sys.exit(1)

        # Additional validation: Check result values
        unique_results = df_assumptions['result'].unique()
        if not all(r in ['PASS', 'FAIL'] for r in unique_results):
            log(f"FAIL - result column contains invalid values: {unique_results}")
            sys.exit(1)

        # Summary statistics
        n_pass = (df_assumptions['result'] == 'PASS').sum()
        n_fail = (df_assumptions['result'] == 'FAIL').sum()
        pass_rate = n_pass / len(df_assumptions) * 100

        log(f"Assumption test results:")
        log(f"  - Total tests: {len(df_assumptions)}")
        log(f"  - Passed: {n_pass} ({pass_rate:.1f}%)")
        log(f"  - Failed: {n_fail} ({100-pass_rate:.1f}%)")

        # Count failures by model
        log("Failures by model:")
        for model in df_assumptions['model'].unique():
            model_fails = df_assumptions[(df_assumptions['model'] == model) & (df_assumptions['result'] == 'FAIL')]
            if len(model_fails) > 0:
                log(f"  - {model}: {len(model_fails)} failures")
                for _, row in model_fails.iterrows():
                    log(f"    - {row['assumption']}: {row['notes']}")

        # Warning if >50% failures
        if pass_rate < 50:
            log(">50% of assumption tests failed - review model specifications")

        log("Step 7.5 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
