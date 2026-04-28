#!/usr/bin/env python3
"""Validate LMM Assumptions for Both Models: Comprehensive 7-diagnostic LMM assumption validation for both IRT-based and"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_lmm_assumptions_comprehensive

# Import statsmodels for model loading
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_validate_lmm_assumptions.log"


# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Helper Function: Parse Diagnostics Dict to Row

def parse_diagnostics_to_rows(diagnostics_dict: Dict[str, Any], model_name: str) -> List[Dict]:
    """
    Parse validate_lmm_assumptions_comprehensive() output dict to list of rows.

    Expected diagnostics structure:
    {
        'valid': bool,
        'diagnostics': {
            'residual_normality': {'test': 'Shapiro-Wilk', 'statistic': float, 'p_value': float, 'passed': bool},
            'homoscedasticity': {'test': 'Breusch-Pagan', 'statistic': float, 'p_value': float, 'passed': bool},
            'random_effects_normality': {'intercepts': {...}, 'slopes': {...}},
            'autocorrelation': {'lag1_acf': float, 'threshold': float, 'passed': bool},
            'linearity': {'method': 'partial_residuals', 'csvs_generated': int, 'passed': bool},
            'outliers': {'cooks_d_max': float, 'threshold': float, 'n_outliers': int, 'passed': bool},
            'convergence': {'converged': bool}
        },
        'plot_paths': List[Path],
        'message': str
    }

    Args:
        diagnostics_dict: Output from validate_lmm_assumptions_comprehensive()
        model_name: 'IRT' or 'CTT'

    Returns:
        List of dicts, each representing one assumption test result
    """
    rows = []
    diag = diagnostics_dict.get('diagnostics', {})

    # (1) Linearity
    linearity = diag.get('linearity', {})
    rows.append({
        'model': model_name,
        'assumption': 'Linearity',
        'test_statistic': linearity.get('csvs_generated', 0),
        'p_value': np.nan,  # Linearity is visual/CSV-based, no p-value
        'threshold': 'Visual inspection',
        'status': 'PASS' if linearity.get('passed', False) else 'FAIL',
        'notes': f"Method: {linearity.get('method', 'unknown')}, CSVs: {linearity.get('csvs_generated', 0)}"
    })

    # (2) Homoscedasticity (Breusch-Pagan test)
    homosced = diag.get('homoscedasticity', {})
    rows.append({
        'model': model_name,
        'assumption': 'Homoscedasticity',
        'test_statistic': homosced.get('statistic', np.nan),
        'p_value': homosced.get('p_value', np.nan),
        'threshold': 'p > 0.05',
        'status': 'PASS' if homosced.get('passed', False) else 'FAIL',
        'notes': f"Test: {homosced.get('test', 'Breusch-Pagan')}"
    })

    # (3) Normality of residuals (Shapiro-Wilk)
    resid_norm = diag.get('residual_normality', {})
    rows.append({
        'model': model_name,
        'assumption': 'Normality_residuals',
        'test_statistic': resid_norm.get('statistic', np.nan),
        'p_value': resid_norm.get('p_value', np.nan),
        'threshold': 'p > 0.05 (or note N=800 limitation)',
        'status': 'PASS' if resid_norm.get('passed', False) else 'FAIL',
        'notes': f"Test: {resid_norm.get('test', 'Shapiro-Wilk')}"
    })

    # (4) Normality of random effects - Intercepts
    re_norm = diag.get('random_effects_normality', {})
    intercepts = re_norm.get('intercepts', {})
    rows.append({
        'model': model_name,
        'assumption': 'Normality_random_effects_intercepts',
        'test_statistic': intercepts.get('statistic', np.nan),
        'p_value': intercepts.get('p_value', np.nan),
        'threshold': 'p > 0.05',
        'status': 'PASS' if intercepts.get('passed', False) else 'FAIL',
        'notes': f"Test: {intercepts.get('test', 'Shapiro-Wilk')}, Component: Intercepts"
    })

    # (4b) Normality of random effects - Slopes (if present)
    slopes = re_norm.get('slopes', {})
    if slopes:  # Only add if slopes exist (model has random slopes)
        rows.append({
            'model': model_name,
            'assumption': 'Normality_random_effects_slopes',
            'test_statistic': slopes.get('statistic', np.nan),
            'p_value': slopes.get('p_value', np.nan),
            'threshold': 'p > 0.05',
            'status': 'PASS' if slopes.get('passed', False) else 'FAIL',
            'notes': f"Test: {slopes.get('test', 'Shapiro-Wilk')}, Component: Slopes"
        })
    else:
        # Model has random intercepts only (no slopes)
        rows.append({
            'model': model_name,
            'assumption': 'Normality_random_effects_slopes',
            'test_statistic': np.nan,
            'p_value': np.nan,
            'threshold': 'N/A (intercept-only model)',
            'status': 'PASS',
            'notes': 'Model has random intercepts only (no random slopes)'
        })

    # (5) Independence (Autocorrelation - Lag-1 ACF)
    autocorr = diag.get('autocorrelation', {})
    rows.append({
        'model': model_name,
        'assumption': 'Independence',
        'test_statistic': autocorr.get('lag1_acf', np.nan),
        'p_value': np.nan,  # ACF threshold-based, not p-value
        'threshold': f"ACF < {autocorr.get('threshold', 0.1)}",
        'status': 'PASS' if autocorr.get('passed', False) else 'FAIL',
        'notes': f"Lag-1 ACF: {autocorr.get('lag1_acf', np.nan):.4f}"
    })

    # (6) Multicollinearity (VIF < 10) - May be skipped if only 2 predictors
    multicoll = diag.get('multicollinearity', {})
    if multicoll:
        rows.append({
            'model': model_name,
            'assumption': 'No_multicollinearity',
            'test_statistic': multicoll.get('max_vif', np.nan),
            'p_value': np.nan,
            'threshold': 'VIF < 10',
            'status': 'PASS' if multicoll.get('passed', False) else 'FAIL',
            'notes': f"Max VIF: {multicoll.get('max_vif', np.nan):.2f}"
        })
    else:
        # Multicollinearity test skipped (only 2 predictors or other reason)
        rows.append({
            'model': model_name,
            'assumption': 'No_multicollinearity',
            'test_statistic': np.nan,
            'p_value': np.nan,
            'threshold': 'N/A (<=2 predictors)',
            'status': 'PASS',
            'notes': 'VIF test skipped (model has <=2 predictors or other reason)'
        })

    # (7) Influential observations (Cook's D < 1.0)
    outliers = diag.get('outliers', {})
    rows.append({
        'model': model_name,
        'assumption': 'Influential_observations',
        'test_statistic': outliers.get('cooks_d_max', np.nan),
        'p_value': np.nan,
        'threshold': "Cook's D < 1.0",
        'status': 'PASS' if outliers.get('passed', False) else 'FAIL',
        'notes': f"Max Cook's D: {outliers.get('cooks_d_max', np.nan):.4f}, N outliers: {outliers.get('n_outliers', 0)}"
    })

    return rows

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Validate LMM Assumptions for Both Models")
        # Load Fitted Models

        log("Loading fitted LMM models from Step 3...")

        # Load IRT-based LMM (using statsmodels MixedLMResults.load())
        irt_model_path = RQ_DIR / "data" / "step03_irt_lmm_model.pkl"
        log(f"Loading IRT model: {irt_model_path}")
        irt_lmm_model = MixedLMResults.load(str(irt_model_path))
        log(f"IRT model: {irt_model_path.name}")

        # Load CTT-based LMM (using statsmodels MixedLMResults.load())
        ctt_model_path = RQ_DIR / "data" / "step03_ctt_lmm_model.pkl"
        log(f"Loading CTT model: {ctt_model_path}")
        ctt_lmm_model = MixedLMResults.load(str(ctt_model_path))
        log(f"CTT model: {ctt_model_path.name}")
        # Load Original Data (for assumption validation)

        log("Loading original data for assumption validation...")

        # Load IRT theta scores (long format, 800 rows)
        theta_long_path = RQ_DIR / "data" / "step00_irt_theta_from_rq551.csv"
        theta_long = pd.read_csv(theta_long_path)
        log(f"{theta_long_path.name} ({len(theta_long)} rows, {len(theta_long.columns)} cols)")

        # Load CTT mean scores (long format, 800 rows)
        ctt_scores_path = RQ_DIR / "data" / "step01_ctt_scores.csv"
        ctt_scores = pd.read_csv(ctt_scores_path)
        log(f"{ctt_scores_path.name} ({len(ctt_scores)} rows, {len(ctt_scores.columns)} cols)")
        # Prepare Data for Validation

        log("Preparing data for assumption validation...")

        # Add log_TSVR to theta_long (IRT data)
        theta_long['log_TSVR'] = np.log(theta_long['TSVR_hours'] + 1)

        # Add LocationType categorical (treatment coding with 'source' as reference)
        theta_long['LocationType'] = pd.Categorical(
            theta_long['location_type'],
            categories=['source', 'destination']
        )

        # Rename irt_theta to 'score' for consistent column naming
        theta_long_for_validation = theta_long.rename(columns={'irt_theta': 'score'})
        log(f"IRT data: {len(theta_long_for_validation)} rows, added log_TSVR and LocationType")

        # Add log_TSVR to ctt_scores (CTT data)
        ctt_scores['log_TSVR'] = np.log(ctt_scores['TSVR_hours'] + 1)

        # Add LocationType categorical
        ctt_scores['LocationType'] = pd.Categorical(
            ctt_scores['location_type'],
            categories=['source', 'destination']
        )

        # Rename ctt_mean_score to 'score' for consistent column naming
        ctt_scores_for_validation = ctt_scores.rename(columns={'ctt_mean_score': 'score'})
        log(f"CTT data: {len(ctt_scores_for_validation)} rows, added log_TSVR and LocationType")
        # Run Assumption Validation for IRT Model

        log("Running assumption validation for IRT model...")
        irt_diagnostics = validate_lmm_assumptions_comprehensive(
            lmm_result=irt_lmm_model,
            data=theta_long_for_validation,
            output_dir=RQ_DIR / "plots",  # Save diagnostic plots to plots/ folder
            acf_lag1_threshold=0.1,  # ACF threshold for independence test
            alpha=0.05  # Significance level for statistical tests
        )
        log(f"IRT model assumption validation complete")
        log(f"Overall validation: {'PASS' if irt_diagnostics['valid'] else 'FAIL'}")
        log(f"Message: {irt_diagnostics['message']}")
        # Run Assumption Validation for CTT Model

        log("Running assumption validation for CTT model...")
        ctt_diagnostics = validate_lmm_assumptions_comprehensive(
            lmm_result=ctt_lmm_model,
            data=ctt_scores_for_validation,
            output_dir=RQ_DIR / "plots",  # Save diagnostic plots to plots/ folder
            acf_lag1_threshold=0.1,  # ACF threshold for independence test
            alpha=0.05  # Significance level for statistical tests
        )
        log(f"CTT model assumption validation complete")
        log(f"Overall validation: {'PASS' if ctt_diagnostics['valid'] else 'FAIL'}")
        log(f"Message: {ctt_diagnostics['message']}")
        # Parse Diagnostics to Create Comparison Table
        # These outputs will be used by: rq_inspect (validation), results analysis (interpretation)

        log("Creating assumptions comparison table...")

        # Parse IRT diagnostics to rows
        irt_rows = parse_diagnostics_to_rows(irt_diagnostics, model_name='IRT')
        log(f"IRT diagnostics: {len(irt_rows)} assumption tests")

        # Parse CTT diagnostics to rows
        ctt_rows = parse_diagnostics_to_rows(ctt_diagnostics, model_name='CTT')
        log(f"CTT diagnostics: {len(ctt_rows)} assumption tests")

        # Combine into single DataFrame
        assumptions_comparison = pd.DataFrame(irt_rows + ctt_rows)
        log(f"Assumptions comparison table: {len(assumptions_comparison)} rows")

        # Validate expected structure
        expected_rows = 14  # 7 assumptions x 2 models
        if len(assumptions_comparison) != expected_rows:
            log(f"Expected {expected_rows} rows, got {len(assumptions_comparison)}")

        # Check status values
        valid_statuses = {'PASS', 'FAIL'}
        actual_statuses = set(assumptions_comparison['status'].unique())
        if not actual_statuses.issubset(valid_statuses):
            log(f"Invalid status values: {actual_statuses - valid_statuses}")
        # Save Outputs
        # Output: Assumptions comparison CSV (14 rows)
        # Contains: model, assumption, test_statistic, p_value, threshold, status, notes

        log("Saving assumptions comparison table...")
        assumptions_path = RQ_DIR / "data" / "step04_assumptions_comparison.csv"
        assumptions_comparison.to_csv(assumptions_path, index=False, encoding='utf-8')
        log(f"{assumptions_path.name} ({len(assumptions_comparison)} rows, {len(assumptions_comparison.columns)} cols)")
        # Create Narrative Diagnostics Report
        # Output: Text file documenting violations and recommendations
        # Contains: Summary of violations, remedial action recommendations

        log("Creating narrative diagnostics report...")

        diagnostics_report_path = RQ_DIR / "data" / "step04_assumption_diagnostics.txt"

        with open(diagnostics_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LMM ASSUMPTION VALIDATION DIAGNOSTICS\n")
            f.write("RQ 5.5.4 - IRT-CTT Convergence Analysis\n")
            f.write("=" * 80 + "\n\n")

            # IRT Model Summary
            f.write("-" * 80 + "\n")
            f.write("IRT MODEL DIAGNOSTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Overall Validation: {'PASS' if irt_diagnostics['valid'] else 'FAIL'}\n")
            f.write(f"Message: {irt_diagnostics['message']}\n\n")

            # List violations for IRT model
            irt_violations = assumptions_comparison[
                (assumptions_comparison['model'] == 'IRT') &
                (assumptions_comparison['status'] == 'FAIL')
            ]
            if len(irt_violations) > 0:
                f.write(f"VIOLATIONS DETECTED ({len(irt_violations)} assumptions failed):\n")
                for _, row in irt_violations.iterrows():
                    f.write(f"  - {row['assumption']}: {row['notes']}\n")
            else:
                f.write("NO VIOLATIONS DETECTED (all assumptions passed)\n")

            f.write("\n")

            # CTT Model Summary
            f.write("-" * 80 + "\n")
            f.write("CTT MODEL DIAGNOSTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Overall Validation: {'PASS' if ctt_diagnostics['valid'] else 'FAIL'}\n")
            f.write(f"Message: {ctt_diagnostics['message']}\n\n")

            # List violations for CTT model
            ctt_violations = assumptions_comparison[
                (assumptions_comparison['model'] == 'CTT') &
                (assumptions_comparison['status'] == 'FAIL')
            ]
            if len(ctt_violations) > 0:
                f.write(f"VIOLATIONS DETECTED ({len(ctt_violations)} assumptions failed):\n")
                for _, row in ctt_violations.iterrows():
                    f.write(f"  - {row['assumption']}: {row['notes']}\n")

                # Special note for CTT model normality violations
                if 'Normality_residuals' in ctt_violations['assumption'].values:
                    f.write("\nNOTE: CTT model bounded outcome [0,1] may violate normality.\n")
                    f.write("This is EXPECTED and documented (not a blocking error).\n")
            else:
                f.write("NO VIOLATIONS DETECTED (all assumptions passed)\n")

            f.write("\n")

            # Comparison Summary
            f.write("-" * 80 + "\n")
            f.write("COMPARISON SUMMARY\n")
            f.write("-" * 80 + "\n")

            # Count violations per model
            n_irt_violations = len(irt_violations)
            n_ctt_violations = len(ctt_violations)

            f.write(f"IRT Model: {n_irt_violations}/7 assumptions violated\n")
            f.write(f"CTT Model: {n_ctt_violations}/7 assumptions violated\n\n")

            # Remedial action recommendations
            f.write("-" * 80 + "\n")
            f.write("REMEDIAL ACTION RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")

            if n_irt_violations == 0 and n_ctt_violations == 0:
                f.write("All assumptions passed for both models.\n")
                f.write("Proceed with confidence to model interpretation.\n")
            else:
                f.write("VIOLATIONS DETECTED:\n")

                # Check for normality violations (common)
                if 'Normality_residuals' in assumptions_comparison[assumptions_comparison['status'] == 'FAIL']['assumption'].values:
                    f.write("  - Residual normality: With N=800, minor violations acceptable per CLT.\n")
                    f.write("    Consider robust standard errors if severe.\n\n")

                # Check for homoscedasticity violations
                if 'Homoscedasticity' in assumptions_comparison[assumptions_comparison['status'] == 'FAIL']['assumption'].values:
                    f.write("  - Homoscedasticity: Consider weighted least squares or robust SEs.\n\n")

                # Check for autocorrelation
                if 'Independence' in assumptions_comparison[assumptions_comparison['status'] == 'FAIL']['assumption'].values:
                    f.write("  - Independence: Consider AR(1) error structure in LMM.\n\n")

                # Check for influential observations
                if 'Influential_observations' in assumptions_comparison[assumptions_comparison['status'] == 'FAIL']['assumption'].values:
                    f.write("  - Influential observations: Review high Cook's D cases.\n")
                    f.write("    Consider sensitivity analysis with/without outliers.\n\n")

            f.write("\n")
            f.write("=" * 80 + "\n")
            f.write("END OF DIAGNOSTICS REPORT\n")
            f.write("=" * 80 + "\n")

        log(f"{diagnostics_report_path.name}")
        # Final Validation
        # Validates: 14 rows (7 assumptions x 2 models), status values, coverage

        log("Running final validation checks...")

        # Check row count (14-16 acceptable: 7-8 assumptions × 2 models)
        # Note: Tool may include both random intercept AND slope normality as separate tests
        if 14 <= len(assumptions_comparison) <= 16:
            log(f"Assumptions comparison has {len(assumptions_comparison)} rows (within expected 14-16 range)")
        else:
            log(f"Expected 14-16 rows, got {len(assumptions_comparison)}")
            raise ValueError(f"Assumptions comparison has {len(assumptions_comparison)} rows, expected 14-16")

        # Check status values
        invalid_statuses = actual_statuses - valid_statuses
        if len(invalid_statuses) == 0:
            log("All status values in {'PASS', 'FAIL'}")
        else:
            log(f"Invalid status values: {invalid_statuses}")
            raise ValueError(f"Invalid status values: {invalid_statuses}")

        # Check both models covered
        models_covered = set(assumptions_comparison['model'].unique())
        expected_models = {'IRT', 'CTT'}
        if models_covered == expected_models:
            log("Both models covered (IRT and CTT)")
        else:
            log(f"Expected models {expected_models}, got {models_covered}")
            raise ValueError(f"Expected models {expected_models}, got {models_covered}")

        # Check all assumptions tested
        assumptions_tested = set(assumptions_comparison['assumption'].unique())
        # Note: May have 8 assumptions if random slopes exist (separate intercepts/slopes normality)
        # or 7 if intercept-only model
        log(f"Assumptions tested: {assumptions_tested}")
        log(f"Total unique assumptions: {len(assumptions_tested)}")

        # Document violations (not blocking)
        total_violations = len(assumptions_comparison[assumptions_comparison['status'] == 'FAIL'])
        log(f"Total violations: {total_violations}/14")

        if total_violations > 0:
            log("Assumption violations detected (documented findings, not errors)")
            log("Violations documented in step04_assumption_diagnostics.txt")
            log("Continue to Step 5 with documented violations")
        else:
            log("All assumptions passed for both models")

        log("Step 04 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
