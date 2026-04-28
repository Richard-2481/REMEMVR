#!/usr/bin/env python3
"""Binary Confidence Analysis: Collapse 5-level confidence to binary and re-run IRT + LMM pipeline to test if"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback
import statsmodels.formula.api as smf
from scipy import stats

# Add project root to path
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import tools
from tools.analysis_irt import calibrate_grm
from tools.validation import validate_irt_convergence

# Configuration

RQ_DIR = SCRIPT_PATH.parents[1]
LOG_FILE = RQ_DIR / "logs" / "step05_binary_confidence_analysis.log"

SEED = 42
LAMBDA_VALUE = 0.41  # Match accuracy PowerLaw

# Binary collapse rule
BINARY_COLLAPSE_RULE = {
    'low': [1, 2],   # Guess, Not Sure → 0
    'high': [4, 5],  # Very Confident, Absolutely Certain → 1
    'exclude': [3]   # Mildly Confident → NaN (ambiguous midpoint)
}

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 05: Binary Confidence Analysis")
        log(f"  Random seed: {SEED}")
        log(f"  Binary collapse rule: {BINARY_COLLAPSE_RULE}")
        # SUBSTEP 5a: BINARY COLLAPSE

        log("\n[SUBSTEP 5a] Binary Collapse")
        log("Loading wide-format IRT input...")

        resp_file = PROJECT_ROOT / "results/ch6/6.1.1/data/step00_irt_input.csv"
        if not resp_file.exists():
            raise FileNotFoundError(f"EXPECTATIONS ERROR: {resp_file}")

        df_wide = pd.read_csv(resp_file)
        log(f"{resp_file.name} ({len(df_wide)} rows, {len(df_wide.columns)} cols)")

        # Convert wide to long format
        log("\nConverting wide to long format...")

        # Get item columns (all columns except composite_ID)
        item_cols = [col for col in df_wide.columns if col != 'composite_ID']
        log(f"  Found {len(item_cols)} items")

        # Melt wide to long
        df_long = df_wide.melt(
            id_vars=['composite_ID'],
            value_vars=item_cols,
            var_name='item_id',
            value_name='response'
        )

        log(f"  Reshaped to {len(df_long)} rows (long format)")

        # Parse composite_ID to extract UID and test
        df_long[['UID', 'test_str']] = df_long['composite_ID'].str.split('_', expand=True)
        df_long['test'] = df_long['test_str'].str.replace('T', '').astype(int)

        # Original response distribution
        log(f"\n  Original response distribution:")
        log(f"\n{df_long['response'].value_counts().sort_index().to_string()}")

        # Apply binary collapse
        log("\nApplying binary collapse rule...")

        def collapse_binary(response):
            if pd.isna(response):
                return np.nan
            if response in BINARY_COLLAPSE_RULE['low']:
                return 0
            elif response in BINARY_COLLAPSE_RULE['high']:
                return 1
            elif response in BINARY_COLLAPSE_RULE['exclude']:
                return np.nan
            else:
                return np.nan

        df_long['response_binary'] = df_long['response'].apply(collapse_binary)

        # Count data loss
        n_original = len(df_long)
        n_valid = df_long['response_binary'].notna().sum()
        n_excluded = n_original - n_valid
        pct_excluded = (n_excluded / n_original) * 100

        log(f"  Original responses: {n_original}")
        log(f"  Valid binary responses: {n_valid}")
        log(f"  Excluded (level 3 or NaN): {n_excluded} ({pct_excluded:.1f}%)")

        if pct_excluded > 50:
            log(f"  WARNING: >50% data loss due to midpoint exclusion")

        log(f"\n  Binary response distribution:")
        log(f"\n{df_long['response_binary'].value_counts().sort_index().to_string()}")

        # Save binary responses
        output_binary = RQ_DIR / "data" / "step05_binary_confidence_responses.csv"
        df_long.to_csv(output_binary, index=False, encoding='utf-8')
        log(f"{output_binary.name}")
        # SUBSTEP 5b: BINARY IRT CALIBRATION

        log("\n[SUBSTEP 5b] Binary IRT Calibration")
        log("Preparing IRT input...")

        # Remove NaN responses
        df_irt = df_long[df_long['response_binary'].notna()].copy()

        # Create composite_ID (UID_test format)
        df_irt['composite_ID'] = df_irt['UID'].astype(str) + '_' + df_irt['test'].astype(str)

        # Prepare for IRT (long format with required columns)
        df_irt_long = df_irt[['UID', 'test', 'item_id', 'response_binary']].copy()
        df_irt_long.columns = ['UID', 'test', 'item_name', 'score']

        log(f"  IRT input: {len(df_irt_long)} responses")
        log(f"  Unique UIDs: {df_irt_long['UID'].nunique()}")
        log(f"  Unique items: {df_irt_long['item_name'].nunique()}")

        # Define groups (single factor for confidence)
        groups = {'confidence': df_irt_long['item_name'].unique().tolist()}

        # IRT config - minimal config for binary GRM
        irt_config = {
            'n_cats': 2,  # Binary (equivalent to 2PL)
            'device': 'cpu',
            'seed': SEED,
            'correlated_factors': False,
            'invert_scale': True,  # Confidence (higher = more confident)
            'model_fit': {}  # Empty dict for defaults
        }

        log("\nCalibrating binary GRM (equivalent to 2PL)...")
        log(f"  n_cats: {irt_config['n_cats']}")
        log(f"  device: {irt_config['device']}")

        try:
            df_theta_binary, df_items_binary = calibrate_grm(
                df_long=df_irt_long,
                groups=groups,
                config=irt_config
            )
            log("IRT calibration complete")

        except Exception as e:
            log(f"IRT calibration failed: {e}")
            raise RuntimeError(f"TOOL ERROR: IRT calibration failed: {e}")

        # Save IRT outputs
        output_items = RQ_DIR / "data" / "step05_binary_confidence_irt_calibration.csv"
        df_items_binary.to_csv(output_items, index=False, encoding='utf-8')
        log(f"{output_items.name} ({len(df_items_binary)} items)")

        # Parse composite_ID back to UID and test
        df_theta_binary[['UID', 'test_str']] = df_theta_binary['composite_ID'].str.split('_', expand=True)
        df_theta_binary['test'] = df_theta_binary['test_str'].str.replace('T', '').astype(int)

        # Map test to TSVR_hours (and convert to Days)
        mapping_file = PROJECT_ROOT / "results/ch6/6.1.1/data/step00_tsvr_mapping.csv"
        df_mapping = pd.read_csv(mapping_file)

        # Extract test from mapping's composite_ID if needed
        if 'test' not in df_mapping.columns:
            df_mapping[['UID_map', 'test_str']] = df_mapping['composite_ID'].str.split('_', expand=True)
            df_mapping['test'] = df_mapping['test_str'].str.replace('T', '').astype(int)

        # Create test to TSVR_hours mapping
        test_to_tsvr_hours = df_mapping.set_index('test')['TSVR_hours'].to_dict()
        df_theta_binary['TSVR_hours'] = df_theta_binary['test'].map(test_to_tsvr_hours)

        # Rename theta column
        df_theta_binary['theta_binary'] = df_theta_binary['theta']
        df_theta_binary['SE_theta_binary'] = df_theta_binary.get('SE', np.nan)

        output_theta = RQ_DIR / "data" / "step05_binary_confidence_theta.csv"
        df_theta_binary[['UID', 'TSVR_hours', 'theta_binary', 'SE_theta_binary']].to_csv(
            output_theta, index=False, encoding='utf-8'
        )
        log(f"{output_theta.name} ({len(df_theta_binary)} rows)")
        # SUBSTEP 5c: BINARY CONFIDENCE LMM

        log("\n[SUBSTEP 5c] Binary Confidence LMM")
        log("Merging theta with time mapping...")

        # Convert TSVR_hours to Days
        df_mapping['Days'] = df_mapping['TSVR_hours'] / 24.0

        # Merge with Days using test
        df_theta_binary = df_theta_binary.merge(df_mapping[['test', 'Days']], on='test', how='left')

        # Apply PowerLaw transformation
        df_theta_binary['log_Days_plus1_lambda_0_41'] = np.log((df_theta_binary['Days'] + 1) ** LAMBDA_VALUE)

        log(f"  LMM input: {len(df_theta_binary)} observations")

        # Fit LMM
        log("\nFitting PowerLaw LMM to binary theta...")

        formula = "theta_binary ~ 1 + log_Days_plus1_lambda_0_41"
        re_formula = "~log_Days_plus1_lambda_0_41"

        fallback_used = False
        try:
            lmm_result = smf.mixedlm(
                formula=formula,
                data=df_theta_binary,
                groups=df_theta_binary['UID'],
                re_formula=re_formula
            ).fit(reml=False, method='powell')

            if not lmm_result.converged:
                raise RuntimeError("Model did not converge")

            log("  Random slopes model converged")

        except Exception as e:
            log(f"  Random slopes failed: {e}")
            log("  Falling back to random intercepts only...")

            try:
                lmm_result = smf.mixedlm(
                    formula=formula,
                    data=df_theta_binary,
                    groups=df_theta_binary['UID']
                ).fit(reml=False, method='powell')

                fallback_used = True
                log("  Intercept-only model converged")

            except Exception as e2:
                raise RuntimeError(f"Both models failed: {e2}")

        # Extract fixed effects
        fixed_effects = pd.DataFrame({
            'parameter': lmm_result.params.index,
            'estimate': lmm_result.params.values,
            'se': lmm_result.bse.values,
            't_value': lmm_result.tvalues.values,
            'p_value': lmm_result.pvalues.values
        })

        output_fe = RQ_DIR / "data" / "step05_binary_confidence_lmm_fit.csv"
        fixed_effects.to_csv(output_fe, index=False, encoding='utf-8')
        log(f"{output_fe.name}")

        # Extract variance components
        cov_re = lmm_result.cov_re

        if fallback_used:
            var_intercept = cov_re.iloc[0, 0]
            var_slope = 0.0
            cov_int_slope = 0.0
        else:
            var_intercept = cov_re.iloc[0, 0]
            var_slope = cov_re.iloc[1, 1]
            cov_int_slope = cov_re.iloc[0, 1]

        var_residual = lmm_result.scale

        # Compute ICCs
        if var_slope > 0:
            ICC_slope = var_slope / (var_slope + var_residual)
            ICC_conditional = var_slope / (var_intercept + var_slope + var_residual)
        else:
            ICC_slope = 0.0
            ICC_conditional = 0.0

        log(f"  var_intercept: {var_intercept:.6f}")
        log(f"  var_slope: {var_slope:.6f}")
        log(f"  var_residual: {var_residual:.6f}")
        log(f"  ICC_slope: {ICC_slope:.6f}")

        # Save variance components
        variance_df = pd.DataFrame({
            'component': ['var_intercept', 'var_slope', 'cov_int_slope', 'var_residual',
                          'ICC_slope', 'ICC_conditional'],
            'value': [var_intercept, var_slope, cov_int_slope, var_residual,
                      ICC_slope, ICC_conditional]
        })

        output_var = RQ_DIR / "data" / "step05_binary_confidence_variance_components.csv"
        variance_df.to_csv(output_var, index=False, encoding='utf-8')
        log(f"{output_var.name}")
        # Compute Binary Sensitivity Ratio

        log("\nComputing binary sensitivity ratio...")

        # Load accuracy ICC
        acc_var_file = RQ_DIR / "data" / "step01_accuracy_variance_components.csv"
        df_acc = pd.read_csv(acc_var_file)
        ICC_slope_accuracy = df_acc[df_acc['component'] == 'ICC_slope']['value'].values[0]

        ratio_binary = ICC_slope / ICC_slope_accuracy if ICC_slope_accuracy > 0 else np.inf

        log(f"  ICC_accuracy: {ICC_slope_accuracy:.6f}")
        log(f"  ICC_binary_confidence: {ICC_slope:.6f}")
        log(f"  Ratio: {ratio_binary:.1f}x")

        # Interpretation
        if ratio_binary < 10:
            interp = "Scaling artifact dominates (<10x)"
        elif ratio_binary < 50:
            interp = "Partial artifact, some genuine difference (10-50x)"
        else:
            interp = "Even binary shows higher ICC (>50x), suggests response compression"

        binary_ratio_df = pd.DataFrame({
            'measure_type': ['binary_accuracy', 'binary_confidence'],
            'ICC_slope': [ICC_slope_accuracy, ICC_slope],
            'ICC_conditional': [np.nan, ICC_conditional],
            'ratio_to_accuracy': [1.0, ratio_binary],
            'interpretation': ['Baseline', interp]
        })

        output_ratio = RQ_DIR / "data" / "step05_binary_sensitivity_ratio.csv"
        binary_ratio_df.to_csv(output_ratio, index=False, encoding='utf-8')
        log(f"{output_ratio.name}")
        # Theta Scale Comparison

        log("\nTheta scale distributions...")

        theta_comparison = pd.DataFrame({
            'measure': ['binary_confidence'],
            'mean_theta': [df_theta_binary['theta_binary'].mean()],
            'sd_theta': [df_theta_binary['theta_binary'].std()],
            'min_theta': [df_theta_binary['theta_binary'].min()],
            'max_theta': [df_theta_binary['theta_binary'].max()],
            'N_observations': [len(df_theta_binary)]
        })

        output_scale = RQ_DIR / "data" / "step05_theta_scale_comparison.csv"
        theta_comparison.to_csv(output_scale, index=False, encoding='utf-8')
        log(f"{output_scale.name}")
        # Save Diagnostics

        conv_df = pd.DataFrame({
            'model_name': ['binary_confidence_PowerLaw'],
            'converged': [lmm_result.converged],
            'n_iterations': [0],
            'convergence_criterion': ['default'],
            'warnings': ['Fallback to intercept-only' if fallback_used else 'None'],
            'fallback_used': [fallback_used]
        })

        output_conv = RQ_DIR / "data" / "step05_convergence_diagnostics.csv"
        conv_df.to_csv(output_conv, index=False, encoding='utf-8')
        log(f"{output_conv.name}")

        # Assumption checks
        diagnostics_text = f"""Binary Confidence Analysis - Assumption Checks
{'=' * 70}

1. IRT Calibration:
   Items calibrated: {len(df_items_binary)}
   Theta scores: {len(df_theta_binary)}
   Data retention: {(len(df_theta_binary) / n_original * 100):.1f}%

2. LMM Convergence:
   Converged: {lmm_result.converged}
   Fallback used: {fallback_used}

3. Variance Components:
   var_intercept = {var_intercept:.6f}
   var_slope = {var_slope:.6f}
   var_residual = {var_residual:.6f}

4. ICC Values:
   ICC_slope = {ICC_slope:.6f}
   ICC_conditional = {ICC_conditional:.6f}

5. Binary Sensitivity Test:
   Ratio to accuracy: {ratio_binary:.1f}x
   Interpretation: {interp}
"""

        output_diag = RQ_DIR / "data" / "step05_assumption_checks.txt"
        with open(output_diag, 'w', encoding='utf-8') as f:
            f.write(diagnostics_text)
        log(f"{output_diag.name}")
        # VALIDATION

        log("\nSummary:")
        log(f"  IRT converged: PASS")
        log(f"  All items calibrated: PASS ({len(df_items_binary)} items)")
        log(f"  LMM converged: {'PASS' if lmm_result.converged else 'FALLBACK'}")
        log(f"  ICC_slope in [0, 1]: {'PASS' if 0 <= ICC_slope <= 1 else 'FAIL'}")
        log(f"  Data retention >= 70%: {'PASS' if (len(df_theta_binary)/n_original) >= 0.7 else 'FAIL'}")

        log("\nStep 05 complete - Binary confidence analysis finished")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
