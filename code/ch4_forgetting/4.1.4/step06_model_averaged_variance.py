#!/usr/bin/env python3
"""
Step 06: Model-Averaged Variance Decomposition (v4.X UPGRADE)

PURPOSE:
Run model-averaged variance decomposition using 17-model kitchen sink comparison.
This addresses the gap identified in rq_status.tsv: "Model_Comparison" and provides
uncertainty quantification across functional forms (PowerLaw vs Log vs Recip+Log).

CRITICAL UPDATE (2025-12-09):
- Original analysis (Steps 1-5) used SINGLE model (Lin+Log)
- NEW v4.X capability: Model-averaged variance decomposition tool
- Addresses functional form uncertainty (RQ 5.1.1 extended testing revealed PowerLaw models competitive)

EXPECTED INPUTS:
  1. ../5.1.1/data/step04_lmm_input.csv (LMM input data with TSVR_hours)
     Columns: [composite_ID, UID, test, Theta, SE, TSVR_hours, Days, Days_squared, log_Days_plus1]
     Description: LMM input data for 17-model comparison

EXPECTED OUTPUTS:
  1. data/step06_model_comparison.csv
     Format: CSV with 17 rows (one per model)
     Columns: [model_name, aic, delta_aic, akaike_weight, n_params]
     Purpose: Model comparison table for kitchen sink analysis

  2. data/step06_competitive_models_metadata.yaml
     Format: YAML
     Content: List of competitive models (ΔAIC < 2.0) with metadata
     Purpose: Documents which models contribute to averaging

  3. data/step06_averaged_variance_components.csv
     Format: CSV with 1 row (unstratified analysis)
     Columns: [var_intercept_avg, var_slope_avg, cov_int_slope_avg, var_resid_avg, cor_int_slope_avg]
     Purpose: Model-averaged variance components

  4. data/step06_averaged_iccs.csv
     Format: CSV with 1 row
     Columns: [icc_intercept_avg, icc_slope_simple_avg, icc_slope_conditional_avg]
     Purpose: Model-averaged ICCs

  5. data/step06_model_specific_variance.csv
     Format: CSV with N rows (one per competitive model)
     Columns: [model_name, var_intercept, var_slope, cov_int_slope, var_resid, cor_int_slope, icc_intercept, icc_slope_simple]
     Purpose: Model-specific variance components for transparency

  6. data/step06_averaged_random_effects.csv
     Format: CSV with 100 rows (one per participant)
     Columns: [UID, intercept_avg, slope_avg]
     Purpose: Model-averaged random intercepts and slopes

  7. logs/step06_model_averaged_variance.log
     Format: Text log
     Content: Kitchen sink comparison results, competitive models, averaging process
     Purpose: Audit trail

VALIDATION CRITERIA:
  1. At least 3 competitive models (ΔAIC < 2.0)
  2. Model weights sum to 1.0 (within tolerance)
  3. var_slope_avg > 0 (positive variance)
  4. No NaN values in averaged components
  5. 100 participants with complete random effects

IMPLEMENTATION NOTES:
  Analysis tool: tools.variance_decomposition.compute_model_averaged_variance_decomposition
  Validation tool: tools.validation.validate_dataframe_structure
  Key parameters: delta_aic_threshold=2.0, stratify_var='intercept_dummy' (unstratified)
  Critical: RQ 5.1.4 is UNSTRATIFIED (single model), not like RQ 5.4.6 (3 congruence levels)
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
from datetime import datetime
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.variance_decomposition import compute_model_averaged_variance_decomposition
from tools.validation import validate_dataframe_structure

# =============================================================================
# Configuration
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_model_averaged_variance.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Input paths
LMM_INPUT = RQ_DIR / "../5.1.1/data/step04_lmm_input.csv"

# Output paths
MODEL_COMPARISON_CSV = RQ_DIR / "data" / "step06_model_comparison.csv"
COMPETITIVE_MODELS_YAML = RQ_DIR / "data" / "step06_competitive_models_metadata.yaml"
AVERAGED_VARIANCE_CSV = RQ_DIR / "data" / "step06_averaged_variance_components.csv"
AVERAGED_ICCS_CSV = RQ_DIR / "data" / "step06_averaged_iccs.csv"
MODEL_SPECIFIC_CSV = RQ_DIR / "data" / "step06_model_specific_variance.csv"
AVERAGED_RANDOM_EFFECTS_CSV = RQ_DIR / "data" / "step06_averaged_random_effects.csv"

MODEL_COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Logging Function
# =============================================================================

def log(msg: str) -> None:
    """Write message to both console and log file (UTF-8 encoding)."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(formatted_msg + "\n")


# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 06: Model-Averaged Variance Decomposition")
        log("=" * 70)

        # =====================================================================
        # STEP 1: Load LMM Input Data
        # =====================================================================
        log("[LOAD] Loading LMM input data from RQ 5.1.1...")
        log(f"  File: {LMM_INPUT}")

        if not LMM_INPUT.exists():
            log(f"[ERROR] LMM input file not found: {LMM_INPUT}")
            log("[FAIL] Cannot proceed without LMM input data")
            sys.exit(1)

        lmm_input = pd.read_csv(LMM_INPUT, encoding='utf-8')
        log(f"[LOADED] {len(lmm_input)} rows, {len(lmm_input.columns)} columns")
        log(f"[INFO] Columns: {list(lmm_input.columns)}")

        # Verify expected columns
        required_cols = ['UID', 'theta', 'TSVR_hours']
        missing_cols = [col for col in required_cols if col not in lmm_input.columns]
        if missing_cols:
            log(f"[ERROR] Missing required columns: {missing_cols}")
            sys.exit(1)

        log(f"[INFO] Unique participants: {lmm_input['UID'].nunique()}")
        log(f"[INFO] Observations per participant: {len(lmm_input) / lmm_input['UID'].nunique():.1f}")

        # =====================================================================
        # STEP 2: Create Dummy Stratification Variable (Unstratified Analysis)
        # =====================================================================
        log("[PREPARE] Creating dummy stratification variable (RQ 5.1.4 is unstratified)...")

        # RQ 5.1.4 analyzes ALL items together (no stratification by domain/congruence/paradigm)
        # But variance_decomposition tool requires stratify_var parameter
        # Solution: Create dummy variable with single level 'all'
        lmm_input['intercept_dummy'] = 'all'
        log("[INFO] Added 'intercept_dummy' column with value 'all' for all rows")

        # =====================================================================
        # STEP 3: Run Model-Averaged Variance Decomposition
        # =====================================================================
        log("[ANALYSIS] Running model-averaged variance decomposition (17 models)...")
        log(f"  Outcome variable: theta")
        log(f"  Time variable: TSVR_hours")
        log(f"  Groups variable: UID")
        log(f"  Stratify variable: intercept_dummy (single level: 'all')")
        log(f"  Delta AIC threshold: 2.0 (competitive models)")
        log(f"  Random effects: Intercepts + Slopes")
        log("")
        log("[INFO] This will take ~2-5 minutes (17 model fits + variance extraction)...")
        log("")

        results = compute_model_averaged_variance_decomposition(
            data=lmm_input,
            outcome_var='theta',
            tsvr_var='TSVR_hours',
            groups_var='UID',
            stratify_var='intercept_dummy',
            stratify_levels=['all'],
            delta_aic_threshold=2.0,
            min_models=3,
            max_models=10,
            re_intercept=True,
            re_slope=True,
            save_dir=None,  # We'll save manually
            log_file=None,  # Using our own log
            return_fitted_models=False,  # Don't return large model objects
            reml=False,  # Use ML for AIC comparison
            handle_convergence_failures='warn',
        )

        log("[SUCCESS] Model-averaged variance decomposition complete")
        log("")

        # =====================================================================
        # STEP 4: Extract and Save Results
        # =====================================================================
        log("[EXTRACT] Extracting results from variance decomposition...")

        # Extract kitchen sink model comparison
        model_comparison = results['model_comparison']
        log(f"[INFO] Kitchen sink comparison: {len(model_comparison)} models tested")

        # Best model info
        best_row = model_comparison.iloc[0]
        log(f"[INFO] Best model: {best_row['model_name']} (AIC={best_row['AIC']:.2f})")

        # Extract competitive models metadata
        competitive_models = results['competitive_models']
        log(f"[INFO] Competitive models (ΔAIC < 2.0): {len(competitive_models)}")
        for model_name in competitive_models:
            model_row = model_comparison[model_comparison['model_name'] == model_name].iloc[0]
            log(f"  - {model_name}: AIC={model_row['AIC']:.2f}, ΔAIC={model_row['delta_AIC']:.2f}, weight={model_row['akaike_weight']:.3f}")

        # Extract variance components (single-row dataframe for unstratified analysis)
        variance_components_df = results['averaged_variance_components']
        variance_avg = variance_components_df.iloc[0]  # Single row
        log("")
        log("[INFO] Model-averaged variance components:")
        log(f"  Columns available: {list(variance_components_df.columns)}")
        log(f"  var_intercept = {variance_avg.get('var_intercept', variance_avg.get('var_int', 'N/A')):.6f}")
        log(f"  var_slope = {variance_avg.get('var_slope', 'N/A'):.6f}")

        # Covariance and correlation might not be in averaged results
        if 'cov_int_slope' in variance_avg.index:
            log(f"  cov_int_slope = {variance_avg['cov_int_slope']:.6f}")
        if 'cor_int_slope' in variance_avg.index:
            log(f"  cor_int_slope = {variance_avg['cor_int_slope']:.6f}")

        if 'var_resid' in variance_avg.index:
            log(f"  var_resid = {variance_avg['var_resid']:.6f}")
        elif 'var_residual' in variance_avg.index:
            log(f"  var_resid = {variance_avg['var_residual']:.6f}")

        # Extract ICCs (single-row dataframe)
        iccs_df = results['averaged_ICCs']
        iccs_avg = iccs_df.iloc[0]  # Single row
        log("")
        log("[INFO] Model-averaged ICCs:")
        log(f"  ICC_intercept = {iccs_avg['ICC_intercept']:.6f} ({iccs_avg['ICC_intercept']*100:.2f}%)")
        log(f"  ICC_slope_simple = {iccs_avg['ICC_slope_simple']:.6f} ({iccs_avg['ICC_slope_simple']*100:.4f}%)")

        # Check if ICC_slope_conditional exists in columns
        if 'ICC_slope_conditional' in iccs_df.columns:
            icc_cond = iccs_avg['ICC_slope_conditional']
            if pd.notna(icc_cond):
                log(f"  ICC_slope_conditional = {icc_cond:.6f}")
            else:
                log(f"  ICC_slope_conditional = N/A")
        else:
            log(f"  ICC_slope_conditional = N/A (not computed)")

        # Extract random effects (dataframe with UID, intercept, slope columns)
        random_effects_df = results['averaged_random_effects']
        log("")
        log(f"[INFO] Model-averaged random effects: {len(random_effects_df)} participants")
        log(f"  Columns available: {list(random_effects_df.columns)}")

        # Find intercept and slope columns (might have different names)
        int_col = None
        slope_col = None
        for col in random_effects_df.columns:
            if 'intercept' in col.lower():
                int_col = col
            elif 'slope' in col.lower():
                slope_col = col

        if int_col and slope_col:
            log(f"  Intercept range: [{random_effects_df[int_col].min():.3f}, {random_effects_df[int_col].max():.3f}]")
            log(f"  Slope range: [{random_effects_df[slope_col].min():.3f}, {random_effects_df[slope_col].max():.3f}]")
        else:
            log(f"  ERROR: Could not find intercept/slope columns")

        # Extract model-specific variance (from stratified_results)
        model_specific_list = []
        stratified_all = results['stratified_results']['all']
        var_by_model = stratified_all['variance_components_by_model']
        icc_by_model = stratified_all['ICCs_by_model']

        log("")
        log("[INFO] Model-specific variance extraction:")
        log(f"  Variance by model columns: {list(var_by_model.columns)}")
        log(f"  ICC by model columns: {list(icc_by_model.columns)}")

        # Determine model identifier column name
        model_col = None
        for col in var_by_model.columns:
            if 'model' in col.lower():
                model_col = col
                break

        if model_col is None:
            log(f"  WARNING: No 'model' column found, skipping model-specific extraction")
        else:
            for model_name in competitive_models:
                var_row = var_by_model[var_by_model[model_col] == model_name].iloc[0]
                icc_row = icc_by_model[icc_by_model[model_col] == model_name].iloc[0]
                model_specific_list.append({
                    'model_name': model_name,
                    'var_intercept': var_row['var_intercept'],
                    'var_slope': var_row['var_slope'],
                    'cov_intercept_slope': var_row.get('cov_intercept_slope', None),
                    'var_residual': var_row.get('var_residual', var_row.get('var_resid', None)),
                    'icc_intercept': icc_row['ICC_intercept'],
                    'icc_slope_simple': icc_row['ICC_slope_simple'],
                })

        log("")
        log("[INFO] Model-specific variance components extracted for transparency")

        # =====================================================================
        # STEP 5: Save All Outputs
        # =====================================================================
        log("[SAVE] Saving results to CSV/YAML files...")

        # Save model comparison
        model_comparison.to_csv(MODEL_COMPARISON_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {MODEL_COMPARISON_CSV}")

        # Save competitive models metadata
        competitive_models_meta = {
            'n_models_tested': len(model_comparison),
            'n_competitive_models': len(competitive_models),
            'delta_aic_threshold': 2.0,
            'competitive_models': competitive_models,
            'generated_timestamp': datetime.now().isoformat(),
        }
        with open(COMPETITIVE_MODELS_YAML, 'w', encoding='utf-8') as f:
            yaml.dump(competitive_models_meta, f, default_flow_style=False, allow_unicode=True)
        log(f"[SAVED] {COMPETITIVE_MODELS_YAML}")

        # Save averaged variance components (already a dataframe)
        variance_components_df.to_csv(AVERAGED_VARIANCE_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {AVERAGED_VARIANCE_CSV}")

        # Save averaged ICCs (already a dataframe)
        iccs_df.to_csv(AVERAGED_ICCS_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {AVERAGED_ICCS_CSV}")

        # Save model-specific variance
        model_specific_df = pd.DataFrame(model_specific_list)
        model_specific_df.to_csv(MODEL_SPECIFIC_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {MODEL_SPECIFIC_CSV}")

        # Save averaged random effects (already has intercept_avg and slope_avg columns)
        random_effects_export = random_effects_df[['UID', 'intercept_avg', 'slope_avg']].copy()
        random_effects_export.to_csv(AVERAGED_RANDOM_EFFECTS_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {AVERAGED_RANDOM_EFFECTS_CSV}")

        # =====================================================================
        # STEP 6: Validation
        # =====================================================================
        log("[VALIDATE] Running output validation...")

        # Validate averaged variance components
        if len(variance_components_df) != 1:
            log(f"[FAIL] Expected 1 row (level='all'), got {len(variance_components_df)}")
            sys.exit(1)
        if variance_components_df.isna().sum().sum() > 0:
            log(f"[FAIL] Variance components contain NaN values")
            sys.exit(1)
        log(f"[PASS] Variance components validated: 1 level, no NaN values")

        # Validate averaged ICCs
        if len(iccs_df) != 1:
            log(f"[FAIL] Expected 1 row (level='all'), got {len(iccs_df)}")
            sys.exit(1)
        if iccs_df[['ICC_intercept', 'ICC_slope_simple']].isna().sum().sum() > 0:
            log(f"[FAIL] ICCs contain NaN values")
            sys.exit(1)
        log(f"[PASS] ICCs validated: 1 level, no NaN values")

        # Validate random effects
        if len(random_effects_export) != 100:
            log(f"[FAIL] Expected 100 participants, got {len(random_effects_export)}")
            sys.exit(1)
        if random_effects_export.isna().sum().sum() > 0:
            log(f"[FAIL] Random effects contain NaN values")
            sys.exit(1)
        log(f"[PASS] Random effects validated: 100 participants, no NaN values")

        # Check for positive variance
        if variance_avg['var_slope'] <= 0:
            log(f"[FAIL] var_slope_avg must be positive, got {variance_avg['var_slope']}")
            sys.exit(1)
        log(f"[PASS] var_slope_avg > 0 ({variance_avg['var_slope']:.6f})")

        # =====================================================================
        # SUMMARY
        # =====================================================================
        log("=" * 70)
        log("[SUCCESS] Step 06 complete: Model-averaged variance decomposition")
        log(f"  Models tested: {len(model_comparison)}")
        log(f"  Competitive models: {len(competitive_models)}")
        log(f"  var_intercept_avg: {variance_avg['var_intercept']:.6f}")
        log(f"  var_slope_avg: {variance_avg['var_slope']:.6f}")
        log(f"  ICC_intercept_avg: {iccs_avg['ICC_intercept']:.4f} ({iccs_avg['ICC_intercept']*100:.2f}%)")
        log(f"  ICC_slope_simple_avg: {iccs_avg['ICC_slope_simple']:.6f} ({iccs_avg['ICC_slope_simple']*100:.4f}%)")
        log(f"  Outputs: 6 CSV/YAML files in data/")
        log("=" * 70)

        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] Unexpected error: {str(e)}")
        log("[TRACEBACK] Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
