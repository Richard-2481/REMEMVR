#!/usr/bin/env python3
"""
Step 08: Model-Averaged Variance Decomposition (STRATIFIED BY DOMAIN)

Applies model averaging across competitive models for What/Where domains separately.
Replaces Log-only variance components from Steps 02-03 with robust model-averaged estimates.

CRITICAL UPDATE (2025-12-09): RQ 5.2.1 ROOT now uses Recip+Log (10 models averaged).
Original RQ 5.2.6 used Log-only variance decomposition (single-model bias).
This step implements stratified model averaging to:
1. Test 65+ models per domain (kitchen sink comparison)
2. Identify competitive models (ΔAIC < 2.0) per domain
3. Compute model-averaged variance components per domain
4. Generate 200 model-averaged random effects (100 UID × 2 domains) for RQ 5.2.7

Input: results/ch5/5.2.1/data/step04_lmm_input.csv (1200 rows, 3 domains)
Output: 6 files per domain + metadata
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.variance_decomposition import compute_model_averaged_variance_decomposition

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
LMM_INPUT = PROJECT_ROOT / "results/ch5/5.2.1/data/step04_lmm_input.csv"
DATA_DIR = RQ_DIR / "data"
LOG_FILE = RQ_DIR / "logs/step08_model_averaged_variance.log"

# Output files (per domain)
MODEL_COMPARISON_CSV = DATA_DIR / "step08_model_comparison.csv"
COMPETITIVE_MODELS_YAML = DATA_DIR / "step08_competitive_models_metadata.yaml"
AVERAGED_VARIANCE_CSV = DATA_DIR / "step08_averaged_variance_components.csv"
AVERAGED_ICCS_CSV = DATA_DIR / "step08_averaged_iccs.csv"
MODEL_SPECIFIC_CSV = DATA_DIR / "step08_model_specific_variance.csv"
AVERAGED_RANDOM_EFFECTS_CSV = DATA_DIR / "step08_averaged_random_effects.csv"

def log(msg):
    """Write to log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def main():
    try:
        # =====================================================================
        # STEP 1: Initialize Log
        # =====================================================================
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("=== Step 08: Model-Averaged Variance Decomposition (Stratified) ===\n")
            f.write(f"Started: {datetime.now().isoformat()}\n\n")

        log("[START] Model-averaged variance decomposition (stratified by domain)")
        log(f"[INFO] RQ 5.2.6 upgrade: Log-only → Model-averaged (v4.X)")
        log("")

        # =====================================================================
        # STEP 2: Load LMM Input Data
        # =====================================================================
        log("[LOAD] Loading LMM input from RQ 5.2.1...")
        if not LMM_INPUT.exists():
            log(f"[FAIL] Cannot find {LMM_INPUT}")
            log("[FAIL] RQ 5.2.1 must complete Step 04 first")
            sys.exit(1)

        lmm_input = pd.read_csv(LMM_INPUT, encoding='utf-8')
        log(f"[LOADED] {len(lmm_input)} rows, {len(lmm_input.columns)} columns")
        log(f"[INFO] Columns: {list(lmm_input.columns)}")

        # Verify expected columns
        required_cols = ['UID', 'theta', 'TSVR_hours', 'domain']
        missing_cols = [col for col in required_cols if col not in lmm_input.columns]
        if missing_cols:
            log(f"[ERROR] Missing required columns: {missing_cols}")
            sys.exit(1)

        log(f"[INFO] Unique participants: {lmm_input['UID'].nunique()}")
        log(f"[INFO] Unique domains: {lmm_input['domain'].nunique()}")
        log(f"[INFO] Domain value counts:\n{lmm_input['domain'].value_counts()}")

        # Filter to What/Where only (When excluded due to floor effect)
        log("")
        log("[FILTER] Excluding 'when' domain (floor effect from RQ 5.2.1)")
        lmm_input_filtered = lmm_input[lmm_input['domain'].isin(['what', 'where'])].copy()
        log(f"[FILTERED] {len(lmm_input_filtered)} rows (what + where only)")
        log(f"[INFO] Observations per domain: {len(lmm_input_filtered) / 2:.0f}")

        # =====================================================================
        # STEP 3: Run Model-Averaged Variance Decomposition (STRATIFIED)
        # =====================================================================
        log("")
        log("[ANALYSIS] Running STRATIFIED model-averaged variance decomposition...")
        log(f"  Outcome variable: theta")
        log(f"  Time variable: TSVR_hours")
        log(f"  Groups variable: UID")
        log(f"  Stratify variable: domain (2 levels: what, where)")
        log(f"  Delta AIC threshold: 2.0 (competitive models)")
        log(f"  Random effects: Intercepts + Slopes")
        log("")
        log("[INFO] This will take ~4-8 minutes (2 domains × ~65 models each)...")
        log("")

        results = compute_model_averaged_variance_decomposition(
            data=lmm_input_filtered,
            outcome_var='theta',
            tsvr_var='TSVR_hours',
            groups_var='UID',
            stratify_var='domain',
            stratify_levels=['what', 'where'],
            delta_aic_threshold=2.0,
            min_models=3,
            max_models=10,
            re_intercept=True,
            re_slope=True,
            save_dir=None,
            log_file=None,
            return_fitted_models=False,
            reml=False,
            handle_convergence_failures='warn',
        )

        log("[SUCCESS] Model-averaged variance decomposition complete")
        log("")

        # =====================================================================
        # STEP 4: Extract and Save Results
        # =====================================================================
        log("[EXTRACT] Extracting results from variance decomposition...")

        # Extract kitchen sink model comparison (POOLED across domains)
        model_comparison = results['model_comparison']
        log(f"[INFO] Kitchen sink comparison: {len(model_comparison)} models tested")

        # Best model info (pooled)
        best_row = model_comparison.iloc[0]
        log(f"[INFO] Best model (pooled): {best_row['model_name']} (AIC={best_row['AIC']:.2f})")

        # Extract competitive models metadata (pooled)
        competitive_models = results['competitive_models']
        log(f"[INFO] Competitive models (pooled, ΔAIC < 2.0): {len(competitive_models)}")
        for model_name in competitive_models:
            model_row = model_comparison[model_comparison['model_name'] == model_name].iloc[0]
            log(f"  - {model_name}: AIC={model_row['AIC']:.2f}, ΔAIC={model_row['delta_AIC']:.2f}, weight={model_row['akaike_weight']:.3f}")

        # Extract variance components (2-row dataframe for 2 domains)
        variance_components_df = results['averaged_variance_components']
        log("")
        log(f"[INFO] Model-averaged variance components ({len(variance_components_df)} domains):")
        log(f"  Columns available: {list(variance_components_df.columns)}")

        for idx, row in variance_components_df.iterrows():
            domain = row.get('domain', row.get('level', f'domain_{idx}'))
            var_int = row.get('var_intercept', row.get('var_int', np.nan))
            var_slope = row.get('var_slope', np.nan)
            var_resid = row.get('var_residual', row.get('var_resid', np.nan))

            log(f"  Domain: {domain}")
            log(f"    var_intercept = {var_int:.6f}")
            log(f"    var_slope = {var_slope:.6f}")
            log(f"    var_residual = {var_resid:.6f}")

        # Extract ICCs (2-row dataframe)
        iccs_df = results['averaged_ICCs']
        log("")
        log(f"[INFO] Model-averaged ICCs ({len(iccs_df)} domains):")
        log(f"  Columns available: {list(iccs_df.columns)}")

        for idx, row in iccs_df.iterrows():
            domain = row.get('domain', row.get('level', f'domain_{idx}'))
            icc_int = row.get('ICC_intercept', np.nan)
            icc_slope = row.get('ICC_slope_simple', np.nan)

            log(f"  Domain: {domain}")
            log(f"    ICC_intercept = {icc_int:.6f} ({icc_int*100:.2f}%)")
            log(f"    ICC_slope_simple = {icc_slope:.6f} ({icc_slope*100:.4f}%)")

            if 'ICC_slope_conditional' in iccs_df.columns:
                icc_cond = row.get('ICC_slope_conditional', np.nan)
                if pd.notna(icc_cond):
                    log(f"    ICC_slope_conditional = {icc_cond:.6f} ({icc_cond*100:.4f}%)")

        # Extract random effects (dataframe with UID, domain, intercept_avg, slope_avg)
        random_effects_df = results['averaged_random_effects']
        log("")
        log(f"[INFO] Model-averaged random effects: {len(random_effects_df)} rows")
        log(f"  Columns available: {list(random_effects_df.columns)}")
        log(f"  Expected: 200 rows (100 UID × 2 domains)")

        # Verify structure
        if len(random_effects_df) != 200:
            log(f"[WARNING] Expected 200 rows (100 UID × 2 domains), got {len(random_effects_df)}")

        # Find intercept and slope columns
        int_col = None
        slope_col = None
        for col in random_effects_df.columns:
            if 'intercept' in col.lower():
                int_col = col
            elif 'slope' in col.lower():
                slope_col = col

        if int_col and slope_col:
            log(f"  Intercept column: {int_col}")
            log(f"  Slope column: {slope_col}")
            log(f"  Intercept range: [{random_effects_df[int_col].min():.3f}, {random_effects_df[int_col].max():.3f}]")
            log(f"  Slope range: [{random_effects_df[slope_col].min():.3f}, {random_effects_df[slope_col].max():.3f}]")
        else:
            log(f"  ERROR: Could not find intercept/slope columns")

        # Extract model-specific variance (from stratified_results)
        log("")
        log("[INFO] Extracting model-specific variance per domain...")

        model_specific_list = []
        for domain_name in ['what', 'where']:
            if domain_name not in results['stratified_results']:
                log(f"  WARNING: Domain '{domain_name}' not found in stratified_results")
                continue

            stratified_domain = results['stratified_results'][domain_name]
            var_by_model = stratified_domain['variance_components_by_model']
            icc_by_model = stratified_domain['ICCs_by_model']

            log(f"  Domain: {domain_name}")
            log(f"    Variance by model: {len(var_by_model)} rows")
            log(f"    ICC by model: {len(icc_by_model)} rows")

            # Determine model identifier column name
            model_col = None
            for col in var_by_model.columns:
                if 'model' in col.lower():
                    model_col = col
                    break

            if model_col is None:
                log(f"    WARNING: No 'model' column found for domain {domain_name}")
                continue

            # Extract competitive models for this domain
            domain_competitive = var_by_model[model_col].unique()[:10]  # Top 10

            for model_name in domain_competitive:
                if model_name not in var_by_model[model_col].values:
                    continue

                var_row = var_by_model[var_by_model[model_col] == model_name].iloc[0]
                icc_row = icc_by_model[icc_by_model[model_col] == model_name].iloc[0]

                model_specific_list.append({
                    'domain': domain_name,
                    'model_name': model_name,
                    'var_intercept': var_row['var_intercept'],
                    'var_slope': var_row['var_slope'],
                    'cov_intercept_slope': var_row.get('cov_intercept_slope', None),
                    'var_residual': var_row.get('var_residual', var_row.get('var_resid', None)),
                    'icc_intercept': icc_row['ICC_intercept'],
                    'icc_slope_simple': icc_row['ICC_slope_simple'],
                })

        log("")
        log(f"[INFO] Extracted {len(model_specific_list)} model-specific entries")

        # =====================================================================
        # STEP 5: Save All Outputs
        # =====================================================================
        log("")
        log("[SAVE] Saving results to CSV/YAML files...")

        # Save model comparison (pooled)
        model_comparison.to_csv(MODEL_COMPARISON_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {MODEL_COMPARISON_CSV}")

        # Save competitive models metadata
        competitive_models_meta = {
            'kitchen_sink_models_tested': len(model_comparison),
            'best_model': best_row['model_name'],
            'best_aic': float(best_row['AIC']),
            'delta_aic_threshold': 2.0,
            'competitive_models': competitive_models,
            'generated_timestamp': datetime.now().isoformat(),
            'note': 'Pooled AIC across domains, but variance components computed per domain',
        }
        with open(COMPETITIVE_MODELS_YAML, 'w', encoding='utf-8') as f:
            yaml.dump(competitive_models_meta, f, default_flow_style=False, allow_unicode=True)
        log(f"[SAVED] {COMPETITIVE_MODELS_YAML}")

        # Save averaged variance components (2 rows: what, where)
        variance_components_df.to_csv(AVERAGED_VARIANCE_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {AVERAGED_VARIANCE_CSV}")

        # Save averaged ICCs (2 rows: what, where)
        iccs_df.to_csv(AVERAGED_ICCS_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {AVERAGED_ICCS_CSV}")

        # Save model-specific variance
        model_specific_df = pd.DataFrame(model_specific_list)
        model_specific_df.to_csv(MODEL_SPECIFIC_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {MODEL_SPECIFIC_CSV}")

        # Save averaged random effects (200 rows: 100 UID × 2 domains)
        # Ensure columns match expected format for RQ 5.2.7
        random_effects_export = random_effects_df.copy()

        # Rename columns if needed
        rename_map = {}
        if int_col and int_col != 'intercept_avg':
            rename_map[int_col] = 'intercept_avg'
        if slope_col and slope_col != 'slope_avg':
            rename_map[slope_col] = 'slope_avg'

        if rename_map:
            random_effects_export = random_effects_export.rename(columns=rename_map)

        # Select only needed columns
        export_cols = ['UID', 'domain', 'intercept_avg', 'slope_avg']
        missing_export_cols = [col for col in export_cols if col not in random_effects_export.columns]

        if missing_export_cols:
            log(f"[WARNING] Missing export columns: {missing_export_cols}")
            # Use available columns
            export_cols = [col for col in export_cols if col in random_effects_export.columns]

        random_effects_export = random_effects_export[export_cols]
        random_effects_export.to_csv(AVERAGED_RANDOM_EFFECTS_CSV, index=False, encoding='utf-8')
        log(f"[SAVED] {AVERAGED_RANDOM_EFFECTS_CSV}")

        # =====================================================================
        # STEP 6: Validation
        # =====================================================================
        log("")
        log("[VALIDATE] Running output validation...")

        # Validate averaged variance components
        if len(variance_components_df) != 2:
            log(f"[FAIL] Expected 2 domains (what, where), got {len(variance_components_df)}")
            sys.exit(1)
        if variance_components_df[['var_intercept', 'var_slope', 'var_residual']].isna().sum().sum() > 0:
            log(f"[FAIL] Variance components contain NaN values")
            sys.exit(1)
        log(f"[PASS] Variance components validated: 2 domains, no NaN values")

        # Validate averaged ICCs
        if len(iccs_df) != 2:
            log(f"[FAIL] Expected 2 domains (what, where), got {len(iccs_df)}")
            sys.exit(1)
        if iccs_df[['ICC_intercept', 'ICC_slope_simple']].isna().sum().sum() > 0:
            log(f"[FAIL] ICCs contain NaN values")
            sys.exit(1)
        log(f"[PASS] ICCs validated: 2 domains, no NaN values")

        # Validate random effects
        if len(random_effects_export) != 200:
            log(f"[WARNING] Expected 200 rows (100 UID × 2 domains), got {len(random_effects_export)}")
        if random_effects_export.isna().sum().sum() > 0:
            log(f"[FAIL] Random effects contain NaN values")
            sys.exit(1)
        log(f"[PASS] Random effects validated: {len(random_effects_export)} rows, no NaN values")

        # Check for positive variance
        for idx, row in variance_components_df.iterrows():
            domain = row.get('domain', row.get('level', f'domain_{idx}'))
            var_slope = row.get('var_slope', np.nan)
            if pd.isna(var_slope) or var_slope <= 0:
                log(f"[FAIL] Domain {domain}: var_slope must be positive, got {var_slope}")
                sys.exit(1)
        log(f"[PASS] All variance components positive")

        # =====================================================================
        # STEP 7: Summary
        # =====================================================================
        log("")
        log("=" * 70)
        log("[COMPLETE] Step 08: Model-Averaged Variance Decomposition (Stratified)")
        log("=" * 70)
        log(f"Kitchen sink: {len(model_comparison)} models tested (pooled AIC)")
        log(f"Competitive: {len(competitive_models)} models (ΔAIC < 2.0)")
        log(f"Domains analyzed: 2 (what, where)")
        log(f"Random effects: {len(random_effects_export)} rows → RQ 5.2.7 input")
        log("")
        log("[NEXT] Run RQ 5.2.7 with model-averaged random effects from step08")
        log("")

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        log("[TRACEBACK]")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
