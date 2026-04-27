#!/usr/bin/env python3
"""
Step 02: Model-Averaged Variance Decomposition (Congruence-Stratified)

PURPOSE:
Compute model-averaged variance decomposition for congruence levels (Common,
Congruent, Incongruent) accounting for functional form uncertainty from RQ 5.4.1.

METHODOLOGY:
Uses Burnham & Anderson (2002) multi-model inference:
1. Identifies competitive models from RQ 5.4.1 (ΔAIC < 2)
2. Fits stratified LMMs for each congruence × model combination
3. Akaike-averages variance components, ICCs, and random effects
4. Provides transparency: saves both model-specific AND averaged results

RATIONALE:
RQ 5.4.1 extended model selection showed:
- Best model (PowerLaw_01): 6.0% Akaike weight << 30% threshold
- 15 competitive models with ΔAIC < 2
- Effective N = 13.96 models (extreme uncertainty)
- Model averaging MANDATORY per Burnham & Anderson (2002)

EXPECTED INPUTS:
- ../../5.4.1/data/step04_lmm_input.csv: LMM input (1200 rows × 6 columns)

EXPECTED OUTPUTS:
- data/step02_averaged_variance_components.csv: Model-averaged variance (3 rows)
- data/step02_averaged_iccs.csv: Model-averaged ICCs (3 rows)
- data/step02_averaged_random_effects.csv: Model-averaged REs (300 rows = 100 UID × 3 congruence)
- data/step02_model_specific_variance.csv: All model×congruence results (transparency)
- data/step02_competitive_models_metadata.yaml: Model selection metadata
- logs/step02_model_averaged_variance.log: Detailed execution log

VALIDATION CRITERIA:
- Minimum 3 competitive models (ΔAIC < 2)
- At least 50% of models converge per congruence level
- Model-averaged ICCs bounded in [0, 1]
- Random effects centered at zero (mean ≈ 0)
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.variance_decomposition import compute_model_averaged_variance_decomposition

# Configuration
RQ_DIR = Path(__file__).resolve().parents[1]
ROOT_RQ_DIR = RQ_DIR.parents[0] / "5.4.1"  # RQ 5.4.1 contains input data
LOG_FILE = RQ_DIR / "logs" / "step02_model_averaged_variance.log"
DATA_DIR = RQ_DIR / "data"

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Step 02: Model-Averaged Variance Decomposition")
        log("=" * 80)

        # =====================================================================
        # STEP 1: Load LMM Input Data from RQ 5.4.1
        # =====================================================================
        log("\n[LOAD] Loading LMM input data from RQ 5.4.1...")

        input_file = ROOT_RQ_DIR / "data" / "step04_lmm_input.csv"
        df_lmm = pd.read_csv(input_file, encoding='utf-8')

        log(f"[LOADED] {input_file.name}")
        log(f"  Rows: {len(df_lmm)}")
        log(f"  Columns: {list(df_lmm.columns)}")
        log(f"  UIDs: {df_lmm['UID'].nunique()}")
        log(f"  Congruence levels: {sorted(df_lmm['congruence'].unique())}")

        # =====================================================================
        # STEP 2: Run Model-Averaged Variance Decomposition
        # =====================================================================
        log("\n[ANALYSIS] Computing model-averaged variance decomposition...")
        log("  Methodology: Burnham & Anderson (2002) multi-model inference")
        log("  Threshold: ΔAIC < 2.0 (competitive models)")
        log("  Max models: 6 (balance between thoroughness and computation)")
        log("  Random effects: Intercept + Slope (full covariance)")
        log("  Convergence: Warn on failure, continue with converged models")

        results = compute_model_averaged_variance_decomposition(
            data=df_lmm,
            outcome_var='theta',
            tsvr_var='TSVR_hours',
            groups_var='UID',
            stratify_var='congruence',
            stratify_levels=['common', 'congruent', 'incongruent'],  # Lowercase per data
            delta_aic_threshold=2.0,
            min_models=3,
            max_models=6,
            re_intercept=True,
            re_slope=True,
            save_dir=DATA_DIR,
            log_file=LOG_FILE,
            return_fitted_models=False,
            reml=False,  # Use ML for model comparison (REML for final estimates)
            handle_convergence_failures='warn',
        )

        log("\n[COMPLETE] Model averaging complete")
        log(f"  Models used: {results['summary_stats']['n_models_competitive']}")
        log(f"  Effective N models: {results['summary_stats']['effective_n_models']:.2f}")

        # =====================================================================
        # STEP 3: Save Model-Averaged Results
        # =====================================================================
        log("\n[SAVE] Saving model-averaged results...")

        # 3.1: Variance components
        variance_file = DATA_DIR / "step02_averaged_variance_components.csv"
        results['averaged_variance_components'].to_csv(variance_file, index=False, encoding='utf-8')
        log(f"[SAVED] {variance_file.name} ({len(results['averaged_variance_components'])} rows)")

        # 3.2: ICCs
        icc_file = DATA_DIR / "step02_averaged_iccs.csv"
        results['averaged_ICCs'].to_csv(icc_file, index=False, encoding='utf-8')
        log(f"[SAVED] {icc_file.name} ({len(results['averaged_ICCs'])} rows)")

        # 3.3: Random effects (for RQ 5.4.7 clustering)
        re_file = DATA_DIR / "step02_averaged_random_effects.csv"
        results['averaged_random_effects'].to_csv(re_file, index=False, encoding='utf-8')
        log(f"[SAVED] {re_file.name} ({len(results['averaged_random_effects'])} rows)")

        # 3.4: Model-specific results (transparency) - combine from stratified_results
        model_specific_dfs = []
        for level in ['common', 'congruent', 'incongruent']:
            level_var = results['stratified_results'][level]['variance_components_by_model'].copy()
            level_var['congruence'] = level
            model_specific_dfs.append(level_var)

        model_specific_variance = pd.concat(model_specific_dfs, ignore_index=True)
        model_specific_file = DATA_DIR / "step02_model_specific_variance.csv"
        model_specific_variance.to_csv(model_specific_file, index=False, encoding='utf-8')
        log(f"[SAVED] {model_specific_file.name} ({len(model_specific_variance)} rows)")

        # =====================================================================
        # STEP 4: Save Model Selection Metadata
        # =====================================================================
        log("\n[SAVE] Saving model selection metadata...")

        # Read competitive_models from the saved CSV
        competitive_models_df = pd.read_csv(DATA_DIR / "competitive_models.csv", encoding='utf-8')

        # Extract model names and weights
        models_used = competitive_models_df['model_name'].tolist()
        weights_dict = dict(zip(
            competitive_models_df['model_name'],
            competitive_models_df['weight_renorm']
        ))

        metadata = {
            'models_used': models_used,
            'effective_n_models': float(results['summary_stats']['effective_n_models']),
            'delta_aic_threshold': 2.0,
            'weights': weights_dict,
            'congruence_levels': ['common', 'congruent', 'incongruent'],
            'n_participants': int(df_lmm['UID'].nunique()),
            'n_observations': int(len(df_lmm)),
            'methodology': 'Burnham & Anderson (2002) multi-model inference',
            'rationale': 'RQ 5.4.1 best model weight 6.0% << 30% threshold → averaging MANDATORY',
            'convergence_rate': float(results['summary_stats']['convergence_rate']),
            'best_model': results['summary_stats']['best_model'],
            'best_model_weight': float(results['summary_stats']['best_model_weight']),
        }

        metadata_file = DATA_DIR / "step02_competitive_models_metadata.yaml"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

        log(f"[SAVED] {metadata_file.name}")

        # =====================================================================
        # STEP 5: Validation
        # =====================================================================
        log("\n[VALIDATION] Validating model-averaged results...")

        # 5.1: Check sufficient models
        n_models = results['summary_stats']['n_models_competitive']
        if n_models < 3:
            log(f"[WARNING] Only {n_models} competitive models - results may be unstable")
        else:
            log(f"[PASS] {n_models} competitive models used")

        # 5.2: Check ICCs bounded
        iccs = results['averaged_ICCs']
        icc_cols = ['ICC_intercept', 'ICC_slope_simple', 'ICC_slope_conditional']

        for col in icc_cols:
            if col in iccs.columns:
                min_val = iccs[col].min()
                max_val = iccs[col].max()

                if min_val < 0 or max_val > 1:
                    log(f"[WARNING] {col} out of bounds: [{min_val:.4f}, {max_val:.4f}]")
                else:
                    log(f"[PASS] {col} bounded in [0, 1]: [{min_val:.4f}, {max_val:.4f}]")

        # 5.3: Check random effects centering
        re_avg = results['averaged_random_effects']
        mean_intercept = re_avg['intercept_avg'].mean()
        mean_slope = re_avg['slope_avg'].mean()

        log(f"[INFO] Random effects means (should be ≈ 0):")
        log(f"  Mean intercept: {mean_intercept:.6f}")
        log(f"  Mean slope: {mean_slope:.6f}")

        if abs(mean_intercept) < 0.01 and abs(mean_slope) < 0.01:
            log("[PASS] Random effects centered at zero")
        else:
            log("[WARNING] Random effects not well-centered (may indicate numerical issues)")

        # =====================================================================
        # STEP 6: Summary Statistics
        # =====================================================================
        log("\n[SUMMARY] Model-averaged variance decomposition results:")
        log("-" * 80)

        for idx, row in results['averaged_variance_components'].iterrows():
            log(f"\n{row['congruence']} congruence:")
            log(f"  var_intercept:   {row['var_intercept']:.6f}")
            log(f"  var_slope:       {row['var_slope']:.6f}")
            log(f"  cov_int_slope:   {row['cov_intercept_slope']:.6f}")
            log(f"  var_residual:    {row['var_residual']:.6f}")

        log("\n" + "-" * 80)

        for idx, row in results['averaged_ICCs'].iterrows():
            log(f"\n{row['congruence']} ICCs:")
            log(f"  ICC_intercept:          {row['ICC_intercept']:.4f}")
            log(f"  ICC_slope_simple:       {row['ICC_slope_simple']:.4f}")
            log(f"  ICC_slope_conditional:  {row['ICC_slope_conditional']:.4f}")

        log("\n" + "=" * 80)
        log("[SUCCESS] Step 02 complete - Model-averaged variance decomposition")
        log("=" * 80)

        sys.exit(0)

    except Exception as e:
        log(f"\n[ERROR] {str(e)}")
        log("\n[TRACEBACK] Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
