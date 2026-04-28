#!/usr/bin/env python3
"""
CONVERGENCE INVESTIGATION SCRIPT
================================
RQ: 5.5.5 - Purified CTT Effects
Purpose: Test if LMM convergence failures due to random slope complexity

CONTEXT:
- Original analysis (step07) used random intercepts + slopes: re_formula='~Time'
- 4/6 models failed to converge (all IRT models + Source_Purified_CTT)
- Only Full CTT models converged successfully
- Need to test if simplified random structure (intercepts-only) resolves failures

TEST APPROACH:
1. Refit all 6 models with intercepts-only: re_formula='~1'
2. Compare convergence rates (slopes vs intercepts-only)
3. For converged models in BOTH specifications, compare AIC
4. Document random slope variance contribution

EXPECTED OUTCOMES:
A. Intercepts-only converges → Slope variance too high for stable estimation
B. Still fails → Data structure issue (4 timepoints insufficient)
C. Converges but AIC worse → Slopes provide better fit despite convergence warnings
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from statsmodels.regression.mixed_linear_model import MixedLM

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "convergence_investigation.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("\n" + "="*80)
        log("CONVERGENCE INVESTIGATION - RQ 5.5.5")
        log("="*80)
        log("Testing random intercepts-only vs intercepts+slopes")
        log("")
        # Load Data
        log("Loading standardized scores and TSVR mapping...")

        standardized_scores_path = RQ_DIR / "data" / "step06_standardized_scores.csv"
        df_scores = pd.read_csv(standardized_scores_path, encoding='utf-8')

        tsvr_path = PROJECT_ROOT / "results" / "ch5" / "5.5.1" / "data" / "step00_tsvr_mapping.csv"
        df_tsvr = pd.read_csv(tsvr_path, encoding='utf-8')

        # Convert TSVR test format
        df_tsvr['test'] = 'T' + df_tsvr['test'].astype(str)

        # Merge
        df_merged = df_scores.merge(
            df_tsvr[['UID', 'test', 'TSVR_hours']],
            on=['UID', 'test'],
            how='left'
        )

        # Create Time variable
        df_merged['Time'] = df_merged['TSVR_hours'] / 24.0

        log(f"{len(df_merged)} observations, Time range: {df_merged['Time'].min():.2f}-{df_merged['Time'].max():.2f} days")
        log("")
        # Define Models to Test
        models = [
            {'name': 'Source_IRT',              'location': 'source',      'outcome': 'irt_z'},
            {'name': 'Source_Full_CTT',         'location': 'source',      'outcome': 'ctt_full_z'},
            {'name': 'Source_Purified_CTT',     'location': 'source',      'outcome': 'ctt_purified_z'},
            {'name': 'Destination_IRT',         'location': 'destination', 'outcome': 'irt_z'},
            {'name': 'Destination_Full_CTT',    'location': 'destination', 'outcome': 'ctt_full_z'},
            {'name': 'Destination_Purified_CTT','location': 'destination', 'outcome': 'ctt_purified_z'},
        ]
        # Fit Both Specifications
        results = []

        for model_spec in models:
            model_name = model_spec['name']
            location = model_spec['location']
            outcome = model_spec['outcome']

            log("-" * 80)
            log(f"{model_name}")
            log("-" * 80)

            # Filter data
            df_filtered = df_merged[df_merged['location_type'] == location].copy()
            df_filtered['score'] = df_filtered[outcome]
            n_obs = len(df_filtered)

            log(f"  N observations: {n_obs}")
            log(f"  N participants: {df_filtered['UID'].nunique()}")
            log(f"  N timepoints per participant: {df_filtered.groupby('UID').size().mean():.1f}")

            # -----------------------------------------------------------------
            # A. ORIGINAL: Random Intercepts + Slopes
            # -----------------------------------------------------------------
            log(f"\n  [A] ORIGINAL SPECIFICATION: Random intercepts + slopes")
            log(f"      Formula: score ~ Time, re_formula='~Time'")

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    md_slopes = MixedLM.from_formula(
                        formula="score ~ Time",
                        data=df_filtered,
                        groups=df_filtered['UID'],
                        re_formula='~Time'
                    )
                    result_slopes = md_slopes.fit(reml=False)

                    aic_slopes = result_slopes.aic
                    converged_slopes = result_slopes.converged

                    # Extract random effects variance
                    if converged_slopes:
                        try:
                            # Random intercept variance
                            var_intercept = result_slopes.cov_re.iloc[0, 0]
                            # Random slope variance (if available)
                            if result_slopes.cov_re.shape[0] >= 2:
                                var_slope = result_slopes.cov_re.iloc[1, 1]
                            else:
                                var_slope = np.nan
                        except:
                            var_intercept = np.nan
                            var_slope = np.nan
                    else:
                        var_intercept = np.nan
                        var_slope = np.nan

                    # Check for convergence warnings
                    warnings_list = [str(warning.message) for warning in w]
                    has_warnings = len(warnings_list) > 0

                    log(f"      ✓ Converged: {converged_slopes}")
                    log(f"      ✓ AIC: {aic_slopes:.2f}")
                    if not np.isnan(var_intercept):
                        log(f"      ✓ Var(Intercept): {var_intercept:.4f}")
                    if not np.isnan(var_slope):
                        log(f"      ✓ Var(Slope): {var_slope:.4f}")
                    if has_warnings:
                        log(f"      ⚠ Warnings: {len(warnings_list)}")
                        for warning_msg in warnings_list[:3]:  # Show first 3
                            log(f"         - {warning_msg[:100]}")

            except Exception as e:
                log(f"      ✗ FAILED: {str(e)[:200]}")
                aic_slopes = np.nan
                converged_slopes = False
                var_intercept = np.nan
                var_slope = np.nan
                has_warnings = False

            # -----------------------------------------------------------------
            # B. SIMPLIFIED: Random Intercepts Only
            # -----------------------------------------------------------------
            log(f"\n  [B] SIMPLIFIED SPECIFICATION: Random intercepts only")
            log(f"      Formula: score ~ Time, re_formula='~1'")

            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")

                    md_intercepts = MixedLM.from_formula(
                        formula="score ~ Time",
                        data=df_filtered,
                        groups=df_filtered['UID'],
                        re_formula='~1'  # Intercepts only
                    )
                    result_intercepts = md_intercepts.fit(reml=False)

                    aic_intercepts = result_intercepts.aic
                    converged_intercepts = result_intercepts.converged

                    # Extract random intercept variance
                    if converged_intercepts:
                        try:
                            var_intercept_simple = result_intercepts.cov_re.iloc[0, 0]
                        except:
                            var_intercept_simple = np.nan
                    else:
                        var_intercept_simple = np.nan

                    # Check for warnings
                    warnings_list_simple = [str(warning.message) for warning in w]
                    has_warnings_simple = len(warnings_list_simple) > 0

                    log(f"      ✓ Converged: {converged_intercepts}")
                    log(f"      ✓ AIC: {aic_intercepts:.2f}")
                    if not np.isnan(var_intercept_simple):
                        log(f"      ✓ Var(Intercept): {var_intercept_simple:.4f}")
                    if has_warnings_simple:
                        log(f"      ⚠ Warnings: {len(warnings_list_simple)}")

            except Exception as e:
                log(f"      ✗ FAILED: {str(e)[:200]}")
                aic_intercepts = np.nan
                converged_intercepts = False
                var_intercept_simple = np.nan
                has_warnings_simple = False

            # -----------------------------------------------------------------
            # C. COMPARISON
            # -----------------------------------------------------------------
            log(f"\n  [C] COMPARISON:")

            # Convergence comparison
            if converged_slopes and converged_intercepts:
                log(f"      ✓ Both specifications converged")
                delta_aic = aic_slopes - aic_intercepts
                log(f"      ✓ ΔAIC (Slopes - Intercepts): {delta_aic:.2f}")

                if delta_aic < -2:
                    log(f"      → Slopes provide BETTER fit (ΔAIC < -2)")
                    recommendation = "Use slopes despite convergence warnings"
                elif abs(delta_aic) <= 2:
                    log(f"      → No meaningful difference (|ΔAIC| ≤ 2)")
                    recommendation = "Use intercepts (simpler, equally good)"
                else:
                    log(f"      → Intercepts provide BETTER fit (ΔAIC > +2)")
                    recommendation = "Use intercepts (better fit, simpler)"

            elif converged_intercepts and not converged_slopes:
                log(f"      ⚠ Only intercepts-only converged")
                log(f"      → Random slope variance too high for stable estimation")
                recommendation = "Use intercepts-only (slopes unstable)"
                delta_aic = np.nan

            elif converged_slopes and not converged_intercepts:
                log(f"      ⚠ Only slopes converged (unusual)")
                log(f"      → Use slopes specification")
                recommendation = "Use slopes (intercepts-only failed)"
                delta_aic = np.nan

            else:
                log(f"      ✗ Neither specification converged")
                log(f"      → Data structure issue (insufficient timepoints?)")
                recommendation = "Cannot use LMM (convergence failures)"
                delta_aic = np.nan

            log(f"\n  {recommendation}")
            log("")

            # Store results
            results.append({
                'model': model_name,
                'location': location,
                'outcome': outcome,
                'n_obs': n_obs,
                # Slopes specification
                'slopes_converged': converged_slopes,
                'slopes_aic': aic_slopes,
                'slopes_var_intercept': var_intercept,
                'slopes_var_slope': var_slope,
                'slopes_has_warnings': has_warnings,
                # Intercepts-only specification
                'intercepts_converged': converged_intercepts,
                'intercepts_aic': aic_intercepts,
                'intercepts_var_intercept': var_intercept_simple,
                'intercepts_has_warnings': has_warnings_simple,
                # Comparison
                'delta_aic': delta_aic,
                'recommendation': recommendation
            })
        # Summary Report
        log("\n" + "="*80)
        log("SUMMARY REPORT")
        log("="*80)

        df_results = pd.DataFrame(results)

        # Overall convergence rates
        n_models = len(df_results)
        n_slopes_converged = df_results['slopes_converged'].sum()
        n_intercepts_converged = df_results['intercepts_converged'].sum()

        log(f"\nCONVERGENCE RATES:")
        log(f"  Slopes specification:      {n_slopes_converged}/{n_models} converged ({100*n_slopes_converged/n_models:.0f}%)")
        log(f"  Intercepts-only:           {n_intercepts_converged}/{n_models} converged ({100*n_intercepts_converged/n_models:.0f}%)")
        log(f"  Improvement with simplification: {n_intercepts_converged - n_slopes_converged} models")

        # Which models benefited from simplification?
        improved = df_results[(~df_results['slopes_converged']) & (df_results['intercepts_converged'])]
        if len(improved) > 0:
            log(f"\nMODELS THAT CONVERGED AFTER SIMPLIFICATION:")
            for _, row in improved.iterrows():
                log(f"  - {row['model']}: Now AIC = {row['intercepts_aic']:.2f}")

        # Which models converged in both?
        both_converged = df_results[df_results['slopes_converged'] & df_results['intercepts_converged']]
        if len(both_converged) > 0:
            log(f"\nMODELS THAT CONVERGED IN BOTH SPECIFICATIONS:")
            for _, row in both_converged.iterrows():
                log(f"  - {row['model']}:")
                log(f"      Slopes AIC: {row['slopes_aic']:.2f}")
                log(f"      Intercepts AIC: {row['intercepts_aic']:.2f}")
                log(f"      ΔAIC: {row['delta_aic']:.2f}")
                log(f"      → {row['recommendation']}")

        # Which models still failed?
        still_failed = df_results[(~df_results['slopes_converged']) & (~df_results['intercepts_converged'])]
        if len(still_failed) > 0:
            log(f"\nMODELS THAT FAILED IN BOTH SPECIFICATIONS:")
            for _, row in still_failed.iterrows():
                log(f"  - {row['model']}: Data structure issue (insufficient timepoints?)")
        # Save Results
        output_path = RQ_DIR / "data" / "convergence_investigation.csv"
        df_results.to_csv(output_path, index=False, encoding='utf-8')
        log(f"\n{output_path.name}")

        log("\nConvergence investigation complete")
        log("="*80)

    except Exception as e:
        log(f"\n{str(e)}")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
