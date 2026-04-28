#!/usr/bin/env python3
"""Recip+Log ROOT Model Verification for RQ 5.2.5: Verify that Purification-Trajectory Paradox findings (RQ 5.2.5) remain robust"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step07b_recip_log_verification.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("STEP 07B: RECIP+LOG ROOT MODEL VERIFICATION")
        log("=" * 80)
        log("")
        log("MOTIVATION:")
        log("  RQ 5.2.1 ROOT model changed from Log-only to Recip+Log")
        log("  Testing if Purification-Trajectory Paradox robust to functional form")
        log("")
        # Load Standardized Outcomes and Add Recip+Log Transformation
        log("[STEP 1] Loading standardized outcomes...")
        log("")

        # Load standardized z-scores from step06
        input_file = RQ_DIR / "data" / "step06_standardized_outcomes.csv"
        data = pd.read_csv(input_file, encoding='utf-8')
        log(f"  Loaded: {input_file.name} ({len(data)} rows)")
        log("")

        # Add recip_TSVR transformation
        data['recip_TSVR'] = 1.0 / (data['TSVR_hours'] + 1)
        log("  Added recip_TSVR transformation: 1 / (TSVR_hours + 1)")

        # Add log_TSVR if not exists
        if 'log_TSVR' not in data.columns:
            data['log_TSVR'] = np.log(data['TSVR_hours'] + 1)
            log("  Added log_TSVR transformation: log(TSVR_hours + 1)")
        else:
            log("  log_TSVR already exists")
        log("")

        log(f"  Data: {data['UID'].nunique()} participants, {data['domain'].nunique()} domains")
        log("")
        # Fit Parallel LMMs with Recip+Log Functional Form
        log("[STEP 2] Fitting parallel LMMs with Recip+Log functional form...")
        log("")

        # Updated formula: recip_TSVR + log_TSVR instead of linear + log
        formula_recip_log = "outcome ~ recip_TSVR + log_TSVR + C(domain) + recip_TSVR:C(domain) + log_TSVR:C(domain)"
        log(f"  Formula: {formula_recip_log}")
        log("  Random effects: ~1 (intercepts only, matching original)")
        log("")

        # Fit 3 models: Full CTT, Purified CTT, IRT theta
        models = {}
        aic_results = []

        for measurement in ['z_full_ctt', 'z_purified_ctt', 'z_irt_theta']:
            log(f"  Fitting {measurement}...")

            # Prepare data with renamed outcome
            model_data = data.copy()
            model_data['outcome'] = model_data[measurement]

            # Fit model
            try:
                model = MixedLM.from_formula(
                    formula_recip_log,
                    data=model_data,
                    groups=model_data['UID'],
                    re_formula="~1"  # Intercepts only (matching original)
                ).fit(method='lbfgs', reml=False)

                log(f"    ✅ Converged: {model.converged}")
                log(f"    AIC: {model.aic:.2f}")
                log("")

                models[measurement] = model
                aic_results.append({
                    'measurement': measurement,
                    'AIC_recip_log': model.aic,
                    'BIC_recip_log': model.bic,
                    'logLik_recip_log': model.llf
                })

            except Exception as e:
                log(f"    ⚠️  Failed to converge: {str(e)}")
                log("")
                models[measurement] = None
                aic_results.append({
                    'measurement': measurement,
                    'AIC_recip_log': np.nan,
                    'BIC_recip_log': np.nan,
                    'logLik_recip_log': np.nan
                })
        # Load Original Log-Only Results for Comparison
        log("[STEP 3] Loading original Log-only results...")
        log("")

        original_file = RQ_DIR / "data" / "step07_lmm_model_comparison.csv"
        if original_file.exists():
            original_aic = pd.read_csv(original_file, encoding='utf-8')
            log(f"  Loaded: {original_file.name}")
            log("")

            # Merge with Recip+Log results
            comparison = pd.merge(
                original_aic[['measurement', 'AIC', 'delta_AIC']].rename(columns={'AIC': 'AIC_log', 'delta_AIC': 'delta_AIC_log'}),
                pd.DataFrame(aic_results),
                on='measurement',
                how='outer'
            )

            # Calculate delta_AIC for Recip+Log (IRT as reference)
            irt_aic_recip_log = comparison.loc[comparison['measurement'] == 'z_irt_theta', 'AIC_recip_log'].values[0]
            comparison['delta_AIC_recip_log'] = comparison['AIC_recip_log'] - irt_aic_recip_log

            # Check if paradox persists
            # Original pattern: Full CTT lowest AIC, Purified CTT highest AIC
            full_lower_than_irt_log = comparison.loc[comparison['measurement'] == 'z_full_ctt', 'delta_AIC_log'].values[0] < 0
            purified_higher_than_irt_log = comparison.loc[comparison['measurement'] == 'z_purified_ctt', 'delta_AIC_log'].values[0] > 0

            full_lower_than_irt_recip = comparison.loc[comparison['measurement'] == 'z_full_ctt', 'delta_AIC_recip_log'].values[0] < 0
            purified_higher_than_irt_recip = comparison.loc[comparison['measurement'] == 'z_purified_ctt', 'delta_AIC_recip_log'].values[0] > 0

            comparison['paradox_log'] = [
                full_lower_than_irt_log if m == 'z_full_ctt' else
                False if m == 'z_irt_theta' else
                purified_higher_than_irt_log
                for m in comparison['measurement']
            ]

            comparison['paradox_recip_log'] = [
                full_lower_than_irt_recip if m == 'z_full_ctt' else
                False if m == 'z_irt_theta' else
                purified_higher_than_irt_recip
                for m in comparison['measurement']
            ]

            comparison['paradox_robust'] = comparison['paradox_log'] & comparison['paradox_recip_log']

            log("  AIC Comparison (Log vs Recip+Log):")
            log("")
            log(f"  {'Measurement':<20} {'AIC (Log)':<15} {'ΔAIC (Log)':<15} {'AIC (Recip+Log)':<18} {'ΔAIC (Recip+Log)':<18} {'Paradox?'}")
            log("  " + "-" * 105)
            for _, row in comparison.iterrows():
                name = row['measurement'].replace('z_', '').replace('_', ' ').title()
                log(f"  {name:<20} {row['AIC_log']:<15.2f} {row['delta_AIC_log']:+15.2f} {row['AIC_recip_log']:<18.2f} {row['delta_AIC_recip_log']:+18.2f} {row['paradox_robust']}")
            log("")

            # Save comparison
            comparison_file = RQ_DIR / "data" / "step07b_lmm_model_comparison_recip_log.csv"
            comparison.to_csv(comparison_file, index=False, encoding='utf-8')
            log(f"  Saved: {comparison_file.name}")
            log("")

        else:
            log(f"  ⚠️  Original AIC file not found: {original_file.name}")
            log("  Cannot compare to original Log-only results")
            log("")
            comparison = pd.DataFrame(aic_results)
        # Save Model Objects
        log("[STEP 4] Saving model objects...")
        log("")

        for measurement, model in models.items():
            if model is not None:
                name = measurement.replace('z_', '')
                model_file = RQ_DIR / "data" / f"step07b_{name}_model_recip_log.pkl"
                model.save(str(model_file))
                log(f"  Saved: {model_file.name}")
        log("")
        # Verification Summary
        log("[STEP 5] VERIFICATION SUMMARY")
        log("=" * 80)
        log("")

        # Check if paradox persists
        if original_file.exists():
            paradox_persistent = comparison['paradox_robust'].iloc[0] and comparison['paradox_robust'].iloc[2]

            if paradox_persistent:
                log("✅ VERIFICATION PASSED: Purification-Trajectory Paradox ROBUST")
                log("")
                log("KEY FINDINGS:")
                log("")
                log("  Original (Log-only):")
                full_log = comparison.loc[comparison['measurement'] == 'z_full_ctt']
                purif_log = comparison.loc[comparison['measurement'] == 'z_purified_ctt']
                log(f"    - Full CTT:     AIC = {full_log['AIC_log'].values[0]:.2f} (ΔAIC = {full_log['delta_AIC_log'].values[0]:+.2f})")
                log(f"    - Purified CTT: AIC = {purif_log['AIC_log'].values[0]:.2f} (ΔAIC = {purif_log['delta_AIC_log'].values[0]:+.2f})")
                log(f"    - Pattern: Full CTT best fit, Purified CTT worst fit")
                log("")
                log("  Updated (Recip+Log):")
                log(f"    - Full CTT:     AIC = {full_log['AIC_recip_log'].values[0]:.2f} (ΔAIC = {full_log['delta_AIC_recip_log'].values[0]:+.2f})")
                log(f"    - Purified CTT: AIC = {purif_log['AIC_recip_log'].values[0]:.2f} (ΔAIC = {purif_log['delta_AIC_recip_log'].values[0]:+.2f})")
                log(f"    - Pattern: SAME (Full CTT best, Purified CTT worst)")
                log("")
                log("CONCLUSION:")
                log("  Purification-Trajectory Paradox persists regardless of functional form.")
                log("  Better correlation (Purified > Full) but WORSE trajectory fit (AIC higher).")
                log("  RQ 5.2.5 findings ROBUST, ready for GOLD status.")
            else:
                log("⚠️ VERIFICATION ALERT: Paradox pattern changed with Recip+Log")
                log("")
                log("FINDINGS:")
                log(f"  Log-only paradox: {comparison['paradox_log'].sum()}/3 measurements show expected pattern")
                log(f"  Recip+Log paradox: {comparison['paradox_recip_log'].sum()}/3 measurements show expected pattern")
                log("")
                log("RECOMMENDATION:")
                log("  Review why Recip+Log changed AIC ordering. May indicate functional form")
                log("  affects purification bias more than expected.")
        else:
            log("⚠️ Cannot verify paradox persistence (original AIC file missing)")

        log("")
        log("=" * 80)
        log("Step 07b complete")
        log(f"All outputs saved to {RQ_DIR / 'data'}")
        sys.exit(0)

    except Exception as e:
        log("")
        log("=" * 80)
        log(f"{str(e)}")
        log("")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
