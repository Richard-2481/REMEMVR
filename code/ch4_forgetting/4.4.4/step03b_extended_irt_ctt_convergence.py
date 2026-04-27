"""
RQ 5.4.4 - Step 03b: Extended IRT-CTT Convergence Across 66 Functional Forms

PURPOSE:
Test whether IRT-CTT convergence (correlations, kappa, agreement) is robust
across all 66 functional forms from kitchen sink model comparison.

APPROACH:
- Fit 66 models for IRT theta (one per functional form)
- Fit 66 models for CTT mean scores (matching functional forms)
- For each of 66 model pairs:
  - Compute Pearson r (correlation of predictions)
  - Compute Cohen's kappa (agreement on fixed effect significance)
  - Compute percent agreement
- Document robustness: "Convergence maintained across X% of functional forms"

INPUT:
- data/step02_merged_irt_ctt.csv (400 rows: IRT theta + CTT scores)
- data/step00_tsvr_mapping.csv (400 rows: TSVR time variable)

OUTPUT:
- data/step03b_extended_convergence.csv (66 rows: model-specific convergence metrics)
- data/step03b_convergence_summary.txt (robustness statistics)
- logs/step03b_extended_convergence.log

CRITICAL:
Uses tools.model_selection.compare_lmm_models_kitchen_sink() adapted for
parallel IRT+CTT model fitting with identical specifications.

Author: Claude Code
Date: 2025-12-09
RQ: ch5/5.4.4
Step: 03b (extended)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Import kitchen_sink model selection tool
from tools.model_selection import compare_lmm_models_kitchen_sink
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

# =============================================================================
# CONFIGURATION
# =============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch5/5.4.4
LOG_FILE = RQ_DIR / "logs" / "step03b_extended_convergence.log"
DATA_DIR = RQ_DIR / "data"

# =============================================================================
# LOGGING
# =============================================================================

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_kappa_from_models(irt_model, ctt_model, alpha=0.05):
    """
    Compute Cohen's kappa comparing fixed effect significance.

    Returns:
        kappa (float): Cohen's kappa score
        agreement_pct (float): Percent agreement
        n_terms (int): Number of terms compared
    """
    try:
        # Extract p-values from both models
        irt_params = irt_model.params
        ctt_params = ctt_model.params

        irt_pvalues = irt_model.pvalues
        ctt_pvalues = ctt_model.pvalues

        # Find common terms
        common_terms = list(set(irt_params.index) & set(ctt_params.index))

        if len(common_terms) == 0:
            return np.nan, np.nan, 0

        # Significance classification
        irt_sig = [irt_pvalues[term] < alpha for term in common_terms]
        ctt_sig = [ctt_pvalues[term] < alpha for term in common_terms]

        # Cohen's kappa
        kappa = cohen_kappa_score(irt_sig, ctt_sig)

        # Percent agreement
        agreement = sum([i == c for i, c in zip(irt_sig, ctt_sig)]) / len(common_terms)

        return kappa, agreement * 100, len(common_terms)

    except Exception as e:
        log(f"    [WARNING] Kappa computation failed: {e}")
        return np.nan, np.nan, 0

def compute_correlation_from_predictions(irt_model, ctt_model, data):
    """
    Compute correlation between IRT and CTT model predictions.

    Returns:
        r (float): Pearson correlation coefficient
        p (float): p-value
    """
    try:
        irt_pred = irt_model.fittedvalues
        ctt_pred = ctt_model.fittedvalues

        r, p = pearsonr(irt_pred, ctt_pred)
        return r, p

    except Exception as e:
        log(f"    [WARNING] Correlation computation failed: {e}")
        return np.nan, np.nan

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

if __name__ == "__main__":
    try:
        log("=" * 80)
        log("[START] Step 03b: Extended IRT-CTT Convergence (66 Models)")
        log("=" * 80)

        # =====================================================================
        # STEP 1: Load Data
        # =====================================================================

        log("[LOAD] Loading merged IRT-CTT data...")
        merged_path = DATA_DIR / "step02_merged_irt_ctt.csv"
        tsvr_path = DATA_DIR / "step00_tsvr_mapping.csv"

        if not merged_path.exists():
            raise FileNotFoundError(f"Merged data missing: {merged_path}")
        if not tsvr_path.exists():
            raise FileNotFoundError(f"TSVR mapping missing: {tsvr_path}")

        merged_df = pd.read_csv(merged_path)
        tsvr_df = pd.read_csv(tsvr_path)

        log(f"  ✓ Loaded {merged_path.name}: {len(merged_df)} rows")
        log(f"  ✓ Loaded {tsvr_path.name}: {len(tsvr_df)} rows")

        # Merge with TSVR
        merged_with_tsvr = merged_df.merge(
            tsvr_df[['composite_ID', 'TSVR_hours']],
            on='composite_ID'
        )

        log(f"  ✓ Merged dataset: {len(merged_with_tsvr)} rows")

        # =====================================================================
        # STEP 2: Prepare IRT and CTT Long-Format Data
        # =====================================================================

        log("[PREPARE] Creating long-format data for IRT and CTT...")

        # IRT input (long format)
        irt_long = merged_with_tsvr[['composite_ID', 'UID', 'TEST', 'TSVR_hours',
                                      'theta_common', 'theta_congruent', 'theta_incongruent']].melt(
            id_vars=['composite_ID', 'UID', 'TEST', 'TSVR_hours'],
            var_name='congruence_var',
            value_name='theta'
        )
        irt_long['congruence'] = irt_long['congruence_var'].str.replace('theta_', '')
        irt_long = irt_long.drop(columns=['congruence_var'])

        log(f"  ✓ IRT long format: {len(irt_long)} rows")

        # CTT input (long format)
        ctt_long = merged_with_tsvr[['composite_ID', 'UID', 'TEST', 'TSVR_hours',
                                      'CTT_common', 'CTT_congruent', 'CTT_incongruent']].melt(
            id_vars=['composite_ID', 'UID', 'TEST', 'TSVR_hours'],
            var_name='congruence_var',
            value_name='CTT_mean'
        )
        ctt_long['congruence'] = ctt_long['congruence_var'].str.replace('CTT_', '')
        ctt_long = ctt_long.drop(columns=['congruence_var'])

        log(f"  ✓ CTT long format: {len(ctt_long)} rows")

        # =====================================================================
        # STEP 3: Run Kitchen Sink for IRT
        # =====================================================================

        log("[ANALYSIS] Running kitchen sink for IRT theta...")
        log("  This will fit 66 models (may take 30-45 minutes)...")

        irt_results = compare_lmm_models_kitchen_sink(
            data=irt_long,
            outcome_var='theta',
            tsvr_var='TSVR_hours',
            groups_var='UID',
            factor1_var='congruence',
            factor1_reference='common',
            factor2_var=None,
            re_formula='~1',  # Random intercepts only for stability
            reml=False,
            save_dir=DATA_DIR,
            log_file=LOG_FILE,
        )

        log("[DONE] IRT kitchen sink complete")
        log(f"  Models fitted: {len(irt_results['comparison'])}")

        # =====================================================================
        # STEP 4: Run Kitchen Sink for CTT
        # =====================================================================

        log("[ANALYSIS] Running kitchen sink for CTT scores...")
        log("  This will fit 66 models (may take 30-45 minutes)...")

        ctt_results = compare_lmm_models_kitchen_sink(
            data=ctt_long,
            outcome_var='CTT_mean',
            tsvr_var='TSVR_hours',
            groups_var='UID',
            factor1_var='congruence',
            factor1_reference='common',
            factor2_var=None,
            re_formula='~1',  # Random intercepts only for stability
            reml=False,
            save_dir=DATA_DIR,
            log_file=LOG_FILE,
        )

        log("[DONE] CTT kitchen sink complete")
        log(f"  Models fitted: {len(ctt_results['comparison'])}")

        # =====================================================================
        # STEP 5: Compute Convergence Metrics for Each Model Pair
        # =====================================================================

        log("[ANALYSIS] Computing convergence metrics for 66 model pairs...")

        convergence_results = []

        irt_models = irt_results['fitted_models']
        ctt_models = ctt_results['fitted_models']

        for model_name in irt_models.keys():
            log(f"  Processing: {model_name}")

            if model_name not in ctt_models:
                log(f"    [SKIP] CTT model not found")
                continue

            irt_model = irt_models[model_name]
            ctt_model = ctt_models[model_name]

            # Check convergence
            if not irt_model.converged or not ctt_model.converged:
                log(f"    [SKIP] Non-convergence (IRT:{irt_model.converged}, CTT:{ctt_model.converged})")
                convergence_results.append({
                    'model_name': model_name,
                    'irt_converged': irt_model.converged,
                    'ctt_converged': ctt_model.converged,
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'kappa': np.nan,
                    'agreement_pct': np.nan,
                    'n_terms': 0,
                    'irt_aic': irt_model.aic,
                    'ctt_aic': ctt_model.aic,
                })
                continue

            # Compute correlation
            r, p = compute_correlation_from_predictions(irt_model, ctt_model, irt_long)

            # Compute kappa
            kappa, agreement_pct, n_terms = compute_kappa_from_models(irt_model, ctt_model)

            convergence_results.append({
                'model_name': model_name,
                'irt_converged': True,
                'ctt_converged': True,
                'correlation': r,
                'p_value': p,
                'kappa': kappa,
                'agreement_pct': agreement_pct,
                'n_terms': n_terms,
                'irt_aic': irt_model.aic,
                'ctt_aic': ctt_model.aic,
            })

            log(f"    r={r:.3f}, kappa={kappa:.3f}, agreement={agreement_pct:.1f}%")

        # Convert to DataFrame
        convergence_df = pd.DataFrame(convergence_results)

        log(f"[DONE] Convergence metrics computed for {len(convergence_df)} model pairs")

        # =====================================================================
        # STEP 6: Compute Summary Statistics
        # =====================================================================

        log("[SUMMARY] Computing robustness statistics...")

        # Filter to converged models only
        converged = convergence_df[
            convergence_df['irt_converged'] & convergence_df['ctt_converged']
        ]

        n_converged = len(converged)
        n_total = len(convergence_df)

        log(f"  Converged models: {n_converged}/{n_total} ({n_converged/n_total*100:.1f}%)")

        if n_converged > 0:
            # Correlation robustness
            r_mean = converged['correlation'].mean()
            r_median = converged['correlation'].median()
            r_min = converged['correlation'].min()
            r_max = converged['correlation'].max()
            r_above_070 = (converged['correlation'] > 0.70).sum()
            r_above_090 = (converged['correlation'] > 0.90).sum()

            log("")
            log("CORRELATION ROBUSTNESS:")
            log(f"  Mean r: {r_mean:.3f}")
            log(f"  Median r: {r_median:.3f}")
            log(f"  Range: [{r_min:.3f}, {r_max:.3f}]")
            log(f"  r > 0.70: {r_above_070}/{n_converged} ({r_above_070/n_converged*100:.1f}%)")
            log(f"  r > 0.90: {r_above_090}/{n_converged} ({r_above_090/n_converged*100:.1f}%)")

            # Kappa robustness
            kappa_mean = converged['kappa'].mean()
            kappa_median = converged['kappa'].median()
            kappa_min = converged['kappa'].min()
            kappa_max = converged['kappa'].max()
            kappa_above_060 = (converged['kappa'] > 0.60).sum()

            log("")
            log("KAPPA ROBUSTNESS:")
            log(f"  Mean kappa: {kappa_mean:.3f}")
            log(f"  Median kappa: {kappa_median:.3f}")
            log(f"  Range: [{kappa_min:.3f}, {kappa_max:.3f}]")
            log(f"  kappa > 0.60: {kappa_above_060}/{n_converged} ({kappa_above_060/n_converged*100:.1f}%)")

            # Agreement robustness
            agreement_mean = converged['agreement_pct'].mean()
            agreement_median = converged['agreement_pct'].median()
            agreement_min = converged['agreement_pct'].min()
            agreement_max = converged['agreement_pct'].max()
            agreement_above_080 = (converged['agreement_pct'] > 80).sum()

            log("")
            log("AGREEMENT ROBUSTNESS:")
            log(f"  Mean agreement: {agreement_mean:.1f}%")
            log(f"  Median agreement: {agreement_median:.1f}%")
            log(f"  Range: [{agreement_min:.1f}%, {agreement_max:.1f}%]")
            log(f"  agreement > 80%: {agreement_above_080}/{n_converged} ({agreement_above_080/n_converged*100:.1f}%)")

        # =====================================================================
        # STEP 7: Save Results
        # =====================================================================

        log("[SAVE] Saving convergence results...")

        # Save detailed results
        output_path = DATA_DIR / "step03b_extended_convergence.csv"
        convergence_df.to_csv(output_path, index=False)
        log(f"  ✓ Saved: {output_path.name}")

        # Save summary
        summary_path = DATA_DIR / "step03b_convergence_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("RQ 5.4.4 - Extended IRT-CTT Convergence Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Models tested: {n_total}\n")
            f.write(f"Models converged: {n_converged} ({n_converged/n_total*100:.1f}%)\n\n")

            if n_converged > 0:
                f.write("CORRELATION ROBUSTNESS:\n")
                f.write(f"  Mean r: {r_mean:.3f}\n")
                f.write(f"  Median r: {r_median:.3f}\n")
                f.write(f"  Range: [{r_min:.3f}, {r_max:.3f}]\n")
                f.write(f"  r > 0.70: {r_above_070}/{n_converged} ({r_above_070/n_converged*100:.1f}%)\n")
                f.write(f"  r > 0.90: {r_above_090}/{n_converged} ({r_above_090/n_converged*100:.1f}%)\n\n")

                f.write("KAPPA ROBUSTNESS:\n")
                f.write(f"  Mean kappa: {kappa_mean:.3f}\n")
                f.write(f"  Median kappa: {kappa_median:.3f}\n")
                f.write(f"  Range: [{kappa_min:.3f}, {kappa_max:.3f}]\n")
                f.write(f"  kappa > 0.60: {kappa_above_060}/{n_converged} ({kappa_above_060/n_converged*100:.1f}%)\n\n")

                f.write("AGREEMENT ROBUSTNESS:\n")
                f.write(f"  Mean agreement: {agreement_mean:.1f}%\n")
                f.write(f"  Median agreement: {agreement_median:.1f}%\n")
                f.write(f"  Range: [{agreement_min:.1f}%, {agreement_max:.1f}%]\n")
                f.write(f"  agreement > 80%: {agreement_above_080}/{n_converged} ({agreement_above_080/n_converged*100:.1f}%)\n")

        log(f"  ✓ Saved: {summary_path.name}")

        log("=" * 80)
        log("[SUCCESS] Step 03b complete")
        log("=" * 80)

    except Exception as e:
        log("[ERROR] Step 03b failed")
        log(f"  {type(e).__name__}: {str(e)}")
        import traceback
        log(traceback.format_exc())
        raise
