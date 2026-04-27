#!/usr/bin/env python3
"""
RQ 6.3.1 Step 06: Compute Post-Hoc Contrasts (Decision D068)

Purpose:
    If Domain × Time interaction significant (p < 0.05), compute pairwise domain slope
    comparisons with Bonferroni correction per Decision D068.

Inputs:
    - data/step05_lmm_coefficients.csv (interaction p-values)
    - data/step04_lmm_input.csv (raw data for re-fitting)

Outputs:
    - data/step06_post_hoc_contrasts.csv (contrast results)
    - data/step06_contrast_decision.txt (decision document)
    - logs/step06_compute_post_hoc_contrasts.log

Author: REMEMVR Analysis Pipeline
Date: 2025-12-07
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import statsmodels.formula.api as smf


def setup_logging(log_path: Path):
    """Setup logging to file and stdout."""
    import logging

    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def log(msg: str, logger):
    """Log message to both file and stdout with flush."""
    logger.info(msg)
    sys.stdout.flush()


def compute_slope_contrasts(model, domain_levels):
    """
    Compute pairwise contrasts for domain × log_TSVR interaction slopes.

    The model has:
    - log_TSVR: slope for reference domain (What)
    - C(congruence)[T.When]:log_TSVR: difference from What slope
    - C(congruence)[T.Where]:log_TSVR: difference from What slope

    We want:
    - What slope = β_log_TSVR
    - When slope = β_log_TSVR + β_When:log_TSVR
    - Where slope = β_log_TSVR + β_Where:log_TSVR

    Contrasts:
    1. When vs What: β_When:log_TSVR
    2. Where vs What: β_Where:log_TSVR
    3. When vs Where: β_When:log_TSVR - β_Where:log_TSVR
    """
    from scipy import stats

    params = model.params
    cov = model.cov_params()

    # Extract coefficients
    beta_log_TSVR = params['log_TSVR']

    # Extract interaction terms
    beta_When_TSVR = params['C(congruence)[T.When]:log_TSVR']
    beta_Where_TSVR = params['C(congruence)[T.Where]:log_TSVR']

    # Contrast 1: When vs What (already in model as interaction)
    contrast1_est = beta_When_TSVR
    contrast1_se = np.sqrt(cov.loc['C(congruence)[T.When]:log_TSVR', 'C(congruence)[T.When]:log_TSVR'])

    # Contrast 2: Where vs What (already in model as interaction)
    contrast2_est = beta_Where_TSVR
    contrast2_se = np.sqrt(cov.loc['C(congruence)[T.Where]:log_TSVR', 'C(congruence)[T.Where]:log_TSVR'])

    # Contrast 3: When vs Where (difference of interactions)
    contrast3_est = beta_When_TSVR - beta_Where_TSVR

    # Variance for difference: Var(A - B) = Var(A) + Var(B) - 2*Cov(A,B)
    var_When = cov.loc['C(congruence)[T.When]:log_TSVR', 'C(congruence)[T.When]:log_TSVR']
    var_Where = cov.loc['C(congruence)[T.Where]:log_TSVR', 'C(congruence)[T.Where]:log_TSVR']
    cov_When_Where = cov.loc['C(congruence)[T.When]:log_TSVR', 'C(congruence)[T.Where]:log_TSVR']
    contrast3_se = np.sqrt(var_When + var_Where - 2*cov_When_Where)

    # Compute z-statistics and p-values
    contrasts = []

    for name, est, se in [
        ('When vs What', contrast1_est, contrast1_se),
        ('Where vs What', contrast2_est, contrast2_se),
        ('When vs Where', contrast3_est, contrast3_se)
    ]:
        z = est / se
        p_uncorrected = 2 * (1 - stats.norm.cdf(abs(z)))
        p_bonferroni = min(1.0, p_uncorrected * 3)  # 3 comparisons

        # Cohen's d: standardized mean difference
        # For regression slopes, d = estimate / residual_sd
        residual_sd = np.sqrt(model.scale)  # Residual variance
        cohens_d = est / residual_sd

        contrasts.append({
            'contrast': name,
            'estimate': est,
            'se': se,
            'z': z,
            'p_uncorrected': p_uncorrected,
            'p_bonferroni': p_bonferroni,
            'cohens_d': cohens_d
        })

    return pd.DataFrame(contrasts)


def main():
    # Setup paths (derived from script location, not hardcoded)
    rq_dir = Path(__file__).resolve().parents[1]  # results/ch6/6.5.1
    data_dir = rq_dir / "data"
    code_dir = rq_dir / "code"
    logs_dir = rq_dir / "logs"

    # Setup logging
    log_path = logs_dir / "step06_compute_post_hoc_contrasts.log"
    logger = setup_logging(log_path)

    log("\n" + "="*60, logger)
    log("[START] Step 06: Compute Post-Hoc Contrasts (Decision D068)", logger)
    log("="*60 + "\n", logger)

    try:
        # Load Step 05 coefficients
        log("[LOAD] Loading Step 05 fixed effects coefficients...", logger)
        coef_path = data_dir / "step05_lmm_coefficients.csv"
        df_coef = pd.read_csv(coef_path)
        log(f"[LOADED] step05_lmm_coefficients.csv ({len(df_coef)} rows)\n", logger)

        # Check interaction significance
        log("[CHECK] Checking Domain × Time interaction significance...", logger)
        interaction_terms = df_coef[df_coef['term'].str.contains(':log_TSVR')]
        log(f"[FOUND] {len(interaction_terms)} interaction terms:\n", logger)
        for _, row in interaction_terms.iterrows():
            log(f"  - {row['term']}: p={row['p_value']:.4f}", logger)

        min_interaction_p = interaction_terms['p_value'].min()
        log(f"\n[DECISION] Minimum interaction p-value: {min_interaction_p:.4f}", logger)
        log(f"[DECISION] Threshold: 0.05", logger)

        if min_interaction_p < 0.05:
            log(f"[DECISION] Result: SIGNIFICANT (p < 0.05)\n", logger)
            log("[COMPUTE] Domain × Time interaction IS significant", logger)
            log("          Post-hoc pairwise contrasts appropriate\n", logger)

            # Load data and re-fit model
            log("[LOAD] Loading LMM input data for model re-fitting...", logger)
            lmm_input_path = data_dir / "step04_lmm_input.csv"
            df_lmm = pd.read_csv(lmm_input_path)
            log(f"[LOADED] step04_lmm_input.csv ({len(df_lmm)} rows)\n", logger)

            # Re-fit model
            log("[FIT] Re-fitting LMM model...", logger)
            log("      Formula: theta ~ C(congruence) * log_TSVR", logger)
            log("      Random effects: ~log_TSVR | UID", logger)
            log("      Method: powell (ML estimation)\n", logger)

            # Suppress convergence warnings (boundary is expected for random effects)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                model = smf.mixedlm(
                    formula="theta ~ C(congruence) * log_TSVR",
                    data=df_lmm,
                    groups=df_lmm["UID"],
                    re_formula="~log_TSVR"
                ).fit(method='powell', maxiter=1000, reml=False)

            log(f"[FITTED] Model converged: {model.converged}", logger)
            log(f"         AIC: {model.aic:.2f}", logger)
            log(f"         BIC: {model.bic:.2f}", logger)
            log(f"         Log-Likelihood: {model.llf:.2f}\n", logger)

            # Compute contrasts
            log("[ANALYSIS] Computing pairwise slope contrasts...", logger)
            log("           Implementing Decision D068: Dual p-value reporting", logger)
            log("           - Family-wise alpha: 0.05", logger)
            log("           - Number of comparisons: 3", logger)
            log("           - Bonferroni-corrected alpha: 0.0167\n", logger)

            df_contrasts = compute_slope_contrasts(
                model=model,
                domain_levels=['What', 'Where', 'When']
            )

            # Save contrasts
            contrasts_path = data_dir / "step06_post_hoc_contrasts.csv"
            df_contrasts.to_csv(contrasts_path, index=False, float_format='%.6f')
            log(f"[SAVED] {contrasts_path.name} ({len(df_contrasts)} rows)\n", logger)

            # Print contrasts summary
            log("="*60, logger)
            log("POST-HOC CONTRASTS SUMMARY (Decision D068)", logger)
            log("="*60 + "\n", logger)

            for _, row in df_contrasts.iterrows():
                log(f"Contrast: {row['contrast']}", logger)
                log(f"  Estimate: {row['estimate']:+.4f}", logger)
                log(f"  SE: {row['se']:.4f}", logger)
                log(f"  z: {row['z']:+.3f}", logger)
                log(f"  p (uncorrected): {row['p_uncorrected']:.4f}", logger)
                log(f"  p (Bonferroni): {row['p_bonferroni']:.4f}", logger)
                sig = "***" if row['p_bonferroni'] < 0.001 else "**" if row['p_bonferroni'] < 0.01 else "*" if row['p_bonferroni'] < 0.05 else "ns"
                log(f"  Significance: {sig}", logger)
                log(f"  Cohen's d: {row['cohens_d']:+.3f}\n", logger)

            # Write decision document
            decision_text = f"""RQ 6.3.1 Step 06: Post-Hoc Contrast Decision

Domain × Time Interaction Test:
  - Minimum interaction p-value: {min_interaction_p:.4f}
  - Threshold: 0.05
  - Result: SIGNIFICANT

Decision: CONTRASTS COMPUTED

Rationale:
  Domain × Time interaction is significant (p < 0.05), indicating
  that confidence decline trajectories differ across episodic memory
  domains. Post-hoc pairwise contrasts are appropriate to identify
  which specific domains differ.

Number of Comparisons: 3
  1. When vs What
  2. Where vs What
  3. When vs Where

Multiple Comparison Correction (Decision D068):
  - Method: Bonferroni
  - Family-wise alpha: 0.05
  - Per-comparison alpha: 0.0167 (0.05/3)
  - Dual p-values reported: uncorrected AND Bonferroni-corrected

Significant Contrasts (Bonferroni-corrected p < 0.05):
"""

            sig_contrasts = df_contrasts[df_contrasts['p_bonferroni'] < 0.05]
            if len(sig_contrasts) > 0:
                for _, row in sig_contrasts.iterrows():
                    decision_text += f"  - {row['contrast']}: estimate={row['estimate']:+.4f}, p={row['p_bonferroni']:.4f}\n"
            else:
                decision_text += "  None (all contrasts non-significant after Bonferroni correction)\n"

            decision_path = data_dir / "step06_contrast_decision.txt"
            decision_path.write_text(decision_text)
            log(f"[SAVED] {decision_path.name}\n", logger)

        else:
            log(f"[DECISION] Result: NOT SIGNIFICANT (p >= 0.05)\n", logger)
            log("[SKIP] Domain × Time interaction is NOT significant", logger)
            log("       Post-hoc contrasts not appropriate (NULL hypothesis supported)\n", logger)

            # Create empty contrasts file
            df_contrasts = pd.DataFrame(columns=[
                'contrast', 'estimate', 'se', 'z', 'p_uncorrected', 'p_bonferroni', 'cohens_d'
            ])
            contrasts_path = data_dir / "step06_post_hoc_contrasts.csv"
            df_contrasts.to_csv(contrasts_path, index=False)
            log(f"[SAVED] {contrasts_path.name} (0 rows - empty as expected)\n", logger)

            # Write decision document
            decision_text = f"""RQ 6.3.1 Step 06: Post-Hoc Contrast Decision

Domain × Time Interaction Test:
  - Minimum interaction p-value: {min_interaction_p:.4f}
  - Threshold: 0.05
  - Result: NOT SIGNIFICANT

Decision: CONTRASTS SKIPPED

Rationale:
  Domain × Time interaction is not significant (p >= 0.05), indicating
  that confidence decline trajectories do NOT differ across episodic memory
  domains. This supports the NULL hypothesis (unitized VR encoding produces
  equivalent forgetting across What/Where/When domains).

  Post-hoc pairwise contrasts are not appropriate when the omnibus interaction
  test is non-significant.

Number of Comparisons: 0 (contrasts not computed)

Result:
  All episodic memory domains show equivalent confidence decline over time.
"""

            decision_path = data_dir / "step06_contrast_decision.txt"
            decision_path.write_text(decision_text)
            log(f"[SAVED] {decision_path.name}\n", logger)

        # Validation
        log("="*60, logger)
        log("[VALIDATION] Checking outputs...", logger)
        log("="*60 + "\n", logger)

        # Check files exist
        assert contrasts_path.exists(), f"Missing {contrasts_path}"
        assert decision_path.exists(), f"Missing {decision_path}"
        log("[PASS] Required files exist", logger)

        # Check contrasts file structure
        df_check = pd.read_csv(contrasts_path)
        expected_cols = ['contrast', 'estimate', 'se', 'z', 'p_uncorrected', 'p_bonferroni', 'cohens_d']
        assert list(df_check.columns) == expected_cols, f"Column mismatch: {df_check.columns}"
        log("[PASS] Contrasts CSV has correct columns", logger)

        if len(df_check) > 0:
            # Check p-value properties
            assert (df_check['p_uncorrected'] >= 0).all() and (df_check['p_uncorrected'] <= 1).all(), "p_uncorrected out of [0,1]"
            assert (df_check['p_bonferroni'] >= 0).all() and (df_check['p_bonferroni'] <= 1).all(), "p_bonferroni out of [0,1]"
            assert (df_check['p_bonferroni'] >= df_check['p_uncorrected']).all(), "Bonferroni < uncorrected"
            log("[PASS] Dual p-values in valid ranges (Decision D068)", logger)

            # Check SE > 0
            assert (df_check['se'] > 0).all(), "SE not positive"
            log("[PASS] Standard errors positive", logger)

            # Check finite values
            assert df_check[['estimate', 'se', 'z', 'cohens_d']].notna().all().all(), "NaN in contrasts"
            log("[PASS] No NaN values in contrasts", logger)
        else:
            log("[PASS] Empty contrasts file (expected for non-significant interaction)", logger)

        log("\n" + "="*60, logger)
        log("[SUCCESS] Step 06 Complete", logger)
        log("="*60 + "\n", logger)

        return 0

    except Exception as e:
        log(f"\n[ERROR] {str(e)}\n", logger)
        import traceback
        log("[TRACEBACK] Full error details:", logger)
        log(traceback.format_exc(), logger)
        return 1


if __name__ == "__main__":
    sys.exit(main())
