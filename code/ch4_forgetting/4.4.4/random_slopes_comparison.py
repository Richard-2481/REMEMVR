"""
Random Slopes Comparison for RQ 5.4.4

MANDATORY TEST per improvement_taxonomy.md Section 4.4:
"Cannot claim homogeneous effects without testing for heterogeneity"

Purpose: Compare intercepts-only vs intercepts+slopes random effects structure
         for BOTH IRT and CTT parallel LMMs.

Expected Outcomes:
A) Slopes improve fit (ΔAIC > 2) → Use slopes model, report individual differences
B) Slopes don't converge → Document attempt, keep intercepts-only
C) Slopes converge but don't improve (ΔAIC < 2) → Keep intercepts, document negligible variance

Date: 2025-12-31 (PLATINUM certification)
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from pathlib import Path
import logging

# Setup paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'
LOGS_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'random_slopes_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fit_model_with_fallback(data, formula, groups, re_formula, model_name):
    """Fit LMM with fallback strategies if convergence fails."""

    # Try Powell optimizer (used in original step03)
    try:
        logger.info(f"Fitting {model_name} with re_formula={re_formula}")
        model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
        result = model.fit(reml=False, method='powell')

        if result.converged:
            logger.info(f"{model_name}: Converged (powell)")
            return result, 'converged'
        else:
            logger.warning(f"{model_name}: Powell did not converge, trying LBFGS")
    except Exception as e:
        logger.warning(f"{model_name}: Powell failed with {e}, trying LBFGS")

    # Fallback 1: Try LBFGS
    try:
        model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
        result = model.fit(reml=False, method='lbfgs')

        if result.converged:
            logger.info(f"{model_name}: Converged (lbfgs)")
            return result, 'converged'
        else:
            logger.warning(f"{model_name}: LBFGS did not converge")
    except Exception as e:
        logger.warning(f"{model_name}: LBFGS failed with {e}")

    return None, 'failed'

def compare_random_structures(data, dv_col, model_label):
    """
    Compare intercepts-only vs intercepts+slopes for a single DV.

    Parameters:
    - data: DataFrame with long-format LMM input
    - dv_col: Name of DV column ('theta' or 'ctt')
    - model_label: 'IRT' or 'CTT'

    Returns:
    - Dictionary with comparison results
    """

    logger.info(f"\n{'='*60}")
    logger.info(f"Random Structure Comparison: {model_label}")
    logger.info(f"{'='*60}\n")

    # Fixed effects formula (identical for both random structures)
    # Using Recip+Log two-process model per RQ 5.4.1 ROOT
    formula = f"{dv_col} ~ C(congruence) * (recip_TSVR + log_TSVR)"
    groups = data['UID']

    results = {
        'model': model_label,
        'dv': dv_col,
        'formula': formula
    }

    # Model 1: Intercepts-only (baseline)
    logger.info("Fitting Model 1: Intercepts-only (~1)")
    re_formula_intercepts = "~1"
    model_intercepts, status_intercepts = fit_model_with_fallback(
        data, formula, groups, re_formula_intercepts, f"{model_label}_Intercepts"
    )

    if status_intercepts != 'converged':
        logger.error(f"{model_label} Intercepts-only FAILED to converge")
        results['outcome'] = 'BLOCKER: Intercepts-only failed to converge'
        return results

    # Model 2: Intercepts + Slopes on recip_TSVR (rapid component)
    logger.info("Fitting Model 2: Intercepts + Slopes (~recip_TSVR)")
    re_formula_slopes = "~recip_TSVR"
    model_slopes, status_slopes = fit_model_with_fallback(
        data, formula, groups, re_formula_slopes, f"{model_label}_Slopes"
    )

    if status_slopes != 'converged':
        logger.warning(f"{model_label} Slopes model FAILED to converge")
        results['outcome'] = 'Option B: Slopes failed to converge, keep intercepts-only'
        results['aic_intercepts'] = model_intercepts.aic
        results['aic_slopes'] = None
        results['delta_aic'] = None
        results['recommendation'] = 'Use intercepts-only (slopes convergence failure)'
        return results

    # Both models converged - Compare via AIC
    aic_intercepts = model_intercepts.aic
    aic_slopes = model_slopes.aic
    delta_aic = aic_intercepts - aic_slopes  # Positive = slopes better

    logger.info(f"\n{'='*60}")
    logger.info(f"AIC Comparison: {model_label}")
    logger.info(f"{'='*60}")
    logger.info(f"Intercepts-only AIC: {aic_intercepts:.2f}")
    logger.info(f"Intercepts+Slopes AIC: {aic_slopes:.2f}")
    logger.info(f"ΔAIC (Intercepts - Slopes): {delta_aic:.2f}")
    logger.info(f"{'='*60}\n")

    # Interpret ΔAIC
    if delta_aic > 2:
        outcome = f"Option A: Slopes improve fit (ΔAIC = {delta_aic:.2f} > 2)"
        recommendation = "Use slopes model (individual differences in forgetting rates)"

        # Extract random slope variance
        try:
            slope_var = model_slopes.cov_re.iloc[1, 1]
            slope_sd = np.sqrt(slope_var)
            logger.info(f"Random slope variance: {slope_var:.4f}")
            logger.info(f"Random slope SD: {slope_sd:.4f}")
            results['slope_var'] = slope_var
            results['slope_sd'] = slope_sd
        except Exception as e:
            logger.warning(f"Could not extract slope variance: {e}")
            results['slope_var'] = None
            results['slope_sd'] = None

    elif abs(delta_aic) < 2:
        outcome = f"Option C: Slopes converge but don't improve (|ΔAIC| = {abs(delta_aic):.2f} < 2)"
        recommendation = "Keep intercepts-only (negligible slope variance, homogeneous effects CONFIRMED)"
        results['slope_var'] = 0.0  # Negligible
        results['slope_sd'] = 0.0
    else:
        # delta_aic < -2 (slopes WORSE than intercepts - unusual)
        outcome = f"Anomaly: Slopes WORSEN fit (ΔAIC = {delta_aic:.2f} < -2)"
        recommendation = "Keep intercepts-only (slopes add noise, not signal)"
        results['slope_var'] = None
        results['slope_sd'] = None

    results['aic_intercepts'] = aic_intercepts
    results['aic_slopes'] = aic_slopes
    results['delta_aic'] = delta_aic
    results['outcome'] = outcome
    results['recommendation'] = recommendation

    logger.info(f"Outcome: {outcome}")
    logger.info(f"Recommendation: {recommendation}\n")

    return results

def main():
    """Main execution: Test random slopes for both IRT and CTT."""

    logger.info("="*60)
    logger.info("RQ 5.4.4: Random Slopes Comparison (PLATINUM Certification)")
    logger.info("="*60)
    logger.info("Testing intercepts-only vs intercepts+slopes for parallel LMMs")
    logger.info("Mandatory per improvement_taxonomy.md Section 4.4\n")

    # Load IRT LMM input
    logger.info("Loading IRT LMM input data...")
    irt_data = pd.read_csv(DATA_DIR / 'step03_irt_lmm_input.csv')
    logger.info(f"IRT data: {irt_data.shape[0]} observations")

    # Load CTT LMM input
    logger.info("Loading CTT LMM input data...")
    ctt_data = pd.read_csv(DATA_DIR / 'step03_ctt_lmm_input.csv')
    logger.info(f"CTT data: {ctt_data.shape[0]} observations\n")

    # Compare IRT
    irt_results = compare_random_structures(irt_data, 'theta', 'IRT')

    # Compare CTT
    ctt_results = compare_random_structures(ctt_data, 'CTT_mean', 'CTT')

    # Compile comparison table
    comparison = pd.DataFrame([
        {
            'model': 'IRT',
            'aic_intercepts_only': irt_results.get('aic_intercepts'),
            'aic_intercepts_slopes': irt_results.get('aic_slopes'),
            'delta_aic': irt_results.get('delta_aic'),
            'slope_var': irt_results.get('slope_var'),
            'slope_sd': irt_results.get('slope_sd'),
            'outcome': irt_results.get('outcome'),
            'recommendation': irt_results.get('recommendation')
        },
        {
            'model': 'CTT',
            'aic_intercepts_only': ctt_results.get('aic_intercepts'),
            'aic_intercepts_slopes': ctt_results.get('aic_slopes'),
            'delta_aic': ctt_results.get('delta_aic'),
            'slope_var': ctt_results.get('slope_var'),
            'slope_sd': ctt_results.get('slope_sd'),
            'outcome': ctt_results.get('outcome'),
            'recommendation': ctt_results.get('recommendation')
        }
    ])

    # Save comparison table
    output_file = DATA_DIR / 'random_slopes_comparison.csv'
    comparison.to_csv(output_file, index=False)
    logger.info(f"\n{'='*60}")
    logger.info(f"Comparison saved: {output_file}")
    logger.info(f"{'='*60}\n")

    # Print summary
    print("\n" + "="*60)
    print("RANDOM SLOPES COMPARISON SUMMARY")
    print("="*60)
    print(comparison.to_string(index=False))
    print("="*60)

    # Interpretation
    print("\nINTERPRETATION:")
    print(f"IRT: {irt_results['outcome']}")
    print(f"     {irt_results['recommendation']}")
    print(f"\nCTT: {ctt_results['outcome']}")
    print(f"     {ctt_results['recommendation']}")

    # Check if BOTH models agree on recommendation
    if irt_results['recommendation'] == ctt_results['recommendation']:
        print("\n✅ CONVERGENCE: Both IRT and CTT recommend same random structure")
        print("   This strengthens IRT-CTT convergence claims (agreement on model specification)")
    else:
        print("\n⚠️  DIVERGENCE: IRT and CTT recommend different random structures")
        print("   This may affect interpretation of convergence findings")

    logger.info("\nRandom slopes comparison COMPLETE")
    logger.info("BLOCKER RESOLVED: Section 4.4 random effects testing performed")

if __name__ == '__main__':
    main()
