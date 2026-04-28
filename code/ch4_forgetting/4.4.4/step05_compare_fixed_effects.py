"""
Step 05: Compare Fixed Effects (IRT vs CTT)
RQ 5.4.4: IRT-CTT Convergence for Schema Congruence-Specific Forgetting

Purpose: Compare fixed effects between IRT and CTT models. Compute Cohen's kappa
         for agreement on significance classifications (threshold: kappa > 0.60).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.analysis_ctt import compute_cohens_kappa_agreement

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOGS_DIR = RQ_DIR / "logs"

# Setup logging
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "step05_compare_fixed_effects.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_fixed_effects(result, model_name):
    """Extract fixed effects from MixedLM result."""

    # Get fixed effects summary
    fe = result.fe_params
    se = np.sqrt(np.diag(result.cov_params()))[:len(fe)]
    z_vals = fe / se
    p_vals = 2 * (1 - pd.Series(z_vals).apply(lambda x: __import__('scipy').stats.norm.cdf(abs(x))))

    # Create DataFrame
    fe_df = pd.DataFrame({
        'term': fe.index,
        'coef': fe.values,
        'se': se,
        'z': z_vals,
        'p_uncorrected': p_vals.values
    })

    # Apply Holm-Bonferroni correction
    n_tests = len(fe_df)
    p_uncorrected = fe_df['p_uncorrected'].values
    sorted_idx = np.argsort(p_uncorrected)
    p_holm = np.zeros(n_tests)

    for rank, idx in enumerate(sorted_idx):
        p_adj = p_uncorrected[idx] * (n_tests - rank)
        p_holm[idx] = min(p_adj, 1.0)

    # Ensure monotonicity (each p_holm >= previous sorted p_holm)
    p_holm_sorted = p_holm[sorted_idx].copy()
    for i in range(1, len(p_holm_sorted)):
        if p_holm_sorted[i] < p_holm_sorted[i-1]:
            p_holm_sorted[i] = p_holm_sorted[i-1]
    p_holm[sorted_idx] = p_holm_sorted

    fe_df['p_holm'] = p_holm
    fe_df['sig'] = fe_df['p_uncorrected'] < 0.05

    return fe_df


def main():
    logger.info("=" * 60)
    logger.info("Step 05: Compare Fixed Effects (IRT vs CTT)")
    logger.info("=" * 60)

    # 1. Load fitted models
    logger.info("Loading fitted models...")

    with open(DATA_DIR / "step03_irt_lmm_model.pkl", 'rb') as f:
        irt_result = pickle.load(f)

    with open(DATA_DIR / "step03_ctt_lmm_model.pkl", 'rb') as f:
        ctt_result = pickle.load(f)

    logger.info("Models loaded successfully")

    # 2. Extract fixed effects
    logger.info("Extracting fixed effects...")

    irt_fe = extract_fixed_effects(irt_result, 'IRT')
    ctt_fe = extract_fixed_effects(ctt_result, 'CTT')

    logger.info(f"IRT fixed effects: {len(irt_fe)} terms")
    logger.info(f"CTT fixed effects: {len(ctt_fe)} terms")

    # 3. Verify same terms
    if set(irt_fe['term']) != set(ctt_fe['term']):
        logger.warning("Term mismatch between IRT and CTT models!")
        logger.warning(f"IRT terms: {list(irt_fe['term'])}")
        logger.warning(f"CTT terms: {list(ctt_fe['term'])}")

    # 4. Compute Cohen's kappa using catalogued tool
    logger.info("Computing Cohen's kappa for agreement...")

    # Align IRT and CTT by term
    irt_fe_aligned = irt_fe.set_index('term')
    ctt_fe_aligned = ctt_fe.set_index('term')

    # Get common terms in same order
    common_terms = [t for t in irt_fe_aligned.index if t in ctt_fe_aligned.index]
    irt_sig = irt_fe_aligned.loc[common_terms, 'sig'].tolist()
    ctt_sig = ctt_fe_aligned.loc[common_terms, 'sig'].tolist()

    kappa_result = compute_cohens_kappa_agreement(
        classifications_1=irt_sig,
        classifications_2=ctt_sig,
        labels=common_terms
    )

    logger.info(f"Cohen's kappa: {kappa_result['kappa']:.3f}")
    logger.info(f"Percent agreement: {kappa_result['agreement_percent']:.1f}%")
    logger.info(f"Interpretation: {kappa_result['interpretation']}")

    # 5. Create side-by-side comparison
    logger.info("Creating coefficient comparison table...")

    # Merge IRT and CTT fixed effects
    comparison = irt_fe.merge(
        ctt_fe,
        on='term',
        suffixes=('_IRT', '_CTT')
    )

    # Add agreement column
    comparison['agreement'] = comparison['sig_IRT'] == comparison['sig_CTT']

    logger.info(f"\nCoefficient comparison:")
    print(comparison[['term', 'coef_IRT', 'coef_CTT', 'p_uncorrected_IRT', 'p_uncorrected_CTT', 'sig_IRT', 'sig_CTT', 'agreement']].to_string())

    # 6. Create agreement metrics summary
    n_effects = kappa_result['n_effects']
    n_agree = int(kappa_result['agreement_percent'] * n_effects / 100)

    agreement_metrics = pd.DataFrame([
        {'metric': 'cohens_kappa', 'value': kappa_result['kappa'], 'threshold': 0.60, 'result': 'PASS' if kappa_result['kappa'] > 0.60 else 'FAIL', 'interpretation': kappa_result['interpretation']},
        {'metric': 'percent_agreement', 'value': kappa_result['agreement_percent'], 'threshold': 80.0, 'result': 'PASS' if kappa_result['agreement_percent'] >= 80.0 else 'FAIL', 'interpretation': f"{n_agree}/{n_effects} terms agree"},
        {'metric': 'n_discordant', 'value': n_effects - n_agree, 'threshold': None, 'result': 'INFO', 'interpretation': 'Terms where IRT and CTT disagree on significance'}
    ])

    # 7. Save outputs
    logger.info("Saving outputs...")

    irt_fe.to_csv(DATA_DIR / "step05_irt_fixed_effects.csv", index=False)
    logger.info(f"Saved: step05_irt_fixed_effects.csv ({len(irt_fe)} rows)")

    ctt_fe.to_csv(DATA_DIR / "step05_ctt_fixed_effects.csv", index=False)
    logger.info(f"Saved: step05_ctt_fixed_effects.csv ({len(ctt_fe)} rows)")

    comparison.to_csv(DATA_DIR / "step05_coefficient_comparison.csv", index=False)
    logger.info(f"Saved: step05_coefficient_comparison.csv ({len(comparison)} rows)")

    agreement_metrics.to_csv(DATA_DIR / "step05_agreement_metrics.csv", index=False)
    logger.info(f"Saved: step05_agreement_metrics.csv ({len(agreement_metrics)} rows)")

    # 8. Print summary
    logger.info("=" * 60)
    logger.info("Agreement Summary:")
    logger.info(f"  Cohen's kappa: {kappa_result['kappa']:.3f} ({kappa_result['interpretation']})")
    logger.info(f"  Percent agreement: {kappa_result['agreement_percent']:.1f}%")
    logger.info(f"  Threshold kappa > 0.60: {'PASS' if kappa_result['kappa'] > 0.60 else 'FAIL'}")
    logger.info(f"  Threshold agreement >= 80%: {'PASS' if kappa_result['agreement_percent'] >= 80.0 else 'FAIL'}")
    logger.info("=" * 60)
    logger.info("Step 05 COMPLETE: Fixed effects compared successfully")
    logger.info("=" * 60)

    return comparison, agreement_metrics


if __name__ == "__main__":
    main()
