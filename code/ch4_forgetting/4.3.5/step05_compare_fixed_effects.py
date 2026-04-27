"""
Step 05: Compare Fixed Effects (IRT vs CTT)

Purpose: Compare fixed effects between IRT and CTT models. Extract coefficients
         with SE, z-values, dual p-values. Compute Cohen's kappa for agreement
         on significance classifications.

Dependencies: Step 03 (fitted LMM models)

Output Files:
    - data/step05_irt_fixed_effects.csv
    - data/step05_ctt_fixed_effects.csv
    - data/step05_coefficient_comparison.csv
    - data/step05_agreement_metrics.csv
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from scipy import stats
from sklearn.metrics import cohen_kappa_score

# Setup paths
RQ_PATH = Path(__file__).parent.parent
DATA_PATH = RQ_PATH / "data"
LOGS_PATH = RQ_PATH / "logs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_PATH / "step05_compare_fixed_effects.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_models():
    """Load fitted LMM models."""
    with open(DATA_PATH / "step03_irt_lmm_model.pkl", 'rb') as f:
        irt_model = pickle.load(f)
    with open(DATA_PATH / "step03_ctt_lmm_model.pkl", 'rb') as f:
        ctt_model = pickle.load(f)

    logger.info("Loaded IRT and CTT models")
    return irt_model, ctt_model

def extract_fixed_effects(model, model_name):
    """Extract fixed effects from model."""
    # Get summary frame
    summary = model.summary().tables[1]

    # Parse the summary table
    results = []

    # Fixed effects are in the summary
    fe = model.fe_params
    se = model.bse_fe
    z_vals = model.tvalues
    p_vals = model.pvalues

    for term in fe.index:
        results.append({
            'term': term,
            'coef': fe[term],
            'se': se[term] if term in se.index else np.nan,
            'z': z_vals[term] if term in z_vals.index else np.nan,
            'p_uncorrected': p_vals[term] if term in p_vals.index else np.nan
        })

    df = pd.DataFrame(results)

    # Apply Bonferroni correction
    n_tests = len(df)
    df['p_bonferroni'] = np.minimum(df['p_uncorrected'] * n_tests, 1.0)

    # Significance classification
    df['sig'] = df['p_uncorrected'] < 0.05

    logger.info(f"{model_name} fixed effects: {len(df)} terms")
    return df

def compute_agreement_metrics(irt_fe, ctt_fe):
    """Compute agreement metrics between IRT and CTT fixed effects."""
    # Merge on term
    comparison = irt_fe.merge(
        ctt_fe,
        on='term',
        suffixes=('_IRT', '_CTT')
    )

    # Rename columns
    comparison = comparison.rename(columns={
        'coef_IRT': 'IRT_coef', 'se_IRT': 'IRT_se', 'z_IRT': 'IRT_z',
        'p_uncorrected_IRT': 'IRT_p_uncorrected', 'p_bonferroni_IRT': 'IRT_p_bonferroni',
        'sig_IRT': 'IRT_sig',
        'coef_CTT': 'CTT_coef', 'se_CTT': 'CTT_se', 'z_CTT': 'CTT_z',
        'p_uncorrected_CTT': 'CTT_p_uncorrected', 'p_bonferroni_CTT': 'CTT_p_bonferroni',
        'sig_CTT': 'CTT_sig'
    })

    # Agreement on significance
    comparison['agreement'] = comparison['IRT_sig'] == comparison['CTT_sig']

    # Cohen's kappa
    kappa = cohen_kappa_score(comparison['IRT_sig'], comparison['CTT_sig'])

    # Percentage agreement
    pct_agreement = comparison['agreement'].mean()

    # Kappa interpretation
    if kappa > 0.80:
        kappa_interp = "Almost Perfect"
    elif kappa > 0.60:
        kappa_interp = "Substantial"
    elif kappa > 0.40:
        kappa_interp = "Moderate"
    else:
        kappa_interp = "Poor"

    agreement_metrics = pd.DataFrame([
        {'metric': 'Cohens_kappa', 'value': kappa, 'threshold': 0.60, 'result': 'PASS' if kappa > 0.60 else 'FAIL', 'interpretation': kappa_interp},
        {'metric': 'Percentage_agreement', 'value': pct_agreement, 'threshold': 0.80, 'result': 'PASS' if pct_agreement >= 0.80 else 'FAIL', 'interpretation': f"{pct_agreement*100:.1f}% terms agree"},
        {'metric': 'Kappa_interpretation', 'value': kappa, 'threshold': 0.60, 'result': 'PASS' if kappa > 0.60 else 'FAIL', 'interpretation': kappa_interp}
    ])

    logger.info(f"Cohen's kappa: {kappa:.3f} ({kappa_interp})")
    logger.info(f"Percentage agreement: {pct_agreement*100:.1f}%")

    return comparison, agreement_metrics

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("STEP 05: COMPARE FIXED EFFECTS (IRT vs CTT)")
    logger.info("=" * 60)

    try:
        # 1. Load models
        logger.info("\n1. Loading models...")
        irt_model, ctt_model = load_models()

        # 2. Extract fixed effects
        logger.info("\n2. Extracting fixed effects...")
        irt_fe = extract_fixed_effects(irt_model, "IRT")
        ctt_fe = extract_fixed_effects(ctt_model, "CTT")

        # 3. Save individual fixed effects
        logger.info("\n3. Saving fixed effects...")
        irt_fe.to_csv(DATA_PATH / "step05_irt_fixed_effects.csv", index=False)
        ctt_fe.to_csv(DATA_PATH / "step05_ctt_fixed_effects.csv", index=False)
        logger.info(f"   Saved: step05_irt_fixed_effects.csv ({len(irt_fe)} terms)")
        logger.info(f"   Saved: step05_ctt_fixed_effects.csv ({len(ctt_fe)} terms)")

        # 4. Compute agreement metrics
        logger.info("\n4. Computing agreement metrics...")
        comparison, agreement_metrics = compute_agreement_metrics(irt_fe, ctt_fe)

        # 5. Save comparison and metrics
        logger.info("\n5. Saving comparison and metrics...")
        comparison.to_csv(DATA_PATH / "step05_coefficient_comparison.csv", index=False)
        agreement_metrics.to_csv(DATA_PATH / "step05_agreement_metrics.csv", index=False)
        logger.info(f"   Saved: step05_coefficient_comparison.csv ({len(comparison)} terms)")
        logger.info(f"   Saved: step05_agreement_metrics.csv ({len(agreement_metrics)} metrics)")

        # Report
        logger.info("\n" + "=" * 60)
        logger.info("STEP 05 COMPLETE: Fixed effects compared")
        logger.info(f"Fixed effects extracted: {len(irt_fe)} terms for IRT model")
        logger.info(f"Fixed effects extracted: {len(ctt_fe)} terms for CTT model")
        kappa = agreement_metrics.loc[agreement_metrics['metric'] == 'Cohens_kappa', 'value'].values[0]
        pct = agreement_metrics.loc[agreement_metrics['metric'] == 'Percentage_agreement', 'value'].values[0]
        logger.info(f"Agreement metrics: Cohen's kappa = {kappa:.3f}, Percentage agreement = {pct*100:.1f}%")
        logger.info("Dual p-values reported per Decision D068")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"STEP 05 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
