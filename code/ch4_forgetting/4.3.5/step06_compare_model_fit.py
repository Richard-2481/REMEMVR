"""
Step 06: Compare Model Fit (AIC/BIC)

Purpose: Compare IRT vs CTT model fit using AIC and BIC. Compute ΔAIC and ΔBIC.
         Interpret per Burnham & Anderson: |Δ| < 2 = equivalent fit,
         2-10 = moderate evidence, > 10 = strong evidence.

Dependencies: Step 03 (fitted LMM models)

Output Files:
    - data/step06_model_fit_comparison.csv
    - data/step06_fit_interpretation.txt
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

# Import the TDD-validated tool
from tools.analysis_ctt import compare_lmm_fit_aic_bic

# Setup paths
RQ_PATH = Path(__file__).parent.parent
DATA_PATH = RQ_PATH / "data"
LOGS_PATH = RQ_PATH / "logs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_PATH / "step06_compare_model_fit.log"),
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

    logger.info(f"IRT model: AIC={irt_model.aic:.2f}, BIC={irt_model.bic:.2f}")
    logger.info(f"CTT model: AIC={ctt_model.aic:.2f}, BIC={ctt_model.bic:.2f}")

    return irt_model, ctt_model

def compare_fit(irt_model, ctt_model):
    """Compare model fit using TDD-validated tool."""
    # Use the TDD-validated tool
    fit_comparison = compare_lmm_fit_aic_bic(
        aic_model1=irt_model.aic,
        bic_model1=irt_model.bic,
        aic_model2=ctt_model.aic,
        bic_model2=ctt_model.bic,
        model1_name="IRT",
        model2_name="CTT"
    )

    logger.info(f"Model fit comparison computed: {len(fit_comparison)} rows")

    return fit_comparison

def write_interpretation(fit_comparison, irt_model, ctt_model):
    """Write detailed interpretation text file."""
    interp_path = DATA_PATH / "step06_fit_interpretation.txt"

    # Extract values
    delta_aic = irt_model.aic - ctt_model.aic
    delta_bic = irt_model.bic - ctt_model.bic

    # Interpretation per Burnham & Anderson
    def interpret_delta(delta):
        abs_delta = abs(delta)
        if abs_delta < 2:
            return "equivalent fit (|Δ| < 2)"
        elif abs_delta < 10:
            return f"moderate evidence for {'IRT' if delta < 0 else 'CTT'} (2 ≤ |Δ| < 10)"
        else:
            return f"strong evidence for {'IRT' if delta < 0 else 'CTT'} (|Δ| ≥ 10)"

    with open(interp_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL FIT INTERPRETATION (IRT vs CTT)\n")
        f.write("=" * 60 + "\n\n")

        f.write("RAW FIT INDICES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"IRT AIC: {irt_model.aic:.2f}\n")
        f.write(f"CTT AIC: {ctt_model.aic:.2f}\n")
        f.write(f"IRT BIC: {irt_model.bic:.2f}\n")
        f.write(f"CTT BIC: {ctt_model.bic:.2f}\n\n")

        f.write("MODEL COMPARISON:\n")
        f.write("-" * 40 + "\n")
        f.write(f"ΔAIC (IRT - CTT): {delta_aic:.2f}\n")
        f.write(f"  Interpretation: {interpret_delta(delta_aic)}\n\n")
        f.write(f"ΔBIC (IRT - CTT): {delta_bic:.2f}\n")
        f.write(f"  Interpretation: {interpret_delta(delta_bic)}\n\n")

        f.write("OVERALL CONCLUSION:\n")
        f.write("-" * 40 + "\n")

        # Overall interpretation
        if abs(delta_aic) < 2 and abs(delta_bic) < 2:
            conclusion = "IRT and CTT models show EQUIVALENT fit to the data."
            implication = "Both measurement approaches are equally valid for paradigm-specific forgetting analysis."
        elif delta_aic < 0 and delta_bic < 0:
            conclusion = f"IRT model shows BETTER fit (ΔAIC={delta_aic:.2f}, ΔBIC={delta_bic:.2f})."
            implication = "IRT theta scores capture forgetting trajectories more precisely than CTT proportion correct."
        else:
            conclusion = f"CTT model shows BETTER fit (ΔAIC={delta_aic:.2f}, ΔBIC={delta_bic:.2f})."
            implication = "CTT proportion correct may capture forgetting patterns differently than IRT."

        f.write(f"{conclusion}\n\n")
        f.write(f"Convergence Implication:\n{implication}\n\n")

        f.write("NOTE: AIC/BIC difference is expected since IRT uses theta (-∞,+∞)\n")
        f.write("      and CTT uses proportion correct [0,1]. The metrics are NOT\n")
        f.write("      directly comparable across scales. Focus on DIRECTION and\n")
        f.write("      AGREEMENT of conclusions rather than absolute values.\n")

    logger.info(f"Interpretation written to {interp_path}")
    return interp_path

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("STEP 06: COMPARE MODEL FIT (AIC/BIC)")
    logger.info("=" * 60)

    try:
        # 1. Load models
        logger.info("\n1. Loading models...")
        irt_model, ctt_model = load_models()

        # 2. Compare fit
        logger.info("\n2. Comparing model fit...")
        fit_comparison = compare_fit(irt_model, ctt_model)

        # 3. Save comparison
        logger.info("\n3. Saving comparison...")
        fit_comparison.to_csv(DATA_PATH / "step06_model_fit_comparison.csv", index=False)
        logger.info(f"   Saved: step06_model_fit_comparison.csv ({len(fit_comparison)} rows)")

        # 4. Write interpretation
        logger.info("\n4. Writing interpretation...")
        write_interpretation(fit_comparison, irt_model, ctt_model)

        # Report
        delta_aic = irt_model.aic - ctt_model.aic
        delta_bic = irt_model.bic - ctt_model.bic

        logger.info("\n" + "=" * 60)
        logger.info("STEP 06 COMPLETE: Model fit compared")
        logger.info(f"Model fit comparison: ΔAIC = {delta_aic:.2f}, ΔBIC = {delta_bic:.2f}")

        if abs(delta_aic) < 2 and abs(delta_bic) < 2:
            logger.info("Interpretation: equivalent fit")
        elif delta_aic < 0:
            logger.info("Interpretation: IRT model shows better fit")
        else:
            logger.info("Interpretation: CTT model shows better fit")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"STEP 06 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
