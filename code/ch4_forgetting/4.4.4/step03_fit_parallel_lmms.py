"""
Step 03: Fit Parallel LMMs (IRT vs CTT)
RQ 5.4.4: IRT-CTT Convergence for Schema Congruence-Specific Forgetting

Purpose: Fit parallel Linear Mixed Models using identical formula for IRT theta
         vs CTT mean scores. Both models use Recip+Log two-process forgetting model
         (congruence * (recip_TSVR + log_TSVR)) with random slopes on recip_TSVR
         per RQ 5.4.1 ROOT model specification.

UPDATED 2025-12-09: Changed from Log-only to Recip+Log two-process forgetting
                    per RQ 5.4.1 ROOT cascade. Random slopes on recip_TSVR (rapid component).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import pickle
import sys
import warnings

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import statsmodels.formula.api as smf

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
        logging.FileHandler(LOGS_DIR / "step03_fit_parallel_lmms.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_lmm_input(merged_df, tsvr_df, dv_col, output_name):
    """Prepare long-format LMM input data."""

    # Merge with TSVR
    merged_with_tsvr = merged_df.merge(tsvr_df[['composite_ID', 'TSVR_hours']], on='composite_ID')

    # Melt to long format for the DV
    if dv_col == 'theta':
        id_vars = ['composite_ID', 'UID', 'TEST', 'TSVR_hours']
        value_vars = ['theta_common', 'theta_congruent', 'theta_incongruent']
        var_name = 'congruence_var'
        value_name = 'theta'
    else:
        id_vars = ['composite_ID', 'UID', 'TEST', 'TSVR_hours']
        value_vars = ['CTT_common', 'CTT_congruent', 'CTT_incongruent']
        var_name = 'congruence_var'
        value_name = 'CTT_mean'

    long_df = merged_with_tsvr[id_vars + value_vars].melt(
        id_vars=id_vars,
        var_name=var_name,
        value_name=value_name
    )

    # Extract congruence level
    if dv_col == 'theta':
        long_df['congruence'] = long_df['congruence_var'].str.replace('theta_', '')
    else:
        long_df['congruence'] = long_df['congruence_var'].str.replace('CTT_', '')

    # Add time transformations for two-process forgetting model
    long_df['recip_TSVR'] = 1.0 / (long_df['TSVR_hours'] + 1)  # Rapid component: 1/(t+1)
    long_df['log_TSVR'] = np.log1p(long_df['TSVR_hours'])      # Slow component: log(t+1)

    # Drop temp column
    long_df = long_df.drop(columns=['congruence_var'])

    return long_df


def fit_lmm_with_fallback(data, formula, groups, re_formula, model_name):
    """Fit LMM with fallback for convergence issues."""

    convergence_log = []

    # Try 1: Full model with random slopes
    try:
        logger.info(f"Fitting {model_name} with random slopes: {formula}, re_formula={re_formula}")
        model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
        result = model.fit(reml=False, method='powell')

        if result.converged:
            convergence_log.append(f"{model_name}: Converged with random slopes (method=powell)")
            logger.info(f"{model_name}: Converged with random slopes")
            return result, convergence_log, 'full'
        else:
            convergence_log.append(f"{model_name}: Powell did not converge, trying LBFGS")

    except Exception as e:
        convergence_log.append(f"{model_name}: Random slopes failed with {str(e)[:100]}")
        logger.warning(f"{model_name}: Random slopes failed: {e}")

    # Try 2: LBFGS optimizer
    try:
        logger.info(f"Trying {model_name} with LBFGS optimizer...")
        model = smf.mixedlm(formula, data, groups=groups, re_formula=re_formula)
        result = model.fit(reml=False, method='lbfgs')

        if result.converged:
            convergence_log.append(f"{model_name}: Converged with random slopes (method=lbfgs)")
            logger.info(f"{model_name}: Converged with LBFGS")
            return result, convergence_log, 'full'

    except Exception as e:
        convergence_log.append(f"{model_name}: LBFGS failed with {str(e)[:100]}")
        logger.warning(f"{model_name}: LBFGS failed: {e}")

    # Try 3: Random intercepts only
    try:
        logger.info(f"Simplifying {model_name} to random intercepts only...")
        model = smf.mixedlm(formula, data, groups=groups)
        result = model.fit(reml=False)

        if result.converged:
            convergence_log.append(f"{model_name}: Converged with intercepts only (simplified)")
            logger.info(f"{model_name}: Converged with intercepts only")
            return result, convergence_log, 'simplified'

    except Exception as e:
        convergence_log.append(f"{model_name}: Intercepts only failed: {str(e)[:100]}")
        logger.error(f"{model_name}: All approaches failed")
        raise ValueError(f"{model_name} failed to converge with any method")

    return None, convergence_log, None


def main():
    logger.info("=" * 60)
    logger.info("Step 03: Fit Parallel LMMs (IRT vs CTT)")
    logger.info("=" * 60)

    # 1. Load merged data
    logger.info("Loading merged IRT-CTT data...")
    merged_df = pd.read_csv(DATA_DIR / "step02_merged_irt_ctt.csv")
    logger.info(f"Loaded merged data: {len(merged_df)} rows")

    # 2. Load TSVR mapping
    logger.info("Loading TSVR mapping...")
    tsvr_df = pd.read_csv(DATA_DIR / "step00_tsvr_mapping.csv")
    logger.info(f"Loaded TSVR: {len(tsvr_df)} rows")

    # 3. Prepare IRT LMM input
    logger.info("Preparing IRT LMM input (long format)...")
    irt_input = prepare_lmm_input(merged_df, tsvr_df, 'theta', 'IRT')
    logger.info(f"IRT input: {len(irt_input)} rows, columns: {list(irt_input.columns)}")

    # 4. Prepare CTT LMM input
    logger.info("Preparing CTT LMM input (long format)...")
    ctt_input = prepare_lmm_input(merged_df, tsvr_df, 'CTT', 'CTT')
    logger.info(f"CTT input: {len(ctt_input)} rows, columns: {list(ctt_input.columns)}")

    # 5. Define model formula (Recip+Log two-process model per RQ 5.4.1 ROOT)
    # Two-process forgetting: rapid 1/(t+1) + slow log(t+1)
    formula = "theta ~ C(congruence) * (recip_TSVR + log_TSVR)"
    re_formula = "~recip_TSVR"  # Random slopes on rapid component

    convergence_logs = []

    # 6. Fit IRT model
    logger.info("Fitting IRT LMM...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        irt_result, irt_conv_log, irt_structure = fit_lmm_with_fallback(
            irt_input, formula, 'UID', re_formula, 'IRT'
        )
    convergence_logs.extend(irt_conv_log)

    # 7. Fit CTT model (with same Recip+Log formula structure)
    logger.info("Fitting CTT LMM...")
    ctt_formula = "CTT_mean ~ C(congruence) * (recip_TSVR + log_TSVR)"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctt_result, ctt_conv_log, ctt_structure = fit_lmm_with_fallback(
            ctt_input, ctt_formula, 'UID', re_formula, 'CTT'
        )
    convergence_logs.extend(ctt_conv_log)

    # 8. Ensure structural equivalence
    logger.info("Checking structural equivalence...")
    if irt_structure != ctt_structure:
        logger.warning(f"Structural mismatch: IRT={irt_structure}, CTT={ctt_structure}")
        logger.warning("Re-fitting with simpler structure for equivalence...")

        # Use the simpler structure for both
        simpler = 'simplified' if 'simplified' in [irt_structure, ctt_structure] else 'full'

        if simpler == 'simplified':
            # Re-fit both with intercepts only
            logger.info("Re-fitting both models with intercepts only for equivalence...")
            irt_result, _, _ = fit_lmm_with_fallback(irt_input, formula, 'UID', None, 'IRT-equiv')
            ctt_result, _, _ = fit_lmm_with_fallback(ctt_input, ctt_formula, 'UID', None, 'CTT-equiv')
            convergence_logs.append("STRUCTURAL EQUIVALENCE: Both models simplified to random intercepts")

    logger.info(f"IRT structure: {irt_structure}, CTT structure: {ctt_structure}")

    # 9. Save model summaries
    logger.info("Generating model summaries...")

    irt_summary = str(irt_result.summary())
    ctt_summary = str(ctt_result.summary())

    # Get n_groups from model
    irt_ngroups = len(irt_input['UID'].unique())
    ctt_ngroups = len(ctt_input['UID'].unique())

    with open(DATA_DIR / "step03_irt_lmm_summary.txt", 'w') as f:
        f.write("IRT LMM Summary for RQ 5.4.4\n")
        f.write("=" * 60 + "\n")
        f.write(f"Formula: {formula}\n")
        f.write(f"Groups: UID\n")
        f.write(f"N observations: {irt_result.nobs}\n")
        f.write(f"N groups: {irt_ngroups}\n")
        f.write(f"Converged: {irt_result.converged}\n")
        f.write(f"AIC: {irt_result.aic:.2f}\n")
        f.write(f"BIC: {irt_result.bic:.2f}\n")
        f.write("=" * 60 + "\n\n")
        f.write(irt_summary)

    with open(DATA_DIR / "step03_ctt_lmm_summary.txt", 'w') as f:
        f.write("CTT LMM Summary for RQ 5.4.4\n")
        f.write("=" * 60 + "\n")
        f.write(f"Formula: {ctt_formula}\n")
        f.write(f"Groups: UID\n")
        f.write(f"N observations: {ctt_result.nobs}\n")
        f.write(f"N groups: {ctt_ngroups}\n")
        f.write(f"Converged: {ctt_result.converged}\n")
        f.write(f"AIC: {ctt_result.aic:.2f}\n")
        f.write(f"BIC: {ctt_result.bic:.2f}\n")
        f.write("=" * 60 + "\n\n")
        f.write(ctt_summary)

    # 10. Save convergence log
    with open(DATA_DIR / "step03_model_convergence_log.txt", 'w') as f:
        f.write("Model Convergence Log for RQ 5.4.4\n")
        f.write("=" * 60 + "\n")
        for entry in convergence_logs:
            f.write(f"{entry}\n")

    # 11. Save model objects
    with open(DATA_DIR / "step03_irt_lmm_model.pkl", 'wb') as f:
        pickle.dump(irt_result, f)
    logger.info("Saved: step03_irt_lmm_model.pkl")

    with open(DATA_DIR / "step03_ctt_lmm_model.pkl", 'wb') as f:
        pickle.dump(ctt_result, f)
    logger.info("Saved: step03_ctt_lmm_model.pkl")

    # 12. Save LMM input data
    irt_input.to_csv(DATA_DIR / "step03_irt_lmm_input.csv", index=False)
    logger.info(f"Saved: step03_irt_lmm_input.csv ({len(irt_input)} rows)")

    ctt_input.to_csv(DATA_DIR / "step03_ctt_lmm_input.csv", index=False)
    logger.info(f"Saved: step03_ctt_lmm_input.csv ({len(ctt_input)} rows)")

    # 13. Print key results
    logger.info("=" * 60)
    logger.info("Model Fit Summary:")
    logger.info(f"  IRT: AIC={irt_result.aic:.2f}, BIC={irt_result.bic:.2f}, converged={irt_result.converged}")
    logger.info(f"  CTT: AIC={ctt_result.aic:.2f}, BIC={ctt_result.bic:.2f}, converged={ctt_result.converged}")
    logger.info("=" * 60)
    logger.info("Step 03 COMPLETE: Both LMMs fitted successfully")
    logger.info("=" * 60)

    return irt_result, ctt_result


if __name__ == "__main__":
    main()
