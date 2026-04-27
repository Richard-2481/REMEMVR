"""
Step 03: Fit Parallel LMMs (IRT vs CTT)

Purpose: Fit parallel Linear Mixed Models using identical model formula for
         IRT theta vs CTT mean scores. Model formula determined by best-fitting
         model from RQ 5.3.1. If either model fails to converge, simplify both
         equally to maintain structural equivalence.

Dependencies: Step 02 (merged IRT-CTT data), Step 00 (TSVR mapping)

Output Files:
    - data/step03_irt_lmm_input.csv (1200 rows)
    - data/step03_ctt_lmm_input.csv (1200 rows)
    - data/step03_irt_lmm_model.pkl
    - data/step03_ctt_lmm_model.pkl
    - data/step03_irt_lmm_summary.txt
    - data/step03_ctt_lmm_summary.txt
    - data/step03_model_convergence_log.txt
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
import statsmodels.formula.api as smf
import warnings

# Setup paths
RQ_PATH = Path(__file__).parent.parent
DATA_PATH = RQ_PATH / "data"
LOGS_PATH = RQ_PATH / "logs"

# Source RQ path for model comparison
SOURCE_RQ_PATH = RQ_PATH.parent / "5.3.1"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_PATH / "step03_fit_parallel_lmms.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_input_files():
    """Load required input files."""
    # Load merged IRT-CTT data
    merged_df = pd.read_csv(DATA_PATH / "step02_merged_irt_ctt.csv")
    logger.info(f"Loaded merged IRT-CTT: {len(merged_df)} rows")

    # Load TSVR mapping
    tsvr_df = pd.read_csv(DATA_PATH / "step00_tsvr_mapping.csv")
    logger.info(f"Loaded TSVR mapping: {len(tsvr_df)} rows")

    return merged_df, tsvr_df

def get_best_model_from_5_3_1():
    """
    Get best model formula from RQ 5.3.1 model comparison.
    Falls back to Log model if file not found.
    """
    model_comparison_path = SOURCE_RQ_PATH / "data" / "step05_lmm_model_comparison.csv"

    if model_comparison_path.exists():
        try:
            model_comparison = pd.read_csv(model_comparison_path)
            # Find model with lowest AIC
            best_idx = model_comparison['AIC'].idxmin()
            best_model = model_comparison.loc[best_idx, 'model']
            best_aic = model_comparison.loc[best_idx, 'AIC']
            logger.info(f"Best model from RQ 5.3.1: {best_model} (AIC={best_aic:.2f})")
            return best_model
        except Exception as e:
            logger.warning(f"Could not read model comparison: {e}")

    # Default to Log model
    logger.info("Using default Log model (RQ 5.3.1 model comparison not found)")
    return "Log"

def prepare_long_format_data(merged_df, tsvr_df):
    """
    Reshape merged data from wide to long format for LMM.
    Create separate IRT and CTT input dataframes.
    """
    # Merge with TSVR for time variable
    merged_with_time = merged_df.merge(
        tsvr_df[['composite_ID', 'TSVR_hours', 'Days']],
        on='composite_ID',
        how='left'
    )

    # Check for missing TSVR values
    missing = merged_with_time['TSVR_hours'].isna().sum()
    if missing > 0:
        logger.warning(f"Missing TSVR_hours values: {missing}")
        merged_with_time = merged_with_time.dropna(subset=['TSVR_hours'])

    # Reshape to long format for IRT
    irt_records = []
    ctt_records = []

    for paradigm in ['IFR', 'ICR', 'IRE']:
        theta_col = f'theta_{paradigm}'
        ctt_col = f'CTT_{paradigm}'

        for _, row in merged_with_time.iterrows():
            irt_records.append({
                'composite_ID': row['composite_ID'],
                'UID': row['UID'],
                'TEST': row['TEST'],
                'TSVR_hours': row['TSVR_hours'],
                'Days': row['Days'],
                'paradigm': paradigm,
                'theta': row[theta_col]
            })

            ctt_records.append({
                'composite_ID': row['composite_ID'],
                'UID': row['UID'],
                'TEST': row['TEST'],
                'TSVR_hours': row['TSVR_hours'],
                'Days': row['Days'],
                'paradigm': paradigm,
                'CTT_mean': row[ctt_col]
            })

    irt_long = pd.DataFrame(irt_records)
    ctt_long = pd.DataFrame(ctt_records)

    logger.info(f"IRT long format: {len(irt_long)} rows")
    logger.info(f"CTT long format: {len(ctt_long)} rows")

    return irt_long, ctt_long

def add_time_transformations(df, best_model):
    """Add time transformation columns based on best model."""
    df = df.copy()

    # Add log transformation (avoid log(0))
    df['log_TSVR'] = np.log(df['TSVR_hours'] + 1)

    # Add quadratic term
    df['TSVR_sq'] = df['TSVR_hours'] ** 2

    logger.info(f"Time transformations added: log_TSVR, TSVR_sq (best model: {best_model})")

    return df

def fit_lmm_model(data, dv_col, model_type="Log"):
    """
    Fit Linear Mixed Model.

    Args:
        data: Long-format DataFrame
        dv_col: Dependent variable column name ('theta' or 'CTT_mean')
        model_type: Model type from RQ 5.3.1 (Log, Linear, etc.)

    Returns:
        Tuple of (model_result, formula_used, convergence_status, simplification_applied)
    """
    # Build formula based on model type
    if model_type == "Log":
        formula = f"{dv_col} ~ C(paradigm) * log_TSVR"
        re_formula = "~log_TSVR"
    elif model_type == "Linear":
        formula = f"{dv_col} ~ C(paradigm) * TSVR_hours"
        re_formula = "~TSVR_hours"
    elif model_type == "Lin+Log":
        formula = f"{dv_col} ~ C(paradigm) * (TSVR_hours + log_TSVR)"
        re_formula = "~TSVR_hours"  # Simplify RE for convergence
    elif model_type == "Quadratic":
        formula = f"{dv_col} ~ C(paradigm) * (TSVR_hours + TSVR_sq)"
        re_formula = "~TSVR_hours"
    else:
        # Default to Log
        formula = f"{dv_col} ~ C(paradigm) * log_TSVR"
        re_formula = "~log_TSVR"

    simplification_applied = None
    convergence_log = []

    # Try full random slopes model first
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                formula,
                data,
                groups=data['UID'],
                re_formula=re_formula
            )
            result = model.fit(reml=False)

            # Check for convergence warnings
            convergence_warnings = [str(warning.message) for warning in w
                                   if 'converg' in str(warning.message).lower()]

            if result.converged and not convergence_warnings:
                logger.info(f"Model converged with random slopes: {re_formula}")
                return result, formula, True, None

            convergence_log.append(f"Attempt 1 (random slopes {re_formula}): warnings={convergence_warnings}")

    except Exception as e:
        convergence_log.append(f"Attempt 1 (random slopes): ERROR - {str(e)}")
        logger.warning(f"Random slopes model failed: {e}")

    # Try random intercepts only
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                formula,
                data,
                groups=data['UID'],
                re_formula="~1"
            )
            result = model.fit(reml=False)

            convergence_warnings = [str(warning.message) for warning in w
                                   if 'converg' in str(warning.message).lower()]

            if result.converged and not convergence_warnings:
                logger.info("Model converged with random intercepts only")
                simplification_applied = "random_intercepts_only"
                return result, formula, True, simplification_applied

            convergence_log.append(f"Attempt 2 (random intercepts): warnings={convergence_warnings}")

    except Exception as e:
        convergence_log.append(f"Attempt 2 (random intercepts): ERROR - {str(e)}")
        logger.warning(f"Random intercepts model failed: {e}")

    # Log all attempts
    for log_entry in convergence_log:
        logger.info(log_entry)

    # Return last attempted model even if not fully converged
    return result, formula, result.converged if 'result' in dir() else False, simplification_applied

def save_model_summary(result, output_path, model_name, formula, converged, simplification):
    """Save model summary to text file."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{model_name} MODEL SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Formula: {formula}\n")
        f.write(f"Convergence: {converged}\n")
        if simplification:
            f.write(f"Simplification Applied: {simplification}\n")
        f.write("\n")

        f.write("MODEL SUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(str(result.summary()) + "\n\n")

        f.write("FIT INDICES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"AIC: {result.aic:.2f}\n")
        f.write(f"BIC: {result.bic:.2f}\n")
        f.write(f"Log-Likelihood: {result.llf:.2f}\n\n")

        f.write("SAMPLE SIZE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"N observations: {result.nobs}\n")
        # Get n_groups from the model object instead
        n_groups = len(result.model.group_labels) if hasattr(result.model, 'group_labels') else 'N/A'
        f.write(f"N groups: {n_groups}\n")

    logger.info(f"Model summary saved to {output_path}")

def save_convergence_log(irt_converged, ctt_converged, irt_simp, ctt_simp, formula):
    """Save convergence log."""
    log_path = DATA_PATH / "step03_model_convergence_log.txt"

    with open(log_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL CONVERGENCE LOG\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Model formula attempted: {formula}\n\n")

        f.write("CONVERGENCE STATUS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"IRT model convergence: {'PASS' if irt_converged else 'FAIL'}\n")
        f.write(f"CTT model convergence: {'PASS' if ctt_converged else 'FAIL'}\n\n")

        f.write("SIMPLIFICATIONS APPLIED:\n")
        f.write("-" * 40 + "\n")
        f.write(f"IRT simplification: {irt_simp or 'None'}\n")
        f.write(f"CTT simplification: {ctt_simp or 'None'}\n\n")

        # Check structural equivalence
        if irt_simp == ctt_simp:
            f.write("Structural equivalence: MAINTAINED\n")
        else:
            f.write("Structural equivalence: WARNING - simplifications differ\n")

        f.write(f"Final random structure: {irt_simp or 'random_slopes'}\n")

    logger.info(f"Convergence log saved to {log_path}")

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("STEP 03: FIT PARALLEL LMMs (IRT vs CTT)")
    logger.info("=" * 60)

    try:
        # 1. Load input files
        logger.info("\n1. Loading input files...")
        merged_df, tsvr_df = load_input_files()

        # 2. Get best model from RQ 5.3.1
        logger.info("\n2. Getting best model from RQ 5.3.1...")
        best_model = get_best_model_from_5_3_1()

        # 3. Prepare long format data
        logger.info("\n3. Preparing long format data...")
        irt_long, ctt_long = prepare_long_format_data(merged_df, tsvr_df)

        # 4. Add time transformations
        logger.info("\n4. Adding time transformations...")
        irt_long = add_time_transformations(irt_long, best_model)
        ctt_long = add_time_transformations(ctt_long, best_model)

        # Save LMM input data
        irt_long.to_csv(DATA_PATH / "step03_irt_lmm_input.csv", index=False)
        ctt_long.to_csv(DATA_PATH / "step03_ctt_lmm_input.csv", index=False)
        logger.info(f"   Saved: step03_irt_lmm_input.csv ({len(irt_long)} rows)")
        logger.info(f"   Saved: step03_ctt_lmm_input.csv ({len(ctt_long)} rows)")

        # 5. Fit IRT model
        logger.info("\n5. Fitting IRT model...")
        irt_result, irt_formula, irt_converged, irt_simp = fit_lmm_model(irt_long, 'theta', best_model)

        # 6. Fit CTT model
        logger.info("\n6. Fitting CTT model...")
        ctt_result, ctt_formula, ctt_converged, ctt_simp = fit_lmm_model(ctt_long, 'CTT_mean', best_model)

        # 7. Verify structural equivalence
        logger.info("\n7. Verifying structural equivalence...")
        if irt_simp != ctt_simp:
            logger.warning("Structural equivalence WARNING: different simplifications applied")
            # Try to equalize by using more conservative simplification
            if irt_simp == "random_intercepts_only" or ctt_simp == "random_intercepts_only":
                logger.info("Re-fitting both with random intercepts only for equivalence...")
                # Re-fit both with random intercepts
                if irt_simp != "random_intercepts_only":
                    irt_result, irt_formula, irt_converged, irt_simp = fit_lmm_model(irt_long, 'theta', "Random_Intercepts")
                if ctt_simp != "random_intercepts_only":
                    ctt_result, ctt_formula, ctt_converged, ctt_simp = fit_lmm_model(ctt_long, 'CTT_mean', "Random_Intercepts")

        # 8. Save models
        logger.info("\n8. Saving models...")
        with open(DATA_PATH / "step03_irt_lmm_model.pkl", 'wb') as f:
            pickle.dump(irt_result, f)
        with open(DATA_PATH / "step03_ctt_lmm_model.pkl", 'wb') as f:
            pickle.dump(ctt_result, f)
        logger.info("   Saved: step03_irt_lmm_model.pkl")
        logger.info("   Saved: step03_ctt_lmm_model.pkl")

        # 9. Save summaries
        logger.info("\n9. Saving model summaries...")
        save_model_summary(irt_result, DATA_PATH / "step03_irt_lmm_summary.txt",
                          "IRT", irt_formula, irt_converged, irt_simp)
        save_model_summary(ctt_result, DATA_PATH / "step03_ctt_lmm_summary.txt",
                          "CTT", ctt_formula, ctt_converged, ctt_simp)

        # 10. Save convergence log
        logger.info("\n10. Saving convergence log...")
        save_convergence_log(irt_converged, ctt_converged, irt_simp, ctt_simp, irt_formula)

        # Report
        logger.info("\n" + "=" * 60)
        logger.info("STEP 03 COMPLETE: Parallel LMMs fitted")
        logger.info(f"IRT model convergence: {irt_converged}")
        logger.info(f"CTT model convergence: {ctt_converged}")
        logger.info(f"Structural equivalence: {'MAINTAINED' if irt_simp == ctt_simp else 'SIMPLIFIED'}")
        logger.info(f"Final random structure: {irt_simp or 'random_slopes'}")
        logger.info(f"IRT AIC: {irt_result.aic:.2f}")
        logger.info(f"CTT AIC: {ctt_result.aic:.2f}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"STEP 03 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
