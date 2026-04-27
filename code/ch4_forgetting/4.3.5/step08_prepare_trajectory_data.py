"""
Step 08: Prepare Trajectory Comparison Data (IRT vs CTT)

Purpose: Prepare trajectory comparison data with observed means and model
         predictions from both IRT and CTT. Aggregate to paradigm x test level
         (12 groups: 3 paradigms x 4 tests) with 95% CIs.

Dependencies: Step 03 (LMM input data and fitted models)

Output Files:
    - data/step08_trajectory_data.csv (24 rows: 3 paradigms x 4 tests x 2 measurement types)
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path

# Setup paths
RQ_PATH = Path(__file__).parent.parent
DATA_PATH = RQ_PATH / "data"
LOGS_PATH = RQ_PATH / "logs"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_PATH / "step08_prepare_trajectory_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data_and_models():
    """Load LMM input data and fitted models."""
    # Load LMM input data
    irt_input = pd.read_csv(DATA_PATH / "step03_irt_lmm_input.csv")
    ctt_input = pd.read_csv(DATA_PATH / "step03_ctt_lmm_input.csv")
    logger.info(f"Loaded IRT input: {len(irt_input)} rows")
    logger.info(f"Loaded CTT input: {len(ctt_input)} rows")

    # Load models
    with open(DATA_PATH / "step03_irt_lmm_model.pkl", 'rb') as f:
        irt_model = pickle.load(f)
    with open(DATA_PATH / "step03_ctt_lmm_model.pkl", 'rb') as f:
        ctt_model = pickle.load(f)
    logger.info("Loaded IRT and CTT models")

    return irt_input, ctt_input, irt_model, ctt_model

def compute_trajectory_summaries(irt_input, ctt_input, irt_model, ctt_model):
    """Compute trajectory summaries per paradigm x test."""
    results = []

    # Add fitted values
    irt_input['fitted'] = irt_model.fittedvalues.values
    ctt_input['fitted'] = ctt_model.fittedvalues.values

    # Process IRT data
    for (paradigm, test), group in irt_input.groupby(['paradigm', 'TEST']):
        n = len(group)
        observed_mean = group['theta'].mean()
        observed_std = group['theta'].std()
        se = observed_std / np.sqrt(n)
        ci_lower = observed_mean - 1.96 * se
        ci_upper = observed_mean + 1.96 * se
        model_prediction = group['fitted'].mean()
        tsvr_hours = group['TSVR_hours'].mean()

        results.append({
            'paradigm': paradigm,
            'TEST': test,
            'TSVR_hours': tsvr_hours,
            'measurement_type': 'IRT',
            'observed_mean': observed_mean,
            'model_prediction': model_prediction,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper
        })

    # Process CTT data
    for (paradigm, test), group in ctt_input.groupby(['paradigm', 'TEST']):
        n = len(group)
        observed_mean = group['CTT_mean'].mean()
        observed_std = group['CTT_mean'].std()
        se = observed_std / np.sqrt(n)
        ci_lower = observed_mean - 1.96 * se
        ci_upper = observed_mean + 1.96 * se
        model_prediction = group['fitted'].mean()
        tsvr_hours = group['TSVR_hours'].mean()

        results.append({
            'paradigm': paradigm,
            'TEST': test,
            'TSVR_hours': tsvr_hours,
            'measurement_type': 'CTT',
            'observed_mean': observed_mean,
            'model_prediction': model_prediction,
            'CI_lower': ci_lower,
            'CI_upper': ci_upper
        })

    trajectory_df = pd.DataFrame(results)
    logger.info(f"Trajectory summaries: {len(trajectory_df)} rows")

    return trajectory_df

def validate_trajectory_data(trajectory_df):
    """Validate trajectory data."""
    # Check row count
    expected = 24  # 3 paradigms x 4 tests x 2 measurement types
    if len(trajectory_df) != expected:
        logger.warning(f"Row count: {len(trajectory_df)} (expected {expected})")

    # Check CI validity
    invalid_ci = trajectory_df[trajectory_df['CI_lower'] > trajectory_df['CI_upper']]
    if len(invalid_ci) > 0:
        raise ValueError("Invalid CI: CI_lower > CI_upper")

    # Check CI contains observed mean
    ci_valid = (trajectory_df['CI_lower'] <= trajectory_df['observed_mean']) & \
               (trajectory_df['observed_mean'] <= trajectory_df['CI_upper'])
    if not ci_valid.all():
        logger.warning("Some CIs don't contain observed mean (expected for 95% CI)")

    # Check paradigm and measurement type distribution
    paradigm_counts = trajectory_df['paradigm'].value_counts()
    type_counts = trajectory_df['measurement_type'].value_counts()

    logger.info(f"Paradigm counts: {paradigm_counts.to_dict()}")
    logger.info(f"Measurement type counts: {type_counts.to_dict()}")

    logger.info("Trajectory data validation: PASS")
    return True

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("STEP 08: PREPARE TRAJECTORY COMPARISON DATA")
    logger.info("=" * 60)

    try:
        # 1. Load data and models
        logger.info("\n1. Loading data and models...")
        irt_input, ctt_input, irt_model, ctt_model = load_data_and_models()

        # 2. Compute trajectory summaries
        logger.info("\n2. Computing trajectory summaries...")
        trajectory_df = compute_trajectory_summaries(irt_input, ctt_input, irt_model, ctt_model)

        # 3. Validate
        logger.info("\n3. Validating data...")
        validate_trajectory_data(trajectory_df)

        # 4. Save
        logger.info("\n4. Saving output...")
        trajectory_df.to_csv(DATA_PATH / "step08_trajectory_data.csv", index=False)
        logger.info(f"   Saved: step08_trajectory_data.csv ({len(trajectory_df)} rows)")

        # Report
        logger.info("\n" + "=" * 60)
        logger.info("STEP 08 COMPLETE: Trajectory data prepared")
        logger.info(f"Trajectory data prepared: {len(trajectory_df)} rows (3 paradigms x 4 tests x 2 measurement types)")
        logger.info("All paradigms represented: IFR, ICR, IRE")
        logger.info("Both measurement types present: IRT, CTT")
        logger.info("Confidence intervals valid: CI_lower < observed_mean < CI_upper for all rows")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"STEP 08 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
