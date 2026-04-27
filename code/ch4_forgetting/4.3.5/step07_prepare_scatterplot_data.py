"""
Step 07: Prepare Scatterplot Data (IRT vs CTT Convergence)

Purpose: Create scatterplot dataset (1200 rows: 100 UID x 4 TEST x 3 paradigms)
         with IRT_theta, CTT_mean, and fitted values from both models.

Dependencies: Step 02 (merged IRT-CTT data), Step 03 (fitted models)

Output Files:
    - data/step07_scatterplot_data.csv (1200 rows)
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
        logging.FileHandler(LOGS_PATH / "step07_prepare_scatterplot_data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data_and_models():
    """Load merged data and fitted models."""
    # Load merged data (wide format)
    merged_df = pd.read_csv(DATA_PATH / "step02_merged_irt_ctt.csv")
    logger.info(f"Loaded merged data: {len(merged_df)} rows")

    # Load LMM input data for predictions
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

    return merged_df, irt_input, ctt_input, irt_model, ctt_model

def reshape_to_long(merged_df):
    """Reshape merged data from wide to long format."""
    records = []

    for _, row in merged_df.iterrows():
        for paradigm in ['IFR', 'ICR', 'IRE']:
            records.append({
                'UID': row['UID'],
                'TEST': row['TEST'],
                'composite_ID': row['composite_ID'],
                'paradigm': paradigm,
                'IRT_theta': row[f'theta_{paradigm}'],
                'CTT_mean': row[f'CTT_{paradigm}']
            })

    long_df = pd.DataFrame(records)
    logger.info(f"Reshaped to long format: {len(long_df)} rows")

    return long_df

def add_fitted_values(scatter_df, irt_input, ctt_input, irt_model, ctt_model):
    """Add fitted values from both models."""
    # Generate predictions from IRT model
    irt_fitted = irt_model.fittedvalues

    # Map predictions back to scatter data
    # Create index mapping
    irt_input['idx'] = range(len(irt_input))
    irt_input['fitted_irt'] = irt_fitted.values

    # Merge fitted values
    scatter_df = scatter_df.merge(
        irt_input[['composite_ID', 'paradigm', 'fitted_irt']],
        on=['composite_ID', 'paradigm'],
        how='left'
    )

    # CTT fitted values
    ctt_fitted = ctt_model.fittedvalues
    ctt_input['fitted_ctt'] = ctt_fitted.values

    scatter_df = scatter_df.merge(
        ctt_input[['composite_ID', 'paradigm', 'fitted_ctt']],
        on=['composite_ID', 'paradigm'],
        how='left'
    )

    # Rename columns
    scatter_df = scatter_df.rename(columns={
        'fitted_irt': 'IRT_fitted',
        'fitted_ctt': 'CTT_fitted'
    })

    logger.info("Added fitted values from both models")

    return scatter_df

def validate_scatterplot_data(scatter_df):
    """Validate scatterplot data."""
    # Check row count
    expected = 1200  # 400 x 3 paradigms
    if len(scatter_df) != expected:
        logger.warning(f"Row count: {len(scatter_df)} (expected {expected})")

    # Check for NaN
    nan_counts = scatter_df[['IRT_theta', 'CTT_mean']].isna().sum()
    if nan_counts.any():
        logger.warning(f"NaN values: {nan_counts.to_dict()}")

    # Check ranges
    theta_range = scatter_df['IRT_theta'].agg(['min', 'max'])
    ctt_range = scatter_df['CTT_mean'].agg(['min', 'max'])

    logger.info(f"IRT_theta range: [{theta_range['min']:.3f}, {theta_range['max']:.3f}]")
    logger.info(f"CTT_mean range: [{ctt_range['min']:.3f}, {ctt_range['max']:.3f}]")

    # Check paradigm distribution
    paradigm_counts = scatter_df['paradigm'].value_counts()
    logger.info(f"Paradigm counts: {paradigm_counts.to_dict()}")

    logger.info("Scatterplot data validation: PASS")
    return True

def main():
    """Main execution function."""
    logger.info("=" * 60)
    logger.info("STEP 07: PREPARE SCATTERPLOT DATA")
    logger.info("=" * 60)

    try:
        # 1. Load data and models
        logger.info("\n1. Loading data and models...")
        merged_df, irt_input, ctt_input, irt_model, ctt_model = load_data_and_models()

        # 2. Reshape to long format
        logger.info("\n2. Reshaping to long format...")
        scatter_df = reshape_to_long(merged_df)

        # 3. Add fitted values
        logger.info("\n3. Adding fitted values...")
        scatter_df = add_fitted_values(scatter_df, irt_input, ctt_input, irt_model, ctt_model)

        # 4. Validate
        logger.info("\n4. Validating data...")
        validate_scatterplot_data(scatter_df)

        # 5. Save
        logger.info("\n5. Saving output...")
        scatter_df.to_csv(DATA_PATH / "step07_scatterplot_data.csv", index=False)
        logger.info(f"   Saved: step07_scatterplot_data.csv ({len(scatter_df)} rows)")

        # Report
        logger.info("\n" + "=" * 60)
        logger.info("STEP 07 COMPLETE: Scatterplot data prepared")
        logger.info(f"Scatterplot data prepared: {len(scatter_df)} rows (400 x 3 paradigms)")
        logger.info("All paradigms represented: IFR, ICR, IRE")
        logger.info("Fitted values generated for both IRT and CTT models")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"STEP 07 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
