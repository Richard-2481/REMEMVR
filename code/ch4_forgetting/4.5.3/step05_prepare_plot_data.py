"""
Step 05: Prepare Age Tertile Plot Data
RQ 5.5.3 - Age Effects on Source-Destination Memory

Purpose: Create age tertiles (Young/Middle/Older based on 33rd and 67th percentiles)
         and aggregate observed theta means by age tertile x location type x test.
         Compute 95% confidence intervals per group.

Input:
- data/step01_lmm_input.csv (800 rows with Age, LocationType, test, theta)

Output:
- data/step05_age_tertile_plot_data.csv (24 rows: 3 tertiles x 2 locations x 4 tests)

Log: logs/step05_prepare_plot_data.log
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup paths
RQ_DIR = Path(__file__).parent.parent
DATA_DIR = RQ_DIR / "data"
LOG_DIR = RQ_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / "step05_prepare_plot_data.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Step 05: Prepare Age Tertile Plot Data")

    # -------------------------------------------------------------------------
    # 1. Load LMM input data
    # -------------------------------------------------------------------------
    logger.info("Loading LMM input data...")

    lmm_input = pd.read_csv(DATA_DIR / "step01_lmm_input.csv")
    logger.info(f"{len(lmm_input)} observations")

    # -------------------------------------------------------------------------
    # 2. Compute age tertile cutoffs
    # -------------------------------------------------------------------------
    logger.info("Computing age tertile cutoffs...")

    # Get unique ages per participant (Age is same for all observations of a participant)
    participant_ages = lmm_input.groupby('UID')['Age'].first()

    # Compute tertile cutoffs
    p33 = participant_ages.quantile(0.33)
    p67 = participant_ages.quantile(0.67)

    logger.info(f"33rd percentile: {p33:.1f} years")
    logger.info(f"67th percentile: {p67:.1f} years")

    # -------------------------------------------------------------------------
    # 3. Assign age tertiles
    # -------------------------------------------------------------------------
    logger.info("Assigning observations to age tertiles...")

    def assign_tertile(age):
        if age <= p33:
            return 'Young'
        elif age <= p67:
            return 'Middle'
        else:
            return 'Older'

    lmm_input['age_tertile'] = lmm_input['Age'].apply(assign_tertile)

    # Count per tertile
    tertile_counts = lmm_input.groupby('age_tertile')['UID'].nunique()
    logger.info("Participants per tertile:")
    for tertile in ['Young', 'Middle', 'Older']:
        count = tertile_counts.get(tertile, 0)
        logger.info(f"  {tertile}: {count} participants")

    # -------------------------------------------------------------------------
    # 4. Aggregate by tertile x location x test
    # -------------------------------------------------------------------------
    logger.info("Computing means and CIs by group...")

    # Group by age_tertile, LocationType, test
    grouped = lmm_input.groupby(['age_tertile', 'LocationType', 'test'])

    plot_data = []
    for (tertile, location, test), group in grouped:
        n = len(group)
        theta_mean = group['theta'].mean()
        theta_std = group['theta'].std()
        se = theta_std / np.sqrt(n) if n > 1 else 0

        ci_lower = theta_mean - 1.96 * se
        ci_upper = theta_mean + 1.96 * se

        plot_data.append({
            'age_tertile': tertile,
            'location': location,
            'test': test,
            'theta_mean': theta_mean,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': n
        })

    plot_df = pd.DataFrame(plot_data)

    # Sort for consistent output
    tertile_order = {'Young': 0, 'Middle': 1, 'Older': 2}
    location_order = {'Source': 0, 'Destination': 1}
    plot_df['tertile_sort'] = plot_df['age_tertile'].map(tertile_order)
    plot_df['location_sort'] = plot_df['location'].map(location_order)
    plot_df = plot_df.sort_values(['tertile_sort', 'location_sort', 'test'])
    plot_df = plot_df.drop(columns=['tertile_sort', 'location_sort'])

    logger.info(f"{len(plot_df)} groups created")

    # -------------------------------------------------------------------------
    # 5. Validation
    # -------------------------------------------------------------------------
    logger.info("Running validation checks...")

    all_pass = True

    # Check 1: Exactly 24 rows
    expected_rows = 3 * 2 * 4  # 3 tertiles x 2 locations x 4 tests
    if len(plot_df) == expected_rows:
        logger.info(f"Row count = {expected_rows}")
    else:
        logger.info(f"Expected {expected_rows} rows, found {len(plot_df)}")
        all_pass = False

    # Check 2: All tertiles present
    tertiles_present = set(plot_df['age_tertile'].unique())
    if tertiles_present == {'Young', 'Middle', 'Older'}:
        logger.info("All 3 age tertiles present")
    else:
        logger.info(f"Missing tertiles: expected Young/Middle/Older, found {tertiles_present}")
        all_pass = False

    # Check 3: All locations present
    locations_present = set(plot_df['location'].unique())
    if locations_present == {'Source', 'Destination'}:
        logger.info("Both locations present")
    else:
        logger.info(f"Missing locations: expected Source/Destination, found {locations_present}")
        all_pass = False

    # Check 4: All tests present
    tests_present = set(plot_df['test'].unique())
    if len(tests_present) == 4:
        logger.info("All 4 tests present")
    else:
        logger.info(f"Missing tests, found: {tests_present}")
        all_pass = False

    # Check 5: No NaN values
    nan_count = plot_df.isna().sum().sum()
    if nan_count == 0:
        logger.info("No NaN values")
    else:
        logger.info(f"Found {nan_count} NaN values")
        all_pass = False

    # Check 6: CI logic (ci_upper > ci_lower)
    if all(plot_df['ci_upper'] > plot_df['ci_lower']):
        logger.info("Confidence intervals valid")
    else:
        logger.info("Invalid confidence intervals")
        all_pass = False

    # Check 7: Minimum group size
    min_n = plot_df['n'].min()
    if min_n >= 5:
        logger.info(f"Minimum group size = {min_n} (>= 5)")
    else:
        logger.info(f"Small group size: min n = {min_n}")
        # Not a failure, just a warning

    if not all_pass:
        raise ValueError("Validation failed - see above for details")

    # -------------------------------------------------------------------------
    # 6. Save output
    # -------------------------------------------------------------------------
    logger.info("Saving plot data...")

    output_path = DATA_DIR / "step05_age_tertile_plot_data.csv"
    plot_df.to_csv(output_path, index=False)
    logger.info(f"{output_path.name} ({len(plot_df)} rows)")

    # -------------------------------------------------------------------------
    # 7. Summary statistics
    # -------------------------------------------------------------------------
    logger.info("Theta means by age tertile and location:")

    summary = plot_df.groupby(['age_tertile', 'location'])['theta_mean'].mean()
    for (tertile, location), mean in summary.items():
        logger.info(f"  {tertile} - {location}: {mean:.3f}")

    logger.info("Step 05 complete - Plot data prepared")


if __name__ == "__main__":
    main()
