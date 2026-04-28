#!/usr/bin/env python3
"""
Step 04: Extract 3-Way Interaction and Prepare Plot Data

Purpose:
- Extract 3-way interaction terms (Time × Difficulty_c × paradigm)
- Test significance at Bonferroni alpha = 0.0033
- Prepare plot data showing trajectories for 2 difficulty levels × 3 paradigms

Inputs:
- data/step03_fixed_effects.csv (fixed effects with dual p-values)
- data/step02_lmm_input.csv (for computing observed means)
- data/step03_lmm_model.pkl (for computing predicted values)

Outputs:
- data/step04_3way_interaction_summary.csv (interaction terms with significance)
- data/step04_difficulty_trajectories_data.csv (plot data: 24 rows = 6 groups × 4 timepoints)

Validation:
- Required interaction terms present
- Dual p-values per Decision D068
- Complete factorial design (all paradigms × difficulty levels × timepoints)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from statsmodels.regression.mixed_linear_model import MixedLMResults
from tools.validation import (
    validate_hypothesis_test_dual_pvalues,
    validate_plot_data_completeness
)

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step04_extract_interaction.log"

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

def main():
    try:
        log("Step 04: Extract 3-Way Interaction and Prepare Plot Data")
        # Load Fixed Effects
        log("Loading fixed effects...")
        fixed_effects_path = RQ_DIR / "data" / "step03_fixed_effects.csv"
        fixed_effects = pd.read_csv(fixed_effects_path)
        log(f"{len(fixed_effects)} fixed effect terms")
        log(f"Columns: {list(fixed_effects.columns)}")
        # Extract 3-Way Interaction Terms
        log("Filtering for 3-way interaction terms...")

        # Filter for Time:Difficulty_c:C(paradigm) interactions
        interaction_mask = fixed_effects['term'].str.contains('Time:Difficulty_c:C\\(paradigm\\)', regex=True)
        interaction_terms = fixed_effects[interaction_mask].copy()

        log(f"Found {len(interaction_terms)} 3-way interaction terms:")
        for term in interaction_terms['term']:
            log(f"  - {term}")

        if len(interaction_terms) == 0:
            log("No 3-way interaction terms found")
            raise ValueError("No 3-way interaction terms found in fixed effects")

        # Add significance flag
        bonferroni_alpha = 0.0033
        interaction_terms['significant_at_0.0033'] = interaction_terms['p_bonferroni'] < bonferroni_alpha

        log(f"Bonferroni alpha: {bonferroni_alpha}")
        for _, row in interaction_terms.iterrows():
            sig_flag = "SIGNIFICANT" if row['significant_at_0.0033'] else "NOT SIGNIFICANT"
            log(f"  {row['term']}: p_bonf = {row['p_bonferroni']:.4f} ({sig_flag})")

        # Save interaction summary
        output_path = RQ_DIR / "data" / "step04_3way_interaction_summary.csv"
        interaction_terms.to_csv(output_path, index=False, encoding='utf-8')
        log(f"{output_path}")
        # Load LMM Input Data and Model
        log("Loading LMM input data for plot preparation...")
        lmm_input_path = RQ_DIR / "data" / "step02_lmm_input.csv"
        df = pd.read_csv(lmm_input_path)
        log(f"{len(df)} rows")

        # Drop rows with missing Response
        df = df.dropna(subset=['Response'])
        log(f"{len(df)} rows after dropping missing Response")

        log("Loading fitted LMM model...")
        model_path = RQ_DIR / "data" / "step03_lmm_model.pkl"
        lmm_model = MixedLMResults.load(str(model_path))
        log("LMM model")
        # Define Difficulty Levels and Timepoints
        log("Defining difficulty levels and timepoints...")

        # Compute SD of Difficulty_c
        difficulty_sd = df['Difficulty_c'].std()
        log(f"Difficulty_c SD: {difficulty_sd:.4f}")

        # Define difficulty levels (Easy = -1 SD, Hard = +1 SD)
        difficulty_levels = {
            'Easy (-1SD)': -1.0 * difficulty_sd,
            'Hard (+1SD)': 1.0 * difficulty_sd
        }

        log("Difficulty levels:")
        for level_name, level_value in difficulty_levels.items():
            log(f"  {level_name}: Difficulty_c = {level_value:.4f}")

        # Define timepoints (nominal days and approximate hours)
        timepoints = [
            {'time_days': 0, 'time_hours_approx': 0},
            {'time_days': 1, 'time_hours_approx': 24},
            {'time_days': 3, 'time_hours_approx': 72},
            {'time_days': 6, 'time_hours_approx': 144}
        ]

        log("Timepoints:")
        for tp in timepoints:
            log(f"  Day {tp['time_days']}: ~{tp['time_hours_approx']} hours")
        # Prepare Plot Data
        log("Computing predicted and observed values...")

        plot_data_rows = []
        paradigms = ['IFR', 'ICR', 'IRE']

        for paradigm in paradigms:
            for difficulty_name, difficulty_value in difficulty_levels.items():
                for timepoint in timepoints:
                    time_hours = timepoint['time_hours_approx']
                    time_days = timepoint['time_days']

                    # Create a prediction dataframe for this combination
                    pred_df = pd.DataFrame({
                        'Time': [time_hours],
                        'Difficulty_c': [difficulty_value],
                        'paradigm': [paradigm],
                        'UID': ['A010']  # Dummy UID for prediction (marginal effect)
                    })

                    # Get predicted response (probability)
                    try:
                        predicted_response = lmm_model.predict(pred_df).values[0]
                    except Exception as e:
                        log(f"Prediction failed for {paradigm}/{difficulty_name}/Day{time_days}: {e}")
                        predicted_response = np.nan

                    # Compute observed mean from data
                    # Find observations close to this timepoint (±6 hours window)
                    time_tolerance = 6.0
                    mask = (
                        (df['paradigm'] == paradigm) &
                        (df['Difficulty_c'] >= difficulty_value - 0.5 * difficulty_sd) &
                        (df['Difficulty_c'] <= difficulty_value + 0.5 * difficulty_sd) &
                        (df['Time'] >= time_hours - time_tolerance) &
                        (df['Time'] <= time_hours + time_tolerance)
                    )

                    subset = df[mask]
                    if len(subset) > 0:
                        observed_mean = subset['Response'].mean()
                        observed_sem = subset['Response'].sem()
                        ci_lower = observed_mean - 1.96 * observed_sem
                        ci_upper = observed_mean + 1.96 * observed_sem
                    else:
                        observed_mean = np.nan
                        ci_lower = np.nan
                        ci_upper = np.nan

                    plot_data_rows.append({
                        'paradigm': paradigm,
                        'difficulty_level': difficulty_name,
                        'time_days': time_days,
                        'predicted_response': predicted_response,
                        'observed_mean': observed_mean,
                        'CI_lower': ci_lower,
                        'CI_upper': ci_upper
                    })

                    log(f"  {paradigm}/{difficulty_name}/Day{time_days}: pred={predicted_response:.3f}, obs={observed_mean:.3f}")

        # Create plot data DataFrame
        plot_data = pd.DataFrame(plot_data_rows)

        log(f"Plot data shape: {plot_data.shape}")
        log(f"Expected: 24 rows (3 paradigms × 2 difficulty levels × 4 timepoints)")
        # Save Plot Data
        log("Saving plot data...")
        plot_data_path = RQ_DIR / "data" / "step04_difficulty_trajectories_data.csv"
        plot_data.to_csv(plot_data_path, index=False, encoding='utf-8')
        log(f"{plot_data_path}")
        # Run Validation Tools
        log("Running validate_hypothesis_test_dual_pvalues...")
        validation_result = validate_hypothesis_test_dual_pvalues(
            interaction_df=interaction_terms,
            required_terms=['Time:Difficulty_c:C(paradigm)[T.IFR]', 'Time:Difficulty_c:C(paradigm)[T.IRE]'],
            alpha_bonferroni=0.0033
        )
        log(f"Hypothesis test result: {validation_result}")

        if not validation_result.get('valid', False):
            log(f"Hypothesis test validation failed: {validation_result}")
            raise ValueError(f"Hypothesis test validation failed: {validation_result.get('message', 'Unknown error')}")

        log("Running validate_plot_data_completeness...")
        plot_validation_result = validate_plot_data_completeness(
            plot_data=plot_data,
            required_domains=['IFR', 'ICR', 'IRE'],
            required_groups=['Easy (-1SD)', 'Hard (+1SD)'],
            domain_col='paradigm',
            group_col='difficulty_level'
        )
        log(f"Plot data result: {plot_validation_result}")

        if not plot_validation_result.get('valid', False):
            log(f"Plot data validation failed: {plot_validation_result}")
            raise ValueError(f"Plot data validation failed: {plot_validation_result.get('message', 'Unknown error')}")

        log("Step 04 complete")
        return 0

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        log("")
        traceback.print_exc()
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        return 1

if __name__ == "__main__":
    sys.exit(main())
