#!/usr/bin/env python3
"""compute_effect_size: Quantify age impact by comparing Day 6 memory for average vs older adults (Age + 1 SD)."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_numeric_range

# Import statsmodels for model loading
from statsmodels.regression.mixed_linear_model import MixedLMResults

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/chX/rqY (derived from script location)
LOG_FILE = RQ_DIR / "logs" / "step04_compute_effect_size.log"


# Effect size parameters
TSVR_DAY6 = 144.0  # Day 6 retention interval (~144 hours)
LOG_OFFSET = 1  # log(TSVR + 1) as in LMM formula

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
    print(msg)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 04: Compute Effect Size (Age Impact on Day 6 Memory)")
        # Load Fitted LMM Model

        log("Loading fitted LMM model from step02...")
        model_path = RQ_DIR / "data" / "step02_lmm_model.pkl"

        # CRITICAL: Use MixedLMResults.load() method (NOT pickle.load())
        # Reason: statsmodels models require special loading to avoid patsy/eval errors
        lmm_result = MixedLMResults.load(str(model_path))

        log(f"LMM model from {model_path}")
        log(f"Model formula: {lmm_result.model.formula}")
        log(f"Converged: {lmm_result.converged}")
        # Load Prepared Data (Extract SD_age and verify TSVR_day6)

        log("Loading prepared data to extract SD(age)...")
        prepared_data_path = RQ_DIR / "data" / "step01_lmm_input_prepared.csv"
        df_prepared = pd.read_csv(prepared_data_path, encoding='utf-8')

        log(f"{prepared_data_path.name} ({len(df_prepared)} rows, {len(df_prepared.columns)} cols)")

        # Extract SD(age) for effect size scenarios
        sd_age = df_prepared['age'].std()
        mean_age = df_prepared['age'].mean()

        log(f"Age statistics: Mean = {mean_age:.2f} years, SD = {sd_age:.2f} years")

        # Verify TSVR_day6 assumption (should be ~144 hours in actual data)
        tsvr_max = df_prepared['TSVR_hours'].max()
        log(f"TSVR range in data: {df_prepared['TSVR_hours'].min():.1f} to {tsvr_max:.1f} hours")
        log(f"Using TSVR_day6 = {TSVR_DAY6} hours for effect size computation")
        # Define Effect Size Scenarios
        # Scenario 1: Average age (Age_c = 0), Day 6 (Time = 144, Time_log = log(145))
        # Scenario 2: Age + 1 SD (Age_c = SD_age), Day 6 (Time = 144, Time_log = log(145))

        log("Defining effect size scenarios...")

        # Compute time transformations for Day 6
        time_day6 = TSVR_DAY6
        time_log_day6 = np.log(TSVR_DAY6 + LOG_OFFSET)

        log(f"Day 6 time transformations: Time = {time_day6:.1f}, Time_log = {time_log_day6:.3f}")

        # Scenario definitions
        scenarios = [
            {
                'scenario': 'Average age',
                'age_c': 0.0,
                'age_years': mean_age,
                'time_hours': time_day6
            },
            {
                'scenario': 'Age + 1 SD',
                'age_c': sd_age,
                'age_years': mean_age + sd_age,
                'time_hours': time_day6
            }
        ]

        log(f"Scenario 1: {scenarios[0]['scenario']} (Age_c = {scenarios[0]['age_c']:.2f}, Age = {scenarios[0]['age_years']:.1f} years)")
        log(f"Scenario 2: {scenarios[1]['scenario']} (Age_c = {scenarios[1]['age_c']:.2f}, Age = {scenarios[1]['age_years']:.1f} years)")
        # Compute Predicted Theta for Each Scenario
        # Method: Use LMM fixed effects to compute theta = intercept + sum(coef_i * predictor_i)
        # Formula: theta ~ (Time + Time_log) * Age_c
        # Expanded: theta ~ 1 + Time + Time_log + Age_c + Time:Age_c + Time_log:Age_c

        log("Computing predicted theta using LMM fixed effects...")

        # Extract fixed effects from LMM
        fixed_effects = lmm_result.fe_params

        log(f"Fixed effects parameters:")
        for param_name, param_value in fixed_effects.items():
            log(f"  {param_name}: {param_value:.4f}")

        # Compute predictions for each scenario
        predictions = []

        for scenario in scenarios:
            age_c = scenario['age_c']
            time = time_day6
            time_log = time_log_day6

            # Manual prediction using fixed effects
            # theta = Intercept + Time*coef_Time + Time_log*coef_Time_log + Age_c*coef_Age_c
            #         + (Time*Age_c)*coef_Time:Age_c + (Time_log*Age_c)*coef_Time_log:Age_c

            theta_pred = (
                fixed_effects['Intercept'] +
                fixed_effects['Time'] * time +
                fixed_effects['Time_log'] * time_log +
                fixed_effects['Age_c'] * age_c +
                fixed_effects['Time:Age_c'] * (time * age_c) +
                fixed_effects['Time_log:Age_c'] * (time_log * age_c)
            )

            scenario['theta_predicted'] = theta_pred
            predictions.append(scenario)

            log(f"{scenario['scenario']}: theta = {theta_pred:.3f}")

        # Create DataFrame with predictions
        df_effect_size = pd.DataFrame(predictions)
        # Compute Decline Metrics
        # Decline = theta_older - theta_avg (expected negative)
        # Decline_percent = (Decline / theta_avg) * 100

        log("Computing age decline metrics...")

        theta_avg = df_effect_size.loc[df_effect_size['scenario'] == 'Average age', 'theta_predicted'].values[0]
        theta_older = df_effect_size.loc[df_effect_size['scenario'] == 'Age + 1 SD', 'theta_predicted'].values[0]

        decline_theta = theta_older - theta_avg
        decline_percent = (decline_theta / theta_avg) * 100 if theta_avg != 0 else np.nan

        log(f"Average age theta at Day 6: {theta_avg:.3f}")
        log(f"Age + 1 SD theta at Day 6: {theta_older:.3f}")
        log(f"Decline (theta): {decline_theta:.3f}")
        log(f"Decline (percent): {decline_percent:.1f}%")
        # Save Effect Size Results
        # Output 1: data/step04_effect_size.csv (comparison table)
        # Output 2: results/step04_effect_size_summary.txt (interpretation)

        log("Saving effect size results...")

        # Save comparison table
        output_csv = RQ_DIR / "data" / "step04_effect_size.csv"
        df_effect_size.to_csv(output_csv, index=False, encoding='utf-8')
        log(f"{output_csv.name} ({len(df_effect_size)} rows)")

        # Create summary text
        summary_text = f"""EFFECT SIZE SUMMARY - Age Impact on Day 6 Memory (RQ 5.1.3)
{'=' * 70}

SCENARIOS COMPARED:
  1. Average age: {scenarios[0]['age_years']:.1f} years (Age_c = {scenarios[0]['age_c']:.2f})
  2. Older adults: {scenarios[1]['age_years']:.1f} years (Age_c = {scenarios[1]['age_c']:.2f})

  Age difference: +1 SD = {sd_age:.2f} years

PREDICTED THETA SCORES AT DAY 6 (TSVR = {TSVR_DAY6:.0f} hours):
  Average age:  {theta_avg:.3f}
  Older adults: {theta_older:.3f}

AGE DECLINE AT DAY 6:
  Absolute decline: {decline_theta:.3f} theta units
  Percentage decline: {decline_percent:.1f}%

INTERPRETATION:
  Older adults (+ 1 SD age) show {'lower' if decline_theta < 0 else 'higher'} memory retention
  at Day 6 compared to average-age participants. A {abs(decline_theta):.3f} theta
  unit decline represents a {abs(decline_percent):.1f}% reduction in memory ability
  after ~6 days of retention.

NOTE:
  - Theta scores are on IRT scale (mean 0, SD 1 in calibration sample)
  - Negative decline indicates worse memory for older adults (expected)
  - Positive decline would indicate better memory for older adults (unexpected)
  - Effect size quantifies practical significance of age effects on forgetting

GENERATED: 2025-11-28
RQ: ch5/5.1.3 (Age effects on baseline memory and forgetting rate)
"""

        # Save summary text
        output_txt = RQ_DIR / "results" / "step04_effect_size_summary.txt"
        output_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        log(f"{output_txt.name}")
        # Validation - Check Predicted Theta Range
        # Criteria: theta_predicted in [-4, 4] (typical IRT range)

        log("Validating predicted theta range...")

        validation_result = validate_numeric_range(
            data=df_effect_size['theta_predicted'].values,
            min_val=-4.0,
            max_val=4.0,
            column_name='theta_predicted'
        )

        if validation_result['valid']:
            log(f"[VALIDATION PASS] theta_predicted in [-4, 4]: {validation_result['message']}")
        else:
            log(f"[VALIDATION FAIL] theta_predicted out of range: {validation_result['message']}")
            raise ValueError(f"Validation failed: {validation_result['message']}")

        # Validate Age_c range
        log("Validating Age_c range...")

        validation_result_age = validate_numeric_range(
            data=df_effect_size['age_c'].values,
            min_val=-30.0,
            max_val=30.0,
            column_name='age_c'
        )

        if validation_result_age['valid']:
            log(f"[VALIDATION PASS] age_c in [-30, 30]: {validation_result_age['message']}")
        else:
            log(f"[VALIDATION FAIL] age_c out of range: {validation_result_age['message']}")
            raise ValueError(f"Validation failed: {validation_result_age['message']}")

        log("Step 04 complete - Effect size computed and validated")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
