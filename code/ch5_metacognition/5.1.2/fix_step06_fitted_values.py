#!/usr/bin/env python3
"""Fix Step 06 fitted values that were left as NaN"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.formula.api import mixedlm

# Paths
RQ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = RQ_DIR / "data"

# Load the piecewise data
df_pw = pd.read_csv(DATA_DIR / "step01_piecewise_input.csv")

# Refit piecewise model to get fitted values
pw_formula = "theta_confidence ~ Time_Early + Time_Late"
pw_model = mixedlm(pw_formula, df_pw, groups=df_pw["UID"])
pw_result = pw_model.fit(reml=False, method='powell')

# Get coefficients for prediction
intercept = pw_result.fe_params['Intercept']
early_slope = pw_result.fe_params['Time_Early']
late_slope = pw_result.fe_params['Time_Late']

# Load plot data files
theta_plot = pd.read_csv(DATA_DIR / "step06_twophase_theta_data.csv")
prob_plot = pd.read_csv(DATA_DIR / "step06_twophase_probability_data.csv")

# Compute fitted values for each row
def compute_fitted(row):
    """Compute fitted theta value based on segment and time."""
    if row['Segment'] == 'Early':
        # For Early: fitted = intercept + early_slope * TSVR_hours
        return intercept + early_slope * row['TSVR_hours']
    else:
        # For Late: fitted = intercept + late_slope * (TSVR_hours - 48)
        return intercept + late_slope * (row['TSVR_hours'] - 48)

# Apply to theta plot
theta_plot['fitted'] = theta_plot.apply(compute_fitted, axis=1)
theta_plot.to_csv(DATA_DIR / "step06_twophase_theta_data.csv", index=False)

# Apply to probability plot (transform fitted theta to probability)
prob_plot['fitted'] = theta_plot.apply(compute_fitted, axis=1).apply(
    lambda theta: 1 / (1 + np.exp(-1.702 * theta))
)
prob_plot.to_csv(DATA_DIR / "step06_twophase_probability_data.csv", index=False)

print(f"Fixed fitted values in plot data files")
print(f"Theta plot: {len(theta_plot)} rows, fitted range: [{theta_plot['fitted'].min():.3f}, {theta_plot['fitted'].max():.3f}]")
print(f"Prob plot: {len(prob_plot)} rows, fitted range: [{prob_plot['fitted'].min():.3f}, {prob_plot['fitted'].max():.3f}]")
