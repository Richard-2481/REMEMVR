#!/usr/bin/env python3
"""
Step 3: Fit Age x Cognitive Test Interaction Models for RQ 7.2.3
Purpose: Test whether cognitive tests predict REMEMVR differently at different ages

Scientific Context:
- Cognitive Reserve Theory: Older adults may rely more on cognitive abilities
- VR Scaffolding Hypothesis: VR may eliminate age differences in prediction
- Testing 4 models: Age x RAVLT, Age x BVMT, Age x NART, Age x RPM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Define paths
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.2.3
OUTPUT_DIR = RQ_DIR / "data"

# Multiple comparison correction
BONFERRONI_ALPHA = 0.05 / 6  # 6 interaction tests

def fit_interaction_model(df, test_name):
    """Fit a single Age x Test interaction model."""
    
    # Define variables
    test_c = f"{test_name}_c"
    interaction = f"Age_c_x_{test_name}_c"
    
    # Prepare data (drop missing values for this test)
    model_df = df[['theta_all', 'Age_c', test_c, interaction]].dropna()
    n_obs = len(model_df)
    
    print(f"\n{test_name} Model (N={n_obs}):")
    print("-" * 40)
    
    # Prepare predictors with constant
    X = model_df[['Age_c', test_c, interaction]]
    X = sm.add_constant(X)
    y = model_df['theta_all']
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    
    # Extract key statistics
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    f_stat = model.fvalue
    f_p = model.f_pvalue
    
    # Get interaction coefficient statistics
    interaction_coef = model.params[interaction]
    interaction_se = model.bse[interaction]
    interaction_t = model.tvalues[interaction]
    interaction_p_uncorr = model.pvalues[interaction]
    interaction_p_bonf = min(1.0, interaction_p_uncorr * 6)  # Bonferroni correction (6 tests)
    
    # Calculate VIF for each predictor
    vif_values = []
    for i in range(1, len(X.columns)):  # Skip constant
        vif = variance_inflation_factor(X.values, i)
        vif_values.append(vif)
    
    # Print model summary
    print(f"R² = {r_squared:.3f}, Adj R² = {adj_r_squared:.3f}")
    print(f"F({model.df_model:.0f}, {model.df_resid:.0f}) = {f_stat:.2f}, p = {f_p:.4f}")
    
    print(f"\nCoefficients:")
    print(f"  Intercept: {model.params['const']:.3f}")
    print(f"  Age_c: {model.params['Age_c']:.4f} (p={model.pvalues['Age_c']:.4f})")
    print(f"  {test_c}: {model.params[test_c]:.4f} (p={model.pvalues[test_c]:.4f})")
    print(f"  {interaction}: {interaction_coef:.5f}")
    print(f"    SE = {interaction_se:.5f}")
    print(f"    t = {interaction_t:.3f}")
    print(f"    p (uncorrected) = {interaction_p_uncorr:.4f}")
    print(f"    p (Bonferroni) = {interaction_p_bonf:.4f}")
    
    # Decision on interaction significance
    if interaction_p_bonf < BONFERRONI_ALPHA:
        print(f"  *** SIGNIFICANT INTERACTION (p < {BONFERRONI_ALPHA:.3f}) ***")
    else:
        print(f"  Interaction not significant at Bonferroni-corrected α = {BONFERRONI_ALPHA:.3f}")
    
    print(f"\nVIF values:")
    for i, col in enumerate(['Age_c', test_c, interaction]):
        print(f"  {col}: {vif_values[i]:.2f}")
    
    # Create results dictionary
    results = {
        'model': f'Age x {test_name}',
        'n_obs': n_obs,
        'R2': r_squared,
        'adj_R2': adj_r_squared,
        'F_stat': f_stat,
        'F_p': f_p,
        'interaction_coef': interaction_coef,
        'interaction_se': interaction_se,
        'interaction_t': interaction_t,
        'interaction_p_uncorr': interaction_p_uncorr,
        'interaction_p_bonf': interaction_p_bonf,
        'VIF_age': vif_values[0],
        'VIF_test': vif_values[1],
        'VIF_interaction': vif_values[2],
        'significant': interaction_p_bonf < BONFERRONI_ALPHA
    }
    
    # Also extract all coefficients for later use
    coef_details = {
        'test_name': test_name,
        'intercept': model.params['const'],
        'age_coef': model.params['Age_c'],
        'test_coef': model.params[test_c],
        'interaction_coef': interaction_coef,
        'age_p': model.pvalues['Age_c'],
        'test_p': model.pvalues[test_c],
        'interaction_p_uncorr': interaction_p_uncorr,
        'interaction_p_bonf': interaction_p_bonf
    }
    
    return results, coef_details, model

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("STEP 3: FIT AGE x COGNITIVE TEST INTERACTION MODELS")
    print("=" * 60)
    print(f"Testing interactions with Bonferroni correction:")
    print(f"  α = 0.05 / 6 tests = {BONFERRONI_ALPHA:.4f}")
    print("=" * 60)
    
    # Load centered data from Step 2
    input_file = OUTPUT_DIR / "step02_centered_predictors.csv"
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} participants from Step 2")
    
    # Fit models for each cognitive test
    all_results = []
    all_coefficients = []
    significant_interactions = []
    
    for test in ['RAVLT', 'BVMT', 'NART', 'RPM', 'RAVLT_Pct_Ret', 'BVMT_Pct_Ret']:
        results, coef_details, model = fit_interaction_model(df, test)
        all_results.append(results)
        all_coefficients.append(coef_details)
        
        if results['significant']:
            significant_interactions.append(test)
    
    # Create summary DataFrames
    results_df = pd.DataFrame(all_results)
    coef_df = pd.DataFrame(all_coefficients)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY OF INTERACTION TESTS")
    print("=" * 60)
    
    if significant_interactions:
        print(f"\nSIGNIFICANT INTERACTIONS FOUND:")
        for test in significant_interactions:
            row = results_df[results_df['model'].str.contains(test)]
            p_val = row['interaction_p_bonf'].values[0]
            coef = row['interaction_coef'].values[0]
            print(f"  - Age x {test}: β = {coef:.5f}, p = {p_val:.4f}")
        print("\nThese require simple slopes analysis (Step 4)")
    else:
        print("\nNO SIGNIFICANT INTERACTIONS")
        print(f"All Age x Test interactions p > {BONFERRONI_ALPHA:.4f} (Bonferroni-corrected)")
        print("\nInterpretation: Cognitive tests predict REMEMVR equally across ages")
        print("This supports VR Scaffolding Hypothesis over Cognitive Reserve Theory")
    
    # Report effect sizes
    print("\n" + "=" * 60)
    print("INTERACTION EFFECT SIZES")
    print("=" * 60)
    
    for _, row in results_df.iterrows():
        test = row['model'].replace('Age x ', '')
        print(f"\n{test}:")
        print(f"  Interaction β = {row['interaction_coef']:.5f}")
        print(f"  SE = {row['interaction_se']:.5f}")
        print(f"  p (uncorrected) = {row['interaction_p_uncorr']:.4f}")
        print(f"  p (Bonferroni) = {row['interaction_p_bonf']:.4f}")
    
    # Check multicollinearity
    print("\n" + "=" * 60)
    print("MULTICOLLINEARITY CHECK (VIF)")
    print("=" * 60)
    
    max_vif = results_df[['VIF_age', 'VIF_test', 'VIF_interaction']].max().max()
    print(f"Maximum VIF across all models: {max_vif:.2f}")
    
    if max_vif < 5:
        print("All VIF < 5: No multicollinearity concerns")
    elif max_vif < 10:
        print("Some VIF between 5-10: Moderate multicollinearity")
    else:
        print("WARNING: VIF > 10 detected - serious multicollinearity")
    
    # Save outputs
    results_df.to_csv(OUTPUT_DIR / "step03_interaction_models.csv", index=False)
    coef_df.to_csv(OUTPUT_DIR / "step03_interaction_coefficients.csv", index=False)
    
    print(f"\nOutputs saved:")
    print(f"  - {OUTPUT_DIR / 'step03_interaction_models.csv'}")
    print(f"  - {OUTPUT_DIR / 'step03_interaction_coefficients.csv'}")
    
    print("\nStep 3 complete: Interaction models fitted and tested")
    
    return results_df, significant_interactions

if __name__ == "__main__":
    main()