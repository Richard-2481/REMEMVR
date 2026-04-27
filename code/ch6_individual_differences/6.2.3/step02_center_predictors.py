#!/usr/bin/env python3
"""
Step 2: Center Predictors and Create Interaction Terms for RQ 7.2.3
Purpose: Center predictors for interpretable interactions and create Age x Test terms

Scientific Context:
- Centering Age at mean allows intercept to represent average-age participant
- Centering cognitive tests at T-score mean (50) for meaningful zero point
- Interaction terms test whether test-REMEMVR relationship varies with age
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# Define paths
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.2.3
OUTPUT_DIR = RQ_DIR / "data"

def center_predictors(df):
    """Center Age and cognitive test predictors."""
    
    # Create copy to avoid modifying original
    df_centered = df.copy()
    
    # Center Age at mean
    age_mean = df['Age'].mean()
    df_centered['Age_c'] = df['Age'] - age_mean
    print(f"Age centered at mean = {age_mean:.1f} years")
    print(f"Age_c range: [{df_centered['Age_c'].min():.1f}, {df_centered['Age_c'].max():.1f}]")
    
    # Center cognitive tests at T-score mean (50)
    t_score_mean = 50.0
    
    for test in ['RAVLT_T', 'BVMT_T', 'NART_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
        centered_col = test.replace('_T', '_c')
        df_centered[centered_col] = df[test] - t_score_mean
        
        # Check centering worked
        actual_mean = df_centered[centered_col].mean()
        print(f"\n{test} centered at {t_score_mean}:")
        print(f"  {centered_col} mean = {actual_mean:.3f} (should be ~0)")
        print(f"  {centered_col} range: [{df_centered[centered_col].min():.1f}, {df_centered[centered_col].max():.1f}]")
    
    return df_centered

def create_interaction_terms(df):
    """Create Age x Cognitive Test interaction terms."""
    
    print("\n" + "=" * 60)
    print("CREATING INTERACTION TERMS")
    print("=" * 60)
    
    # Create interaction terms
    interaction_terms = []
    
    for test in ['RAVLT', 'BVMT', 'NART', 'RPM', 'RAVLT_Pct_Ret', 'BVMT_Pct_Ret']:
        test_col = f"{test}_c"
        interaction_col = f"Age_c_x_{test}_c"
        
        # Handle missing NART values
        if test == 'NART' and df[test_col].isnull().any():
            print(f"\nWARNING: {test_col} has {df[test_col].isnull().sum()} missing values")
            print("  Interaction term will have same missing pattern")
        
        df[interaction_col] = df['Age_c'] * df[test_col]
        interaction_terms.append(interaction_col)
        
        # Report interaction statistics
        valid_data = df[interaction_col].dropna()
        print(f"\n{interaction_col}:")
        print(f"  Mean = {valid_data.mean():.3f}")
        print(f"  SD = {valid_data.std():.3f}")
        print(f"  Range: [{valid_data.min():.2f}, {valid_data.max():.2f}]")
    
    return df, interaction_terms

def check_multicollinearity(df):
    """Check correlation matrix for multicollinearity concerns."""
    
    print("\n" + "=" * 60)
    print("MULTICOLLINEARITY CHECK")
    print("=" * 60)
    
    # Select predictors (centered vars and interactions)
    predictors = ['Age_c', 'RAVLT_c', 'BVMT_c', 'NART_c', 'RPM_c',
                  'RAVLT_Pct_Ret_c', 'BVMT_Pct_Ret_c',
                  'Age_c_x_RAVLT_c', 'Age_c_x_BVMT_c', 'Age_c_x_NART_c', 'Age_c_x_RPM_c',
                  'Age_c_x_RAVLT_Pct_Ret_c', 'Age_c_x_BVMT_Pct_Ret_c']
    
    # Compute correlation matrix (handling missing NART)
    corr_matrix = df[predictors].corr()
    
    # Find high correlations (excluding diagonal)
    high_corr_threshold = 0.70
    high_corrs = []
    
    for i in range(len(predictors)):
        for j in range(i+1, len(predictors)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > high_corr_threshold and not pd.isna(corr_val):
                high_corrs.append((predictors[i], predictors[j], corr_val))
    
    if high_corrs:
        print(f"WARNING: Found {len(high_corrs)} correlations > {high_corr_threshold}:")
        for var1, var2, corr in high_corrs:
            print(f"  {var1} <-> {var2}: r = {corr:.3f}")
    else:
        print(f"No correlations exceed {high_corr_threshold} threshold")
        print("Multicollinearity not a concern")
    
    # Print correlation matrix for main effects
    main_effects = ['Age_c', 'RAVLT_c', 'BVMT_c', 'NART_c', 'RPM_c',
                     'RAVLT_Pct_Ret_c', 'BVMT_Pct_Ret_c']
    print("\nMain effect correlations:")
    print(df[main_effects].corr().round(3))
    
    return corr_matrix

def main():
    """Main execution function."""
    
    print("=" * 60)
    print("STEP 2: CENTER PREDICTORS AND CREATE INTERACTIONS")
    print("=" * 60)
    
    # Load merged data from Step 1
    input_file = OUTPUT_DIR / "step01_merged_data.csv"
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} participants from Step 1")
    
    # Center predictors
    df_centered = center_predictors(df)
    
    # Create interaction terms
    df_final, interaction_cols = create_interaction_terms(df_centered)
    
    # Check multicollinearity
    corr_matrix = check_multicollinearity(df_final)
    
    # Verify centering worked
    print("\n" + "=" * 60)
    print("CENTERING VERIFICATION")
    print("=" * 60)
    
    centered_vars = ['Age_c', 'RAVLT_c', 'BVMT_c', 'NART_c', 'RPM_c',
                      'RAVLT_Pct_Ret_c', 'BVMT_Pct_Ret_c']
    for var in centered_vars:
        valid_mean = df_final[var].dropna().mean()
        print(f"{var} mean: {valid_mean:.6f} (should be ~0)")
    
    # Save outputs
    df_final.to_csv(OUTPUT_DIR / "step02_centered_predictors.csv", index=False)
    corr_matrix.to_csv(OUTPUT_DIR / "step02_correlation_matrix.csv")
    
    print(f"\nOutputs saved:")
    print(f"  - {OUTPUT_DIR / 'step02_centered_predictors.csv'}")
    print(f"  - {OUTPUT_DIR / 'step02_correlation_matrix.csv'}")
    
    # Report final dataset info
    print(f"\nFinal dataset shape: {df_final.shape}")
    print(f"Complete cases (no missing): {df_final.dropna().shape[0]}")
    print(f"Cases with NART missing: {df_final['NART_c'].isnull().sum()}")
    
    print("\nStep 2 complete: Predictors centered and interactions created")
    
    return df_final

if __name__ == "__main__":
    main()