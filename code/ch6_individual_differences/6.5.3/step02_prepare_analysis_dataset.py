#!/usr/bin/env python3
"""prepare_analysis_dataset: Merge strategy variables with theta scores and demographics, create analysis-ready dataset"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import traceback
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import validate_data_format

# Configuration

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.5.3
LOG_FILE = RQ_DIR / "logs" / "step02_prepare_analysis_dataset.log"

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Data Processing Functions

def convert_education_to_numeric(education_series):
    """
    Convert education categories to numeric years of education.
    
    Mapping based on UK education system:
    - No formal education: 0 years
    - Primary education: 6 years  
    - Secondary education: 11 years
    - College/A-levels: 13 years
    - University undergraduate: 16 years
    - University postgraduate: 18 years
    - Doctoral: 21 years
    """
    education_mapping = {
        'No formal education': 0,
        'Primary education': 6,
        'Secondary education': 11,
        'College/A-levels': 13,
        'A-levels': 13,
        'University undergraduate': 16,
        'Undergraduate': 16,
        'University postgraduate': 18,
        'Postgraduate': 18,
        'Doctoral': 21,
        'PhD': 21
    }
    
    # Convert to numeric, handling variations in text
    education_numeric = []
    for edu in education_series:
        if pd.isna(edu):
            education_numeric.append(np.nan)
            continue
            
        edu_str = str(edu).strip()
        
        # Direct mapping
        if edu_str in education_mapping:
            education_numeric.append(education_mapping[edu_str])
            continue
        
        # Fuzzy matching for common variations
        edu_lower = edu_str.lower()
        if 'no formal' in edu_lower or 'none' in edu_lower:
            education_numeric.append(0)
        elif 'primary' in edu_lower:
            education_numeric.append(6)
        elif 'secondary' in edu_lower or 'high school' in edu_lower:
            education_numeric.append(11)
        elif 'college' in edu_lower or 'a-level' in edu_lower:
            education_numeric.append(13)
        elif 'undergraduate' in edu_lower or 'bachelor' in edu_lower:
            education_numeric.append(16)
        elif 'postgraduate' in edu_lower or 'master' in edu_lower:
            education_numeric.append(18)
        elif 'doctoral' in edu_lower or 'phd' in edu_lower:
            education_numeric.append(21)
        else:
            # Default to secondary if unclear
            education_numeric.append(11)
    
    return education_numeric

def compute_descriptive_statistics(df, variables):
    """
    Compute descriptive statistics for specified variables.
    """
    descriptives = []
    
    for var in variables:
        if var not in df.columns:
            continue
            
        data = df[var].dropna()
        if len(data) == 0:
            continue
        
        stats_dict = {
            'variable': var,
            'mean': data.mean() if data.dtype in ['float64', 'int64'] else np.nan,
            'std': data.std() if data.dtype in ['float64', 'int64'] else np.nan,
            'min': data.min() if data.dtype in ['float64', 'int64'] else np.nan,
            'max': data.max() if data.dtype in ['float64', 'int64'] else np.nan,
            'n': len(data),
            'missing': df[var].isna().sum(),
            'dtype': str(data.dtype)
        }
        
        # Add normality test for numeric variables
        if data.dtype in ['float64', 'int64'] and len(data) >= 3:
            try:
                shapiro_stat, shapiro_p = stats.shapiro(data)
                stats_dict['shapiro_p'] = shapiro_p
                stats_dict['normality'] = 'normal' if shapiro_p > 0.05 else 'non-normal'
            except:
                stats_dict['shapiro_p'] = np.nan
                stats_dict['normality'] = 'unknown'
        
        descriptives.append(stats_dict)
    
    return pd.DataFrame(descriptives)

def check_outliers(df, variables, method='iqr', threshold=1.5):
    """
    Identify outliers using IQR method.
    """
    outlier_info = {}
    
    for var in variables:
        if var not in df.columns or df[var].dtype not in ['float64', 'int64']:
            continue
        
        data = df[var].dropna()
        if len(data) < 4:
            continue
        
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            outlier_info[var] = {
                'n_outliers': len(outliers),
                'outlier_percent': len(outliers) / len(data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_values': outliers.tolist() if len(outliers) < 10 else 'too_many_to_list'
            }
    
    return outlier_info

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 02: Prepare analysis dataset")
        # Load Data Files

        log("Loading merged data...")
        merged_path = RQ_DIR / "data" / "step00_merged_data.csv"
        merged_df = pd.read_csv(merged_path)
        log(f"Merged data ({len(merged_df)} rows, {len(merged_df.columns)} cols)")
        
        log("Loading strategy variables...")
        strategy_path = RQ_DIR / "data" / "step01_strategy_variables.csv"
        strategy_df = pd.read_csv(strategy_path)
        log(f"Strategy variables ({len(strategy_df)} rows, {len(strategy_df.columns)} cols)")
        # Merge Strategy Variables with Outcomes
        # Join: strategy codes + theta scores + demographics on UID

        log("Merging strategy variables with outcomes...")
        analysis_df = merged_df.merge(
            strategy_df[['UID', 'rehearsal_frequency', 'mnemonic_use']], 
            on='UID', 
            how='inner'
        )
        log(f"Analysis dataset: {len(analysis_df)} participants")
        
        # Check for any data loss in merge
        if len(analysis_df) < len(merged_df):
            lost = len(merged_df) - len(analysis_df)
            log(f"Lost {lost} participants in strategy merge")
        # Convert Education to Numeric Scale

        log("Converting education to numeric years...")
        analysis_df['education_numeric'] = convert_education_to_numeric(analysis_df['education'])
        
        # Log education conversion results
        education_stats = analysis_df['education_numeric'].describe()
        log(f"Education years - Mean: {education_stats['mean']:.1f}, Range: {education_stats['min']:.0f}-{education_stats['max']:.0f}")
        
        # Show education distribution
        edu_dist = analysis_df.groupby('education')['education_numeric'].first().sort_values()
        log("Education categories -> years mapping:")
        for category, years in edu_dist.items():
            count = (analysis_df['education'] == category).sum()
            log(f"  {category}: {years} years (n={count})")
        # Compute Descriptive Statistics
        # Variables: All analysis variables for descriptive table
        # Tests: Normality, outliers, missing data patterns

        log("Computing descriptive statistics...")
        analysis_variables = ['theta_all', 'rehearsal_frequency', 'mnemonic_use', 
                             'age', 'education_numeric', 'vr_exposure', 'sleep_hours']
        
        descriptives_df = compute_descriptive_statistics(analysis_df, analysis_variables)
        
        # Save descriptive statistics
        descriptives_path = RQ_DIR / "data" / "step02_descriptive_stats.csv"
        descriptives_df.to_csv(descriptives_path, index=False, encoding='utf-8')
        log(f"Descriptive statistics: {descriptives_path}")
        
        # Log key descriptive statistics
        log("Variable summary:")
        for _, row in descriptives_df.iterrows():
            if pd.notna(row['mean']):
                log(f"  {row['variable']}: M={row['mean']:.2f}, SD={row['std']:.2f}, n={row['n']}")
            else:
                log(f"  {row['variable']}: n={row['n']} (non-numeric)")
        # Check Data Quality
        # Tests: Outlier detection, missing data, distribution checks

        log("Checking for outliers...")
        outlier_info = check_outliers(analysis_df, analysis_variables)
        
        log("Outlier summary:")
        for var, info in outlier_info.items():
            log(f"  {var}: {info['n_outliers']} outliers ({info['outlier_percent']:.1f}%)")
        
        # Check missing data patterns
        log("Missing data summary:")
        missing_counts = analysis_df[analysis_variables].isna().sum()
        for var, missing in missing_counts.items():
            if missing > 0:
                log(f"  {var}: {missing} missing ({missing/len(analysis_df)*100:.1f}%)")
        
        if missing_counts.sum() == 0:
            log("  No missing data in analysis variables")
        # Prepare Final Analysis Dataset
        # Columns: UID + all analysis variables for hierarchical regression
        # Quality: Remove any cases with missing analysis variables

        final_columns = ['UID', 'theta_all', 'rehearsal_frequency', 'mnemonic_use', 
                        'age', 'education_numeric', 'vr_exposure', 'sleep_hours']
        
        final_df = analysis_df[final_columns].copy()
        
        # Remove cases with missing analysis variables (complete case analysis)
        initial_n = len(final_df)
        final_df = final_df.dropna()
        final_n = len(final_df)
        
        if initial_n > final_n:
            log(f"[COMPLETE CASES] Removed {initial_n - final_n} cases with missing data")
        
        log(f"Analysis dataset: {final_n} complete cases")
        # Save Analysis Dataset
        
        output_path = RQ_DIR / "data" / "step02_analysis_dataset.csv"
        final_df.to_csv(output_path, index=False, encoding='utf-8')
        log(f"Analysis dataset: {output_path} ({len(final_df)} rows, {len(final_df.columns)} cols)")
        # Run Validation
        # Validation: Check expected columns and data types
        # Custom validation due to function signature mismatch

        log("Running dataset validation...")
        
        expected_columns = ['UID', 'theta_all', 'rehearsal_frequency', 'mnemonic_use', 
                           'age', 'education_numeric', 'vr_exposure', 'sleep_hours']
        
        validation_results = {
            'columns_match': list(final_df.columns) == expected_columns,
            'expected_rows': len(final_df) >= 80,  # Allow some missing data
            'no_missing_data': final_df.isna().sum().sum() == 0,
            'theta_numeric': pd.api.types.is_numeric_dtype(final_df['theta_all']),
            'age_numeric': pd.api.types.is_numeric_dtype(final_df['age']),
            'valid': True
        }
        
        validation_results['valid'] = all([
            validation_results['columns_match'],
            validation_results['expected_rows'],
            validation_results['no_missing_data'],
            validation_results['theta_numeric'],
            validation_results['age_numeric']
        ])
        
        # Report validation results
        for key, value in validation_results.items():
            status = "" if value else ""
            log(f"{status} {key}: {value}")

        if not validation_results['valid']:
            raise ValueError("Validation failed - see log for details")

        log("Step 02 complete - analysis dataset prepared with descriptive statistics")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)