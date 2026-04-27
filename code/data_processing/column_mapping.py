"""
Column mapping and data cleaning for Ch7 analysis
Maps original column names to clean, code-friendly names
"""

import pandas as pd
import numpy as np

# Mapping for dfnonvr.csv (participant-level data)
NONVR_COLUMN_MAP = {
    # Cognitive tests
    'RAVLT trial 1 score': 'ravlt_t1',
    'RAVLT trial 2 score': 'ravlt_t2',
    'RAVLT trial 3 score': 'ravlt_t3',
    'RAVLT trial 4 score': 'ravlt_t4',
    'RAVLT trial 5 score': 'ravlt_t5',
    'RAVLT distraction trial score': 'ravlt_distraction',
    'RAVLT free recall score': 'ravlt_free_recall',
    'RAVLT delayed recall score': 'ravlt_delayed_recall',
    
    'BVMT trial 1 score': 'bvmt_t1',
    'BVMT trial 2 score': 'bvmt_t2',
    'BVMT trial 3 score': 'bvmt_t3',
    'BVMT delayed recall score': 'bvmt_delayed_recall',
    'BVMT total recall': 'bvmt_total',
    'BVMT learning': 'bvmt_learning',
    'BVMT percent retained': 'bvmt_percent_retained',
    
    'NART Score': 'nart_score',
    'RPM Score': 'rpm_score',
    
    # Demographics
    'Age in years': 'age',
    'Sex 0=female 1=male': 'sex',
    'Education level ( High school (Year 9 or lower)\n High school (Year 10)\n High school (Year 12)\n Certificate 1 & 2\n Certificate 3 & 4\n Diploma / Advanced Diploma\n Bachelors Degree\n Graduate Certificate / Diploma\n Masters Degree\n Doctoral Degree)': 'education_text',
    'VR Usage (Never\n Less than 1 hour\n 1 - 10 hours\n 10 - 50 hours\n More than 50 hours)': 'vr_experience_text',
    'Typical sleep hours': 'typical_sleep',
    
    # DASS
    'Total DASS Anxiety Items': 'dass_anxiety',
    'Total DASS Stress Items': 'dass_stress',
    # Note: DASS Depression seems to be missing
    
    # Keep UID as is
    'UID': 'uid'
}

# Mapping for dfdata.csv (test-level data)
DATA_COLUMN_MAP = {
    'UID': 'uid',
    'TEST': 'test_number',
    'Time since VR': 'tsvr_hours',
    'Hours slept night before': 'sleep_hours',
    'Sleep quality -1=bad 1=good': 'sleep_quality',
}

def clean_education(edu_text):
    """Convert education text to numeric years"""
    education_map = {
        'high school (year 9 or lower)': 9,
        'high school (year 10)': 10,
        'high school (year 12)': 12,
        'certificate 1 & 2': 12.5,
        'certificate 3 & 4': 13,
        'diploma / advanced diploma': 14,
        'bachelors degree': 16,
        'graduate certificate / diploma': 17,
        'masters degree': 18,
        'doctoral degree': 21
    }
    
    if pd.isna(edu_text):
        return np.nan
    
    edu_lower = str(edu_text).lower().strip()
    for key, value in education_map.items():
        if key in edu_lower:
            return value
    return np.nan

def clean_vr_experience(vr_text):
    """Convert VR experience text to numeric scale"""
    vr_map = {
        'never': 0,
        'less than 1 hour': 1,
        '1 - 10 hours': 2,
        '10 - 50 hours': 3,
        'more than 50 hours': 4
    }
    
    if pd.isna(vr_text):
        return np.nan
    
    vr_lower = str(vr_text).lower().strip()
    for key, value in vr_map.items():
        if key in vr_lower:
            return value
    return np.nan

def clean_dfnonvr():
    """Load and clean participant-level data"""
    df = pd.read_csv('data/dfnonvr.csv')
    
    # Rename columns
    df_clean = df[list(NONVR_COLUMN_MAP.keys())].copy()
    df_clean.columns = [NONVR_COLUMN_MAP[col] for col in df_clean.columns]
    
    # Convert education and VR experience to numeric
    df_clean['education_years'] = df_clean['education_text'].apply(clean_education)
    df_clean['vr_experience'] = df_clean['vr_experience_text'].apply(clean_vr_experience)
    
    # Drop text columns
    df_clean = df_clean.drop(['education_text', 'vr_experience_text'], axis=1)
    
    # Compute derived RAVLT scores
    df_clean['ravlt_total'] = df_clean[['ravlt_t1', 'ravlt_t2', 'ravlt_t3', 'ravlt_t4', 'ravlt_t5']].sum(axis=1)
    df_clean['ravlt_learning'] = df_clean['ravlt_t5'] - df_clean['ravlt_t1']
    df_clean['ravlt_forgetting'] = df_clean['ravlt_t5'] - df_clean['ravlt_delayed_recall']
    
    # Add DASS Depression as 0 (missing from data)
    df_clean['dass_depression'] = 0  # Placeholder since it's missing
    
    return df_clean

def clean_dfdata():
    """Load and clean test-level data"""
    df = pd.read_csv('data/dfdata.csv')
    
    # Start with mapped columns
    df_clean = pd.DataFrame()
    for old_col, new_col in DATA_COLUMN_MAP.items():
        if old_col in df.columns:
            df_clean[new_col] = df[old_col]
    
    # Add all TQ_ columns (test questions) with cleaner names
    tq_cols = [col for col in df.columns if col.startswith('TQ_')]
    for col in tq_cols:
        # Clean up TQ column names (e.g., TQ_RFR-N-OBJ1 -> tq_rfr_n_obj1)
        clean_name = col.lower().replace('-', '_')
        df_clean[clean_name] = df[col]
    
    # Add confidence columns if they exist (currently none found)
    # Will need to identify these later
    
    return df_clean

if __name__ == "__main__":
    # Clean both dataframes
    print("Cleaning dfnonvr.csv...")
    df_nonvr_clean = clean_dfnonvr()
    df_nonvr_clean.to_csv('data/dfnonvr_clean.csv', index=False)
    print(f"Saved clean participant data: {df_nonvr_clean.shape}")
    print(f"Columns: {list(df_nonvr_clean.columns)[:20]}...")
    
    print("\nCleaning dfdata.csv...")
    df_data_clean = clean_dfdata()
    df_data_clean.to_csv('data/dfdata_clean.csv', index=False)
    print(f"Saved clean test data: {df_data_clean.shape}")
    print(f"Columns: {list(df_data_clean.columns)[:10]}...")