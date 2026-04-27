#!/usr/bin/env python3
"""
Step 01: Extract and Prepare Domain-Specific Data (v2 - Percent Retention)
RQ: ch7/7.1.3
Purpose: Extract domain-specific theta scores from Ch5 outputs and merge with cognitive test data
Output: Domain theta scores and merged dataset for analysis

v2 CHANGES (2026-03-22):
1. RAVLT ceiling fix: participants with unadministered trials stored as 0.
   Substitutes 15 where trial N == 0 and trial N-1 >= 14 (ceiling performance).
2. BVMT Total recomputed explicitly from sum(trials 1-3) instead of pre-computed column.
3. Added RAVLT Percent Retention (Delayed Recall / best available trial x 100).
4. Added BVMT Percent Retained (from pre-computed column in dfnonvr.csv).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy import stats

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # Go up 4 levels from code file
sys.path.insert(0, str(PROJECT_ROOT))

# Import missing data utilities
try:
    sys.path.insert(0, str(PROJECT_ROOT / "results" / "ch7"))
    from missing_data_handler import analyze_missing_pattern, create_missing_data_report
except ImportError:
    # Utilities not available - continue without
    pass


# =============================================================================
# Configuration
# =============================================================================
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch7/7.1.3
LOG_FILE = RQ_DIR / "logs" / "step01_extract_data.log"

# Output files
OUTPUT_THETA = RQ_DIR / "data" / "step01_domain_theta_scores.csv"
OUTPUT_MERGED = RQ_DIR / "data" / "step01_merged_dataset.csv"
OUTPUT_STATS = RQ_DIR / "data" / "step01_descriptive_stats.csv"

# Ensure directories exist
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_THETA.parent.mkdir(parents=True, exist_ok=True)

def log(msg):
    """Write to both log file and console."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

def compute_t_score(raw_scores):
    """Convert raw scores to T-scores (M=50, SD=10)."""
    if len(raw_scores) == 0:
        return []
    mean = np.nanmean(raw_scores)
    std = np.nanstd(raw_scores)
    if std == 0:
        return np.full_like(raw_scores, 50.0)
    return 50 + 10 * (raw_scores - mean) / std

def fix_ravlt_ceiling(df, log_fn):
    """Fix RAVLT ceiling effects: substitute 15 for unadministered trials (stored as 0).

    Logic: If a participant scored >= 14 on trial N-1 and trial N == 0,
    trial N was not administered (ceiling). Substitute 15.
    Affected: A103 (trials 4,5), A064 (trial 5).
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    fixes_applied = 0

    for idx in df.index:
        for i in range(2, 5):  # Check trials 3,4,5 (index 2,3,4)
            current_col = trial_cols[i]
            prev_col = trial_cols[i - 1]
            if df.at[idx, current_col] == 0 and df.at[idx, prev_col] >= 14:
                uid = df.at[idx, 'UID']
                old_val = df.at[idx, current_col]
                df.at[idx, current_col] = 15
                fixes_applied += 1
                log_fn(f"[CEILING FIX] {uid}: {current_col} {old_val} -> 15 "
                       f"(prev trial = {df.at[idx, prev_col]})")

    log_fn(f"[CEILING FIX] Total fixes applied: {fixes_applied}")
    return df

def compute_ravlt_percent_retention(df, log_fn):
    """Compute RAVLT Percent Retention = Delayed Recall / best available trial x 100.

    Denominator: last non-zero trial score (trial 5 -> 4 -> 3 fallback).
    After ceiling fix, this should always be trial 5 for all participants.
    """
    trial_cols = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
    dr_col = 'ravlt-delayed-recall-score'

    pct_ret = np.full(len(df), np.nan)
    for i, idx in enumerate(df.index):
        dr = df.at[idx, dr_col]
        # Find best available trial (last non-zero, working backwards)
        denom = np.nan
        for trial_col in reversed(trial_cols):
            val = df.at[idx, trial_col]
            if val > 0:
                denom = val
                break
        if denom > 0:
            pct_ret[i] = (dr / denom) * 100

    n_valid = np.sum(~np.isnan(pct_ret))
    log_fn(f"[COMPUTED] RAVLT Percent Retention: {n_valid}/{len(df)} valid, "
           f"M={np.nanmean(pct_ret):.1f}%, SD={np.nanstd(pct_ret):.1f}%")
    return pct_ret

# =============================================================================
# Main Analysis
# =============================================================================

if __name__ == "__main__":
    try:
        log("[START] Step 01: Extract and Prepare Domain-Specific Data")
        log(f"[SETUP] RQ Directory: {RQ_DIR}")
        
        # =========================================================================
        # STEP 1: Load Ch5 5.2.1 theta scores
        # =========================================================================
        log("\n[STEP 1] Loading Ch5 5.2.1 domain theta scores...")
        
        ch5_theta_file = PROJECT_ROOT / "results" / "ch5" / "5.2.1" / "data" / "step03_theta_scores.csv"
        theta_df = pd.read_csv(ch5_theta_file)
        
        log(f"[INFO] Loaded theta scores: {theta_df.shape}")
        log(f"[INFO] Columns: {theta_df.columns.tolist()}")
        
        # Extract UID from composite_ID (format: UID_test)
        theta_df['UID'] = theta_df['composite_ID'].str.split('_').str[0]
        theta_df['test'] = theta_df['composite_ID'].str.split('_').str[1]
        
        log(f"[INFO] Unique participants: {theta_df['UID'].nunique()}")
        log(f"[INFO] Unique tests: {theta_df['test'].unique().tolist()}")
        
        # =========================================================================
        # STEP 2: Aggregate theta scores by UID and domain
        # =========================================================================
        log("\n[STEP 2] Aggregating theta scores by UID and domain...")
        
        # Calculate mean theta scores across tests for each domain
        theta_agg = theta_df.groupby('UID').agg({
            'theta_what': 'mean',
            'theta_where': 'mean',
            'theta_when': 'mean'
        }).reset_index()
        
        log(f"[INFO] Aggregated theta scores: {theta_agg.shape}")
        
        # Reshape to long format for domain-specific analysis
        theta_long = pd.melt(
            theta_agg,
            id_vars=['UID'],
            value_vars=['theta_what', 'theta_where', 'theta_when'],
            var_name='domain_col',
            value_name='theta_mean'
        )
        
        # Clean domain names
        theta_long['domain'] = theta_long['domain_col'].str.replace('theta_', '').str.capitalize()
        theta_long = theta_long.drop('domain_col', axis=1)
        
        # Calculate standard errors (using SD across tests)
        theta_se = theta_df.groupby('UID').agg({
            'theta_what': lambda x: x.std() / np.sqrt(len(x)),
            'theta_where': lambda x: x.std() / np.sqrt(len(x)),
            'theta_when': lambda x: x.std() / np.sqrt(len(x))
        }).reset_index()
        
        theta_se_long = pd.melt(
            theta_se,
            id_vars=['UID'],
            value_vars=['theta_what', 'theta_where', 'theta_when'],
            var_name='domain_col',
            value_name='theta_se'
        )
        theta_se_long['domain'] = theta_se_long['domain_col'].str.replace('theta_', '').str.capitalize()
        theta_se_long = theta_se_long.drop('domain_col', axis=1)
        
        # Merge mean and SE
        domain_theta = pd.merge(
            theta_long,
            theta_se_long[['UID', 'domain', 'theta_se']],
            on=['UID', 'domain']
        )
        
        log(f"[INFO] Domain theta scores shape: {domain_theta.shape}")
        log(f"[INFO] Domains: {domain_theta['domain'].unique().tolist()}")
        
        # Save domain theta scores
        domain_theta.to_csv(OUTPUT_THETA, index=False)
        log(f"[OUTPUT] Domain theta scores saved to: {OUTPUT_THETA}")
        
        # =========================================================================
        # STEP 3: Load and prepare cognitive test data (v2 with ceiling fix + percent retention)
        # =========================================================================
        log("\n[STEP 3] Loading cognitive test data from dfnonvr.csv...")

        dfnonvr_file = PROJECT_ROOT / "data" / "dfnonvr.csv"
        df_cog = pd.read_csv(dfnonvr_file)

        log(f"[INFO] Loaded cognitive data: {df_cog.shape}")

        # --- RAVLT ceiling fix BEFORE computing totals ---
        log("\n[CEILING] Applying RAVLT ceiling fix (0 -> 15 for unadministered trials)...")
        df_cog = fix_ravlt_ceiling(df_cog, log)

        # --- RAVLT Total: Sum trials 1-5 (after ceiling fix) ---
        ravlt_trials = [f'ravlt-trial-{i}-score' for i in range(1, 6)]
        missing_cols = [c for c in ravlt_trials if c not in df_cog.columns]
        if missing_cols:
            raise ValueError(f"Missing RAVLT trial columns: {missing_cols}")

        df_cog['RAVLT_Total'] = df_cog[ravlt_trials].sum(axis=1)
        log(f"[SUCCESS] RAVLT Total (ceiling-fixed): "
            f"M={df_cog['RAVLT_Total'].mean():.1f}, SD={df_cog['RAVLT_Total'].std():.1f}")

        # --- RAVLT Percent Retention (new predictor) ---
        df_cog['RAVLT_Pct_Ret'] = compute_ravlt_percent_retention(df_cog, log)

        # --- BVMT Total: Explicitly sum trials 1-3 ---
        bvmt_trial_cols = [f'bvmt-trial-{i}-score' for i in range(1, 4)]
        missing_cols = [c for c in bvmt_trial_cols if c not in df_cog.columns]
        if missing_cols:
            raise ValueError(f"Missing BVMT trial columns: {missing_cols}")

        df_cog['BVMT_Total'] = df_cog[bvmt_trial_cols].sum(axis=1)
        log(f"[SUCCESS] BVMT Total (sum trials 1-3): "
            f"M={df_cog['BVMT_Total'].mean():.1f}, SD={df_cog['BVMT_Total'].std():.1f}")

        # --- BVMT Percent Retained (new predictor, pre-computed in CSV) ---
        if 'bvmt-percent-retained' not in df_cog.columns:
            raise ValueError("Column 'bvmt-percent-retained' not found in dfnonvr.csv")
        df_cog['BVMT_Pct_Ret'] = df_cog['bvmt-percent-retained']
        log(f"[SUCCESS] BVMT Percent Retained: "
            f"M={df_cog['BVMT_Pct_Ret'].mean():.1f}%, SD={df_cog['BVMT_Pct_Ret'].std():.1f}%")

        # --- RPM ---
        if 'rpm-score' in df_cog.columns:
            df_cog['RPM_Total'] = df_cog['rpm-score']
            log(f"[SUCCESS] RPM score extracted")
        else:
            raise ValueError("RPM score column not found")

        # --- Convert ALL scores to T-scores ---
        log("\n[PROCESS] Converting raw scores to T-scores (M=50, SD=10)...")

        raw_to_t = {
            'RAVLT_Total': 'RAVLT_T',
            'RAVLT_Pct_Ret': 'RAVLT_Pct_Ret_T',
            'BVMT_Total': 'BVMT_T',
            'BVMT_Pct_Ret': 'BVMT_Pct_Ret_T',
            'RPM_Total': 'RPM_T',
        }

        for raw_col, t_col in raw_to_t.items():
            df_cog[t_col] = compute_t_score(df_cog[raw_col].values)
            mean = df_cog[t_col].mean()
            std = df_cog[t_col].std()
            log(f"  {t_col:20} M={mean:.1f}, SD={std:.1f}")

        # Select relevant columns
        cog_cols = ['UID', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_T', 'BVMT_Pct_Ret_T', 'RPM_T']
        existing_cols = [col for col in cog_cols if col in df_cog.columns]

        cognitive_df = df_cog[existing_cols].copy()

        log(f"[INFO] Cognitive test data shape: {cognitive_df.shape}")
        log(f"[INFO] Available columns: {cognitive_df.columns.tolist()}")
        
        # =========================================================================
        # STEP 4: Merge domain theta with cognitive tests
        # =========================================================================
        log("\n[STEP 4] Merging domain theta scores with cognitive tests...")
        
        # Ensure UID is string in both dataframes
        domain_theta['UID'] = domain_theta['UID'].astype(str)
        cognitive_df['UID'] = cognitive_df['UID'].astype(str)
        
        # Merge
        merged_df = pd.merge(
            domain_theta,
            cognitive_df,
            on='UID',
            how='inner'
        )
        
        log(f"[INFO] Merged dataset shape: {merged_df.shape}")
        log(f"[INFO] Unique participants: {merged_df['UID'].nunique()}")
        log(f"[INFO] Domains: {merged_df['domain'].value_counts().to_dict()}")
        
        # Check for missing values
        missing_counts = merged_df.isnull().sum()
        if missing_counts.any():
            log(f"[WARNING] Missing values detected:")
            for col, count in missing_counts[missing_counts > 0].items():
                log(f"  - {col}: {count} missing")
        else:
            log(f"[INFO] No missing values in merged dataset")
            
        # Save merged dataset
        merged_df.to_csv(OUTPUT_MERGED, index=False)
        log(f"[OUTPUT] Merged dataset saved to: {OUTPUT_MERGED}")
        
        # =========================================================================
        # STEP 5: Calculate descriptive statistics by domain
        # =========================================================================
        log("\n[STEP 5] Calculating descriptive statistics...")
        
        stats_list = []
        for domain in ['What', 'Where', 'When']:
            domain_data = merged_df[merged_df['domain'] == domain]
            
            stats_dict = {
                'domain': domain,
                'n': len(domain_data),
                'theta_mean': domain_data['theta_mean'].mean(),
                'theta_sd': domain_data['theta_mean'].std(),
                'theta_min': domain_data['theta_mean'].min(),
                'theta_max': domain_data['theta_mean'].max(),
                'theta_q25': domain_data['theta_mean'].quantile(0.25),
                'theta_q50': domain_data['theta_mean'].quantile(0.50),
                'theta_q75': domain_data['theta_mean'].quantile(0.75)
            }
            
            stats_list.append(stats_dict)
            log(f"[INFO] {domain}: n={stats_dict['n']}, mean={stats_dict['theta_mean']:.3f}, sd={stats_dict['theta_sd']:.3f}")
            
        stats_df = pd.DataFrame(stats_list)
        stats_df.to_csv(OUTPUT_STATS, index=False)
        log(f"[OUTPUT] Descriptive statistics saved to: {OUTPUT_STATS}")
        
        # Check for outliers using IQR method
        log("\n[INFO] Checking for outliers (IQR method)...")
        for domain in ['What', 'Where', 'When']:
            domain_data = merged_df[merged_df['domain'] == domain]['theta_mean']
            Q1 = domain_data.quantile(0.25)
            Q3 = domain_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = domain_data[(domain_data < lower_bound) | (domain_data > upper_bound)]
            if len(outliers) > 0:
                log(f"[WARNING] {domain}: {len(outliers)} outliers detected (bounds: [{lower_bound:.3f}, {upper_bound:.3f}])")
            else:
                log(f"[INFO] {domain}: No outliers detected")
                
        log("\n[COMPLETE] Step 01 completed successfully")
        log(f"[SUMMARY] Created {len(merged_df)} records (100 participants × 3 domains)")
        
    except Exception as e:
        log(f"[CRITICAL ERROR] Unexpected error: {e}")
        import traceback
        log(f"[TRACEBACK] {traceback.format_exc()}")
        sys.exit(1)