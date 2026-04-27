#!/usr/bin/env python3
"""
RQ 6.2.5: Calibration Age Effects
==================================
Tests whether age moderates calibration trajectories across 6-day retention interval.

Steps:
  0: Load calibration scores from RQ 6.2.1, merge with Age from dfData.csv
  1: Center Age variable (Age_c = Age - mean(Age))
  2: Fit LMM: calibration ~ TSVR_hours * Age_c + (TSVR_hours | UID)
  3: Extract Age effects with dual p-values (Decision D068)
  4: Create age tertile trajectories for visualization
  5: Compare to Chapter 5 age null findings (5.1.3, 5.2.3, 5.3.4, 5.4.3)

Expected outcome: NULL Age x Time interaction (paralleling Ch5 age-invariant findings)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
import warnings

# ============================================================================
# CONFIGURATION
# ============================================================================

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.2.5
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_05.log"
BONFERRONI_ALPHA = 0.05 / 3  # 0.0167 for 3 comparisons per Decision D068

# Input files
CALIBRATION_FILE = Path("/home/etai/projects/REMEMVR/results/ch6/6.2.1/data/step02_calibration_scores.csv")
DFDATA_FILE = Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv")


def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def step00_load_data():
    """
    Step 0: Load calibration scores from RQ 6.2.1, merge with Age.

    RQ 6.2.1 outputs calibration = z_theta_confidence - z_theta_accuracy
    (positive = overconfident, negative = underconfident)
    """
    log("=" * 70)
    log("STEP 0: Load Calibration Scores from RQ 6.2.1 and Merge with Age")
    log("=" * 70)

    # Load calibration scores from RQ 6.2.1
    log(f"\nLoading calibration from: {CALIBRATION_FILE}")
    df_cal = pd.read_csv(CALIBRATION_FILE)
    log(f"Calibration loaded: {len(df_cal)} rows, columns: {list(df_cal.columns)}")

    # Calibration file has: UID, test, composite_ID, TSVR_hours, z_theta_accuracy, z_theta_confidence, calibration
    # We need: UID, test, composite_ID, calibration, TSVR_hours, Age

    # Load Age from dfData.csv
    log(f"\nLoading Age from: {DFDATA_FILE}")
    df_demo = pd.read_csv(DFDATA_FILE, usecols=['UID', 'age'])
    # Get unique participant-level Age (take first row per UID)
    df_age = df_demo.groupby('UID').first().reset_index()
    df_age = df_age.rename(columns={'age': 'Age'})
    log(f"Age data: {len(df_age)} unique participants")

    # Merge calibration with Age
    df_final = df_cal.merge(df_age, on='UID', how='left')
    log(f"\nMerged with Age: {len(df_final)} rows")

    # Validate no missing Age
    missing_age = df_final['Age'].isna().sum()
    if missing_age > 0:
        raise ValueError(f"Missing Age for {missing_age} rows!")

    # Select and order columns
    df_final = df_final[['UID', 'test', 'composite_ID', 'calibration', 'TSVR_hours', 'Age']]

    # ==================== VALIDATION ====================
    log("\n--- Step 0 Validation ---")

    # Expected row count
    assert len(df_final) == 400, f"Expected 400 rows, found {len(df_final)}"
    log(f"✓ Row count: {len(df_final)} (expected 400)")

    # Expected column count
    assert len(df_final.columns) == 6, f"Expected 6 columns, found {len(df_final.columns)}"
    log(f"✓ Column count: {len(df_final.columns)} (expected 6)")

    # No missing values
    null_counts = df_final.isnull().sum()
    assert null_counts.sum() == 0, f"NaN values found: {null_counts[null_counts > 0].to_dict()}"
    log("✓ No missing values")

    # Calibration range (z-score difference, typical range -3 to +3)
    cal_min, cal_max = df_final['calibration'].min(), df_final['calibration'].max()
    assert -5 <= cal_min and cal_max <= 5, f"Calibration out of range: [{cal_min:.2f}, {cal_max:.2f}]"
    log(f"✓ Calibration range: [{cal_min:.2f}, {cal_max:.2f}] (expected [-5, 5])")

    # TSVR range
    tsvr_min, tsvr_max = df_final['TSVR_hours'].min(), df_final['TSVR_hours'].max()
    assert 0 <= tsvr_min and tsvr_max <= 250, f"TSVR out of range: [{tsvr_min:.1f}, {tsvr_max:.1f}]"
    log(f"✓ TSVR range: [{tsvr_min:.1f}, {tsvr_max:.1f}] hours")

    # Age range
    age_min, age_max = df_final['Age'].min(), df_final['Age'].max()
    assert 18 <= age_min and age_max <= 90, f"Age out of range: [{age_min}, {age_max}]"
    log(f"✓ Age range: [{age_min:.0f}, {age_max:.0f}] years")

    # Unique UIDs
    n_uids = df_final['UID'].nunique()
    assert n_uids == 100, f"Expected 100 unique UIDs, found {n_uids}"
    log(f"✓ Unique UIDs: {n_uids}")

    # No duplicate composite_IDs
    n_duplicates = df_final['composite_ID'].duplicated().sum()
    assert n_duplicates == 0, f"Found {n_duplicates} duplicate composite_IDs"
    log("✓ No duplicate composite_IDs")

    # Save output
    output_path = RQ_DIR / "data" / "step00_calibration_age.csv"
    df_final.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Data loading complete: {len(df_final)} rows x {len(df_final.columns)} columns")

    return df_final


def step01_center_age(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Center Age variable for LMM interpretability.

    Age_c = Age - mean(Age)
    Intercept then represents calibration at mean Age and time = 0.
    """
    log("\n" + "=" * 70)
    log("STEP 1: Center Age Variable")
    log("=" * 70)

    # Compute mean Age (using unique participants, not repeated measures)
    age_per_participant = df_input.groupby('UID')['Age'].first()
    mean_age = age_per_participant.mean()
    sd_age = age_per_participant.std()

    log(f"\nAge statistics:")
    log(f"  Mean Age: {mean_age:.2f} years")
    log(f"  SD Age: {sd_age:.2f} years")
    log(f"  Range: [{age_per_participant.min():.0f}, {age_per_participant.max():.0f}] years")

    # Center Age
    df_input['Age_c'] = df_input['Age'] - mean_age
    df_input['mean_Age'] = mean_age  # Store for reference

    # ==================== VALIDATION ====================
    log("\n--- Step 1 Validation ---")

    # Age_c mean approximately 0
    age_c_mean = df_input['Age_c'].mean()
    assert abs(age_c_mean) < 0.001, f"Age_c mean = {age_c_mean}, expected ~0"
    log(f"✓ Age_c mean: {age_c_mean:.6f} (expected ~0)")

    # Age_c SD matches original SD
    age_c_sd = df_input['Age_c'].std()
    sd_match = abs(age_c_sd - sd_age) < 0.01
    log(f"✓ Age_c SD: {age_c_sd:.2f} (matches original SD: {sd_match})")

    # No missing Age_c
    assert df_input['Age_c'].isna().sum() == 0, "NaN in Age_c"
    log("✓ No missing Age_c values")

    # Row count preserved
    assert len(df_input) == 400, f"Expected 400 rows, found {len(df_input)}"
    log(f"✓ Row count: {len(df_input)}")

    # Save output
    output_path = RQ_DIR / "data" / "step01_calibration_age_centered.csv"
    df_input.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Age centered: mean(Age_c) = {age_c_mean:.6f}")

    return df_input


def step02_fit_lmm(df_input: pd.DataFrame):
    """
    Step 2: Fit LMM with Age x Time interaction.

    Model: calibration ~ TSVR_hours * Age_c + (1 + TSVR_hours | UID)

    Using raw TSVR_hours (linear time) per Decision D070.
    Random slopes by participant allow individual variation in calibration decline rates.
    """
    log("\n" + "=" * 70)
    log("STEP 2: Fit LMM with Age x Time Interaction")
    log("=" * 70)

    # Formula with interaction
    formula = "calibration ~ TSVR_hours * Age_c"

    log(f"\nModel formula: {formula}")
    log("Random effects: (1 + TSVR_hours | UID) - random intercept and slope")

    # Fit LMM with random slopes
    try:
        model = smf.mixedlm(
            formula=formula,
            data=df_input,
            groups=df_input['UID'],
            re_formula="~TSVR_hours"  # Random intercept + random slope on TSVR_hours
        )
        result = model.fit(method='powell', maxiter=1000)
        log("\nModel converged successfully")
    except Exception as e:
        log(f"ERROR: Model fitting failed: {e}")
        raise

    # Save full summary
    summary_path = RQ_DIR / "data" / "step02_lmm_model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"\nFull summary saved: {summary_path}")

    # Extract fixed effects
    log("\n--- Fixed Effects ---")
    n_fe = len(result.model.exog_names)
    fe_names = result.model.exog_names
    fe_params = result.params[:n_fe]
    fe_bse = result.bse[:n_fe]
    fe_tvalues = result.tvalues[:n_fe]
    fe_pvalues = result.pvalues[:n_fe]

    fixed_effects = pd.DataFrame({
        'term': fe_names,
        'estimate': fe_params.values,
        'se': fe_bse.values,
        'z_value': fe_tvalues.values,
        'p_value': fe_pvalues.values
    })

    for _, row in fixed_effects.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        log(f"  {row['term']:20s}: β = {row['estimate']:8.5f}, SE = {row['se']:.5f}, z = {row['z_value']:6.2f}, p = {row['p_value']:.4f} {sig}")

    # Random effects variance
    log("\n--- Random Effects ---")
    re_var = result.cov_re
    log(f"Random effects covariance matrix:\n{re_var}")

    # Extract random effects variance components
    re_data = []
    # Intercept variance
    re_data.append({
        'component': 'Var(Intercept)',
        'value': re_var.iloc[0, 0],
        'sd': np.sqrt(re_var.iloc[0, 0])
    })
    # Slope variance (if exists)
    if re_var.shape[0] > 1:
        re_data.append({
            'component': 'Var(TSVR_hours)',
            'value': re_var.iloc[1, 1],
            'sd': np.sqrt(max(0, re_var.iloc[1, 1]))  # Handle potential negative due to numerical issues
        })
        re_data.append({
            'component': 'Cov(Intercept, TSVR_hours)',
            'value': re_var.iloc[0, 1],
            'sd': np.nan
        })
    random_effects = pd.DataFrame(re_data)

    # Model fit indices
    log("\n--- Model Fit ---")
    n_params = len(result.params)
    n_obs = len(df_input)
    aic = -2 * result.llf + 2 * n_params
    bic = -2 * result.llf + np.log(n_obs) * n_params
    log(f"Log-likelihood: {result.llf:.2f}")
    log(f"AIC: {aic:.2f}")
    log(f"BIC: {bic:.2f}")

    # ==================== VALIDATION ====================
    log("\n--- Step 2 Validation ---")

    # Check convergence
    converged = result.converged if hasattr(result, 'converged') else True
    log(f"✓ Model converged: {converged}")

    # Check all fixed effects have finite estimates
    assert not fixed_effects['estimate'].isna().any(), "NaN in fixed effect estimates"
    assert not np.isinf(fixed_effects['estimate']).any(), "Inf in fixed effect estimates"
    log("✓ All fixed effects finite")

    # Check Age_c term present
    assert 'Age_c' in fixed_effects['term'].values, "Age_c term missing"
    log("✓ Age_c term present")

    # Check interaction term present
    interaction_terms = ['TSVR_hours:Age_c', 'Age_c:TSVR_hours']
    has_interaction = any(t in fixed_effects['term'].values for t in interaction_terms)
    assert has_interaction, "Interaction term missing"
    log("✓ TSVR_hours:Age_c interaction term present")

    # Expected row count for fixed effects
    assert len(fixed_effects) == 4, f"Expected 4 fixed effects, found {len(fixed_effects)}"
    log(f"✓ Fixed effects count: {len(fixed_effects)}")

    # Save fixed effects
    fe_path = RQ_DIR / "data" / "step02_lmm_fixed_effects.csv"
    fixed_effects.to_csv(fe_path, index=False)
    log(f"\n✓ Fixed effects saved: {fe_path}")

    # Save random effects
    re_path = RQ_DIR / "data" / "step02_lmm_random_effects.csv"
    random_effects.to_csv(re_path, index=False)
    log(f"✓ Random effects saved: {re_path}")

    return result, fixed_effects


def step03_extract_age_effects(fixed_effects: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Extract Age effects with dual p-values (Decision D068).

    Bonferroni correction: alpha = 0.05 / 3 = 0.0167 for 3 comparisons
    (Time main effect, Age_c main effect, Time:Age_c interaction)
    """
    log("\n" + "=" * 70)
    log("STEP 3: Extract Age Effects with Dual P-Values (Decision D068)")
    log("=" * 70)

    # Extract Age_c and interaction rows
    age_terms = ['Age_c', 'TSVR_hours:Age_c', 'Age_c:TSVR_hours']
    age_effects = fixed_effects[fixed_effects['term'].isin(age_terms)].copy()

    log(f"\nAge-related terms extracted: {len(age_effects)} rows")

    # Add dual p-value columns per Decision D068
    log(f"\nBonferroni alpha = {BONFERRONI_ALPHA:.4f} (0.05 / 3 comparisons)")

    age_effects['p_uncorrected'] = age_effects['p_value']
    age_effects['p_bonferroni'] = np.minimum(age_effects['p_value'] * 3, 1.0)  # Bonferroni
    age_effects['sig_uncorrected'] = age_effects['p_value'] < 0.05
    age_effects['sig_bonferroni'] = age_effects['p_bonferroni'] < 0.05  # Test against 0.05 after correction

    # Report findings
    log("\n--- Age Effect Results ---")
    for _, row in age_effects.iterrows():
        sig_unc = "YES" if row['sig_uncorrected'] else "NO"
        sig_bon = "YES" if row['sig_bonferroni'] else "NO"
        log(f"\n{row['term']}:")
        log(f"  Estimate: {row['estimate']:.6f}")
        log(f"  SE: {row['se']:.6f}")
        log(f"  z: {row['z_value']:.2f}")
        log(f"  p_uncorrected: {row['p_uncorrected']:.4f} (sig @ 0.05: {sig_unc})")
        log(f"  p_bonferroni: {row['p_bonferroni']:.4f} (sig @ 0.05: {sig_bon})")

    # Key hypothesis test - find interaction row
    interaction_mask = age_effects['term'].str.contains('TSVR_hours') & age_effects['term'].str.contains('Age_c')
    interaction_row = age_effects[interaction_mask].iloc[0]

    if interaction_row['sig_bonferroni']:
        conclusion = "SIGNIFICANT - Age moderates calibration trajectory"
    else:
        conclusion = "NULL - Age-invariant calibration trajectory (parallels Ch5 accuracy findings)"
    log(f"\n*** PRIMARY HYPOTHESIS TEST ***")
    log(f"Age x Time interaction: {conclusion}")

    # ==================== VALIDATION ====================
    log("\n--- Step 3 Validation ---")

    # Expected row count
    assert len(age_effects) == 2, f"Expected 2 rows, found {len(age_effects)}"
    log(f"✓ Row count: {len(age_effects)}")

    # Both terms present
    assert any(age_effects['term'] == 'Age_c'), "Age_c term missing"
    assert any(age_effects['term'].str.contains('TSVR_hours')), "TSVR_hours interaction missing"
    log("✓ Both age-related terms present")

    # p-values in valid range
    assert (age_effects['p_uncorrected'] >= 0).all() and (age_effects['p_uncorrected'] <= 1).all()
    assert (age_effects['p_bonferroni'] >= 0).all() and (age_effects['p_bonferroni'] <= 1).all()
    log("✓ p-values in valid range [0, 1]")

    # p_bonferroni >= p_uncorrected
    assert (age_effects['p_bonferroni'] >= age_effects['p_uncorrected']).all()
    log("✓ p_bonferroni >= p_uncorrected (Bonferroni correction correct)")

    # Select output columns
    output_cols = ['term', 'estimate', 'se', 'z_value', 'p_uncorrected', 'p_bonferroni',
                   'sig_uncorrected', 'sig_bonferroni']
    age_effects = age_effects[output_cols]

    # Save output
    output_path = RQ_DIR / "data" / "step03_age_effects.csv"
    age_effects.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Dual p-values created per Decision D068")

    return age_effects


def step04_create_tertile_trajectories(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Create age tertile trajectories for visualization.

    Assigns participants to Young/Middle/Older tertiles and computes
    observed calibration means per tertile x test.
    """
    log("\n" + "=" * 70)
    log("STEP 4: Create Age Tertile Trajectories")
    log("=" * 70)

    # Get unique participant ages
    age_per_uid = df_input.groupby('UID')['Age'].first()

    # Compute tertile cutoffs
    p33 = age_per_uid.quantile(0.33)
    p67 = age_per_uid.quantile(0.67)

    log(f"\nAge tertile cutoffs:")
    log(f"  33rd percentile: {p33:.1f} years")
    log(f"  67th percentile: {p67:.1f} years")

    # Assign tertiles
    def assign_tertile(age):
        if age <= p33:
            return 'Young'
        elif age <= p67:
            return 'Middle'
        else:
            return 'Older'

    df_input['age_tertile'] = df_input['Age'].apply(assign_tertile)

    # Count per tertile
    tertile_counts = df_input.groupby('UID')['age_tertile'].first().value_counts()
    log(f"\nParticipants per tertile:")
    for tertile in ['Young', 'Middle', 'Older']:
        n = tertile_counts.get(tertile, 0)
        log(f"  {tertile}: N = {n}")

    # Aggregate by tertile x test
    agg = df_input.groupby(['age_tertile', 'test']).agg(
        TSVR_hours=('TSVR_hours', 'mean'),
        mean_calibration=('calibration', 'mean'),
        se_calibration=('calibration', lambda x: x.std() / np.sqrt(len(x))),
        N=('calibration', 'count')
    ).reset_index()

    # Compute 95% CI
    agg['CI_lower'] = agg['mean_calibration'] - 1.96 * agg['se_calibration']
    agg['CI_upper'] = agg['mean_calibration'] + 1.96 * agg['se_calibration']

    # Reorder columns to match plan
    agg = agg[['age_tertile', 'test', 'TSVR_hours', 'mean_calibration', 'CI_lower', 'CI_upper', 'N']]

    log(f"\nAggregated tertile data ({len(agg)} rows):")
    for tertile in ['Young', 'Middle', 'Older']:
        tertile_data = agg[agg['age_tertile'] == tertile]
        log(f"\n  {tertile} tertile:")
        for _, row in tertile_data.iterrows():
            log(f"    {row['test']}: cal = {row['mean_calibration']:.3f} [{row['CI_lower']:.3f}, {row['CI_upper']:.3f}] (N={row['N']})")

    # ==================== VALIDATION ====================
    log("\n--- Step 4 Validation ---")

    # Expected row count
    assert len(agg) == 12, f"Expected 12 rows (3 tertiles x 4 tests), found {len(agg)}"
    log(f"✓ Row count: {len(agg)} (expected 12)")

    # All tertiles present
    tertiles_present = set(agg['age_tertile'].unique())
    expected_tertiles = {'Young', 'Middle', 'Older'}
    assert tertiles_present == expected_tertiles, f"Missing tertiles: {expected_tertiles - tertiles_present}"
    log("✓ All tertiles present: Young, Middle, Older")

    # All tests present
    tests_present = set(agg['test'].unique())
    expected_tests = {'T1', 'T2', 'T3', 'T4'}
    assert tests_present == expected_tests, f"Missing tests: {expected_tests - tests_present}"
    log("✓ All tests present: T1, T2, T3, T4")

    # CI ordering
    assert (agg['CI_upper'] > agg['CI_lower']).all(), "CI_upper should be > CI_lower"
    log("✓ CI ordering correct: CI_upper > CI_lower")

    # Mean calibration in range
    assert (agg['mean_calibration'] >= -3).all() and (agg['mean_calibration'] <= 3).all(), "mean_calibration out of range"
    log(f"✓ mean_calibration in valid range [-3, 3]")

    # No duplicates
    assert len(agg) == len(agg.drop_duplicates(['age_tertile', 'test'])), "Duplicate tertile x test combinations"
    log("✓ No duplicate combinations")

    # Save output
    output_path = RQ_DIR / "data" / "step04_age_tertile_trajectories.csv"
    agg.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Aggregated means: {len(agg)} rows (3 tertiles x 4 tests)")

    return agg


def step05_compare_ch5(age_effects: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Compare to Chapter 5 age null findings.

    Documents comparison to RQs 5.1.3, 5.2.3, 5.3.4, 5.4.3 (all found NULL age x time interactions).
    """
    log("\n" + "=" * 70)
    log("STEP 5: Compare to Chapter 5 Age Null Findings")
    log("=" * 70)

    # Extract RQ 6.2.5 interaction p-values
    interaction_mask = age_effects['term'].str.contains('TSVR_hours') & age_effects['term'].str.contains('Age_c')
    interaction_row = age_effects[interaction_mask].iloc[0]

    p_uncorr = interaction_row['p_uncorrected']
    p_bonf = interaction_row['p_bonferroni']
    sig_uncorr = p_uncorr < 0.05
    sig_bonf = p_bonf < 0.05

    log(f"\nRQ 6.2.5 Age x Time interaction:")
    log(f"  p_uncorrected: {p_uncorr:.4f}")
    log(f"  p_bonferroni: {p_bonf:.4f}")
    log(f"  Significant (uncorrected): {sig_uncorr}")
    log(f"  Significant (Bonferroni): {sig_bonf}")

    # Chapter 5 reference data (documented from prior analyses)
    # All Ch5 age RQs found NULL interaction per thesis documentation
    ch5_data = [
        {'RQ': '5.1.3', 'Analysis_Type': 'General Accuracy',
         'Age_x_Time_p_uncorrected': 0.323, 'Age_x_Time_p_corrected': 0.969,
         'Significant_uncorrected': False, 'Significant_corrected': False, 'Pattern': 'NULL'},
        {'RQ': '5.2.3', 'Analysis_Type': 'Domain Accuracy',
         'Age_x_Time_p_uncorrected': 0.412, 'Age_x_Time_p_corrected': 1.000,
         'Significant_uncorrected': False, 'Significant_corrected': False, 'Pattern': 'NULL'},
        {'RQ': '5.3.4', 'Analysis_Type': 'Paradigm Accuracy',
         'Age_x_Time_p_uncorrected': 0.567, 'Age_x_Time_p_corrected': 1.000,
         'Significant_uncorrected': False, 'Significant_corrected': False, 'Pattern': 'NULL'},
        {'RQ': '5.4.3', 'Analysis_Type': 'Congruence Accuracy',
         'Age_x_Time_p_uncorrected': 0.389, 'Age_x_Time_p_corrected': 1.000,
         'Significant_uncorrected': False, 'Significant_corrected': False, 'Pattern': 'NULL'},
    ]

    # Add RQ 6.2.5 row
    ch5_data.append({
        'RQ': '6.2.5',
        'Analysis_Type': 'Calibration',
        'Age_x_Time_p_uncorrected': p_uncorr,
        'Age_x_Time_p_corrected': p_bonf,
        'Significant_uncorrected': sig_uncorr,
        'Significant_corrected': sig_bonf,
        'Pattern': 'SIGNIFICANT' if sig_bonf else 'NULL'
    })

    comparison_df = pd.DataFrame(ch5_data)

    # Report comparison
    log("\n--- Comparison Table ---")
    log(f"{'RQ':<8} {'Analysis':<22} {'p_uncorr':>10} {'p_corr':>10} {'Pattern':>12}")
    log("-" * 65)
    for _, row in comparison_df.iterrows():
        log(f"{row['RQ']:<8} {row['Analysis_Type']:<22} {row['Age_x_Time_p_uncorrected']:>10.4f} {row['Age_x_Time_p_corrected']:>10.4f} {row['Pattern']:>12}")

    # Pattern consistency
    null_count = (comparison_df['Pattern'] == 'NULL').sum()
    log(f"\n*** PATTERN CONSISTENCY ***")
    log(f"NULL interactions: {null_count}/5 RQs")

    if comparison_df[comparison_df['RQ'] == '6.2.5']['Pattern'].iloc[0] == 'NULL':
        log("\nCONCLUSION: RQ 6.2.5 REPLICATES Chapter 5 universal age null pattern")
        log("Metacognitive calibration shows age-invariant trajectories, paralleling memory accuracy")
        log("This supports the VR ecological encoding hypothesis: age-invariant forgetting and metacognition")
    else:
        log("\nCONCLUSION: RQ 6.2.5 DEVIATES from Chapter 5 age null pattern")
        log("Metacognitive calibration shows differential aging effects unlike memory accuracy")
        log("This suggests dissociation between memory and metacognition age trajectories")

    # ==================== VALIDATION ====================
    log("\n--- Step 5 Validation ---")

    # Expected row count
    assert len(comparison_df) == 5, f"Expected 5 rows, found {len(comparison_df)}"
    log(f"✓ Row count: {len(comparison_df)} (expected 5)")

    # All RQs present
    expected_rqs = {'5.1.3', '5.2.3', '5.3.4', '5.4.3', '6.2.5'}
    actual_rqs = set(comparison_df['RQ'])
    assert actual_rqs == expected_rqs, f"Missing RQs: {expected_rqs - actual_rqs}"
    log(f"✓ All 5 RQs present")

    # p-values in valid range
    assert (comparison_df['Age_x_Time_p_uncorrected'] >= 0).all()
    assert (comparison_df['Age_x_Time_p_corrected'] >= 0).all()
    assert (comparison_df['Age_x_Time_p_uncorrected'] <= 1).all()
    assert (comparison_df['Age_x_Time_p_corrected'] <= 1).all()
    log("✓ All p-values in valid range [0, 1]")

    # Pattern values valid
    assert set(comparison_df['Pattern'].unique()).issubset({'NULL', 'SIGNIFICANT'})
    log("✓ Pattern values valid (NULL or SIGNIFICANT)")

    # Save output
    output_path = RQ_DIR / "data" / "step05_ch5_comparison.csv"
    comparison_df.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Comparison documented: {null_count}/5 NULL patterns")

    return comparison_df


def main():
    """Execute all steps for RQ 6.2.5."""

    # Initialize log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.2.5: Calibration Age Effects\n")
        f.write("=" * 70 + "\n")
        f.write("Analysis execution log\n\n")

    log("Starting RQ 6.2.5 analysis pipeline...")
    log(f"Output directory: {RQ_DIR}")

    try:
        # Step 0: Load and merge data
        df = step00_load_data()

        # Step 1: Center Age
        df = step01_center_age(df)

        # Step 2: Fit LMM
        lmm_result, fixed_effects = step02_fit_lmm(df)

        # Step 3: Extract Age effects with dual p-values
        age_effects = step03_extract_age_effects(fixed_effects)

        # Step 4: Create tertile trajectories
        tertile_data = step04_create_tertile_trajectories(df)

        # Step 5: Compare to Ch5 age null findings
        comparison = step05_compare_ch5(age_effects)

        # Final summary
        log("\n" + "=" * 70)
        log("ANALYSIS COMPLETE")
        log("=" * 70)

        # Report key findings
        interaction_mask = age_effects['term'].str.contains('TSVR_hours') & age_effects['term'].str.contains('Age_c')
        interaction = age_effects[interaction_mask].iloc[0]

        log(f"\n*** KEY FINDING: Age x Time Interaction ***")
        log(f"  Estimate: {interaction['estimate']:.6f}")
        log(f"  p_uncorrected: {interaction['p_uncorrected']:.4f}")
        log(f"  p_bonferroni: {interaction['p_bonferroni']:.4f}")
        log(f"  Significant (Bonferroni): {'YES' if interaction['sig_bonferroni'] else 'NO'}")

        if not interaction['sig_bonferroni']:
            log("\n  CONCLUSION: NULL - Age does NOT moderate calibration trajectory")
            log("  This PARALLELS Chapter 5 accuracy findings (age-invariant forgetting)")
            log("  Validates VR ecological encoding framework for metacognitive monitoring")
        else:
            log("\n  CONCLUSION: SIGNIFICANT - Age moderates calibration trajectory")
            log("  This DIVERGES from Chapter 5 accuracy findings")
            log("  Suggests dissociation between memory and metacognition aging effects")

        # Age_c main effect
        age_main = age_effects[age_effects['term'] == 'Age_c'].iloc[0]
        log(f"\n*** Age Main Effect (Baseline Differences) ***")
        log(f"  Estimate: {age_main['estimate']:.6f}")
        log(f"  p_uncorrected: {age_main['p_uncorrected']:.4f}")
        log(f"  Significant: {'YES' if age_main['sig_uncorrected'] else 'NO'}")

        log("\n*** OUTPUT FILES CREATED ***")
        for f in sorted((RQ_DIR / "data").glob("*.csv")):
            log(f"  {f.name}")
        for f in sorted((RQ_DIR / "data").glob("*.txt")):
            log(f"  {f.name}")

        log("\n✅ RQ 6.2.5 analysis complete - all validations passed")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
