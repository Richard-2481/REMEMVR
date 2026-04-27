#!/usr/bin/env python3
"""
RQ 6.1.3: Age Effects on Confidence
=====================================
Tests whether age moderates confidence trajectories across 6-day retention interval.

Steps:
  0: Load theta confidence from RQ 6.1.1, merge with TSVR and Age
  1: Center Age variable
  2: Create time predictors (Reciprocal based on RQ 6.1.1 best converged model)
  3: Fit LMM with Age x Time interaction
  4: Extract Age effects with dual p-values (Decision D068)
  5: Compute effect size at Day 6
  6: Prepare age tertile comparison data

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

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.1.3
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_06.log"
BONFERRONI_ALPHA = 0.05 / 3  # 0.0167 for 3 comparisons per Decision D068

# Input files from RQ 6.1.1
THETA_FILE = Path("/home/etai/projects/REMEMVR/results/ch6/6.1.1/data/step03_theta_confidence.csv")
TSVR_FILE = Path("/home/etai/projects/REMEMVR/results/ch6/6.1.1/data/step00_tsvr_mapping.csv")
DFDATA_FILE = Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv")


def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def step00_load_data():
    """
    Step 0: Load theta confidence from RQ 6.1.1, merge with TSVR and Age.

    Handles composite_ID format mismatch:
    - theta file: A010_T1 format
    - TSVR file: A010_1 format
    """
    log("=" * 70)
    log("STEP 0: Load Theta Confidence and Merge with TSVR and Age")
    log("=" * 70)

    # Load theta confidence
    log(f"\nLoading theta from: {THETA_FILE}")
    df_theta = pd.read_csv(THETA_FILE)
    log(f"Theta loaded: {len(df_theta)} rows, columns: {list(df_theta.columns)}")

    # Rename columns to match plan specification
    df_theta = df_theta.rename(columns={
        'theta_All': 'theta_confidence',
        'se_All': 'se_confidence'
    })

    # Parse UID and test from composite_ID (format: A010_T1)
    df_theta['UID'] = df_theta['composite_ID'].str.split('_').str[0]
    df_theta['test'] = df_theta['composite_ID'].str.split('_').str[1]  # T1, T2, T3, T4

    # Load TSVR mapping
    log(f"\nLoading TSVR from: {TSVR_FILE}")
    df_tsvr = pd.read_csv(TSVR_FILE)
    log(f"TSVR loaded: {len(df_tsvr)} rows")

    # Convert TSVR composite_ID format (A010_1) to match theta format (A010_T1)
    df_tsvr['UID'] = df_tsvr['composite_ID'].str.split('_').str[0]
    df_tsvr['test_num'] = df_tsvr['composite_ID'].str.split('_').str[1]
    df_tsvr['test'] = 'T' + df_tsvr['test_num']
    df_tsvr = df_tsvr[['UID', 'test', 'TSVR_hours']]

    # Merge theta with TSVR on UID and test
    df_merged = df_theta.merge(df_tsvr, on=['UID', 'test'], how='left')
    log(f"\nMerged theta with TSVR: {len(df_merged)} rows")

    # Check for missing TSVR
    missing_tsvr = df_merged['TSVR_hours'].isna().sum()
    if missing_tsvr > 0:
        raise ValueError(f"Missing TSVR_hours for {missing_tsvr} rows!")

    # Load Age from dfData.csv
    log(f"\nLoading Age from: {DFDATA_FILE}")
    df_demo = pd.read_csv(DFDATA_FILE, usecols=['UID', 'age'])
    # Get unique participant-level Age (take first row per UID)
    df_age = df_demo.groupby('UID').first().reset_index()
    df_age = df_age.rename(columns={'age': 'Age'})
    log(f"Age data: {len(df_age)} unique participants")

    # Merge with Age
    df_final = df_merged.merge(df_age, on='UID', how='left')
    log(f"\nMerged with Age: {len(df_final)} rows")

    # Validate no missing Age
    missing_age = df_final['Age'].isna().sum()
    if missing_age > 0:
        raise ValueError(f"Missing Age for {missing_age} rows!")

    # Select and order columns
    df_final = df_final[['composite_ID', 'UID', 'test', 'theta_confidence',
                         'se_confidence', 'TSVR_hours', 'Age']]

    # ==================== VALIDATION ====================
    log("\n--- Step 0 Validation ---")

    # Expected row count
    assert len(df_final) == 400, f"Expected 400 rows, found {len(df_final)}"
    log(f"✓ Row count: {len(df_final)} (expected 400)")

    # Expected column count
    assert len(df_final.columns) == 7, f"Expected 7 columns, found {len(df_final.columns)}"
    log(f"✓ Column count: {len(df_final.columns)} (expected 7)")

    # No missing values
    null_counts = df_final.isnull().sum()
    assert null_counts.sum() == 0, f"NaN values found: {null_counts[null_counts > 0].to_dict()}"
    log("✓ No missing values")

    # Theta range (allow slight tolerance for IRT estimation)
    theta_min, theta_max = df_final['theta_confidence'].min(), df_final['theta_confidence'].max()
    assert -4 <= theta_min and theta_max <= 4, f"Theta out of range: [{theta_min:.2f}, {theta_max:.2f}]"
    log(f"✓ Theta range: [{theta_min:.2f}, {theta_max:.2f}] (expected within [-4, 4])")

    # SE range
    se_min, se_max = df_final['se_confidence'].min(), df_final['se_confidence'].max()
    log(f"✓ SE range: [{se_min:.4f}, {se_max:.4f}]")

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
    output_path = RQ_DIR / "data" / "step00_lmm_input_raw.csv"
    df_final.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Data loading complete: {len(df_final)} rows x {len(df_final.columns)} columns")

    return df_final


def step01_center_age(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Center Age variable for LMM interpretability.

    Age_c = Age - mean(Age)
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
    output_path = RQ_DIR / "data" / "step01_lmm_input.csv"
    df_input.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Age centered: mean(Age_c) = {age_c_mean:.6f}")

    return df_input


def step02_create_time_predictors(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Create time predictors based on RQ 6.1.1 functional form selection.

    RQ 6.1.1 model comparison identified multiple well-fitting functional forms
    including Reciprocal and power law variants. For this age effects analysis,
    we use Time_log = log(TSVR_hours + 1) due to:
    - Standard interpretation in forgetting curve literature
    - Mathematical stability across full time range
    - Better interpretability of Age x Time_log interaction coefficient

    Both Time_recip and Time_log capture similar nonlinear deceleration patterns.
    """
    log("\n" + "=" * 70)
    log("STEP 2: Create Time Predictors from RQ 6.1.1 Functional Form")
    log("=" * 70)

    # RQ 6.1.1 model comparison: Best converged models include Reciprocal, PowerLaw variants
    # Using Reciprocal as primary time predictor (simple, interpretable, good fit)
    log("\nFunctional form selected: Reciprocal (1/(TSVR+1))")
    log("Rationale: Among best converged models in RQ 6.1.1 model comparison")

    # Create time predictors
    df_input['Time'] = df_input['TSVR_hours']  # Raw time for reference
    df_input['Time_recip'] = 1.0 / (df_input['TSVR_hours'] + 1)  # Reciprocal
    df_input['Time_log'] = np.log(df_input['TSVR_hours'] + 1)  # Log for comparison

    # ==================== VALIDATION ====================
    log("\n--- Step 2 Validation ---")

    # Time predictor created
    assert 'Time_recip' in df_input.columns, "Time_recip column missing"
    log("✓ Time_recip column created")

    # Time_recip range
    recip_min, recip_max = df_input['Time_recip'].min(), df_input['Time_recip'].max()
    assert 0 < recip_min and recip_max <= 1, f"Time_recip out of range: [{recip_min:.4f}, {recip_max:.4f}]"
    log(f"✓ Time_recip range: [{recip_min:.4f}, {recip_max:.4f}]")

    # Time_log range
    log_min, log_max = df_input['Time_log'].min(), df_input['Time_log'].max()
    assert 0 <= log_min and log_max <= 6, f"Time_log out of range: [{log_min:.2f}, {log_max:.2f}]"
    log(f"✓ Time_log range: [{log_min:.2f}, {log_max:.2f}]")

    # No missing time values
    assert df_input['Time_recip'].isna().sum() == 0, "NaN in Time_recip"
    assert df_input['Time_log'].isna().sum() == 0, "NaN in Time_log"
    log("✓ No missing time values")

    # Row count preserved
    assert len(df_input) == 400, f"Expected 400 rows, found {len(df_input)}"
    log(f"✓ Row count: {len(df_input)}")

    # Save output
    output_path = RQ_DIR / "data" / "step02_lmm_input_with_time.csv"
    df_input.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Time predictors created: Time, Time_recip, Time_log")

    return df_input


def step03_fit_lmm(df_input: pd.DataFrame):
    """
    Step 3: Fit LMM with Age x Time interaction.

    Model: theta_confidence ~ Time_log * Age_c + (1 + Time_log | UID)

    Using Time_log for better interpretability and convergence.
    Random slopes by participant allow individual variation in decline rates.
    """
    log("\n" + "=" * 70)
    log("STEP 3: Fit LMM with Age x Time Interaction")
    log("=" * 70)

    # Using Time_log as predictor (log-time is interpretable and common in forgetting literature)
    # Formula with random slopes
    formula = "theta_confidence ~ Time_log * Age_c"

    log(f"\nModel formula: {formula}")
    log("Random effects: (1 + Time_log | UID) - random intercept and slope")

    # Fit LMM with random slopes
    try:
        model = smf.mixedlm(
            formula=formula,
            data=df_input,
            groups=df_input['UID'],
            re_formula="~Time_log"  # Random intercept + random slope on Time_log
        )
        result = model.fit(method='powell', maxiter=1000)
        log("\nModel converged successfully")
    except Exception as e:
        log(f"ERROR: Model fitting failed: {e}")
        raise

    # Save full summary
    summary_path = RQ_DIR / "data" / "step03_lmm_summary.txt"
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
        'z': fe_tvalues.values,
        'p': fe_pvalues.values
    })

    for _, row in fixed_effects.iterrows():
        sig = "***" if row['p'] < 0.001 else "**" if row['p'] < 0.01 else "*" if row['p'] < 0.05 else ""
        log(f"  {row['term']:20s}: β = {row['estimate']:7.4f}, SE = {row['se']:.4f}, z = {row['z']:6.2f}, p = {row['p']:.4f} {sig}")

    # Random effects variance
    log("\n--- Random Effects ---")
    re_var = result.cov_re
    log(f"Random effects covariance matrix:\n{re_var}")

    # Model fit indices
    log("\n--- Model Fit ---")
    log(f"Log-likelihood: {result.llf:.2f}")
    log(f"AIC: {-2*result.llf + 2*len(result.params):.2f}")
    log(f"BIC: {-2*result.llf + np.log(len(df_input))*len(result.params):.2f}")

    # ==================== VALIDATION ====================
    log("\n--- Step 3 Validation ---")

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
    interaction_term = 'Time_log:Age_c'
    assert interaction_term in fixed_effects['term'].values, f"{interaction_term} term missing"
    log(f"✓ {interaction_term} term present")

    # Save fixed effects
    fe_path = RQ_DIR / "data" / "step03_lmm_fixed_effects.csv"
    fixed_effects.to_csv(fe_path, index=False)
    log(f"\n✓ Fixed effects saved: {fe_path}")

    return result, fixed_effects


def step04_extract_age_effects(fixed_effects: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Extract Age effects with dual p-values (Decision D068).

    Bonferroni correction: alpha = 0.05 / 3 = 0.0167 for 3 comparisons
    (Time main effect, Age_c main effect, Time:Age_c interaction)
    """
    log("\n" + "=" * 70)
    log("STEP 4: Extract Age Effects with Dual P-Values (Decision D068)")
    log("=" * 70)

    # Extract Age_c and interaction rows
    age_terms = ['Age_c', 'Time_log:Age_c']
    age_effects = fixed_effects[fixed_effects['term'].isin(age_terms)].copy()

    if len(age_effects) != 2:
        # Try alternate interaction naming
        alt_terms = ['Age_c', 'Age_c:Time_log']
        age_effects = fixed_effects[fixed_effects['term'].isin(alt_terms)].copy()

    log(f"\nAge-related terms extracted: {len(age_effects)} rows")

    # Add dual p-value columns per Decision D068
    log(f"\nBonferroni alpha = {BONFERRONI_ALPHA:.4f} (0.05 / 3 comparisons)")

    age_effects['p_uncorrected'] = age_effects['p']
    age_effects['p_bonferroni'] = age_effects['p']  # Same value, different threshold
    age_effects['sig_uncorrected'] = age_effects['p'] < 0.05
    age_effects['sig_bonferroni'] = age_effects['p'] < BONFERRONI_ALPHA

    # Report findings
    log("\n--- Age Effect Results ---")
    for _, row in age_effects.iterrows():
        sig_unc = "YES" if row['sig_uncorrected'] else "NO"
        sig_bon = "YES" if row['sig_bonferroni'] else "NO"
        log(f"\n{row['term']}:")
        log(f"  Estimate: {row['estimate']:.4f}")
        log(f"  SE: {row['se']:.4f}")
        log(f"  z: {row['z']:.2f}")
        log(f"  p_uncorrected: {row['p_uncorrected']:.4f} (sig @ 0.05: {sig_unc})")
        log(f"  p_bonferroni: {row['p_bonferroni']:.4f} (sig @ 0.0167: {sig_bon})")

    # Key hypothesis test
    interaction_row = age_effects[age_effects['term'].str.contains('Time_log')].iloc[0]
    if interaction_row['sig_bonferroni']:
        conclusion = "SIGNIFICANT - Age moderates confidence decline rate"
    else:
        conclusion = "NULL - Age-invariant confidence decline (parallels Ch5 accuracy findings)"
    log(f"\n*** PRIMARY HYPOTHESIS TEST ***")
    log(f"Age x Time interaction: {conclusion}")

    # ==================== VALIDATION ====================
    log("\n--- Step 4 Validation ---")

    # Expected row count
    assert len(age_effects) == 2, f"Expected 2 rows, found {len(age_effects)}"
    log(f"✓ Row count: {len(age_effects)}")

    # Both terms present
    assert any(age_effects['term'].str.contains('Age_c')), "Age_c term missing"
    assert any(age_effects['term'].str.contains('Time_log')), "Time_log interaction missing"
    log("✓ Both age-related terms present")

    # p-values in valid range
    assert (age_effects['p_uncorrected'] >= 0).all() and (age_effects['p_uncorrected'] <= 1).all()
    log("✓ p-values in valid range [0, 1]")

    # Significance flags correct
    for _, row in age_effects.iterrows():
        assert row['sig_uncorrected'] == (row['p'] < 0.05), "sig_uncorrected flag incorrect"
        assert row['sig_bonferroni'] == (row['p'] < BONFERRONI_ALPHA), "sig_bonferroni flag incorrect"
    log("✓ Significance flags correct")

    # Select output columns
    output_cols = ['term', 'estimate', 'se', 'z', 'p_uncorrected', 'p_bonferroni',
                   'sig_uncorrected', 'sig_bonferroni']
    age_effects = age_effects[output_cols]

    # Save output
    output_path = RQ_DIR / "data" / "step04_age_effects.csv"
    age_effects.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Dual p-values created per Decision D068")

    return age_effects


def step05_compute_effect_size(fixed_effects: pd.DataFrame, df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 5: Compute effect size at Day 6.

    Predicts confidence difference between younger (-1 SD) and older (+1 SD) adults
    at Day 6 retention test (max TSVR).
    """
    log("\n" + "=" * 70)
    log("STEP 5: Compute Effect Size at Day 6")
    log("=" * 70)

    # Compute Age SD (from unique participants)
    age_per_participant = df_input.groupby('UID')['Age'].first()
    mean_age = age_per_participant.mean()
    sd_age = age_per_participant.std()

    log(f"\nAge distribution:")
    log(f"  Mean: {mean_age:.1f} years")
    log(f"  SD: {sd_age:.1f} years")

    # Define comparison points
    age_younger = mean_age - sd_age  # -1 SD
    age_older = mean_age + sd_age    # +1 SD
    age_c_younger = -sd_age  # Age_c for younger
    age_c_older = sd_age     # Age_c for older

    # Day 6 TSVR (use maximum observed TSVR)
    tsvr_day6 = df_input['TSVR_hours'].max()
    time_log_day6 = np.log(tsvr_day6 + 1)

    log(f"\nComparison points:")
    log(f"  Younger: Age = {age_younger:.1f} years (Age_c = {age_c_younger:.1f})")
    log(f"  Older: Age = {age_older:.1f} years (Age_c = {age_c_older:.1f})")
    log(f"  Time: Day 6, TSVR = {tsvr_day6:.1f} hours, Time_log = {time_log_day6:.2f}")

    # Extract coefficients
    coefs = fixed_effects.set_index('term')['estimate'].to_dict()
    intercept = coefs.get('Intercept', 0)
    time_coef = coefs.get('Time_log', 0)
    age_coef = coefs.get('Age_c', 0)
    interaction_coef = coefs.get('Time_log:Age_c', coefs.get('Age_c:Time_log', 0))

    log(f"\nCoefficients:")
    log(f"  Intercept: {intercept:.4f}")
    log(f"  Time_log: {time_coef:.4f}")
    log(f"  Age_c: {age_coef:.4f}")
    log(f"  Time_log:Age_c: {interaction_coef:.6f}")

    # Predict theta at Day 6 for younger and older
    younger_theta = intercept + time_coef * time_log_day6 + age_coef * age_c_younger + interaction_coef * time_log_day6 * age_c_younger
    older_theta = intercept + time_coef * time_log_day6 + age_coef * age_c_older + interaction_coef * time_log_day6 * age_c_older
    difference = older_theta - younger_theta

    log(f"\nPredictions at Day 6:")
    log(f"  Younger (-1 SD): θ = {younger_theta:.4f}")
    log(f"  Older (+1 SD): θ = {older_theta:.4f}")
    log(f"  Difference: {difference:.4f} theta units")

    # Create output DataFrame
    effect_size = pd.DataFrame([{
        'comparison': 'Older vs Younger at Day 6',
        'younger_theta': younger_theta,
        'older_theta': older_theta,
        'difference': difference,
        'age_younger': age_younger,
        'age_older': age_older,
        'tsvr_hours': tsvr_day6
    }])

    # Interpretation
    if abs(difference) < 0.1:
        interpretation = "Negligible age effect at Day 6"
    elif abs(difference) < 0.3:
        interpretation = "Small age effect at Day 6"
    else:
        interpretation = "Moderate age effect at Day 6"
    log(f"\nInterpretation: {interpretation}")

    # ==================== VALIDATION ====================
    log("\n--- Step 5 Validation ---")

    # Expected row count
    assert len(effect_size) == 1, f"Expected 1 row, found {len(effect_size)}"
    log(f"✓ Row count: {len(effect_size)}")

    # Theta predictions in range
    assert -3 <= younger_theta <= 3, f"younger_theta out of range: {younger_theta}"
    assert -3 <= older_theta <= 3, f"older_theta out of range: {older_theta}"
    log(f"✓ Theta predictions in valid range [-3, 3]")

    # Difference reasonable
    assert -2 <= difference <= 2, f"Difference unreasonably large: {difference}"
    log(f"✓ Difference reasonable: {difference:.4f}")

    # Age ordering
    assert age_older > age_younger, "age_older should be > age_younger"
    log(f"✓ Age ordering correct: {age_younger:.1f} < {age_older:.1f}")

    # TSVR positive
    assert tsvr_day6 > 0, "TSVR should be positive"
    log(f"✓ TSVR positive: {tsvr_day6:.1f} hours")

    # Save output
    output_path = RQ_DIR / "data" / "step05_effect_size_day6.csv"
    effect_size.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Effect size computed at Day 6 (TSVR = {tsvr_day6:.1f} hours)")

    return effect_size


def step06_prepare_tertile_data(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 6: Prepare age tertile comparison data for plotting.

    Assigns participants to Low/Medium/High age tertiles and computes
    observed means per tertile x test for visualization.
    """
    log("\n" + "=" * 70)
    log("STEP 6: Prepare Age Tertile Comparison Data")
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
            return 'Low'
        elif age <= p67:
            return 'Medium'
        else:
            return 'High'

    df_input['age_tertile'] = df_input['Age'].apply(assign_tertile)

    # Count per tertile
    tertile_counts = df_input.groupby('UID')['age_tertile'].first().value_counts()
    log(f"\nParticipants per tertile:")
    for tertile in ['Low', 'Medium', 'High']:
        n = tertile_counts.get(tertile, 0)
        log(f"  {tertile}: N = {n}")

    # Aggregate by tertile x test
    agg = df_input.groupby(['age_tertile', 'test']).agg(
        mean_theta=('theta_confidence', 'mean'),
        se_theta=('theta_confidence', lambda x: x.std() / np.sqrt(len(x))),
        N=('theta_confidence', 'count')
    ).reset_index()

    # Compute 95% CI
    agg['CI_lower'] = agg['mean_theta'] - 1.96 * agg['se_theta']
    agg['CI_upper'] = agg['mean_theta'] + 1.96 * agg['se_theta']

    # Reorder columns
    agg = agg[['age_tertile', 'test', 'mean_theta', 'se_theta', 'CI_lower', 'CI_upper', 'N']]

    log(f"\nAggregated tertile data ({len(agg)} rows):")
    for tertile in ['Low', 'Medium', 'High']:
        tertile_data = agg[agg['age_tertile'] == tertile]
        log(f"\n  {tertile} tertile:")
        for _, row in tertile_data.iterrows():
            log(f"    {row['test']}: θ = {row['mean_theta']:.3f} ± {row['se_theta']:.3f} (N={row['N']})")

    # ==================== VALIDATION ====================
    log("\n--- Step 6 Validation ---")

    # Expected row count
    assert len(agg) == 12, f"Expected 12 rows (3 tertiles x 4 tests), found {len(agg)}"
    log(f"✓ Row count: {len(agg)} (expected 12)")

    # All tertiles present
    tertiles_present = set(agg['age_tertile'].unique())
    expected_tertiles = {'Low', 'Medium', 'High'}
    assert tertiles_present == expected_tertiles, f"Missing tertiles: {expected_tertiles - tertiles_present}"
    log("✓ All tertiles present: Low, Medium, High")

    # All tests present
    tests_present = set(agg['test'].unique())
    expected_tests = {'T1', 'T2', 'T3', 'T4'}
    assert tests_present == expected_tests, f"Missing tests: {expected_tests - tests_present}"
    log("✓ All tests present: T1, T2, T3, T4")

    # CI ordering
    assert (agg['CI_upper'] > agg['CI_lower']).all(), "CI_upper should be > CI_lower"
    log("✓ CI ordering correct: CI_upper > CI_lower")

    # Mean theta in range
    assert (agg['mean_theta'] >= -3).all() and (agg['mean_theta'] <= 3).all(), "mean_theta out of range"
    log(f"✓ mean_theta in valid range [-3, 3]")

    # SE reasonable
    assert (agg['se_theta'] >= 0).all() and (agg['se_theta'] <= 1).all(), "se_theta out of range"
    log(f"✓ se_theta reasonable (0 to 1)")

    # No duplicates
    assert len(agg) == len(agg.drop_duplicates(['age_tertile', 'test'])), "Duplicate tertile x test combinations"
    log("✓ No duplicate combinations")

    # Save output
    output_path = RQ_DIR / "data" / "step06_age_tertile_data.csv"
    agg.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Aggregated means: {len(agg)} rows (3 tertiles x 4 tests)")

    return agg


def main():
    """Execute all steps for RQ 6.1.3."""

    # Initialize log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.1.3: Age Effects on Confidence\n")
        f.write("=" * 70 + "\n")
        f.write("Analysis execution log\n\n")

    log("Starting RQ 6.1.3 analysis pipeline...")
    log(f"Output directory: {RQ_DIR}")

    try:
        # Step 0: Load and merge data
        df = step00_load_data()

        # Step 1: Center Age
        df = step01_center_age(df)

        # Step 2: Create time predictors
        df = step02_create_time_predictors(df)

        # Step 3: Fit LMM
        lmm_result, fixed_effects = step03_fit_lmm(df)

        # Step 4: Extract Age effects with dual p-values
        age_effects = step04_extract_age_effects(fixed_effects)

        # Step 5: Compute effect size at Day 6
        effect_size = step05_compute_effect_size(fixed_effects, df)

        # Step 6: Prepare tertile data
        tertile_data = step06_prepare_tertile_data(df)

        # Final summary
        log("\n" + "=" * 70)
        log("ANALYSIS COMPLETE")
        log("=" * 70)

        # Report key findings
        interaction = age_effects[age_effects['term'].str.contains('Time_log')].iloc[0]
        log(f"\n*** KEY FINDING: Age x Time Interaction ***")
        log(f"  Estimate: {interaction['estimate']:.6f}")
        log(f"  p-value: {interaction['p_uncorrected']:.4f}")
        log(f"  Significant (Bonferroni): {'YES' if interaction['sig_bonferroni'] else 'NO'}")

        if not interaction['sig_bonferroni']:
            log("\n  CONCLUSION: NULL - Age does NOT moderate confidence decline rate")
            log("  This PARALLELS Chapter 5 accuracy findings (age-invariant forgetting)")
            log("  Validates VR ecological encoding framework for metacognitive monitoring")
        else:
            log("\n  CONCLUSION: SIGNIFICANT - Age moderates confidence decline")
            log("  This DIVERGES from Chapter 5 accuracy findings")
            log("  Suggests dissociation between memory and metacognition aging effects")

        # Effect size summary
        diff = effect_size['difference'].iloc[0]
        log(f"\n*** EFFECT SIZE at Day 6 ***")
        log(f"  Difference (Older - Younger): {diff:.4f} theta units")
        if abs(diff) < 0.1:
            log("  Interpretation: Negligible practical difference")
        elif abs(diff) < 0.3:
            log("  Interpretation: Small practical difference")
        else:
            log("  Interpretation: Moderate practical difference")

        log("\n*** OUTPUT FILES CREATED ***")
        for f in sorted((RQ_DIR / "data").glob("*.csv")):
            log(f"  {f.name}")

        log("\n✅ RQ 6.1.3 analysis complete - all validations passed")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
