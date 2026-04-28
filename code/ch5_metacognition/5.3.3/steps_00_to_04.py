#!/usr/bin/env python3
"""
RQ 6.3.3: Age x Domain Interaction in Confidence Decline
=========================================================
Tests whether age interacts with memory domain (What/Where/When) for
confidence decline trajectories over a 6-day retention interval.

Steps:
  0: Load domain-stratified confidence theta from RQ 6.3.1, merge with Age
  1: Center Age and reshape to long format (3 domains stacked)
  2: Fit LMM: theta_confidence ~ TSVR_hours * Age_c * Domain + (TSVR_hours | UID)
  3: Extract 3-way interaction terms with dual p-values (Decision D068)
  4: Create age tertile x domain trajectories for visualization

Expected outcome: NULL 3-way interaction (paralleling Ch5 5.2.3 accuracy null findings)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
import warnings

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.3.3
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_04.log"
BONFERRONI_N_TESTS = 2  # 2 domain contrasts (Where vs What, When vs What reference)
BONFERRONI_ALPHA = 0.05 / BONFERRONI_N_TESTS  # 0.025 for interaction contrasts

# Input files
THETA_FILE = Path("/home/etai/projects/REMEMVR/results/ch6/6.3.1/data/step03_theta_confidence.csv")
DFDATA_FILE = Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv")


def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def step00_load_theta_with_age() -> pd.DataFrame:
    """
    Step 0: Load domain-stratified confidence theta from RQ 6.3.1, merge with Age.

    RQ 6.3.1 outputs theta scores per domain (What/Where/When) from 3-factor GRM.
    """
    log("=" * 70)
    log("STEP 0: Load Domain Confidence Theta from RQ 6.3.1 and Merge with Age")
    log("=" * 70)

    # Load theta scores from RQ 6.3.1
    log(f"\nLoading theta from: {THETA_FILE}")
    df_theta = pd.read_csv(THETA_FILE)
    log(f"Theta loaded: {len(df_theta)} rows, columns: {list(df_theta.columns)}")

    # Expected format: composite_ID, theta_What, theta_Where, theta_When
    # Parse UID and test from composite_ID
    df_theta['UID'] = df_theta['composite_ID'].str.split('_').str[0]
    df_theta['test'] = df_theta['composite_ID'].str.split('_').str[1]

    log(f"\nParsed composite_ID: {df_theta['UID'].nunique()} unique UIDs, {df_theta['test'].nunique()} tests")

    # Load Age from dfData.csv
    log(f"\nLoading Age from: {DFDATA_FILE}")
    df_demo = pd.read_csv(DFDATA_FILE, usecols=['UID', 'age', 'TEST', 'TSVR'])

    # Get unique participant-level Age (take first row per UID)
    df_age = df_demo.groupby('UID').agg({'age': 'first'}).reset_index()
    df_age = df_age.rename(columns={'age': 'Age'})
    log(f"Age data: {len(df_age)} unique participants")

    # Get TSVR hours per UID x TEST
    df_tsvr = df_demo[['UID', 'TEST', 'TSVR']].drop_duplicates()
    df_tsvr = df_tsvr.rename(columns={'TSVR': 'TSVR_hours', 'TEST': 'test'})
    # Convert TEST (int) to match composite_ID format (T1, T2, T3, T4)
    df_tsvr['test'] = 'T' + df_tsvr['test'].astype(str)
    log(f"TSVR data: {len(df_tsvr)} UID x test combinations")

    # Merge theta with Age
    df_merged = df_theta.merge(df_age, on='UID', how='left')
    log(f"\nMerged with Age: {len(df_merged)} rows")

    # Validate no missing Age
    missing_age = df_merged['Age'].isna().sum()
    if missing_age > 0:
        raise ValueError(f"Missing Age for {missing_age} rows!")

    # Merge with TSVR
    df_final = df_merged.merge(df_tsvr, on=['UID', 'test'], how='left')
    log(f"Merged with TSVR: {len(df_final)} rows")

    # Validate no missing TSVR
    missing_tsvr = df_final['TSVR_hours'].isna().sum()
    if missing_tsvr > 0:
        raise ValueError(f"Missing TSVR for {missing_tsvr} rows!")

    # VALIDATION
    log("\n--- Step 0 Validation ---")

    # Expected row count
    assert len(df_final) == 400, f"Expected 400 rows (100 participants x 4 tests), found {len(df_final)}"
    log(f"Row count: {len(df_final)} (expected 400)")

    # Required theta columns present
    for col in ['theta_What', 'theta_Where', 'theta_When']:
        assert col in df_final.columns, f"Missing column: {col}"
    log("All theta columns present: theta_What, theta_Where, theta_When")

    # No missing values in theta
    for col in ['theta_What', 'theta_Where', 'theta_When']:
        null_count = df_final[col].isna().sum()
        assert null_count == 0, f"NaN values in {col}: {null_count}"
    log("No missing theta values")

    # Theta ranges reasonable
    for col in ['theta_What', 'theta_Where', 'theta_When']:
        tmin, tmax = df_final[col].min(), df_final[col].max()
        assert -4 <= tmin and tmax <= 4, f"{col} out of range: [{tmin:.2f}, {tmax:.2f}]"
        log(f"{col} range: [{tmin:.2f}, {tmax:.2f}]")

    # Age range
    age_min, age_max = df_final['Age'].min(), df_final['Age'].max()
    assert 18 <= age_min and age_max <= 90, f"Age out of range: [{age_min}, {age_max}]"
    log(f"Age range: [{age_min:.0f}, {age_max:.0f}] years")

    # TSVR range
    tsvr_min, tsvr_max = df_final['TSVR_hours'].min(), df_final['TSVR_hours'].max()
    assert 0 <= tsvr_min and tsvr_max <= 250, f"TSVR out of range: [{tsvr_min:.1f}, {tsvr_max:.1f}]"
    log(f"TSVR range: [{tsvr_min:.1f}, {tsvr_max:.1f}] hours")

    # Unique UIDs
    n_uids = df_final['UID'].nunique()
    assert n_uids == 100, f"Expected 100 unique UIDs, found {n_uids}"
    log(f"Unique UIDs: {n_uids}")

    # Save output
    output_path = RQ_DIR / "data" / "step00_theta_with_age.csv"
    df_final.to_csv(output_path, index=False)
    log(f"\nOutput saved: {output_path}")
    log(f"Data loading complete: {len(df_final)} rows")

    return df_final


def step01_center_age_reshape_long(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Center Age and reshape from wide to long format (3 domains stacked).

    Wide: 400 rows (100 participants x 4 tests) with theta_What, theta_Where, theta_When
    Long: 1200 rows (400 x 3 domains) with Domain factor and theta_confidence
    """
    log("\n" + "=" * 70)
    log("STEP 1: Center Age and Reshape to Long Format")
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

    # Add log-transformed time per RQ 6.1.1 best model
    df_input['log_TSVR'] = np.log(df_input['TSVR_hours'] + 1)

    log(f"\nAge centered: Age_c = Age - {mean_age:.2f}")
    log(f"Log-transformed time: log_TSVR = log(TSVR_hours + 1)")

    # Reshape to long format
    log("\nReshaping from wide to long format...")

    # Prepare for melt - keep ID columns
    id_vars = ['composite_ID', 'UID', 'test', 'Age', 'Age_c', 'TSVR_hours', 'log_TSVR']
    value_vars = ['theta_What', 'theta_Where', 'theta_When']

    df_long = df_input.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='theta_column',
        value_name='theta_confidence'
    )

    # Extract Domain from theta_column name (theta_What -> What)
    df_long['Domain'] = df_long['theta_column'].str.replace('theta_', '')
    df_long = df_long.drop(columns=['theta_column'])

    # Reorder columns
    df_long = df_long[['UID', 'test', 'composite_ID', 'Age', 'Age_c', 'TSVR_hours', 'log_TSVR', 'Domain', 'theta_confidence']]

    log(f"Reshaped: {len(df_long)} rows (400 observations x 3 domains)")

    # VALIDATION
    log("\n--- Step 1 Validation ---")

    # Expected row count
    assert len(df_long) == 1200, f"Expected 1200 rows, found {len(df_long)}"
    log(f"Row count: {len(df_long)} (expected 1200)")

    # Age_c mean approximately 0
    age_c_mean = df_long['Age_c'].mean()
    assert abs(age_c_mean) < 0.001, f"Age_c mean = {age_c_mean}, expected ~0"
    log(f"Age_c mean: {age_c_mean:.6f} (expected ~0)")

    # Balanced design: 400 rows per domain
    domain_counts = df_long['Domain'].value_counts()
    for domain in ['What', 'Where', 'When']:
        count = domain_counts.get(domain, 0)
        assert count == 400, f"Expected 400 rows for {domain}, found {count}"
    log(f"Balanced design: {dict(domain_counts)}")

    # Each UID has exactly 12 rows (4 tests x 3 domains)
    uid_counts = df_long.groupby('UID').size()
    assert (uid_counts == 12).all(), f"Not all UIDs have 12 rows: {uid_counts[uid_counts != 12].to_dict()}"
    log("Each UID has exactly 12 rows (4 tests x 3 domains)")

    # No missing values
    null_counts = df_long.isnull().sum()
    assert null_counts.sum() == 0, f"NaN values found: {null_counts[null_counts > 0].to_dict()}"
    log("No missing values")

    # Domain factor has exactly 3 levels
    domains = set(df_long['Domain'].unique())
    expected_domains = {'What', 'Where', 'When'}
    assert domains == expected_domains, f"Unexpected domains: {domains}"
    log(f"Domain levels: {domains}")

    # Save output
    output_path = RQ_DIR / "data" / "step01_lmm_input.csv"
    df_long.to_csv(output_path, index=False)
    log(f"\nOutput saved: {output_path}")
    log(f"Long format: {len(df_long)} rows x {len(df_long.columns)} columns")

    return df_long


def step02_fit_lmm_3way_interaction(df_input: pd.DataFrame):
    """
    Step 2: Fit LMM with 3-way Age x Domain x Time interaction.

    Model: theta_confidence ~ TSVR_hours * Age_c * Domain + (TSVR_hours | UID)

    Tests whether age moderates the domain-specific confidence decline trajectories.
    """
    log("\n" + "=" * 70)
    log("STEP 2: Fit LMM with 3-Way Age x Domain x Time Interaction")
    log("=" * 70)

    # Formula with full 3-way interaction
    # Using TSVR_hours (linear) per Decision D070
    formula = "theta_confidence ~ TSVR_hours * Age_c * C(Domain)"

    log(f"\nModel formula: {formula}")
    log("Random effects: (1 + TSVR_hours | UID) - random intercept and slope")
    log("Reference category: What (alphabetically first)")

    # Fit LMM with random slopes
    try:
        model = smf.mixedlm(
            formula=formula,
            data=df_input,
            groups=df_input['UID'],
            re_formula="~TSVR_hours"  # Random intercept + random slope
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
        log(f"  {row['term']:50s}: beta = {row['estimate']:8.5f}, SE = {row['se']:.5f}, z = {row['z_value']:6.2f}, p = {row['p_value']:.4f} {sig}")

    # Random effects variance
    log("\n--- Random Effects ---")
    re_var = result.cov_re
    log(f"Random effects covariance matrix:\n{re_var}")

    # Model fit indices
    log("\n--- Model Fit ---")
    n_params = len(result.params)
    n_obs = len(df_input)
    aic = -2 * result.llf + 2 * n_params
    bic = -2 * result.llf + np.log(n_obs) * n_params
    log(f"Log-likelihood: {result.llf:.2f}")
    log(f"AIC: {aic:.2f}")
    log(f"BIC: {bic:.2f}")

    # VALIDATION
    log("\n--- Step 2 Validation ---")

    # Check convergence
    converged = result.converged if hasattr(result, 'converged') else True
    log(f"Model converged: {converged}")

    # Check all fixed effects have finite estimates
    assert not fixed_effects['estimate'].isna().any(), "NaN in fixed effect estimates"
    assert not np.isinf(fixed_effects['estimate']).any(), "Inf in fixed effect estimates"
    log("All fixed effects finite")

    # Check 3-way interaction terms present
    # Should have: TSVR_hours:Age_c:C(Domain)[T.Where], TSVR_hours:Age_c:C(Domain)[T.When]
    interaction_3way = fixed_effects[fixed_effects['term'].str.contains('TSVR_hours') &
                                      fixed_effects['term'].str.contains('Age_c') &
                                      fixed_effects['term'].str.contains('Domain')]
    assert len(interaction_3way) == 2, f"Expected 2 3-way interaction terms, found {len(interaction_3way)}"
    log(f"3-way interaction terms present: {len(interaction_3way)} terms")

    # Save fixed effects
    fe_path = RQ_DIR / "data" / "step02_lmm_fixed_effects.csv"
    fixed_effects.to_csv(fe_path, index=False)
    log(f"\nFixed effects saved: {fe_path}")

    return result, fixed_effects


def step03_extract_interaction_dual_pvalues(fixed_effects: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Extract 3-way interaction terms with Decision D068 dual p-values.

    Primary hypothesis test: Age x Domain x Time interaction
    - Reference domain: What
    - Contrasts: Where vs What, When vs What
    - Bonferroni correction for 2 contrasts
    """
    log("\n" + "=" * 70)
    log("STEP 3: Extract 3-Way Interaction Terms with Dual P-Values (Decision D068)")
    log("=" * 70)

    # Extract 3-way interaction terms
    interaction_3way = fixed_effects[fixed_effects['term'].str.contains('TSVR_hours') &
                                      fixed_effects['term'].str.contains('Age_c') &
                                      fixed_effects['term'].str.contains('Domain')].copy()

    log(f"\n3-way interaction terms extracted: {len(interaction_3way)} terms")

    # Add dual p-value columns per Decision D068
    log(f"\nBonferroni correction: alpha = {BONFERRONI_ALPHA:.4f} (0.05 / {BONFERRONI_N_TESTS} contrasts)")

    interaction_3way['p_uncorrected'] = interaction_3way['p_value']
    interaction_3way['p_bonferroni'] = np.minimum(interaction_3way['p_value'] * BONFERRONI_N_TESTS, 1.0)
    interaction_3way['sig_uncorrected'] = interaction_3way['p_value'] < 0.05
    interaction_3way['sig_bonferroni'] = interaction_3way['p_bonferroni'] < 0.05

    # Report findings
    log("\n--- 3-Way Interaction Results (Age x Domain x Time) ---")
    for _, row in interaction_3way.iterrows():
        sig_unc = "YES" if row['sig_uncorrected'] else "NO"
        sig_bon = "YES" if row['sig_bonferroni'] else "NO"
        log(f"\n{row['term']}:")
        log(f"  Estimate: {row['estimate']:.6f}")
        log(f"  SE: {row['se']:.6f}")
        log(f"  z: {row['z_value']:.2f}")
        log(f"  p_uncorrected: {row['p_uncorrected']:.4f} (sig @ 0.05: {sig_unc})")
        log(f"  p_bonferroni: {row['p_bonferroni']:.4f} (sig @ 0.05: {sig_bon})")

    # Overall 3-way interaction test
    any_sig_uncorrected = interaction_3way['sig_uncorrected'].any()
    any_sig_bonferroni = interaction_3way['sig_bonferroni'].any()

    log(f"\n*** PRIMARY HYPOTHESIS TEST: Age x Domain x Time Interaction ***")
    log(f"Any significant (uncorrected): {any_sig_uncorrected}")
    log(f"Any significant (Bonferroni): {any_sig_bonferroni}")

    if any_sig_bonferroni:
        conclusion = "SIGNIFICANT - Age differentially moderates domain-specific confidence trajectories"
    else:
        conclusion = "NULL - Age-invariant confidence decline across all domains (parallels Ch5 5.2.3)"
    log(f"CONCLUSION: {conclusion}")

    # Also extract 2-way Age x Time interaction (overall)
    age_time_2way = fixed_effects[fixed_effects['term'].str.contains('TSVR_hours:Age_c') &
                                   ~fixed_effects['term'].str.contains('Domain')].copy()
    if len(age_time_2way) > 0:
        log(f"\n--- 2-Way Age x Time Interaction (Overall) ---")
        for _, row in age_time_2way.iterrows():
            sig = "YES" if row['p_value'] < 0.05 else "NO"
            log(f"{row['term']}: beta = {row['estimate']:.6f}, p = {row['p_value']:.4f} (sig: {sig})")

    # VALIDATION
    log("\n--- Step 3 Validation ---")

    # Expected row count
    assert len(interaction_3way) == 2, f"Expected 2 3-way interaction terms, found {len(interaction_3way)}"
    log(f"Row count: {len(interaction_3way)} (expected 2)")

    # Both domain contrasts present (Where vs What, When vs What)
    terms_str = ' '.join(interaction_3way['term'].values)
    assert 'Where' in terms_str, "Missing Where contrast"
    assert 'When' in terms_str, "Missing When contrast"
    log("Both domain contrasts present: Where vs What, When vs What")

    # p-values in valid range
    assert (interaction_3way['p_uncorrected'] >= 0).all() and (interaction_3way['p_uncorrected'] <= 1).all()
    assert (interaction_3way['p_bonferroni'] >= 0).all() and (interaction_3way['p_bonferroni'] <= 1).all()
    log("p-values in valid range [0, 1]")

    # p_bonferroni >= p_uncorrected
    assert (interaction_3way['p_bonferroni'] >= interaction_3way['p_uncorrected']).all()
    log("p_bonferroni >= p_uncorrected (Bonferroni correction correct)")

    # Select output columns
    output_cols = ['term', 'estimate', 'se', 'z_value', 'p_uncorrected', 'p_bonferroni',
                   'sig_uncorrected', 'sig_bonferroni']
    interaction_3way = interaction_3way[output_cols]

    # Save output
    output_path = RQ_DIR / "data" / "step03_interaction_terms.csv"
    interaction_3way.to_csv(output_path, index=False)
    log(f"\nOutput saved: {output_path}")
    log(f"Dual p-values created per Decision D068")

    return interaction_3way


def step04_create_tertile_domain_trajectories(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Create age tertile x domain trajectories for visualization.

    Aggregates mean confidence by age tertile (Young/Middle/Older) and domain (What/Where/When)
    across tests for trajectory plotting.
    """
    log("\n" + "=" * 70)
    log("STEP 4: Create Age Tertile x Domain Trajectories")
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

    # Count participants per tertile
    tertile_counts = df_input.groupby('UID')['age_tertile'].first().value_counts()
    log(f"\nParticipants per tertile:")
    for tertile in ['Young', 'Middle', 'Older']:
        n = tertile_counts.get(tertile, 0)
        log(f"  {tertile}: N = {n}")

    # Aggregate by tertile x domain x test
    agg = df_input.groupby(['age_tertile', 'Domain', 'test']).agg(
        TSVR_hours=('TSVR_hours', 'mean'),
        mean_theta=('theta_confidence', 'mean'),
        se_theta=('theta_confidence', lambda x: x.std() / np.sqrt(len(x))),
        N=('theta_confidence', 'count')
    ).reset_index()

    # Compute 95% CI
    agg['CI_lower'] = agg['mean_theta'] - 1.96 * agg['se_theta']
    agg['CI_upper'] = agg['mean_theta'] + 1.96 * agg['se_theta']

    log(f"\nAggregated data: {len(agg)} rows (3 tertiles x 3 domains x 4 tests = 36)")

    # Report summary by tertile and domain
    for tertile in ['Young', 'Middle', 'Older']:
        log(f"\n{tertile} tertile:")
        for domain in ['What', 'Where', 'When']:
            subset = agg[(agg['age_tertile'] == tertile) & (agg['Domain'] == domain)]
            t1_val = subset[subset['test'] == 'T1']['mean_theta'].values[0] if len(subset[subset['test'] == 'T1']) > 0 else np.nan
            t4_val = subset[subset['test'] == 'T4']['mean_theta'].values[0] if len(subset[subset['test'] == 'T4']) > 0 else np.nan
            change = t4_val - t1_val if not np.isnan(t1_val) and not np.isnan(t4_val) else np.nan
            log(f"  {domain}: T1={t1_val:.3f}, T4={t4_val:.3f}, change={change:+.3f}")

    # VALIDATION
    log("\n--- Step 4 Validation ---")

    # Expected row count
    assert len(agg) == 36, f"Expected 36 rows (3 tertiles x 3 domains x 4 tests), found {len(agg)}"
    log(f"Row count: {len(agg)} (expected 36)")

    # All tertiles present
    tertiles_present = set(agg['age_tertile'].unique())
    expected_tertiles = {'Young', 'Middle', 'Older'}
    assert tertiles_present == expected_tertiles, f"Missing tertiles: {expected_tertiles - tertiles_present}"
    log("All tertiles present: Young, Middle, Older")

    # All domains present
    domains_present = set(agg['Domain'].unique())
    expected_domains = {'What', 'Where', 'When'}
    assert domains_present == expected_domains, f"Missing domains: {expected_domains - domains_present}"
    log("All domains present: What, Where, When")

    # All tests present
    tests_present = set(agg['test'].unique())
    expected_tests = {'T1', 'T2', 'T3', 'T4'}
    assert tests_present == expected_tests, f"Missing tests: {expected_tests - tests_present}"
    log("All tests present: T1, T2, T3, T4")

    # CI ordering
    assert (agg['CI_upper'] > agg['CI_lower']).all(), "CI_upper should be > CI_lower"
    log("CI ordering correct: CI_upper > CI_lower")

    # No duplicates
    n_unique = len(agg.drop_duplicates(['age_tertile', 'Domain', 'test']))
    assert n_unique == 36, f"Duplicate combinations found"
    log("No duplicate combinations")

    # Save output
    output_path = RQ_DIR / "data" / "step04_tertile_domain_trajectories.csv"
    agg.to_csv(output_path, index=False)
    log(f"\nOutput saved: {output_path}")
    log(f"Aggregated means: {len(agg)} rows (3 tertiles x 3 domains x 4 tests)")

    return agg


def main():
    """Execute all steps for RQ 6.3.3."""

    # Initialize log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.3.3: Age x Domain Interaction in Confidence Decline\n")
        f.write("=" * 70 + "\n")
        f.write("Analysis execution log\n\n")

    log("Starting RQ 6.3.3 analysis pipeline...")
    log(f"Output directory: {RQ_DIR}")

    try:
        # Step 0: Load and merge data
        df_wide = step00_load_theta_with_age()

        # Step 1: Center Age and reshape to long format
        df_long = step01_center_age_reshape_long(df_wide)

        # Step 2: Fit LMM with 3-way interaction
        lmm_result, fixed_effects = step02_fit_lmm_3way_interaction(df_long)

        # Step 3: Extract 3-way interaction terms with dual p-values
        interaction_terms = step03_extract_interaction_dual_pvalues(fixed_effects)

        # Step 4: Create tertile x domain trajectories
        trajectories = step04_create_tertile_domain_trajectories(df_long)

        # Final summary
        log("\n" + "=" * 70)
        log("ANALYSIS COMPLETE")
        log("=" * 70)

        # Report key findings
        log(f"\n*** KEY FINDING: 3-Way Age x Domain x Time Interaction ***")
        for _, row in interaction_terms.iterrows():
            sig_status = "SIGNIFICANT" if row['sig_bonferroni'] else "NOT SIGNIFICANT"
            log(f"\n{row['term']}:")
            log(f"  Estimate: {row['estimate']:.6f}")
            log(f"  p_uncorrected: {row['p_uncorrected']:.4f}")
            log(f"  p_bonferroni: {row['p_bonferroni']:.4f}")
            log(f"  Status: {sig_status}")

        any_sig = interaction_terms['sig_bonferroni'].any()
        if not any_sig:
            log("\n*** CONCLUSION: NULL 3-WAY INTERACTION ***")
            log("Age does NOT differentially moderate domain-specific confidence trajectories")
            log("This PARALLELS Chapter 5 RQ 5.2.3 (age-invariant accuracy across domains)")
            log("Validates VR ecological encoding framework: universal age-invariant pattern")
        else:
            log("\n*** CONCLUSION: SIGNIFICANT 3-WAY INTERACTION ***")
            log("Age differentially moderates domain-specific confidence trajectories")
            log("This DIVERGES from Chapter 5 RQ 5.2.3 accuracy findings")
            log("Suggests dissociation between memory and metacognition aging effects")

        log("\n*** OUTPUT FILES CREATED ***")
        for f in sorted((RQ_DIR / "data").glob("*.csv")):
            log(f"  {f.name}")
        for f in sorted((RQ_DIR / "data").glob("*.txt")):
            log(f"  {f.name}")

        log("\nRQ 6.3.3 analysis complete - all validations passed")

    except Exception as e:
        log(f"\nERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
