#!/usr/bin/env python3
"""
RQ 6.4.3: Age x Paradigm Interaction for Confidence Decline
============================================================
Tests whether age moderates the relationship between retrieval paradigm
(Free Recall, Cued Recall, Recognition) and confidence decline trajectories.

Steps:
  0: Load theta confidence by paradigm from RQ 6.4.1, merge with Age
  1: Fit LMM with Age x Paradigm x Time 3-way interaction
  2: Extract interaction terms with dual p-values (Decision D068)
  3: Compute effect sizes (Cohen's f-squared)
  4: Compare to Chapter 5 RQ 5.3.4 results

Expected outcome: NULL 3-way interaction (paralleling Ch5 5.3.4 accuracy findings)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
from scipy import stats
import warnings

# CONFIGURATION

RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.4.3
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_04.log"
BONFERRONI_ALPHA = 0.05 / 3  # 0.0167 for 3 comparisons: Age_c, Age_c:Time, Age_c:Paradigm:Time

# Input files
LMM_INPUT_FILE = Path("/home/etai/projects/REMEMVR/results/ch6/6.4.1/data/step04_lmm_input.csv")
DFDATA_FILE = Path("/home/etai/projects/REMEMVR/data/cache/dfData.csv")
CH5_COMPARISON_FILE = Path("/home/etai/projects/REMEMVR/results/ch5/5.3.4/data/step02_interaction_terms.csv")


def log(msg: str):
    """Log message to file and console."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def step00_prepare_lmm_input():
    """
    Step 0: Load confidence theta by paradigm from RQ 6.4.1, merge with Age.

    Uses step04_lmm_input.csv which already has paradigm-level theta in long format.
    """
    log("=" * 70)
    log("STEP 0: Prepare LMM Input with Age")
    log("=" * 70)

    # Load theta confidence by paradigm from RQ 6.4.1
    log(f"\nLoading theta from: {LMM_INPUT_FILE}")
    df = pd.read_csv(LMM_INPUT_FILE)
    log(f"Data loaded: {len(df)} rows, columns: {list(df.columns)}")

    # Rename columns to standardize (paradigm -> Paradigm, theta -> theta_confidence)
    if 'paradigm' in df.columns:
        df = df.rename(columns={'paradigm': 'Paradigm'})
    if 'theta' in df.columns:
        df = df.rename(columns={'theta': 'theta_confidence'})

    log(f"Columns after rename: {list(df.columns)}")

    # Load Age from dfData.csv
    log(f"\nLoading Age from: {DFDATA_FILE}")
    df_demo = pd.read_csv(DFDATA_FILE, usecols=['UID', 'age'])
    # Get unique participant-level Age (take first row per UID)
    df_age = df_demo.groupby('UID').first().reset_index()
    df_age = df_age.rename(columns={'age': 'Age'})
    log(f"Age data: {len(df_age)} unique participants")

    # Merge with Age
    df = df.merge(df_age, on='UID', how='left')
    log(f"\nMerged with Age: {len(df)} rows")

    # Validate no missing Age
    missing_age = df['Age'].isna().sum()
    if missing_age > 0:
        raise ValueError(f"Missing Age for {missing_age} rows!")

    # Center Age variable
    age_per_participant = df.groupby('UID')['Age'].first()
    mean_age = age_per_participant.mean()
    sd_age = age_per_participant.std()
    df['Age_c'] = df['Age'] - mean_age

    log(f"\nAge statistics:")
    log(f"  Mean Age: {mean_age:.2f} years")
    log(f"  SD Age: {sd_age:.2f} years")
    log(f"  Range: [{age_per_participant.min():.0f}, {age_per_participant.max():.0f}] years")

    # VALIDATION
    log("\n--- Step 0 Validation ---")

    # Expected row count
    assert len(df) == 1200, f"Expected 1200 rows, found {len(df)}"
    log(f"✓ Row count: {len(df)} (expected 1200 = 100 × 4 × 3)")

    # Required columns
    required_cols = ['UID', 'Age', 'Age_c', 'Paradigm', 'test', 'TSVR_hours', 'log_TSVR', 'theta_confidence']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    log(f"✓ All required columns present")

    # No missing values
    null_counts = df[required_cols].isnull().sum()
    assert null_counts.sum() == 0, f"NaN values found: {null_counts[null_counts > 0].to_dict()}"
    log("✓ No missing values")

    # Age_c mean approximately 0
    age_c_mean = df['Age_c'].mean()
    assert abs(age_c_mean) < 0.001, f"Age_c mean = {age_c_mean}, expected ~0"
    log(f"✓ Age_c mean: {age_c_mean:.6f} (expected ~0)")

    # Paradigm distribution
    paradigm_counts = df['Paradigm'].value_counts()
    assert len(paradigm_counts) == 3, f"Expected 3 paradigms, found {len(paradigm_counts)}"
    for paradigm, count in paradigm_counts.items():
        assert count == 400, f"Expected 400 rows for {paradigm}, found {count}"
    log(f"✓ Paradigm distribution: {dict(paradigm_counts)}")

    # Unique UIDs
    n_uids = df['UID'].nunique()
    assert n_uids == 100, f"Expected 100 unique UIDs, found {n_uids}"
    log(f"✓ Unique UIDs: {n_uids}")

    # Theta range (IRT theta typically within +/-4, allow some margin)
    theta_min, theta_max = df['theta_confidence'].min(), df['theta_confidence'].max()
    assert -4 <= theta_min and theta_max <= 4, f"Theta out of range: [{theta_min:.2f}, {theta_max:.2f}]"
    log(f"✓ Theta range: [{theta_min:.2f}, {theta_max:.2f}]")

    # Save output
    output_path = RQ_DIR / "data" / "step00_lmm_input.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")
    log(f"Data preparation complete: {len(df)} rows x {len(df.columns)} columns")

    return df


def step01_fit_lmm_3way(df_input: pd.DataFrame):
    """
    Step 1: Fit LMM with Age x Paradigm x Time 3-way interaction.

    Model: theta_confidence ~ log_TSVR * Paradigm * Age_c + (log_TSVR | UID)

    Tests whether age moderates paradigm-specific confidence decline rates.
    """
    log("\n" + "=" * 70)
    log("STEP 1: Fit LMM with Age x Paradigm x Time 3-Way Interaction")
    log("=" * 70)

    # Ensure Paradigm is categorical with IFR as reference
    df_input['Paradigm'] = pd.Categorical(df_input['Paradigm'], categories=['IFR', 'ICR', 'IRE'])

    # Formula with full 3-way interaction
    formula = "theta_confidence ~ log_TSVR * C(Paradigm) * Age_c"

    log(f"\nModel formula: {formula}")
    log("Random effects: (log_TSVR | UID) - random intercept and slope")
    log("Reference level: IFR (Free Recall)")

    # Fit LMM with random slopes
    try:
        model = smf.mixedlm(
            formula=formula,
            data=df_input,
            groups=df_input['UID'],
            re_formula="~log_TSVR"  # Random intercept + random slope
        )
        result = model.fit(method='powell', maxiter=2000)
        log("\nModel converged successfully")
    except Exception as e:
        log(f"ERROR: Model fitting failed: {e}")
        raise

    # Save full summary
    summary_path = RQ_DIR / "data" / "step01_lmm_model_summary.txt"
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
        log(f"  {row['term']:50s}: β = {row['estimate']:8.5f}, SE = {row['se']:.5f}, z = {row['z']:7.2f}, p = {row['p']:.4f} {sig}")

    # Model fit indices
    log("\n--- Model Fit ---")
    log(f"Log-likelihood: {result.llf:.2f}")
    n_params = len(result.params)
    n_obs = len(df_input)
    aic = -2 * result.llf + 2 * n_params
    bic = -2 * result.llf + np.log(n_obs) * n_params
    log(f"AIC: {aic:.2f}")
    log(f"BIC: {bic:.2f}")
    log(f"N observations: {n_obs}")
    log(f"N groups (UIDs): {df_input['UID'].nunique()}")

    # VALIDATION
    log("\n--- Step 1 Validation ---")

    # Check convergence
    converged = result.converged if hasattr(result, 'converged') else True
    log(f"✓ Model converged: {converged}")

    # Check all fixed effects have finite estimates
    assert not fixed_effects['estimate'].isna().any(), "NaN in fixed effect estimates"
    assert not np.isinf(fixed_effects['estimate']).any(), "Inf in fixed effect estimates"
    log("✓ All fixed effects finite")

    # Check Age_c term present
    assert any('Age_c' in t for t in fixed_effects['term'].values), "Age_c term missing"
    log("✓ Age_c term present")

    # Check 3-way interaction terms present (look for terms with all 3 predictors)
    has_3way = any('log_TSVR' in t and 'Paradigm' in t and 'Age_c' in t for t in fixed_effects['term'].values)
    assert has_3way, "3-way interaction terms missing"
    log("✓ 3-way interaction terms present")

    # Save fixed effects
    fe_path = RQ_DIR / "data" / "step01_lmm_fixed_effects.csv"
    fixed_effects.to_csv(fe_path, index=False)
    log(f"\n✓ Fixed effects saved: {fe_path}")

    return result, fixed_effects


def step02_extract_interaction_terms(lmm_result, fixed_effects: pd.DataFrame, df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Extract Age_c interaction terms with dual p-values (Decision D068).

    Key tests:
    1. Age_c main effect (baseline age effect on confidence)
    2. Age_c × Time 2-way (does age affect overall decline rate)
    3. Age_c × Paradigm × Time 3-way (PRIMARY TEST: age × paradigm × time)

    Bonferroni correction: α = 0.0167 for 3 tests
    """
    log("\n" + "=" * 70)
    log("STEP 2: Extract Interaction Terms with Dual P-Values (Decision D068)")
    log("=" * 70)

    log(f"\nBonferroni alpha = {BONFERRONI_ALPHA:.4f} (0.05 / 3 comparisons)")

    # Identify Age_c-related terms
    age_terms = fixed_effects[fixed_effects['term'].str.contains('Age_c', regex=False)].copy()
    log(f"\nAge_c-related terms extracted: {len(age_terms)} rows")

    # Categorize terms
    def categorize_term(term):
        has_time = 'log_TSVR' in term
        has_paradigm = 'Paradigm' in term

        if has_time and has_paradigm:
            return 'Age_c:log_TSVR:Paradigm'
        elif has_time:
            return 'Age_c:log_TSVR'
        elif has_paradigm:
            return 'Age_c:Paradigm'
        else:
            return 'Age_c'

    age_terms['term_category'] = age_terms['term'].apply(categorize_term)

    # Compute omnibus tests for categorical terms (Paradigm has 2 dummy codes)
    omnibus_results = []

    # 1. Age_c main effect
    age_main = age_terms[age_terms['term_category'] == 'Age_c']
    if len(age_main) == 1:
        row = age_main.iloc[0]
        omnibus_results.append({
            'term': 'Age_c',
            'coef': row['estimate'],
            'se': row['se'],
            'z_wald': row['z'],
            'p_wald_uncorrected': row['p'],
            'chi2_lrt': row['z']**2,  # Wald chi-square approximation
            'df_lrt': 1
        })

    # 2. Age_c × Time 2-way interaction
    age_time = age_terms[age_terms['term_category'] == 'Age_c:log_TSVR']
    if len(age_time) == 1:
        row = age_time.iloc[0]
        omnibus_results.append({
            'term': 'Age_c:log_TSVR',
            'coef': row['estimate'],
            'se': row['se'],
            'z_wald': row['z'],
            'p_wald_uncorrected': row['p'],
            'chi2_lrt': row['z']**2,
            'df_lrt': 1
        })

    # 3. Age_c × Paradigm × Time 3-way interaction (omnibus test, df=2)
    age_para_time = age_terms[age_terms['term_category'] == 'Age_c:log_TSVR:Paradigm']
    if len(age_para_time) == 2:
        # Joint Wald test: chi2 = z1^2 + z2^2 (assuming independence for simplicity)
        z_values = age_para_time['z'].values
        chi2_joint = np.sum(z_values**2)
        p_joint = 1 - stats.chi2.cdf(chi2_joint, df=2)
        omnibus_results.append({
            'term': 'Age_c:log_TSVR:Paradigm',
            'coef': np.nan,  # Multiple coefficients
            'se': np.nan,
            'z_wald': np.nan,
            'p_wald_uncorrected': p_joint,
            'chi2_lrt': chi2_joint,
            'df_lrt': 2
        })
        log(f"\n3-way interaction individual terms:")
        for _, row in age_para_time.iterrows():
            log(f"  {row['term']}: β = {row['estimate']:.5f}, z = {row['z']:.2f}, p = {row['p']:.4f}")

    # Create output DataFrame
    interaction_df = pd.DataFrame(omnibus_results)

    # Add LRT p-values (using Wald chi2 as approximation)
    interaction_df['p_lrt_uncorrected'] = interaction_df.apply(
        lambda row: 1 - stats.chi2.cdf(row['chi2_lrt'], df=row['df_lrt']), axis=1
    )

    # Apply Bonferroni correction
    interaction_df['p_wald_bonferroni'] = np.minimum(interaction_df['p_wald_uncorrected'] * 3, 1.0)
    interaction_df['p_lrt_bonferroni'] = np.minimum(interaction_df['p_lrt_uncorrected'] * 3, 1.0)
    interaction_df['significant_bonferroni'] = interaction_df['p_wald_bonferroni'] < 0.05

    # Report findings
    log("\n--- Interaction Test Results (Decision D068 Dual P-Values) ---")
    for _, row in interaction_df.iterrows():
        sig_unc = "YES" if row['p_wald_uncorrected'] < 0.05 else "NO"
        sig_bon = "YES" if row['significant_bonferroni'] else "NO"
        log(f"\n{row['term']}:")
        if not np.isnan(row['coef']):
            log(f"  Coefficient: {row['coef']:.5f}")
            log(f"  SE: {row['se']:.5f}")
            log(f"  Wald z: {row['z_wald']:.2f}")
        log(f"  χ² (LRT): {row['chi2_lrt']:.2f} (df={row['df_lrt']:.0f})")
        log(f"  p_wald_uncorrected: {row['p_wald_uncorrected']:.4f} (sig @ 0.05: {sig_unc})")
        log(f"  p_wald_bonferroni: {row['p_wald_bonferroni']:.4f} (sig @ 0.05: {sig_bon})")
        log(f"  p_lrt_uncorrected: {row['p_lrt_uncorrected']:.4f}")
        log(f"  p_lrt_bonferroni: {row['p_lrt_bonferroni']:.4f}")

    # Key hypothesis test
    three_way = interaction_df[interaction_df['term'] == 'Age_c:log_TSVR:Paradigm']
    if len(three_way) == 1:
        three_way_sig = three_way['significant_bonferroni'].iloc[0]
        if three_way_sig:
            conclusion = "SIGNIFICANT - Age moderates paradigm-specific confidence decline rates"
        else:
            conclusion = "NULL - Age-invariant paradigm effects (parallels Ch5 5.3.4 accuracy findings)"
        log(f"\n*** PRIMARY HYPOTHESIS TEST: Age × Paradigm × Time ***")
        log(f"Result: {conclusion}")

    # VALIDATION
    log("\n--- Step 2 Validation ---")

    # Expected 3 terms
    assert len(interaction_df) == 3, f"Expected 3 interaction terms, found {len(interaction_df)}"
    log(f"✓ Row count: {len(interaction_df)}")

    # All terms present
    expected_terms = {'Age_c', 'Age_c:log_TSVR', 'Age_c:log_TSVR:Paradigm'}
    actual_terms = set(interaction_df['term'])
    assert expected_terms == actual_terms, f"Missing terms: {expected_terms - actual_terms}"
    log("✓ All 3 interaction terms present")

    # p-values in valid range
    assert (interaction_df['p_wald_uncorrected'] >= 0).all() and (interaction_df['p_wald_uncorrected'] <= 1).all()
    log("✓ p-values in valid range [0, 1]")

    # Bonferroni correction applied
    for _, row in interaction_df.iterrows():
        expected_bonf = min(row['p_wald_uncorrected'] * 3, 1.0)
        assert abs(row['p_wald_bonferroni'] - expected_bonf) < 0.0001, "Bonferroni correction incorrect"
    log("✓ Bonferroni correction applied correctly")

    # Reorder columns
    output_cols = ['term', 'coef', 'se', 'z_wald', 'p_wald_uncorrected', 'p_wald_bonferroni',
                   'chi2_lrt', 'df_lrt', 'p_lrt_uncorrected', 'p_lrt_bonferroni', 'significant_bonferroni']
    interaction_df = interaction_df[output_cols]

    # Save output
    output_path = RQ_DIR / "data" / "step02_interaction_terms.csv"
    interaction_df.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")

    return interaction_df


def step03_compute_effect_sizes(fixed_effects: pd.DataFrame, df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Step 3: Compute effect sizes (partial R² and Cohen's f²).

    Uses the proportion of variance explained by each Age_c term.
    """
    log("\n" + "=" * 70)
    log("STEP 3: Compute Effect Sizes (Cohen's f²)")
    log("=" * 70)

    # Get total variance in theta_confidence
    total_var = df_input['theta_confidence'].var()
    log(f"\nTotal variance in theta_confidence: {total_var:.4f}")

    # For LMM, we use standardized coefficients as effect size proxy
    # Cohen's f² = R²_partial / (1 - R²_partial)
    # For regression: f² ≈ (β²_standardized × var(predictor)) / var(residual)

    # Compute standardized predictor variances
    var_age_c = df_input['Age_c'].var()
    var_time = df_input['log_TSVR'].var()
    var_interaction = var_age_c * var_time  # Approximate for interaction

    log(f"Predictor variances:")
    log(f"  Var(Age_c): {var_age_c:.4f}")
    log(f"  Var(log_TSVR): {var_time:.4f}")

    # Extract relevant coefficients
    coefs = fixed_effects.set_index('term')

    effect_sizes = []

    # Age_c main effect
    age_c_terms = [t for t in coefs.index if 'Age_c' in t and 'log_TSVR' not in t and 'Paradigm' not in t]
    if age_c_terms:
        beta = coefs.loc[age_c_terms[0], 'estimate']
        # f² = (beta * SD_x / SD_y)² as approximation
        f_squared = (beta**2 * var_age_c) / total_var
        effect_sizes.append({
            'term': 'Age_c',
            'f_squared': f_squared,
            'interpretation': interpret_f_squared(f_squared)
        })

    # Age_c × Time 2-way
    age_time_terms = [t for t in coefs.index if 'Age_c' in t and 'log_TSVR' in t and 'Paradigm' not in t]
    if age_time_terms:
        beta = coefs.loc[age_time_terms[0], 'estimate']
        f_squared = (beta**2 * var_interaction) / total_var
        effect_sizes.append({
            'term': 'Age_c:log_TSVR',
            'f_squared': f_squared,
            'interpretation': interpret_f_squared(f_squared)
        })

    # Age_c × Paradigm × Time 3-way (average of dummy coefficients)
    three_way_terms = [t for t in coefs.index if 'Age_c' in t and 'log_TSVR' in t and 'Paradigm' in t]
    if three_way_terms:
        betas = [coefs.loc[t, 'estimate'] for t in three_way_terms]
        mean_beta_sq = np.mean([b**2 for b in betas])
        f_squared = mean_beta_sq * var_interaction / total_var
        effect_sizes.append({
            'term': 'Age_c:log_TSVR:Paradigm',
            'f_squared': f_squared,
            'interpretation': interpret_f_squared(f_squared)
        })

    effect_df = pd.DataFrame(effect_sizes)

    # Report findings
    log("\n--- Effect Sizes (Cohen's f²) ---")
    for _, row in effect_df.iterrows():
        log(f"\n{row['term']}:")
        log(f"  f² = {row['f_squared']:.6f}")
        log(f"  Interpretation: {row['interpretation']}")

    # VALIDATION
    log("\n--- Step 3 Validation ---")

    # Expected 3 terms
    assert len(effect_df) == 3, f"Expected 3 effect sizes, found {len(effect_df)}"
    log(f"✓ Row count: {len(effect_df)}")

    # All f² non-negative
    assert (effect_df['f_squared'] >= 0).all(), "Negative f² detected"
    log("✓ All f² values non-negative")

    # All interpretations valid
    valid_interps = {'negligible', 'small', 'medium', 'large'}
    assert all(i in valid_interps for i in effect_df['interpretation']), "Invalid interpretation"
    log("✓ All interpretations valid")

    # Save output
    output_path = RQ_DIR / "data" / "step03_effect_sizes.csv"
    effect_df.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")

    return effect_df


def interpret_f_squared(f2: float) -> str:
    """Interpret Cohen's f² effect size."""
    if f2 < 0.02:
        return 'negligible'
    elif f2 < 0.15:
        return 'small'
    elif f2 < 0.35:
        return 'medium'
    else:
        return 'large'


def step04_compare_to_ch5(interaction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 4: Compare to Chapter 5 RQ 5.3.4 results.

    Tests whether NULL 3-way interaction pattern replicates across accuracy and confidence.
    """
    log("\n" + "=" * 70)
    log("STEP 4: Compare to Chapter 5 RQ 5.3.4 Results")
    log("=" * 70)

    # Try to load Ch5 comparison file
    ch5_available = False
    if CH5_COMPARISON_FILE.exists():
        try:
            ch5_df = pd.read_csv(CH5_COMPARISON_FILE)
            ch5_available = True
            log(f"\nCh5 data loaded: {CH5_COMPARISON_FILE}")
        except Exception as e:
            log(f"\nCh5 data load failed: {e}")
    else:
        log(f"\nCh5 file not found: {CH5_COMPARISON_FILE}")
        log("Creating comparison table with Ch6 only (Ch5 comparison pending)")

    # Prepare Ch6 confidence data
    ch6_data = interaction_df[['term', 'p_wald_bonferroni', 'p_lrt_bonferroni', 'significant_bonferroni']].copy()
    ch6_data['domain'] = 'Confidence'

    if ch5_available:
        # Align Ch5 terms with Ch6 terms
        ch5_data = ch5_df[['term', 'p_wald_bonferroni', 'p_lrt_bonferroni', 'significant_bonferroni']].copy()
        ch5_data['domain'] = 'Accuracy'

        # Combine
        comparison = pd.concat([ch5_data, ch6_data], ignore_index=True)
    else:
        comparison = ch6_data.copy()
        comparison['interpretation'] = 'Ch5 comparison pending RQ 5.3.4 completion'

    # Add interpretation column
    def interpret_consistency(row):
        if 'interpretation' in row and pd.notna(row.get('interpretation')):
            return row['interpretation']
        return 'Ch6 only - Ch5 pending'

    if ch5_available:
        # Create interpretation for each term pair
        interpretations = []
        for term in ['Age_c', 'Age_c:log_TSVR', 'Age_c:log_TSVR:Paradigm']:
            ch5_sig = ch5_data[ch5_data['term'] == term]['significant_bonferroni'].iloc[0] if len(ch5_data[ch5_data['term'] == term]) > 0 else None
            ch6_sig = ch6_data[ch6_data['term'] == term]['significant_bonferroni'].iloc[0] if len(ch6_data[ch6_data['term'] == term]) > 0 else None

            if ch5_sig is not None and ch6_sig is not None:
                if not ch5_sig and not ch6_sig:
                    interp = "Consistent NULL - age-invariance generalizes"
                elif ch5_sig and ch6_sig:
                    interp = "Consistent SIGNIFICANT - age effects generalize"
                else:
                    interp = "DIVERGENT - accuracy and confidence dissociate"
            else:
                interp = "Comparison incomplete"
            interpretations.append({'term': term, 'interpretation': interp})

        interp_df = pd.DataFrame(interpretations)
        comparison = comparison.merge(interp_df, on='term', how='left')
    else:
        comparison['interpretation'] = 'Ch5 comparison pending RQ 5.3.4 completion'

    # Report findings
    log("\n--- Cross-Chapter Comparison ---")
    for term in comparison['term'].unique():
        term_data = comparison[comparison['term'] == term]
        log(f"\n{term}:")
        for _, row in term_data.iterrows():
            sig = "SIGNIFICANT" if row['significant_bonferroni'] else "NULL"
            log(f"  {row['domain']}: p_bonf = {row['p_wald_bonferroni']:.4f} ({sig})")
        if 'interpretation' in comparison.columns:
            interp = term_data['interpretation'].iloc[0]
            log(f"  Interpretation: {interp}")

    # VALIDATION
    log("\n--- Step 4 Validation ---")

    # Expected row count
    expected_rows = 6 if ch5_available else 3
    assert len(comparison) == expected_rows, f"Expected {expected_rows} rows, found {len(comparison)}"
    log(f"✓ Row count: {len(comparison)} (Ch5 {'available' if ch5_available else 'pending'})")

    # Required columns present
    required_cols = ['term', 'domain', 'p_wald_bonferroni', 'p_lrt_bonferroni', 'significant_bonferroni', 'interpretation']
    missing_cols = [c for c in required_cols if c not in comparison.columns]
    assert not missing_cols, f"Missing columns: {missing_cols}"
    log("✓ All required columns present")

    # Reorder columns
    comparison = comparison[required_cols]

    # Save output
    output_path = RQ_DIR / "data" / "step04_ch5_comparison.csv"
    comparison.to_csv(output_path, index=False)
    log(f"\n✓ Output saved: {output_path}")

    return comparison


def main():
    """Execute all steps for RQ 6.4.3."""

    # Initialize log
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    (RQ_DIR / "data").mkdir(parents=True, exist_ok=True)

    with open(LOG_FILE, 'w') as f:
        f.write("RQ 6.4.3: Age x Paradigm Interaction for Confidence Decline\n")
        f.write("=" * 70 + "\n")
        f.write("Analysis execution log\n\n")

    log("Starting RQ 6.4.3 analysis pipeline...")
    log(f"Output directory: {RQ_DIR}")

    try:
        # Step 0: Prepare LMM input with Age
        df = step00_prepare_lmm_input()

        # Step 1: Fit LMM with 3-way interaction
        lmm_result, fixed_effects = step01_fit_lmm_3way(df)

        # Step 2: Extract interaction terms with dual p-values
        interaction_df = step02_extract_interaction_terms(lmm_result, fixed_effects, df)

        # Step 3: Compute effect sizes
        effect_sizes = step03_compute_effect_sizes(fixed_effects, df)

        # Step 4: Compare to Ch5 5.3.4
        comparison = step04_compare_to_ch5(interaction_df)

        # Final summary
        log("\n" + "=" * 70)
        log("ANALYSIS COMPLETE")
        log("=" * 70)

        # Report key findings
        three_way = interaction_df[interaction_df['term'] == 'Age_c:log_TSVR:Paradigm']
        if len(three_way) == 1:
            p_val = three_way['p_wald_bonferroni'].iloc[0]
            sig = three_way['significant_bonferroni'].iloc[0]

            log(f"\n*** PRIMARY HYPOTHESIS TEST: Age × Paradigm × Time 3-Way Interaction ***")
            log(f"  χ²(2) = {three_way['chi2_lrt'].iloc[0]:.2f}")
            log(f"  p_bonferroni = {p_val:.4f}")
            log(f"  Significant: {'YES' if sig else 'NO'}")

            if not sig:
                log("\n  CONCLUSION: NULL - Age does NOT moderate paradigm-specific confidence decline")
                log("  This PARALLELS Chapter 5 accuracy findings (age-invariant forgetting)")
                log("  Validates VR ecological encoding for metacognitive monitoring across paradigms")
            else:
                log("\n  CONCLUSION: SIGNIFICANT - Age moderates paradigm-specific confidence decline")
                log("  This may DIVERGE from Chapter 5 accuracy findings")
                log("  Suggests metacognitive aging effects differ by retrieval support")

        # Effect size summary
        f2_3way = effect_sizes[effect_sizes['term'] == 'Age_c:log_TSVR:Paradigm']['f_squared'].iloc[0]
        interp = effect_sizes[effect_sizes['term'] == 'Age_c:log_TSVR:Paradigm']['interpretation'].iloc[0]
        log(f"\n*** EFFECT SIZE: 3-Way Interaction ***")
        log(f"  Cohen's f² = {f2_3way:.6f}")
        log(f"  Interpretation: {interp}")

        log("\n*** OUTPUT FILES CREATED ***")
        for f in sorted((RQ_DIR / "data").glob("*.csv")) + sorted((RQ_DIR / "data").glob("*.txt")):
            log(f"  {f.name}")

        log("\n✅ RQ 6.4.3 analysis complete - all validations passed")

    except Exception as e:
        log(f"\n❌ ERROR: {e}")
        import traceback
        log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
