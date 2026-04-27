#!/usr/bin/env python3
"""
RQ 6.5.2: Schema Confidence Calibration
Steps 00-02: Merge theta scores, compute calibration, fit LMM

Hypothesis: Congruent items show OVERCONFIDENCE (schema-driven familiarity inflates
confidence without corresponding accuracy gains - Ch5 5.4.1 found NULL schema effects
on accuracy).

Created: 2025-12-12
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]  # results/ch6/6.5.2
PROJECT_ROOT = RQ_DIR.parents[2]  # REMEMVR root
LOG_FILE = RQ_DIR / "logs" / "steps_00_to_02.log"

# Ensure logs directory exists
(RQ_DIR / "logs").mkdir(exist_ok=True)

def log(msg: str):
    """Log message to file and stdout."""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# =============================================================================
# STEP 00: Merge Accuracy and Confidence Theta Scores
# =============================================================================
def step00_merge_accuracy_confidence():
    """Merge accuracy theta (RQ 5.4.1) and confidence theta (RQ 6.5.1)."""
    log("\n" + "="*60)
    log("STEP 00: Merge Accuracy and Confidence Theta Scores")
    log("="*60)

    # Read accuracy theta from RQ 5.4.1
    acc_path = PROJECT_ROOT / "results/ch5/5.4.1/data/step03_theta_scores.csv"
    log(f"Reading accuracy theta: {acc_path}")
    df_acc = pd.read_csv(acc_path)
    log(f"  Accuracy data: {len(df_acc)} rows, columns: {list(df_acc.columns)}")

    # Read confidence theta from RQ 6.5.1
    conf_path = PROJECT_ROOT / "results/ch6/6.5.1/data/step03_theta_confidence.csv"
    log(f"Reading confidence theta: {conf_path}")
    df_conf = pd.read_csv(conf_path)
    log(f"  Confidence data: {len(df_conf)} rows, columns: {list(df_conf.columns)}")

    # Read TSVR mapping from RQ 6.5.1
    tsvr_path = PROJECT_ROOT / "results/ch6/6.5.1/data/step00_tsvr_mapping.csv"
    log(f"Reading TSVR mapping: {tsvr_path}")
    df_tsvr = pd.read_csv(tsvr_path)
    log(f"  TSVR data: {len(df_tsvr)} rows")

    # CRITICAL: Normalize composite_ID formats
    # Accuracy uses: A010_1 (UID_test_number)
    # Confidence uses: A010_T1 (UID_T+test_number)
    # Normalize to: A010_T1 format

    def normalize_composite_id(cid):
        """Convert A010_1 format to A010_T1 format."""
        if '_T' in str(cid):
            return cid  # Already in correct format
        parts = str(cid).split('_')
        if len(parts) == 2:
            uid, test_num = parts
            return f"{uid}_T{test_num}"
        return cid

    df_acc['composite_ID_norm'] = df_acc['composite_ID'].apply(normalize_composite_id)
    df_conf['composite_ID_norm'] = df_conf['composite_ID']  # Already in T format

    log(f"  Normalized accuracy IDs: {df_acc['composite_ID_norm'].iloc[:3].tolist()}")
    log(f"  Confidence IDs: {df_conf['composite_ID_norm'].iloc[:3].tolist()}")

    # Merge on normalized composite_ID
    df_merged = pd.merge(
        df_acc[['composite_ID_norm', 'theta_common', 'theta_congruent', 'theta_incongruent',
                'se_common', 'se_congruent', 'se_incongruent']],
        df_conf[['composite_ID_norm', 'theta_Common', 'theta_Congruent', 'theta_Incongruent']],
        on='composite_ID_norm',
        how='inner'
    )

    log(f"  Merged: {len(df_merged)} rows (expected ~400)")

    # Merge with TSVR for timing
    df_merged = pd.merge(
        df_merged,
        df_tsvr[['composite_ID', 'UID', 'TSVR_hours', 'test']],
        left_on='composite_ID_norm',
        right_on='composite_ID',
        how='inner'
    )

    log(f"  After TSVR merge: {len(df_merged)} rows")

    # Reshape to long format (3 rows per composite_ID)
    records = []
    for _, row in df_merged.iterrows():
        for cong in ['common', 'congruent', 'incongruent']:
            # Accuracy column names (lowercase)
            theta_acc = row[f'theta_{cong}']
            se_acc = row[f'se_{cong}']

            # Confidence column names (Title case in source)
            cong_title = cong.capitalize()
            theta_conf = row[f'theta_{cong_title}']

            records.append({
                'composite_ID': row['composite_ID_norm'],
                'UID': row['UID'],
                'test': row['test'],
                'congruence': cong_title,  # Common, Congruent, Incongruent
                'theta_accuracy': theta_acc,
                'se_accuracy': se_acc,
                'theta_confidence': theta_conf,
                'se_confidence': np.nan,  # Not available in source
                'TSVR_hours': row['TSVR_hours']
            })

    df_long = pd.DataFrame(records)
    log(f"  Long format: {len(df_long)} rows (expected ~1200 = 400 x 3)")

    # Validation
    n_unique_ids = df_long['composite_ID'].nunique()
    n_congruence_levels = df_long['congruence'].nunique()
    log(f"  Unique composite_IDs: {n_unique_ids}")
    log(f"  Congruence levels: {df_long['congruence'].unique().tolist()}")

    # Check for missing values
    nan_acc = df_long['theta_accuracy'].isna().sum()
    nan_conf = df_long['theta_confidence'].isna().sum()
    log(f"  NaN in theta_accuracy: {nan_acc}")
    log(f"  NaN in theta_confidence: {nan_conf}")

    if nan_acc > 0 or nan_conf > 0:
        log("WARNING: NaN values detected in theta columns")

    # Save
    out_path = RQ_DIR / "data" / "step00_merged_accuracy_confidence.csv"
    df_long.to_csv(out_path, index=False)
    log(f"  Saved: {out_path}")
    log(f"STEP 00 COMPLETE: {len(df_long)} observations merged")

    return df_long

# =============================================================================
# STEP 01: Compute Calibration Scores
# =============================================================================
def step01_compute_calibration(df: pd.DataFrame):
    """Compute calibration = z(confidence) - z(accuracy) within each congruence level."""
    log("\n" + "="*60)
    log("STEP 01: Compute Calibration Scores")
    log("="*60)

    # Z-standardize WITHIN each congruence level
    df = df.copy()

    for cong in ['Common', 'Congruent', 'Incongruent']:
        mask = df['congruence'] == cong

        # Z-standardize accuracy
        acc_vals = df.loc[mask, 'theta_accuracy']
        df.loc[mask, 'theta_accuracy_z'] = (acc_vals - acc_vals.mean()) / acc_vals.std()

        # Z-standardize confidence
        conf_vals = df.loc[mask, 'theta_confidence']
        df.loc[mask, 'theta_confidence_z'] = (conf_vals - conf_vals.mean()) / conf_vals.std()

        log(f"  {cong}:")
        log(f"    Accuracy: mean={acc_vals.mean():.3f}, SD={acc_vals.std():.3f}")
        log(f"    Confidence: mean={conf_vals.mean():.3f}, SD={conf_vals.std():.3f}")
        log(f"    Z-acc: mean={df.loc[mask, 'theta_accuracy_z'].mean():.6f}, SD={df.loc[mask, 'theta_accuracy_z'].std():.3f}")
        log(f"    Z-conf: mean={df.loc[mask, 'theta_confidence_z'].mean():.6f}, SD={df.loc[mask, 'theta_confidence_z'].std():.3f}")

    # Compute calibration: positive = overconfidence, negative = underconfidence
    df['calibration'] = df['theta_confidence_z'] - df['theta_accuracy_z']

    # Summary statistics
    log("\nCalibration Summary:")
    for cong in ['Common', 'Congruent', 'Incongruent']:
        mask = df['congruence'] == cong
        cal = df.loc[mask, 'calibration']
        log(f"  {cong}: mean={cal.mean():.4f}, SD={cal.std():.4f}, range=[{cal.min():.3f}, {cal.max():.3f}]")

    # Add log_TSVR for LMM (common transformation)
    df['log_TSVR'] = np.log(df['TSVR_hours'] + 1)

    # Validation: check standardization
    for cong in ['Common', 'Congruent', 'Incongruent']:
        mask = df['congruence'] == cong
        acc_z_mean = df.loc[mask, 'theta_accuracy_z'].mean()
        acc_z_std = df.loc[mask, 'theta_accuracy_z'].std()
        conf_z_mean = df.loc[mask, 'theta_confidence_z'].mean()
        conf_z_std = df.loc[mask, 'theta_confidence_z'].std()

        if abs(acc_z_mean) > 0.01 or abs(acc_z_std - 1) > 0.01:
            log(f"WARNING: Z-standardization issue for {cong} accuracy: mean={acc_z_mean:.6f}, SD={acc_z_std:.6f}")
        if abs(conf_z_mean) > 0.01 or abs(conf_z_std - 1) > 0.01:
            log(f"WARNING: Z-standardization issue for {cong} confidence: mean={conf_z_mean:.6f}, SD={conf_z_std:.6f}")

    # Save
    out_cols = ['composite_ID', 'UID', 'test', 'congruence', 'TSVR_hours', 'log_TSVR',
                'theta_accuracy', 'theta_confidence', 'theta_accuracy_z',
                'theta_confidence_z', 'calibration']
    out_path = RQ_DIR / "data" / "step01_calibration_by_congruence.csv"
    df[out_cols].to_csv(out_path, index=False)
    log(f"\nSaved: {out_path}")
    log(f"STEP 01 COMPLETE: Calibration computed for {len(df)} observations")

    return df

# =============================================================================
# STEP 02: Fit LMM and Test Congruence Effects
# =============================================================================
def step02_fit_lmm_congruence(df: pd.DataFrame):
    """Fit LMM testing Congruence × Time interaction on calibration."""
    log("\n" + "="*60)
    log("STEP 02: Fit LMM and Test Congruence Effects")
    log("="*60)

    # Prepare data
    df = df.copy()
    df['congruence'] = pd.Categorical(
        df['congruence'],
        categories=['Common', 'Congruent', 'Incongruent'],
        ordered=False
    )

    # Reference level = Common (baseline)
    log(f"Reference level: Common")
    log(f"Observations: {len(df)}")
    log(f"Unique participants: {df['UID'].nunique()}")

    # Fit LMM with random slopes
    formula = "calibration ~ C(congruence, Treatment('Common')) * log_TSVR"
    log(f"Formula: {formula}")
    log(f"Random effects: ~log_TSVR | UID")

    try:
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df['UID'],
            re_formula="~log_TSVR"
        )
        result = model.fit(method='powell', reml=False)
        log(f"Model converged: {result.converged}")

    except Exception as e:
        log(f"ERROR: Model fitting failed: {e}")
        # Try simpler model with intercept-only random effects
        log("Attempting intercept-only random effects...")
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df['UID'],
            re_formula="~1"
        )
        result = model.fit(method='powell', reml=False)
        log(f"Intercept-only model converged: {result.converged}")

    # Save model summary
    summary_path = RQ_DIR / "data" / "step02_lmm_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(str(result.summary()))
    log(f"Saved: {summary_path}")

    # Extract fixed effects
    n_fe = len(result.model.exog_names)
    fe_names = result.model.exog_names
    fe_params = result.params[:n_fe]
    fe_se = result.bse[:n_fe]
    fe_z = fe_params / fe_se
    fe_pvals = result.pvalues[:n_fe]

    log("\nFixed Effects:")
    for i, name in enumerate(fe_names):
        log(f"  {name}: β={fe_params[i]:.4f}, SE={fe_se[i]:.4f}, z={fe_z[i]:.3f}, p={fe_pvals[i]:.4f}")

    # Create effects dataframe
    effects_df = pd.DataFrame({
        'effect': fe_names,
        'coefficient': fe_params.values,
        'SE': fe_se.values,
        'z': fe_z.values,
        'p_parametric': fe_pvals.values,
        'p_bootstrap': np.nan,  # Would need bootstrap for this
        'CI_lower': fe_params.values - 1.96 * fe_se.values,
        'CI_upper': fe_params.values + 1.96 * fe_se.values
    })
    effects_path = RQ_DIR / "data" / "step02_congruence_effects.csv"
    effects_df.to_csv(effects_path, index=False)
    log(f"Saved: {effects_path}")

    # Post-hoc contrasts (from model coefficients)
    # Congruent vs Common = coefficient for Congruent dummy
    # Incongruent vs Common = coefficient for Incongruent dummy
    # Congruent vs Incongruent = Congruent_coef - Incongruent_coef

    contrasts = []

    # Find coefficient indices
    cong_idx = [i for i, n in enumerate(fe_names) if 'Congruent' in n and 'Incongruent' not in n and ':' not in n]
    incong_idx = [i for i, n in enumerate(fe_names) if 'Incongruent' in n and ':' not in n]

    if cong_idx and incong_idx:
        cong_idx = cong_idx[0]
        incong_idx = incong_idx[0]

        # Congruent - Common
        est1 = fe_params[cong_idx]
        se1 = fe_se[cong_idx]
        z1 = fe_z[cong_idx]
        p1 = fe_pvals[cong_idx]
        contrasts.append({
            'contrast': 'Congruent - Common',
            'estimate': est1,
            'SE': se1,
            'z': z1,
            'p_uncorrected': p1,
            'p_bonferroni': min(p1 * 3, 1.0),
            'CI_lower': est1 - 1.96 * se1,
            'CI_upper': est1 + 1.96 * se1
        })

        # Incongruent - Common
        est2 = fe_params[incong_idx]
        se2 = fe_se[incong_idx]
        z2 = fe_z[incong_idx]
        p2 = fe_pvals[incong_idx]
        contrasts.append({
            'contrast': 'Incongruent - Common',
            'estimate': est2,
            'SE': se2,
            'z': z2,
            'p_uncorrected': p2,
            'p_bonferroni': min(p2 * 3, 1.0),
            'CI_lower': est2 - 1.96 * se2,
            'CI_upper': est2 + 1.96 * se2
        })

        # Congruent - Incongruent (need to compute from coefficients)
        est3 = est1 - est2
        # SE for difference: sqrt(var1 + var2 - 2*cov)
        # Approximate with independent: sqrt(se1^2 + se2^2)
        se3 = np.sqrt(se1**2 + se2**2)
        z3 = est3 / se3
        p3 = 2 * (1 - stats.norm.cdf(abs(z3)))
        contrasts.append({
            'contrast': 'Congruent - Incongruent',
            'estimate': est3,
            'SE': se3,
            'z': z3,
            'p_uncorrected': p3,
            'p_bonferroni': min(p3 * 3, 1.0),
            'CI_lower': est3 - 1.96 * se3,
            'CI_upper': est3 + 1.96 * se3
        })

    contrasts_df = pd.DataFrame(contrasts)
    contrasts_path = RQ_DIR / "data" / "step02_post_hoc_contrasts.csv"
    contrasts_df.to_csv(contrasts_path, index=False)
    log(f"Saved: {contrasts_path}")

    log("\nPost-hoc Contrasts (Bonferroni-corrected alpha = 0.0167):")
    for _, row in contrasts_df.iterrows():
        sig = "***" if row['p_bonferroni'] < 0.001 else ("**" if row['p_bonferroni'] < 0.01 else ("*" if row['p_bonferroni'] < 0.05 else ""))
        log(f"  {row['contrast']}: est={row['estimate']:.4f}, SE={row['SE']:.4f}, z={row['z']:.3f}, p_bonf={row['p_bonferroni']:.4f} {sig}")

    # Effect sizes (Cohen's f²)
    # f² = R²_model / (1 - R²_model) for fixed effects
    # Approximate using variance explained by each term

    # Calculate variance explained
    y_pred = result.fittedvalues
    y_obs = df['calibration']
    ss_res = np.sum((y_obs - y_pred)**2)
    ss_tot = np.sum((y_obs - y_obs.mean())**2)
    r_squared = 1 - ss_res/ss_tot

    log(f"\nModel R²: {r_squared:.4f}")

    # Approximate f² for each fixed effect using coefficient magnitude relative to residual variance
    sigma_sq = result.scale  # Residual variance

    effect_sizes = []
    for i, name in enumerate(fe_names):
        if name == 'Intercept':
            continue
        # Cohen's f² approximation: (β²) / (σ² * (1 + f²_other_effects))
        # Simplified: f² ≈ β² / σ²
        f_sq = (fe_params[i]**2) / sigma_sq

        if f_sq < 0.02:
            interp = "negligible"
        elif f_sq < 0.15:
            interp = "small"
        elif f_sq < 0.35:
            interp = "medium"
        else:
            interp = "large"

        effect_sizes.append({
            'effect': name,
            'f_squared': f_sq,
            'interpretation': interp
        })

    effect_sizes_df = pd.DataFrame(effect_sizes)
    effect_sizes_path = RQ_DIR / "data" / "step02_effect_sizes.csv"
    effect_sizes_df.to_csv(effect_sizes_path, index=False)
    log(f"Saved: {effect_sizes_path}")

    log("\nEffect Sizes (Cohen's f²):")
    for _, row in effect_sizes_df.iterrows():
        log(f"  {row['effect']}: f²={row['f_squared']:.4f} ({row['interpretation']})")

    # Hypothesis test summary
    log("\n" + "="*60)
    log("HYPOTHESIS TEST SUMMARY")
    log("="*60)

    # Check overconfidence hypothesis: Congruent > Common
    if len(contrasts) >= 1:
        cong_common = contrasts_df[contrasts_df['contrast'] == 'Congruent - Common'].iloc[0]
        if cong_common['estimate'] > 0 and cong_common['p_bonferroni'] < 0.05:
            log("HYPOTHESIS SUPPORTED: Congruent shows significant overconfidence vs Common")
            log(f"  Difference: {cong_common['estimate']:.4f} (p_bonf={cong_common['p_bonferroni']:.4f})")
        elif cong_common['estimate'] > 0:
            log("HYPOTHESIS DIRECTION CORRECT but not significant: Congruent > Common (trend)")
            log(f"  Difference: {cong_common['estimate']:.4f} (p_bonf={cong_common['p_bonferroni']:.4f})")
        else:
            log("HYPOTHESIS REFUTED: Congruent does NOT show overconfidence vs Common")
            log(f"  Difference: {cong_common['estimate']:.4f} (p_bonf={cong_common['p_bonferroni']:.4f})")

    # Check interaction effects
    interaction_terms = [i for i, n in enumerate(fe_names) if ':' in n]
    if interaction_terms:
        log("\nCongruence × Time Interaction:")
        for idx in interaction_terms:
            name = fe_names[idx]
            p = fe_pvals[idx]
            sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"
            log(f"  {name}: p={p:.4f} ({sig})")

    log(f"\nSTEP 02 COMPLETE")

    return result, effects_df, contrasts_df

# =============================================================================
# MAIN
# =============================================================================
def main():
    log(f"RQ 6.5.2: Schema Confidence Calibration")
    log(f"Started: {pd.Timestamp.now()}")
    log(f"RQ_DIR: {RQ_DIR}")

    # Step 00: Merge
    df_merged = step00_merge_accuracy_confidence()

    # Step 01: Compute calibration
    df_cal = step01_compute_calibration(df_merged)

    # Step 02: Fit LMM
    result, effects_df, contrasts_df = step02_fit_lmm_congruence(df_cal)

    log("\n" + "="*60)
    log("ALL STEPS COMPLETE")
    log(f"Finished: {pd.Timestamp.now()}")
    log("="*60)

    return 0

if __name__ == "__main__":
    sys.exit(main())
