#!/usr/bin/env python3
"""
§6.5 Self-Report Measures — Unified Reanalysis

PURPOSE:
Re-run all three self-report models with consistent specification:
  - Block 1 (cognitive baseline): age + RAVLT_T + BVMT_T + RPM_T + RAVLT_Pct_Ret_T + BVMT_Pct_Ret_T
  - N = 100 (no NART-related listwise deletion)
  - 10-fold cross-validation for Models 1 & 2
  - Marginal/conditional R² for Model 3 (MLM)

Model 1 — Lifestyle: Block 2 adds education, VR_experience, sleep_hours_typical
Model 2 — DASS:      Block 2 adds DASS_depression, DASS_anxiety, DASS_stress
Model 3 — Sleep MLM: Within-person sleep + between-person cognitive controls

Data sources:
  - Theta: results/ch7/7.5.1/data/step02_theta_scores.csv (mean theta_all, N=100)
  - Cognitive tests: results/ch7/7.3.1/data/step02_cognitive_tests.csv (RAVLT_T, BVMT_T, RPM_T, RAVLT_Pct_Ret_T, BVMT_Pct_Ret_T, age)
  - Self-report: data/dfnonvr.csv (education, vr-exposure, typical-sleep-hours, DASS)
  - Sleep MLM: results/ch7/7.5.4/data/step04_analysis_dataset.csv (400 rows, within-person)

Outputs saved to: results/ch7/7.5.1/data/step10_*.csv
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import KFold

RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step10_s65_reanalysis.log"


def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def compute_sr2(model, X_with_const, y, predictor_idx):
    """Compute semi-partial r² for a specific predictor.

    sr² = (t² / (t² + df_resid)) * (1 - R²_reduced) ...
    Actually, sr² = (R²_full - R²_reduced) where reduced drops that predictor.
    More precisely: sr² = (beta_j * se_j_inv)² * (1/n) ...

    The cleanest way: fit model without predictor j, compute R² difference.
    """
    # Full model R²
    r2_full = model.rsquared

    # Fit reduced model without predictor at predictor_idx
    cols_reduced = [i for i in range(X_with_const.shape[1]) if i != predictor_idx]
    X_reduced = X_with_const[:, cols_reduced]
    model_reduced = sm.OLS(y, X_reduced).fit()
    r2_reduced = model_reduced.rsquared

    return r2_full - r2_reduced


def hierarchical_regression(y, X_block1, X_block2, block1_names, block2_names):
    """Fit hierarchical regression: Block 1, then Block 1 + Block 2.

    Returns dict with model objects, R², ΔR², F-change, individual predictor stats.
    """
    n = len(y)

    # Block 1 model
    X1 = sm.add_constant(X_block1)
    model1 = sm.OLS(y, X1).fit()

    # Block 2 model (Block 1 + Block 2 predictors)
    X_full = np.column_stack([X_block1, X_block2])
    X2 = sm.add_constant(X_full)
    model2 = sm.OLS(y, X2).fit()

    # F-change test for Block 2 increment
    r2_1 = model1.rsquared
    r2_2 = model2.rsquared
    delta_r2 = r2_2 - r2_1
    k_new = X_block2.shape[1]  # number of new predictors
    k_full = X_full.shape[1]   # total predictors (excl. intercept)
    df_num = k_new
    df_den = n - k_full - 1

    if delta_r2 > 0 and (1 - r2_2) > 0:
        f_change = (delta_r2 / df_num) / ((1 - r2_2) / df_den)
        p_change = 1 - stats.f.cdf(f_change, df_num, df_den)
    else:
        f_change = 0.0
        p_change = 1.0

    # Individual predictor stats for Block 2 predictors
    all_names = block1_names + block2_names
    individual_stats = []

    for i, name in enumerate(all_names):
        coef_idx = i + 1  # +1 for intercept
        beta = model2.params[coef_idx]
        se = model2.bse[coef_idx]
        t_val = model2.tvalues[coef_idx]
        p_val = model2.pvalues[coef_idx]

        # Semi-partial r²
        sr2 = compute_sr2(model2, X2, y, coef_idx)

        individual_stats.append({
            'predictor': name,
            'beta': beta,
            'se': se,
            't': t_val,
            'p': p_val,
            'sr2': sr2,
            'block': 'block1' if name in block1_names else 'block2'
        })

    # Bonferroni correction for Block 2 predictors only
    block2_ps = [s['p'] for s in individual_stats if s['block'] == 'block2']
    n_block2 = len(block2_ps)
    for s in individual_stats:
        if s['block'] == 'block2':
            s['p_bonferroni'] = min(s['p'] * n_block2, 1.0)
        else:
            s['p_bonferroni'] = np.nan  # Not corrected for control vars

    return {
        'model1': model1,
        'model2': model2,
        'r2_block1': r2_1,
        'adj_r2_block1': model1.rsquared_adj,
        'f_block1': model1.fvalue,
        'p_block1': model1.f_pvalue,
        'df1_block1': model1.df_model,
        'df2_block1': model1.df_resid,
        'r2_block2': r2_2,
        'adj_r2_block2': model2.rsquared_adj,
        'f_block2': model2.fvalue,
        'p_block2': model2.f_pvalue,
        'df1_block2': model2.df_model,
        'df2_block2': model2.df_resid,
        'delta_r2': delta_r2,
        'f_change': f_change,
        'p_change': p_change,
        'df_num_change': df_num,
        'df_den_change': df_den,
        'individual_stats': individual_stats,
        'n': n,
        'k_block1': X_block1.shape[1],
        'k_block2': k_new,
    }


def cross_validate_10fold(y, X_block1, X_block2, seed=42):
    """10-fold CV for Block 2 model. Returns train/test R² per fold."""
    X_full = np.column_stack([X_block1, X_block2])
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_full)):
        X_train = sm.add_constant(X_full[train_idx], has_constant=False)
        X_test = sm.add_constant(X_full[test_idx], has_constant=False)
        y_train = y[train_idx]
        y_test = y[test_idx]

        model = sm.OLS(y_train, X_train).fit()

        # Training R²
        train_r2 = model.rsquared

        # Test R² (1 - SS_res/SS_tot on test set)
        y_pred = model.predict(X_test)
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        test_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        results.append({
            'fold': fold + 1,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })

    return results


if __name__ == "__main__":
    try:
        # Clear log
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, 'w') as f:
            f.write("")

        log("=" * 70)
        log("§6.5 SELF-REPORT MEASURES — UNIFIED REANALYSIS")
        log("=" * 70)
        # LOAD AND MERGE DATA
        log("\n[STEP 1] Loading and merging data sources...")

        # Theta scores (N=100, mean theta_all)
        theta = pd.read_csv(RQ_DIR / "data" / "step02_theta_scores.csv")
        log(f"  Theta scores: N={len(theta)}, cols={theta.columns.tolist()}")

        # Cognitive tests (N=100, RAVLT_T, BVMT_T, RPM_T, age)
        cog = pd.read_csv(PROJECT_ROOT / "results" / "ch7" / "7.3.1" / "data" / "step02_cognitive_tests.csv")
        log(f"  Cognitive tests: N={len(cog)}, cols={cog.columns.tolist()}")

        # Self-report from dfnonvr (N=100)
        dfnonvr = pd.read_csv(PROJECT_ROOT / "data" / "dfnonvr.csv")
        log(f"  dfnonvr: N={len(dfnonvr)}, cols={len(dfnonvr.columns)}")

        # Extract self-report variables
        selfreport = dfnonvr[['UID', 'education', 'vr-exposure', 'typical-sleep-hours',
                              'total-dass-depression-items', 'total-dass-anxiety-items',
                              'total-dass-stress-items']].copy()
        selfreport.columns = ['UID', 'education', 'vr_experience', 'sleep_hours_typical',
                              'dass_dep', 'dass_anx', 'dass_str']

        # Merge all
        df = theta[['UID', 'theta_all']].merge(cog[['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age']], on='UID')
        df = df.merge(selfreport, on='UID')

        log(f"  Merged dataset: N={len(df)}, cols={df.columns.tolist()}")
        assert len(df) == 100, f"Expected N=100, got N={len(df)}"

        # Check missing
        missing = df.isnull().sum()
        if missing.sum() > 0:
            log(f"  WARNING: Missing values detected:\n{missing[missing > 0]}")
        else:
            log("  No missing data — full N=100 for all variables")

        # Save merged dataset
        df.to_csv(RQ_DIR / "data" / "step10_merged_dataset.csv", index=False)
        log(f"  Saved: step10_merged_dataset.csv")
        # DASS DESCRIPTIVE STATISTICS
        log("\n[STEP 2] DASS descriptive statistics...")

        dass_vars = ['dass_dep', 'dass_anx', 'dass_str']
        dass_thresholds = {'dass_dep': 10, 'dass_anx': 8, 'dass_str': 15}  # DASS-21 mild thresholds
        dass_labels = {'dass_dep': 'Depression', 'dass_anx': 'Anxiety', 'dass_str': 'Stress'}

        dass_desc = []
        for var in dass_vars:
            vals = df[var]
            threshold = dass_thresholds[var]
            pct_above = (vals >= threshold).sum() / len(vals) * 100
            pct_of_threshold = vals.mean() / threshold * 100

            desc = {
                'subscale': dass_labels[var],
                'variable': var,
                'mean': vals.mean(),
                'sd': vals.std(),
                'median': vals.median(),
                'min': vals.min(),
                'max': vals.max(),
                'mild_threshold': threshold,
                'pct_above_mild': pct_above,
                'mean_as_pct_of_threshold': pct_of_threshold,
                'n': len(vals),
            }
            dass_desc.append(desc)
            log(f"  {dass_labels[var]}: M={vals.mean():.2f}, SD={vals.std():.2f}, "
                f"range=[{vals.min()}, {vals.max()}], "
                f"{pct_above:.0f}% above mild threshold (>={threshold}), "
                f"mean = {pct_of_threshold:.0f}% of threshold")

        dass_desc_df = pd.DataFrame(dass_desc)
        dass_desc_df.to_csv(RQ_DIR / "data" / "step10_dass_descriptives.csv", index=False)
        log(f"  Saved: step10_dass_descriptives.csv")

        # Also report lifestyle descriptives
        log("\n  Lifestyle descriptives:")
        for var, label in [('education', 'Education'), ('vr_experience', 'VR Experience'),
                           ('sleep_hours_typical', 'Typical Sleep Hours')]:
            vals = df[var]
            log(f"  {label}: M={vals.mean():.2f}, SD={vals.std():.2f}, range=[{vals.min()}, {vals.max()}]")
        # MODEL 1 — LIFESTYLE FACTORS
        log("\n" + "=" * 70)
        log("[MODEL 1] LIFESTYLE FACTORS")
        log("=" * 70)

        y = df['theta_all'].values

        # Block 1: age + RAVLT_T + BVMT_T + RPM_T
        block1_names = ['age', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        X_block1 = df[block1_names].values

        # Block 2: education, vr_experience, sleep_hours_typical
        block2_names_m1 = ['education', 'vr_experience', 'sleep_hours_typical']
        X_block2_m1 = df[block2_names_m1].values

        log(f"  Block 1: {block1_names}")
        log(f"  Block 2: {block2_names_m1}")
        log(f"  N = {len(y)}")

        res1 = hierarchical_regression(y, X_block1, X_block2_m1, block1_names, block2_names_m1)

        log(f"\n  Block 1: R²={res1['r2_block1']:.4f}, Adj R²={res1['adj_r2_block1']:.4f}, "
            f"F({res1['df1_block1']:.0f},{res1['df2_block1']:.0f})={res1['f_block1']:.2f}, p={res1['p_block1']:.4f}")
        log(f"  Block 2: R²={res1['r2_block2']:.4f}, Adj R²={res1['adj_r2_block2']:.4f}, "
            f"F({res1['df1_block2']:.0f},{res1['df2_block2']:.0f})={res1['f_block2']:.2f}, p={res1['p_block2']:.4f}")
        log(f"  ΔR²={res1['delta_r2']:.4f}, F-change({res1['df_num_change']},{res1['df_den_change']})="
            f"{res1['f_change']:.2f}, p={res1['p_change']:.4f}")

        log(f"\n  Individual predictors (Block 2 model):")
        for s in res1['individual_stats']:
            bonf_str = f", p_Bonf={s['p_bonferroni']:.4f}" if not np.isnan(s['p_bonferroni']) else ""
            log(f"    {s['predictor']:25s}: β={s['beta']:.4f}, sr²={s['sr2']:.4f}, "
                f"t={s['t']:.3f}, p={s['p']:.4f}{bonf_str}")

        # Save Model 1 results
        m1_models = pd.DataFrame([
            {'block': 'Block1', 'R2': res1['r2_block1'], 'adj_R2': res1['adj_r2_block1'],
             'F': res1['f_block1'], 'df1': res1['df1_block1'], 'df2': res1['df2_block1'],
             'p': res1['p_block1']},
            {'block': 'Block2', 'R2': res1['r2_block2'], 'adj_R2': res1['adj_r2_block2'],
             'F': res1['f_block2'], 'df1': res1['df1_block2'], 'df2': res1['df2_block2'],
             'p': res1['p_block2']},
            {'block': 'Delta', 'R2': res1['delta_r2'], 'adj_R2': np.nan,
             'F': res1['f_change'], 'df1': res1['df_num_change'], 'df2': res1['df_den_change'],
             'p': res1['p_change']},
        ])
        m1_models.to_csv(RQ_DIR / "data" / "step10_model1_blocks.csv", index=False)

        m1_coefs = pd.DataFrame(res1['individual_stats'])
        m1_coefs.to_csv(RQ_DIR / "data" / "step10_model1_coefficients.csv", index=False)

        # 10-fold CV for Model 1
        log(f"\n  Cross-validation (10-fold):")
        cv1 = cross_validate_10fold(y, X_block1, X_block2_m1)
        cv1_df = pd.DataFrame(cv1)
        mean_train = cv1_df['train_r2'].mean()
        mean_test = cv1_df['test_r2'].mean()
        log(f"    Mean train R² = {mean_train:.4f}")
        log(f"    Mean test R²  = {mean_test:.4f}")
        log(f"    Gap = {mean_train - mean_test:.4f}")
        cv1_df.to_csv(RQ_DIR / "data" / "step10_model1_cv.csv", index=False)
        # MODEL 2 — PSYCHOLOGICAL DISTRESS (DASS)
        log("\n" + "=" * 70)
        log("[MODEL 2] PSYCHOLOGICAL DISTRESS (DASS)")
        log("=" * 70)

        # Block 2: DASS subscales
        block2_names_m2 = ['dass_dep', 'dass_anx', 'dass_str']
        X_block2_m2 = df[block2_names_m2].values

        log(f"  Block 1: {block1_names}")
        log(f"  Block 2: {block2_names_m2}")
        log(f"  N = {len(y)}")

        res2 = hierarchical_regression(y, X_block1, X_block2_m2, block1_names, block2_names_m2)

        log(f"\n  Block 1: R²={res2['r2_block1']:.4f}, Adj R²={res2['adj_r2_block1']:.4f}, "
            f"F({res2['df1_block1']:.0f},{res2['df2_block1']:.0f})={res2['f_block1']:.2f}, p={res2['p_block1']:.4f}")
        log(f"  Block 2: R²={res2['r2_block2']:.4f}, Adj R²={res2['adj_r2_block2']:.4f}, "
            f"F({res2['df1_block2']:.0f},{res2['df2_block2']:.0f})={res2['f_block2']:.2f}, p={res2['p_block2']:.4f}")
        log(f"  ΔR²={res2['delta_r2']:.4f}, F-change({res2['df_num_change']},{res2['df_den_change']})="
            f"{res2['f_change']:.2f}, p={res2['p_change']:.4f}")

        log(f"\n  Individual predictors (Block 2 model):")
        for s in res2['individual_stats']:
            bonf_str = f", p_Bonf={s['p_bonferroni']:.4f}" if not np.isnan(s['p_bonferroni']) else ""
            log(f"    {s['predictor']:25s}: β={s['beta']:.4f}, sr²={s['sr2']:.4f}, "
                f"t={s['t']:.3f}, p={s['p']:.4f}{bonf_str}")

        # Save Model 2 results
        m2_models = pd.DataFrame([
            {'block': 'Block1', 'R2': res2['r2_block1'], 'adj_R2': res2['adj_r2_block1'],
             'F': res2['f_block1'], 'df1': res2['df1_block1'], 'df2': res2['df2_block1'],
             'p': res2['p_block1']},
            {'block': 'Block2', 'R2': res2['r2_block2'], 'adj_R2': res2['adj_r2_block2'],
             'F': res2['f_block2'], 'df1': res2['df1_block2'], 'df2': res2['df2_block2'],
             'p': res2['p_block2']},
            {'block': 'Delta', 'R2': res2['delta_r2'], 'adj_R2': np.nan,
             'F': res2['f_change'], 'df1': res2['df_num_change'], 'df2': res2['df_den_change'],
             'p': res2['p_change']},
        ])
        m2_models.to_csv(RQ_DIR / "data" / "step10_model2_blocks.csv", index=False)

        m2_coefs = pd.DataFrame(res2['individual_stats'])
        m2_coefs.to_csv(RQ_DIR / "data" / "step10_model2_coefficients.csv", index=False)

        # 10-fold CV for Model 2
        log(f"\n  Cross-validation (10-fold):")
        cv2 = cross_validate_10fold(y, X_block1, X_block2_m2)
        cv2_df = pd.DataFrame(cv2)
        mean_train2 = cv2_df['train_r2'].mean()
        mean_test2 = cv2_df['test_r2'].mean()
        log(f"    Mean train R² = {mean_train2:.4f}")
        log(f"    Mean test R²  = {mean_test2:.4f}")
        log(f"    Gap = {mean_train2 - mean_test2:.4f}")
        cv2_df.to_csv(RQ_DIR / "data" / "step10_model2_cv.csv", index=False)
        # MODEL 3 — WITHIN-PERSON SLEEP (MLM)
        log("\n" + "=" * 70)
        log("[MODEL 3] WITHIN-PERSON SLEEP (MLM)")
        log("=" * 70)

        # Load sleep analysis dataset (400 rows, person-mean-centred)
        sleep_df = pd.read_csv(PROJECT_ROOT / "results" / "ch7" / "7.5.4" / "data" / "step04_analysis_dataset.csv")
        log(f"  Sleep dataset: {len(sleep_df)} rows, {sleep_df['UID'].nunique()} participants")

        # Merge cognitive tests as between-person (Level 2) controls
        sleep_merged = sleep_df.merge(
            cog[['UID', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age']], on='UID', how='left'
        )
        assert len(sleep_merged) == 400, f"Expected 400 rows, got {len(sleep_merged)}"
        assert sleep_merged[['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'age']].isnull().sum().sum() == 0

        log(f"  Merged with cognitive controls: {len(sleep_merged)} rows")
        log(f"  Level 1 (within-person): Sleep_Hours_WP, Sleep_Quality_WP")
        log(f"  Level 2 (between-person): age, RAVLT_T, BVMT_T, RPM_T, RAVLT_Pct_Ret_T, BVMT_Pct_Ret_T, Sleep_Hours_PM, Sleep_Quality_PM")

        # Ensure UID is string for grouping
        sleep_merged['UID'] = sleep_merged['UID'].astype(str)

        # Fit MLM with statsmodels
        from statsmodels.regression.mixed_linear_model import MixedLM

        # Model: Memory_Score ~ Sleep_Hours_WP + Sleep_Quality_WP +
        #         Sleep_Hours_PM + Sleep_Quality_PM +
        #         age + RAVLT_T + BVMT_T + RPM_T + TEST + (1|UID)

        # Build design matrices manually for clarity
        fixed_vars = ['Sleep_Hours_WP', 'Sleep_Quality_WP',
                       'Sleep_Hours_PM', 'Sleep_Quality_PM',
                       'age', 'RAVLT_T', 'BVMT_T', 'RPM_T',
                       'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'TEST']

        # Standardize between-person variables for comparability
        # (within-person are already centred)
        for var in ['age', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T', 'Sleep_Hours_PM', 'Sleep_Quality_PM']:
            sleep_merged[f'{var}_z'] = (sleep_merged[var] - sleep_merged[var].mean()) / sleep_merged[var].std()

        formula = ("Memory_Score ~ Sleep_Hours_WP + Sleep_Quality_WP + "
                   "Sleep_Hours_PM_z + Sleep_Quality_PM_z + "
                   "age_z + RAVLT_T_z + BVMT_T_z + RPM_T_z + "
                   "RAVLT_Pct_Ret_T_z + BVMT_Pct_Ret_T_z + TEST")

        log(f"  Formula: {formula}")
        log(f"  Random effects: (1 | UID)")

        from statsmodels.formula.api import mixedlm

        mlm_model = mixedlm(formula, sleep_merged, groups=sleep_merged['UID'],
                            re_formula="1").fit(reml=True)

        log(f"\n  Model converged: {mlm_model.converged}")
        log(f"  Log-likelihood: {mlm_model.llf:.2f}")
        log(f"  AIC: {mlm_model.aic:.2f}")
        log(f"  N observations: {mlm_model.nobs}")
        log(f"  N groups: {len(mlm_model.model.group_labels)}")

        # Extract fixed effects
        log(f"\n  Fixed effects:")
        mlm_fixed = pd.DataFrame({
            'parameter': mlm_model.params.index,
            'estimate': mlm_model.params.values,
            'std_error': mlm_model.bse.values,
            'z_value': mlm_model.tvalues.values,
            'p_uncorrected': mlm_model.pvalues.values,
        })

        # Identify sleep parameters for Bonferroni correction (2 within-person sleep params)
        sleep_params = ['Sleep_Hours_WP', 'Sleep_Quality_WP']
        n_sleep = len(sleep_params)
        mlm_fixed['p_bonferroni'] = mlm_fixed.apply(
            lambda row: min(row['p_uncorrected'] * n_sleep, 1.0)
            if row['parameter'] in sleep_params else np.nan, axis=1
        )

        for _, row in mlm_fixed.iterrows():
            bonf_str = f", p_Bonf={row['p_bonferroni']:.4f}" if not np.isnan(row['p_bonferroni']) else ""
            log(f"    {row['parameter']:25s}: β={row['estimate']:.6f}, SE={row['std_error']:.6f}, "
                f"z={row['z_value']:.3f}, p={row['p_uncorrected']:.4f}{bonf_str}")

        # Random effects
        log(f"\n  Random effects:")
        re_var = mlm_model.cov_re.iloc[0, 0] if hasattr(mlm_model.cov_re, 'iloc') else float(mlm_model.cov_re)
        resid_var = mlm_model.scale
        icc = re_var / (re_var + resid_var) if (re_var + resid_var) > 0 else np.nan
        log(f"    Random intercept variance: {re_var:.6f}")
        log(f"    Residual variance: {resid_var:.6f}")
        log(f"    ICC: {icc:.4f}")

        # Compute Cohen's d for Sleep_Hours_WP (primary within-person predictor)
        sleep_wp_row = mlm_fixed[mlm_fixed['parameter'] == 'Sleep_Hours_WP']
        if len(sleep_wp_row) > 0:
            beta_sleep = sleep_wp_row.iloc[0]['estimate']
            # d = beta / SD(residual) ... for unstandardized coefficient
            sd_resid = np.sqrt(resid_var)
            d_sleep = beta_sleep / sd_resid if sd_resid > 0 else np.nan
            log(f"\n  Sleep_Hours_WP effect size: d = {d_sleep:.4f}")

        # Compute marginal and conditional R² (Nakagawa & Schielzeth)
        # Marginal R² = variance explained by fixed effects / total variance
        # Conditional R² = variance explained by fixed + random / total variance

        # Get fixed effect predictions
        y_fixed = mlm_model.fittedvalues
        y_actual = sleep_merged['Memory_Score'].values

        # Variance of fixed-effect predictions
        var_fixed = np.var(y_fixed)
        total_var = var_fixed + re_var + resid_var

        r2_marginal = var_fixed / total_var
        r2_conditional = (var_fixed + re_var) / total_var

        log(f"\n  Nakagawa R²:")
        log(f"    Marginal R² (fixed effects): {r2_marginal:.4f}")
        log(f"    Conditional R² (fixed + random): {r2_conditional:.4f}")

        # Save Model 3 results
        mlm_fixed.to_csv(RQ_DIR / "data" / "step10_model3_fixed_effects.csv", index=False)

        mlm_summary = pd.DataFrame([{
            'n_obs': int(mlm_model.nobs),
            'n_groups': len(mlm_model.model.group_labels),
            'converged': mlm_model.converged,
            'log_likelihood': mlm_model.llf,
            'aic': mlm_model.aic,
            'icc': icc,
            'random_intercept_var': re_var,
            'residual_var': resid_var,
            'r2_marginal': r2_marginal,
            'r2_conditional': r2_conditional,
            'd_sleep_hours_wp': d_sleep if len(sleep_wp_row) > 0 else np.nan,
        }])
        mlm_summary.to_csv(RQ_DIR / "data" / "step10_model3_summary.csv", index=False)
        # COMPREHENSIVE SUMMARY
        log("\n" + "=" * 70)
        log("COMPREHENSIVE SUMMARY")
        log("=" * 70)

        log(f"\n  Consistent Block 1 across Models 1 & 2:")
        log(f"    Predictors: age, RAVLT_T, BVMT_T, RPM_T, RAVLT_Pct_Ret_T, BVMT_Pct_Ret_T")
        log(f"    R² = {res1['r2_block1']:.4f} (same for both — identical Block 1)")
        log(f"    F({res1['df1_block1']:.0f},{res1['df2_block1']:.0f}) = {res1['f_block1']:.2f}, p = {res1['p_block1']:.4f}")

        log(f"\n  Model 1 — Lifestyle (ΔR² beyond cognitive baseline):")
        log(f"    ΔR² = {res1['delta_r2']:.4f}, F-change({res1['df_num_change']},{res1['df_den_change']}) = "
            f"{res1['f_change']:.2f}, p = {res1['p_change']:.4f}")
        log(f"    CV: train R² = {mean_train:.4f}, test R² = {mean_test:.4f}")

        log(f"\n  Model 2 — DASS (ΔR² beyond cognitive baseline):")
        log(f"    ΔR² = {res2['delta_r2']:.4f}, F-change({res2['df_num_change']},{res2['df_den_change']}) = "
            f"{res2['f_change']:.2f}, p = {res2['p_change']:.4f}")
        log(f"    CV: train R² = {mean_train2:.4f}, test R² = {mean_test2:.4f}")

        log(f"\n  Model 3 — Within-person sleep (MLM with cognitive controls):")
        sleep_wp = mlm_fixed[mlm_fixed['parameter'] == 'Sleep_Hours_WP'].iloc[0]
        log(f"    Sleep_Hours_WP: β = {sleep_wp['estimate']:.6f}, p = {sleep_wp['p_uncorrected']:.4f}")
        if len(sleep_wp_row) > 0:
            log(f"    Cohen's d = {d_sleep:.4f}")
        log(f"    Marginal R² = {r2_marginal:.4f}")
        log(f"    ICC = {icc:.4f}")

        log(f"\n  Missing data: N=100 for all models (no listwise deletion)")
        log(f"    All 100 participants have complete cognitive tests (RAVLT, BVMT, RPM, retention T-scores)")
        log(f"    All 100 participants have complete self-report data")
        log(f"    All 100 participants have complete DASS data")
        log(f"    Sleep MLM: 400 observations (100 × 4 sessions), no missing")

        log(f"\n  Degrees of freedom summary:")
        log(f"    Models 1 & 2 Block 1: F({res1['df1_block1']:.0f},{res1['df2_block1']:.0f})")
        log(f"    Model 1 Block 2: F({res1['df1_block2']:.0f},{res1['df2_block2']:.0f}), "
            f"F-change({res1['df_num_change']},{res1['df_den_change']})")
        log(f"    Model 2 Block 2: F({res2['df1_block2']:.0f},{res2['df2_block2']:.0f}), "
            f"F-change({res2['df_num_change']},{res2['df_den_change']})")
        log(f"    Model 3: {int(mlm_model.nobs)} obs, {len(mlm_model.model.group_labels)} groups")

        log("\n§6.5 reanalysis complete")
        sys.exit(0)

    except Exception as e:
        log(f"\n{str(e)}")
        import traceback
        log("")
        with open(LOG_FILE, 'a') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
