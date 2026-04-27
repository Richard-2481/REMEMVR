#!/usr/bin/env python3
"""
Step ID: step12
Step Name: accuracy_hierarchical_regression
RQ: results/ch7/7.3.1
Generated: 2026-02-26

PURPOSE:
Hierarchical regression predicting ACCURACY theta from the same 6 predictors
as the confidence model (step05). Enables direct accuracy-vs-confidence comparison.

Model 1 (Demographics): age, sex, education
Model 2 (Full): age, sex, education, RAVLT_T, BVMT_T, RPM_T

EXPECTED INPUTS:
  - results/ch7/7.3.1/data/step11_accuracy_analysis_dataset.csv

EXPECTED OUTPUTS:
  - results/ch7/7.3.1/data/step12_accuracy_hierarchical_models.csv
  - results/ch7/7.3.1/data/step12_accuracy_individual_predictors.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.utils import resample
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step12_accuracy_hierarchical_regression.log"

def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def bootstrap_r_squared_ci(X, y, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """Bootstrap confidence intervals for R-squared."""
    np.random.seed(random_state)
    n_obs = len(y)
    r_squared_values = []
    for _ in range(n_bootstrap):
        boot_indices = resample(range(n_obs), n_samples=n_obs, random_state=None)
        X_boot = X.iloc[boot_indices] if hasattr(X, 'iloc') else X[boot_indices]
        y_boot = y.iloc[boot_indices] if hasattr(y, 'iloc') else y[boot_indices]
        try:
            X_reg = sm.add_constant(X_boot)
            model = sm.OLS(y_boot, X_reg).fit()
            r_squared_values.append(model.rsquared)
        except:
            continue
    alpha = 1 - confidence_level
    ci_lower = np.percentile(r_squared_values, (alpha / 2) * 100)
    ci_upper = np.percentile(r_squared_values, (1 - alpha / 2) * 100)
    return ci_lower, ci_upper


if __name__ == "__main__":
    try:
        # Clear log
        LOG_FILE.write_text("")

        log("[START] Step 12: Accuracy Hierarchical Regression (6-predictor model)")

        # =====================================================================
        # LOAD DATA
        # =====================================================================
        input_path = RQ_DIR / "data" / "step11_accuracy_analysis_dataset.csv"
        df = pd.read_csv(input_path)
        log(f"[LOADED] {input_path.name} ({len(df)} rows, {len(df.columns)} cols)")

        required_cols = ['accuracy_theta', 'age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        log(f"[VERIFY] All required columns present")

        # =====================================================================
        # PREPARE PREDICTOR BLOCKS
        # =====================================================================
        df_analysis = df.copy()
        # Sex is already numeric (0/1) in step02_cognitive_tests.csv
        df_analysis['sex_dummy'] = df_analysis['sex']

        block_1_predictors = ['age', 'sex_dummy', 'education']
        block_2_predictors = ['age', 'sex_dummy', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']

        X_block1 = df_analysis[block_1_predictors]
        X_block2 = df_analysis[block_2_predictors]
        y = df_analysis['accuracy_theta']

        log(f"[BLOCKS] Block 1 (Demographics): {block_1_predictors}")
        log(f"[BLOCKS] Block 2 (Demographics + Cognitive): {block_2_predictors}")
        log(f"[BLOCKS] Outcome variable: accuracy_theta (N={len(y)}, mean={y.mean():.4f}, sd={y.std():.4f})")

        # =====================================================================
        # FIT HIERARCHICAL REGRESSION
        # =====================================================================
        log("[ANALYSIS] Running hierarchical regression...")

        # Model 1: Demographics only
        X1_const = sm.add_constant(X_block1)
        model1 = sm.OLS(y, X1_const).fit()

        # Model 2: Demographics + Cognitive
        X2_const = sm.add_constant(X_block2)
        model2 = sm.OLS(y, X2_const).fit()

        log(f"[RESULTS] Model 1: R²={model1.rsquared:.4f}, Adj R²={model1.rsquared_adj:.4f}, "
            f"F({int(model1.df_model)},{int(model1.df_resid)})={model1.fvalue:.3f}, p={model1.f_pvalue:.6f}")
        log(f"[RESULTS] Model 2: R²={model2.rsquared:.4f}, Adj R²={model2.rsquared_adj:.4f}, "
            f"F({int(model2.df_model)},{int(model2.df_resid)})={model2.fvalue:.3f}, p={model2.f_pvalue:.6f}")

        # Hierarchical F-test
        r2_change = model2.rsquared - model1.rsquared
        df_change = model2.df_model - model1.df_model
        df_error = model2.df_resid
        f_change = (r2_change / df_change) / ((1 - model2.rsquared) / df_error)
        p_change = 1 - stats.f.cdf(f_change, df_change, df_error)

        log(f"[HIERARCHICAL] ΔR² = {r2_change:.4f}")
        log(f"[HIERARCHICAL] F-change({int(df_change)},{int(df_error)}) = {f_change:.3f}")
        log(f"[HIERARCHICAL] p-change = {p_change:.6f}")

        # =====================================================================
        # BOOTSTRAP CONFIDENCE INTERVALS
        # =====================================================================
        log("[BOOTSTRAP] Computing bootstrap 95% CIs (n=1000, seed=42)...")
        ci1_lower, ci1_upper = bootstrap_r_squared_ci(X_block1, y, n_bootstrap=1000, random_state=42)
        ci2_lower, ci2_upper = bootstrap_r_squared_ci(X_block2, y, n_bootstrap=1000, random_state=42)
        log(f"[BOOTSTRAP] Demographics 95% CI: [{ci1_lower:.4f}, {ci1_upper:.4f}]")
        log(f"[BOOTSTRAP] Full model 95% CI: [{ci2_lower:.4f}, {ci2_upper:.4f}]")

        # =====================================================================
        # EFFECT SIZES
        # =====================================================================
        f2_demo = model1.rsquared / (1 - model1.rsquared) if model1.rsquared < 1 else np.inf
        f2_full = model2.rsquared / (1 - model2.rsquared) if model2.rsquared < 1 else np.inf
        log(f"[EFFECT_SIZE] Demographics Cohen's f² = {f2_demo:.4f}")
        log(f"[EFFECT_SIZE] Full model Cohen's f² = {f2_full:.4f}")

        # =====================================================================
        # SAVE HIERARCHICAL MODEL COMPARISON
        # =====================================================================
        hier_df = pd.DataFrame([
            {'model': 'Demographics', 'R_squared': model1.rsquared, 'adj_R_squared': model1.rsquared_adj,
             'F_stat': model1.fvalue, 'p_value': model1.f_pvalue,
             'ci_lower': ci1_lower, 'ci_upper': ci1_upper, 'cohens_f2': f2_demo},
            {'model': 'Cognitive', 'R_squared': model2.rsquared, 'adj_R_squared': model2.rsquared_adj,
             'F_stat': model2.fvalue, 'p_value': model2.f_pvalue,
             'ci_lower': ci2_lower, 'ci_upper': ci2_upper, 'cohens_f2': f2_full}
        ])
        hier_path = RQ_DIR / "data" / "step12_accuracy_hierarchical_models.csv"
        hier_df.to_csv(hier_path, index=False)
        log(f"[SAVED] {hier_path.name}")

        # =====================================================================
        # INDIVIDUAL PREDICTORS (sr², VIF, dual p-values)
        # =====================================================================
        log("[INDIVIDUAL] Computing individual predictor statistics...")

        # Full model is model2
        X_full = df_analysis[['age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']]
        predictor_names = list(X_full.columns)

        # VIF
        X_vif = sm.add_constant(X_full)
        vif_values = [variance_inflation_factor(X_vif.values, i+1) for i in range(len(predictor_names))]

        # Semi-partial r² (drop each predictor and compare R²)
        full_r2 = model2.rsquared
        sr2_values = []
        for predictor in predictor_names:
            X_reduced = X_full.drop(columns=[predictor])
            X_red_const = sm.add_constant(X_reduced)
            reduced_model = sm.OLS(y, X_red_const).fit()
            sr2 = full_r2 - reduced_model.rsquared
            sr2_values.append(sr2)

        # p-values from full model (skip intercept)
        p_uncorrected = [model2.pvalues[i+1] for i in range(len(predictor_names))]

        # Bonferroni correction (Decision D068): correct only cognitive tests (3 tests)
        n_cog_tests = 5
        p_bonferroni = []
        for i, pred in enumerate(predictor_names):
            if pred in ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
                p_bonferroni.append(min(p_uncorrected[i] * n_cog_tests, 1.0))
            else:
                p_bonferroni.append(p_uncorrected[i])

        # FDR correction across all 6 predictors
        _, p_fdr, _, _ = multipletests(p_uncorrected, method='fdr_bh')

        # Confidence intervals
        conf_int = model2.conf_int()

        # Build results
        individual_results = []
        for i, pred in enumerate(predictor_names):
            row = {
                'predictor': pred,
                'beta': model2.params[i+1],
                'se': model2.bse[i+1],
                'ci_lower': conf_int.iloc[i+1, 0],
                'ci_upper': conf_int.iloc[i+1, 1],
                'sr2': sr2_values[i],
                'p_uncorrected': p_uncorrected[i],
                'p_bonferroni': p_bonferroni[i],
                'p_fdr': p_fdr[i],
                'vif': vif_values[i]
            }
            individual_results.append(row)
            log(f"[PREDICTOR] {pred}: β={row['beta']:.4f}, sr²={row['sr2']:.6f}, "
                f"p_uncorr={row['p_uncorrected']:.4f}, p_bonf={row['p_bonferroni']:.4f}, VIF={row['vif']:.3f}")

        ind_df = pd.DataFrame(individual_results)
        ind_path = RQ_DIR / "data" / "step12_accuracy_individual_predictors.csv"
        ind_df.to_csv(ind_path, index=False)
        log(f"[SAVED] {ind_path.name} ({len(ind_df)} rows)")

        # =====================================================================
        # SUMMARY COMPARISON
        # =====================================================================
        log("")
        log("=" * 60)
        log("ACCURACY vs CONFIDENCE MODEL COMPARISON")
        log("=" * 60)
        log(f"{'Metric':<30} {'Accuracy':<15} {'Confidence':<15}")
        log(f"{'-'*60}")
        log(f"{'R²':<30} {model2.rsquared:<15.4f} {'0.1875':<15}")
        log(f"{'Adj R²':<30} {model2.rsquared_adj:<15.4f} {'0.1351':<15}")
        log(f"{'F(6,93)':<30} {model2.fvalue:<15.3f} {'3.578':<15}")
        log(f"{'p':<30} {model2.f_pvalue:<15.6f} {'0.003105':<15}")
        log(f"{'ΔR² (cognitive)':<30} {r2_change:<15.4f} {'0.1673':<15}")
        log(f"{'Cohens f-squared':<30} {f2_full:<15.4f} {'0.2308':<15}")
        log(f"{'-'*60}")
        for i, pred in enumerate(predictor_names):
            if pred in ['RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
                log(f"{'sr² ' + pred:<30} {sr2_values[i]:<15.6f}")

        log("")
        log("[SUCCESS] Step 12: Accuracy hierarchical regression complete")
        sys.exit(0)

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
