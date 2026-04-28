#!/usr/bin/env python3
"""
Step ID: step13
Step Name: formal_dissociation_tests
RQ: results/ch7/7.3.1

PURPOSE:
Formal tests of the accuracy-confidence dissociation using MATCHED models
(identical 6-predictor specification, N=100 for both).

Produces:
  1. Steiger's Z for RPM (sr2 0.090 vs 0.042) and BVMT (sr2 0.018 vs 0.048)
  2. Bootstrap sr2 difference CI for RPM (B=5000, BCa)
  3. Cross-validation for the accuracy model (5-fold)
  4. Omnibus test: R2 difference (accuracy 0.249 vs confidence 0.188)

Supersedes RQ 7.9.1 step01-02 which used mismatched models (4-predictor vs 6-predictor).

INPUTS:
  - results/ch7/7.3.1/data/step11_accuracy_analysis_dataset.csv  (accuracy + predictors, N=100)
  - results/ch7/7.3.1/data/step04_analysis_dataset.csv  (confidence + predictors, N=100)

OUTPUTS:
  - results/ch7/7.3.1/data/step13_steiger_z_tests.csv
  - results/ch7/7.3.1/data/step13_bootstrap_sr2_difference.csv
  - results/ch7/7.3.1/data/step13_accuracy_cross_validation.csv
  - results/ch7/7.3.1/data/step13_omnibus_r2_comparison.csv
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step13_formal_dissociation_tests.log"


def log(msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def compute_sr2(y, X_df, predictor_name):
    """Compute sr2 for a named predictor by comparing full vs reduced model."""
    X_full = sm.add_constant(X_df)
    r2_full = sm.OLS(y, X_full).fit().rsquared

    X_reduced = sm.add_constant(X_df.drop(columns=[predictor_name]))
    r2_reduced = sm.OLS(y, X_reduced).fit().rsquared

    return r2_full - r2_reduced


def steiger_z_test(r_ya, r_yb, r_ab, N):
    """
    Steiger (1980) test for comparing two dependent correlations.
    r_ya: correlation between predictor and outcome A
    r_yb: correlation between predictor and outcome B
    r_ab: correlation between outcome A and outcome B
    N: sample size
    """
    det_R = 1 - r_ya**2 - r_yb**2 - r_ab**2 + 2 * r_ya * r_yb * r_ab

    if det_R <= 0:
        det_R = abs(det_R) + 1e-10

    denominator = np.sqrt(2 * (1 - r_ab) * det_R)

    if denominator < 1e-10:
        return 0.0, 1.0

    Z = (r_ya - r_yb) * np.sqrt(N - 3) / denominator
    p = 2 * (1 - stats.norm.cdf(abs(Z)))
    return Z, p


if __name__ == "__main__":
    try:
        LOG_FILE.write_text("")
        log("Step 13: Formal Dissociation Tests (Matched 6-Predictor Models)")
        # LOAD DATA
        acc_df = pd.read_csv(RQ_DIR / "data" / "step11_accuracy_analysis_dataset.csv")
        conf_df = pd.read_csv(RQ_DIR / "data" / "step04_analysis_dataset.csv")

        log(f"Accuracy dataset: {acc_df.shape}")
        log(f"Confidence dataset: {conf_df.shape}")

        # Merge on UID to ensure identical participant ordering
        merged = pd.merge(
            acc_df[['UID', 'accuracy_theta']],
            conf_df[['UID', 'confidence_theta', 'age', 'sex', 'education',
                      'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']],
            on='UID', how='inner'
        )
        merged = merged.dropna()
        N = len(merged)
        log(f"N={N} participants with both accuracy and confidence theta")

        predictor_cols = ['age', 'sex', 'education', 'RAVLT_T', 'BVMT_T', 'RPM_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']
        X = merged[predictor_cols]
        y_acc = merged['accuracy_theta']
        y_conf = merged['confidence_theta']

        # Verify R2 matches step12
        model_acc = sm.OLS(y_acc, sm.add_constant(X)).fit()
        model_conf = sm.OLS(y_conf, sm.add_constant(X)).fit()
        log(f"Accuracy R2 = {model_acc.rsquared:.6f} (expect ~0.249)")
        log(f"Confidence R2 = {model_conf.rsquared:.6f} (expect ~0.188)")
        # TEST 1: STEIGER'S Z FOR RPM AND BVMT
        log("")
        log("=" * 70)
        log("TEST 1: Steiger's Z Tests for Dependent Correlations")
        log("=" * 70)

        steiger_results = []

        for predictor in ['RPM_T', 'BVMT_T', 'RAVLT_T', 'RAVLT_Pct_Ret_T', 'BVMT_Pct_Ret_T']:
            pred_vals = merged[predictor].values
            acc_vals = y_acc.values
            conf_vals = y_conf.values

            r_pred_acc = np.corrcoef(pred_vals, acc_vals)[0, 1]
            r_pred_conf = np.corrcoef(pred_vals, conf_vals)[0, 1]
            r_acc_conf = np.corrcoef(acc_vals, conf_vals)[0, 1]

            Z, p = steiger_z_test(r_pred_acc, r_pred_conf, r_acc_conf, N)

            # sr2 values from matched models
            sr2_acc = compute_sr2(y_acc, X, predictor)
            sr2_conf = compute_sr2(y_conf, X, predictor)

            log(f"\n{predictor}:")
            log(f"  r({predictor}-accuracy)   = {r_pred_acc:.4f}")
            log(f"  r({predictor}-confidence) = {r_pred_conf:.4f}")
            log(f"  r(accuracy-confidence)   = {r_acc_conf:.4f}")
            log(f"  sr2 accuracy  = {sr2_acc:.6f}")
            log(f"  sr2 confidence = {sr2_conf:.6f}")
            log(f"  Steiger Z = {Z:.4f}, p = {p:.6f}")
            log(f"  {'*** SIGNIFICANT (p < .05) ***' if p < 0.05 else '  n.s.'}")

            steiger_results.append({
                'predictor': predictor,
                'r_pred_accuracy': r_pred_acc,
                'r_pred_confidence': r_pred_conf,
                'r_acc_conf': r_acc_conf,
                'sr2_accuracy': sr2_acc,
                'sr2_confidence': sr2_conf,
                'sr2_difference': sr2_acc - sr2_conf,
                'N': N,
                'steiger_Z': Z,
                'p_value': p,
                'significant_05': p < 0.05
            })

        steiger_df = pd.DataFrame(steiger_results)
        steiger_path = RQ_DIR / "data" / "step13_steiger_z_tests.csv"
        steiger_df.to_csv(steiger_path, index=False)
        log(f"\n{steiger_path.name}")
        # TEST 2: BOOTSTRAP SR2 DIFFERENCE FOR RPM (B=5000, BCa)
        log("")
        log("=" * 70)
        log("TEST 2: Bootstrap sr2 Difference for RPM (B=5000)")
        log("=" * 70)

        # Point estimate
        sr2_rpm_acc = compute_sr2(y_acc, X, 'RPM_T')
        sr2_rpm_conf = compute_sr2(y_conf, X, 'RPM_T')
        delta_point = sr2_rpm_acc - sr2_rpm_conf
        log(f"Point estimate: {sr2_rpm_acc:.6f} - {sr2_rpm_conf:.6f} = {delta_point:.6f}")

        # Also do BVMT
        sr2_bvmt_acc = compute_sr2(y_acc, X, 'BVMT_T')
        sr2_bvmt_conf = compute_sr2(y_conf, X, 'BVMT_T')
        delta_bvmt_point = sr2_bvmt_acc - sr2_bvmt_conf
        log(f"BVMT point estimate: {sr2_bvmt_acc:.6f} - {sr2_bvmt_conf:.6f} = {delta_bvmt_point:.6f}")

        B = 5000
        rng = np.random.RandomState(42)
        X_vals = X.values
        y_acc_vals = y_acc.values
        y_conf_vals = y_conf.values

        boot_deltas_rpm = []
        boot_deltas_bvmt = []
        n_failed = 0

        log(f"Running bootstrap (B={B})...")
        for b in range(B):
            idx = rng.choice(N, size=N, replace=True)
            X_b = pd.DataFrame(X_vals[idx], columns=predictor_cols)
            y_acc_b = pd.Series(y_acc_vals[idx])
            y_conf_b = pd.Series(y_conf_vals[idx])

            try:
                sr2_acc_b = compute_sr2(y_acc_b, X_b, 'RPM_T')
                sr2_conf_b = compute_sr2(y_conf_b, X_b, 'RPM_T')
                boot_deltas_rpm.append(sr2_acc_b - sr2_conf_b)

                sr2_bvmt_acc_b = compute_sr2(y_acc_b, X_b, 'BVMT_T')
                sr2_bvmt_conf_b = compute_sr2(y_conf_b, X_b, 'BVMT_T')
                boot_deltas_bvmt.append(sr2_bvmt_acc_b - sr2_bvmt_conf_b)
            except Exception:
                n_failed += 1
                continue

            if (b + 1) % 1000 == 0:
                log(f"  Progress: {b+1}/{B}")

        log(f"Bootstrap complete: {len(boot_deltas_rpm)} successful, {n_failed} failed")

        boot_deltas_rpm = np.array(boot_deltas_rpm)
        boot_deltas_bvmt = np.array(boot_deltas_bvmt)

        # --- RPM: Percentile CI ---
        pct_lower_rpm = np.percentile(boot_deltas_rpm, 2.5)
        pct_upper_rpm = np.percentile(boot_deltas_rpm, 97.5)
        prop_positive_rpm = np.mean(boot_deltas_rpm > 0)

        # --- RPM: BCa CI ---
        z0_rpm = stats.norm.ppf(np.mean(boot_deltas_rpm < delta_point))

        # Jackknife for acceleration
        theta_jack_rpm = np.zeros(N)
        for i in range(N):
            idx_j = np.concatenate([np.arange(i), np.arange(i + 1, N)])
            X_j = pd.DataFrame(X_vals[idx_j], columns=predictor_cols)
            y_acc_j = pd.Series(y_acc_vals[idx_j])
            y_conf_j = pd.Series(y_conf_vals[idx_j])
            sr2_acc_j = compute_sr2(y_acc_j, X_j, 'RPM_T')
            sr2_conf_j = compute_sr2(y_conf_j, X_j, 'RPM_T')
            theta_jack_rpm[i] = sr2_acc_j - sr2_conf_j

        theta_bar_rpm = theta_jack_rpm.mean()
        diffs_rpm = theta_bar_rpm - theta_jack_rpm
        a_hat_rpm = np.sum(diffs_rpm**3) / (6 * np.sum(diffs_rpm**2)**1.5 + 1e-10)

        bca_bounds_rpm = []
        for alpha in [0.025, 0.975]:
            z_alpha = stats.norm.ppf(alpha)
            adj = stats.norm.cdf(z0_rpm + (z0_rpm + z_alpha) / (1 - a_hat_rpm * (z0_rpm + z_alpha)))
            adj = np.clip(adj, 0.001, 0.999)
            bca_bounds_rpm.append(np.percentile(boot_deltas_rpm, adj * 100))

        bca_lower_rpm, bca_upper_rpm = bca_bounds_rpm

        log(f"\nRPM sr2 difference:")
        log(f"  Point estimate: {delta_point:.6f}")
        log(f"  Percentile 95% CI: [{pct_lower_rpm:.6f}, {pct_upper_rpm:.6f}]")
        log(f"  BCa 95% CI: [{bca_lower_rpm:.6f}, {bca_upper_rpm:.6f}]")
        log(f"  Proportion positive: {prop_positive_rpm:.4f}")
        log(f"  {'*** CI EXCLUDES ZERO ***' if pct_lower_rpm > 0 else '  CI includes zero'}")

        # --- BVMT: Percentile CI ---
        pct_lower_bvmt = np.percentile(boot_deltas_bvmt, 2.5)
        pct_upper_bvmt = np.percentile(boot_deltas_bvmt, 97.5)
        prop_positive_bvmt = np.mean(boot_deltas_bvmt > 0)

        log(f"\nBVMT sr2 difference (accuracy - confidence):")
        log(f"  Point estimate: {delta_bvmt_point:.6f}")
        log(f"  Percentile 95% CI: [{pct_lower_bvmt:.6f}, {pct_upper_bvmt:.6f}]")
        log(f"  Proportion positive: {prop_positive_bvmt:.4f}")
        log(f"  Note: NEGATIVE means confidence > accuracy (reversed direction)")

        # Save bootstrap results
        boot_summary = pd.DataFrame([
            {'predictor': 'RPM_T', 'delta_point': delta_point,
             'pct_ci_lower': pct_lower_rpm, 'pct_ci_upper': pct_upper_rpm,
             'bca_ci_lower': bca_lower_rpm, 'bca_ci_upper': bca_upper_rpm,
             'prop_positive': prop_positive_rpm, 'B': len(boot_deltas_rpm)},
            {'predictor': 'BVMT_T', 'delta_point': delta_bvmt_point,
             'pct_ci_lower': pct_lower_bvmt, 'pct_ci_upper': pct_upper_bvmt,
             'bca_ci_lower': np.nan, 'bca_ci_upper': np.nan,
             'prop_positive': prop_positive_bvmt, 'B': len(boot_deltas_bvmt)}
        ])
        boot_path = RQ_DIR / "data" / "step13_bootstrap_sr2_difference.csv"
        boot_summary.to_csv(boot_path, index=False)
        log(f"\n{boot_path.name}")
        # TEST 3: CROSS-VALIDATION FOR ACCURACY MODEL (5-fold)
        log("")
        log("=" * 70)
        log("TEST 3: 5-Fold Cross-Validation for Accuracy Model")
        log("=" * 70)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y_acc.iloc[train_idx]
            y_test = y_acc.iloc[test_idx]

            model_train = sm.OLS(y_train, sm.add_constant(X_train)).fit()
            y_pred = model_train.predict(sm.add_constant(X_test))

            ss_res = np.sum((y_test - y_pred)**2)
            ss_tot = np.sum((y_test - y_test.mean())**2)
            test_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            mae = np.mean(np.abs(y_test - y_pred))

            cv_results.append({
                'fold': fold + 1,
                'train_r2': model_train.rsquared,
                'test_r2': test_r2,
                'rmse': rmse,
                'mae': mae,
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            })

            log(f"  Fold {fold+1}: train R2={model_train.rsquared:.4f}, "
                f"test R2={test_r2:.4f}, RMSE={rmse:.4f}")

        cv_df = pd.DataFrame(cv_results)

        mean_train = cv_df['train_r2'].mean()
        mean_test = cv_df['test_r2'].mean()
        sd_test = cv_df['test_r2'].std()
        gap = mean_train - mean_test

        log(f"\nSummary:")
        log(f"  Mean train R2: {mean_train:.4f} +/- {cv_df['train_r2'].std():.4f}")
        log(f"  Mean test R2:  {mean_test:.4f} +/- {sd_test:.4f}")
        log(f"  Train-test gap: {gap:.4f}")
        log(f"  {'OVERFITTING CONCERN' if gap > 0.10 else 'Acceptable generalization'}")

        cv_path = RQ_DIR / "data" / "step13_accuracy_cross_validation.csv"
        cv_df.to_csv(cv_path, index=False)
        log(f"{cv_path.name}")
        # TEST 4: OMNIBUS R2 DIFFERENCE (Bootstrap)
        log("")
        log("=" * 70)
        log("TEST 4: Omnibus R2 Difference (Accuracy vs Confidence)")
        log("=" * 70)

        r2_acc = model_acc.rsquared
        r2_conf = model_conf.rsquared
        r2_diff_point = r2_acc - r2_conf

        log(f"Point estimate: R2_acc ({r2_acc:.4f}) - R2_conf ({r2_conf:.4f}) = {r2_diff_point:.4f}")

        # Bootstrap the R2 difference
        B_omnibus = 5000
        rng2 = np.random.RandomState(42)
        boot_r2_diffs = []

        log(f"Running bootstrap (B={B_omnibus})...")
        for b in range(B_omnibus):
            idx = rng2.choice(N, size=N, replace=True)
            X_b = sm.add_constant(pd.DataFrame(X_vals[idx], columns=predictor_cols))
            y_acc_b = y_acc_vals[idx]
            y_conf_b = y_conf_vals[idx]

            try:
                r2_acc_b = sm.OLS(y_acc_b, X_b).fit().rsquared
                r2_conf_b = sm.OLS(y_conf_b, X_b).fit().rsquared
                boot_r2_diffs.append(r2_acc_b - r2_conf_b)
            except Exception:
                continue

            if (b + 1) % 1000 == 0:
                log(f"  Progress: {b+1}/{B_omnibus}")

        boot_r2_diffs = np.array(boot_r2_diffs)

        pct_lower_r2 = np.percentile(boot_r2_diffs, 2.5)
        pct_upper_r2 = np.percentile(boot_r2_diffs, 97.5)
        prop_positive_r2 = np.mean(boot_r2_diffs > 0)

        log(f"\nOmnibus R2 difference:")
        log(f"  Point estimate: {r2_diff_point:.4f}")
        log(f"  Percentile 95% CI: [{pct_lower_r2:.4f}, {pct_upper_r2:.4f}]")
        log(f"  Proportion positive: {prop_positive_r2:.4f}")
        log(f"  {'*** CI EXCLUDES ZERO ***' if pct_lower_r2 > 0 else '  CI includes zero'}")

        omnibus_df = pd.DataFrame([{
            'r2_accuracy': r2_acc,
            'r2_confidence': r2_conf,
            'r2_difference': r2_diff_point,
            'pct_ci_lower': pct_lower_r2,
            'pct_ci_upper': pct_upper_r2,
            'prop_positive': prop_positive_r2,
            'B': len(boot_r2_diffs)
        }])
        omnibus_path = RQ_DIR / "data" / "step13_omnibus_r2_comparison.csv"
        omnibus_df.to_csv(omnibus_path, index=False)
        log(f"{omnibus_path.name}")
        # FINAL SUMMARY
        log("")
        log("=" * 70)
        log("SUMMARY OF ALL DISSOCIATION TESTS")
        log("=" * 70)
        log(f"All tests use matched 6-predictor models, N={N}")
        log("")
        log("1. STEIGER'S Z TESTS:")
        for _, row in steiger_df.iterrows():
            sig = "***" if row['p_value'] < 0.05 else "n.s."
            log(f"   {row['predictor']}: Z={row['steiger_Z']:.3f}, p={row['p_value']:.4f} {sig}")
        log("")
        log("2. BOOTSTRAP SR2 DIFFERENCE (RPM):")
        log(f"   Delta = {delta_point:.4f}, 95% CI [{pct_lower_rpm:.4f}, {pct_upper_rpm:.4f}], "
            f"BCa [{bca_lower_rpm:.4f}, {bca_upper_rpm:.4f}]")
        log(f"   {prop_positive_rpm*100:.1f}% of bootstrap samples favour accuracy > confidence")
        log("")
        log("3. CROSS-VALIDATION (ACCURACY MODEL):")
        log(f"   Mean test R2 = {mean_test:.4f} (train = {mean_train:.4f}, gap = {gap:.4f})")
        log("")
        log("4. OMNIBUS R2 DIFFERENCE:")
        log(f"   Delta R2 = {r2_diff_point:.4f}, 95% CI [{pct_lower_r2:.4f}, {pct_upper_r2:.4f}]")
        log(f"   {prop_positive_r2*100:.1f}% of bootstrap samples favour accuracy > confidence")
        log("")

        # Overall interpretation
        rpm_sig = steiger_df[steiger_df['predictor'] == 'RPM_T']['p_value'].iloc[0] < 0.05
        rpm_ci_excludes_zero = pct_lower_rpm > 0
        omnibus_ci_excludes_zero = pct_lower_r2 > 0

        if rpm_sig or rpm_ci_excludes_zero:
            log("INTERPRETATION: RPM dissociation reaches significance with matched models")
            log("  -> S6.3 can frame as SIGNIFICANT dissociation (not just trend)")
        else:
            log("INTERPRETATION: RPM dissociation remains non-significant")
            log("  -> S6.3 retains 'numerical trend' framing")

        log("")
        log("Step 13: Formal dissociation tests complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
