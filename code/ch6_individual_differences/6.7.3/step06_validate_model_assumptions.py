#!/usr/bin/env python3
"""validate_model_assumptions: Comprehensive assumption checking for all 5 regression models:"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# RQ directory
RQ_DIR = Path(__file__).resolve().parents[1]
LOG_FILE = RQ_DIR / "logs" / "step06_validate_model_assumptions.log"
OUTPUT_TESTS = RQ_DIR / "data" / "step06_assumption_tests.csv"
OUTPUT_DIAG = RQ_DIR / "data" / "step06_diagnostic_data.csv"
OUTPUT_ROBUST = RQ_DIR / "data" / "step06_robust_results.csv"

# Configuration

VIF_THRESHOLD = 5.0
COOKS_D_THRESHOLD = 4.0 / 100  # 4/N where N=100

# Logging Function

def log(msg):
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)

# Main Analysis

if __name__ == "__main__":
    try:
        log("Step 06: Validate Model Assumptions")
        # Load Data
        log("Loading analysis dataset...")

        data_path = RQ_DIR / "data" / "step03_analysis_dataset.csv"
        df = pd.read_csv(data_path)

        log(f"{data_path.name} ({len(df)} participants)")
        # Define Models
        models_spec = [
            {'name': 'Model_1_Total', 'predictors': ['Total_z']},
            {'name': 'Model_2_Learning', 'predictors': ['Learning_z']},
            {'name': 'Model_3_LearningSlope', 'predictors': ['LearningSlope_z']},
            {'name': 'Model_4_Forgetting', 'predictors': ['Forgetting_z']},
            {'name': 'Model_5_Recognition', 'predictors': ['Recognition_z']},
            {'name': 'Model_6_PctRet', 'predictors': ['PctRet_z']},
            {'name': 'Model_7_Combined', 'predictors': ['Total_z', 'Learning_z']}
        ]

        assumption_tests = []
        diagnostic_data = []
        robust_results = []
        # Fit Models and Run Diagnostics
        for spec in models_spec:
            model_name = spec['name']
            predictors = spec['predictors']

            log(f"{model_name}: {', '.join(predictors)}")

            # Prepare data
            X = df[predictors]
            y = df['theta_all']
            X_with_const = sm.add_constant(X)

            # Fit model
            model = sm.OLS(y, X_with_const).fit()

            # Extract residuals and fitted values
            residuals = model.resid
            fitted = model.fittedvalues
            std_residuals = residuals / np.std(residuals)
            # TEST 1: Normality (Shapiro-Wilk)
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
            log(f"  Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")

            if shapiro_p > 0.05:
                log(f"  Residuals normally distributed (p={shapiro_p:.4f})")
            else:
                log(f"  Residuals non-normal (p={shapiro_p:.4f})")
            # TEST 2: Homoscedasticity (Breusch-Pagan)
            bp_test = het_breuschpagan(residuals, X_with_const)
            bp_stat, bp_p = bp_test[0], bp_test[1]
            log(f"  Breusch-Pagan: LM={bp_stat:.4f}, p={bp_p:.4f}")

            if bp_p > 0.05:
                log(f"  Homoscedasticity assumption met (p={bp_p:.4f})")
                heteroscedasticity_detected = False
            else:
                log(f"  Heteroscedasticity detected (p={bp_p:.4f})")
                heteroscedasticity_detected = True
            # TEST 3: Independence (Durbin-Watson)
            dw_stat = durbin_watson(residuals)
            log(f"  Durbin-Watson: DW={dw_stat:.4f}")

            if 1.5 <= dw_stat <= 2.5:
                log(f"  No autocorrelation (DW={dw_stat:.4f})")
            else:
                log(f"  Possible autocorrelation (DW={dw_stat:.4f})")
            # TEST 4: Multicollinearity (VIF) - Only for Model 5
            if len(predictors) > 1:
                vif_data = pd.DataFrame()
                vif_data['predictor'] = X.columns
                vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

                max_vif = vif_data['VIF'].max()
                log(f"  Max VIF={max_vif:.2f}")

                if max_vif < VIF_THRESHOLD:
                    log(f"  No multicollinearity (max VIF={max_vif:.2f})")
                else:
                    log(f"  Multicollinearity detected (max VIF={max_vif:.2f})")
            else:
                max_vif = np.nan
                log(f"  N/A (single predictor)")
            # TEST 5: Outliers (Cook's D)
            influence = model.get_influence()
            cooks_d = influence.cooks_distance[0]
            outlier_count = np.sum(cooks_d > COOKS_D_THRESHOLD)

            log(f"  Cook's D: max={np.max(cooks_d):.4f}, threshold={COOKS_D_THRESHOLD:.4f}")

            if outlier_count == 0:
                log(f"  No influential outliers")
            else:
                log(f"  {outlier_count} influential outliers detected (Cook's D > {COOKS_D_THRESHOLD:.4f})")

            # Store assumption test results
            assumption_tests.append({
                'model': model_name,
                'shapiro_stat': shapiro_stat,
                'shapiro_p': shapiro_p,
                'bp_stat': bp_stat,
                'bp_p': bp_p,
                'dw_stat': dw_stat,
                'max_vif': max_vif,
                'outlier_count': outlier_count
            })

            # Store diagnostic data (for plotting)
            for i in range(len(df)):
                diagnostic_data.append({
                    'model': model_name,
                    'fitted': fitted.iloc[i],
                    'residuals': residuals.iloc[i],
                    'std_residuals': std_residuals[i],
                    'cooks_d': cooks_d[i]
                })
            # ROBUST STANDARD ERRORS (if heteroscedasticity detected)
            if heteroscedasticity_detected:
                log(f"  Computing HC3 robust standard errors...")
                model_robust = sm.OLS(y, X_with_const).fit(cov_type='HC3')

                for i, param_name in enumerate(model_robust.params.index):
                    if param_name == 'const':
                        continue

                    beta = model_robust.params[i]
                    se_robust = model_robust.bse[i]
                    p_robust = model_robust.pvalues[i]

                    robust_results.append({
                        'model': model_name,
                        'predictor': param_name,
                        'beta': beta,
                        'se_robust': se_robust,
                        'p_robust': p_robust
                    })

                log(f"  HC3 standard errors computed")

        log("All assumption tests complete")
        # Save Results
        log("Saving assumption test results...")

        tests_df = pd.DataFrame(assumption_tests)
        tests_df.to_csv(OUTPUT_TESTS, index=False, encoding='utf-8')
        log(f"{OUTPUT_TESTS} ({len(tests_df)} models)")

        log("Saving diagnostic data...")

        diag_df = pd.DataFrame(diagnostic_data)
        diag_df.to_csv(OUTPUT_DIAG, index=False, encoding='utf-8')
        log(f"{OUTPUT_DIAG} ({len(diag_df)} observations)")

        if robust_results:
            log("Saving robust SE results...")
            robust_df = pd.DataFrame(robust_results)
            robust_df.to_csv(OUTPUT_ROBUST, index=False, encoding='utf-8')
            log(f"{OUTPUT_ROBUST} ({len(robust_df)} coefficients)")
        else:
            log("No heteroscedasticity detected - robust SE not needed")
        # Summary
        log("Assumption violations:")
        violations = 0

        for _, row in tests_df.iterrows():
            model_violations = []
            if row['shapiro_p'] < 0.05:
                model_violations.append("Non-normality")
            if row['bp_p'] < 0.05:
                model_violations.append("Heteroscedasticity")
            if row['dw_stat'] < 1.5 or row['dw_stat'] > 2.5:
                model_violations.append("Autocorrelation")
            if not np.isnan(row['max_vif']) and row['max_vif'] > VIF_THRESHOLD:
                model_violations.append("Multicollinearity")
            if row['outlier_count'] > 0:
                model_violations.append(f"{int(row['outlier_count'])} outliers")

            if model_violations:
                log(f"  {row['model']}: {', '.join(model_violations)}")
                violations += 1
            else:
                log(f"  {row['model']}: All assumptions met")

        if violations == 0:
            log("All models meet assumptions")
        else:
            log(f"{violations}/{len(models_spec)} models have assumption violations")

        log("Step 06 complete")
        sys.exit(0)

    except Exception as e:
        log(f"{str(e)}")
        log("Full error details:")
        import traceback
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            traceback.print_exc(file=f)
        traceback.print_exc()
        sys.exit(1)
