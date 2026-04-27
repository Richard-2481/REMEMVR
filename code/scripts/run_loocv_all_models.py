"""
LOOCV with Ridge regularisation for all Ch7 regression models.
Saves per-observation predictions and summary stats to each RQ's data/ folder.
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

BASE = Path("/home/etai/projects/REMEMVR")
RIDGE_ALPHAS = np.logspace(-3, 3, 50)

# Define all models: (rq, model_name, data_file, predictors, outcome)
MODELS = [
    ("7.1.1", "cognitive_battery",
     "results/ch7/7.1.1/data/step03_merged_analysis.csv",
     ["RAVLT_T", "RAVLT_Pct_Ret_T", "BVMT_T", "BVMT_Pct_Ret_T", "NART_T", "RPM_T"],
     "theta_mean"),

    ("7.1.4", "block2_demographics_cognitive",
     "results/ch7/7.1.4/data/step05_merged_predictors.csv",
     ["age", "sex_binary", "education", "RAVLT_T", "RAVLT_Pct_Ret", "BVMT_T", "BVMT_Pct_Ret", "RPM_T"],
     "theta"),

    ("7.1.4", "block3_full",
     "results/ch7/7.1.4/data/step05_merged_predictors.csv",
     ["age", "sex_binary", "education", "RAVLT_T", "RAVLT_Pct_Ret", "BVMT_T", "BVMT_Pct_Ret", "RPM_T",
      "Sleep", "VR_Exp", "DASS_Dep", "DASS_Anx", "DASS_Str"],
     "theta"),

    ("7.2.1", "age_only",
     "results/ch7/7.2.1/data/step01_analysis_dataset.csv",
     ["Age_std"],
     "theta_all"),

    ("7.2.1", "age_plus_cognitive",
     "results/ch7/7.2.1/data/step01_analysis_dataset.csv",
     ["Age_std", "RPM_T_std", "BVMT_T_std", "RAVLT_T_std"],
     "theta_all"),

    ("7.3.1", "confidence_prediction",
     "results/ch7/7.3.1/data/step04_analysis_dataset.csv",
     ["age", "sex", "education", "RAVLT_T", "BVMT_T", "RPM_T", "RAVLT_Pct_Ret_T", "BVMT_Pct_Ret_T"],
     "confidence_theta"),

    ("7.3.2", "calibration_prediction",
     "results/ch7/7.3.2/data/step03_analysis_dataset.csv",
     ["age", "sex", "education", "RAVLT_T", "BVMT_T", "RPM_T", "RAVLT_Pct_Ret_T", "BVMT_Pct_Ret_T"],
     "calibration_quality"),

    ("7.3.3", "hce_prediction",
     "results/ch7/7.3.3/data/step03_analysis_dataset.csv",
     ["age_c", "sex", "education", "ravlt_c", "bvmt_c", "rpm_c", "ravlt_pct_ret_c", "bvmt_pct_ret_c"],
     "hce_rate"),

    ("7.3.4", "dass_accuracy",
     "results/ch7/7.3.4/data/step02_analysis_dataset.csv",
     ["z_Dep", "z_Anx", "z_Str"],
     "theta_accuracy"),

    ("7.3.4", "dass_confidence",
     "results/ch7/7.3.4/data/step02_analysis_dataset.csv",
     ["z_Dep", "z_Anx", "z_Str"],
     "confidence"),

    ("7.3.4", "dass_calibration",
     "results/ch7/7.3.4/data/step02_analysis_dataset.csv",
     ["z_Dep", "z_Anx", "z_Str"],
     "calibration"),

    ("7.5.1", "lifestyle_prediction",
     "results/ch7/7.5.1/data/step03_analysis_dataset.csv",
     ["Age_z", "Education_z", "VR_Experience_z", "Typical_Sleep_z"],
     "theta_all"),
]


def run_loocv(df, predictors, outcome, model_name):
    """Run LOOCV for OLS and Ridge on the given data."""
    cols = predictors + [outcome]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns {missing} — skipping {model_name}")
        return None, None

    sub = df[cols].dropna()
    X = sub[predictors].values
    y = sub[outcome].values
    n, p = X.shape

    if n < p + 2:
        print(f"  WARNING: n={n} too small for p={p} — skipping {model_name}")
        return None, None

    loo = LeaveOneOut()

    # OLS pipeline
    ols_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    y_pred_ols = cross_val_predict(ols_pipe, X, y, cv=loo)
    ols_loocv_r2 = r2_score(y, y_pred_ols)

    # Ridge pipeline
    ridge_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS))
    ])
    y_pred_ridge = cross_val_predict(ridge_pipe, X, y, cv=loo)
    ridge_loocv_r2 = r2_score(y, y_pred_ridge)

    # In-sample metrics (OLS)
    ols_pipe.fit(X, y)
    y_hat_insample = ols_pipe.predict(X)
    in_sample_r2 = r2_score(y, y_hat_insample)
    adjusted_r2 = 1 - (1 - in_sample_r2) * (n - 1) / (n - p - 1)

    # Full-data Ridge for alpha
    ridge_pipe.fit(X, y)
    ridge_alpha = ridge_pipe.named_steps["ridge"].alpha_

    # Shrinkage
    shrinkage_ols = in_sample_r2 - ols_loocv_r2
    shrinkage_ridge = in_sample_r2 - ridge_loocv_r2

    # Per-observation predictions
    pred_df = pd.DataFrame({
        "observation_idx": range(n),
        "y_true": y,
        "y_pred_ols": y_pred_ols,
        "y_pred_ridge": y_pred_ridge,
    })

    # Summary row
    summary = {
        "model_name": model_name,
        "n": n,
        "p": p,
        "in_sample_r2": round(in_sample_r2, 4),
        "adjusted_r2": round(adjusted_r2, 4),
        "ols_loocv_r2": round(ols_loocv_r2, 4),
        "ridge_loocv_r2": round(ridge_loocv_r2, 4),
        "ridge_alpha": round(ridge_alpha, 4),
        "shrinkage_ols": round(shrinkage_ols, 4),
        "shrinkage_ridge": round(shrinkage_ridge, 4),
    }

    return pred_df, summary


def main():
    all_summaries = []
    # Group models by RQ for file output
    rq_predictions = {}
    rq_summaries = {}

    for rq, model_name, data_file, predictors, outcome in MODELS:
        filepath = BASE / data_file
        print(f"\n[{rq}] {model_name}: {filepath.name} → {outcome}")

        if not filepath.exists():
            print(f"  ERROR: File not found: {filepath}")
            continue

        df = pd.read_csv(filepath)
        print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} cols")

        pred_df, summary = run_loocv(df, predictors, outcome, model_name)
        if pred_df is None:
            continue

        print(f"  N={summary['n']}, p={summary['p']}")
        print(f"  In-sample R²={summary['in_sample_r2']:.4f}, Adj R²={summary['adjusted_r2']:.4f}")
        print(f"  OLS LOOCV R²={summary['ols_loocv_r2']:.4f}, Ridge LOOCV R²={summary['ridge_loocv_r2']:.4f}")
        print(f"  Ridge α={summary['ridge_alpha']:.4f}")
        print(f"  Shrinkage: OLS={summary['shrinkage_ols']:.4f}, Ridge={summary['shrinkage_ridge']:.4f}")

        rq_predictions.setdefault(rq, []).append(pred_df.assign(model_name=model_name))
        rq_summaries.setdefault(rq, []).append(summary)
        all_summaries.append({**summary, "rq": rq})

    # Save files per RQ
    print("\n" + "=" * 80)
    print("SAVING FILES")
    print("=" * 80)

    for rq in rq_predictions:
        out_dir = BASE / f"results/ch7/{rq}/data"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Predictions
        pred_path = out_dir / "step_loocv_predictions.csv"
        combined_pred = pd.concat(rq_predictions[rq], ignore_index=True)
        combined_pred.to_csv(pred_path, index=False)
        print(f"  Saved: {pred_path} ({len(combined_pred)} rows)")

        # Summary
        sum_path = out_dir / "step_loocv_summary.csv"
        sum_df = pd.DataFrame(rq_summaries[rq])
        sum_df.to_csv(sum_path, index=False)
        print(f"  Saved: {sum_path} ({len(sum_df)} rows)")

    # Print final summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    summary_df = pd.DataFrame(all_summaries)
    cols = ["rq", "model_name", "n", "p", "in_sample_r2", "adjusted_r2",
            "ols_loocv_r2", "ridge_loocv_r2", "ridge_alpha", "shrinkage_ols", "shrinkage_ridge"]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
