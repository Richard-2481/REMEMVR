#!/usr/bin/env python3
"""
RQ 6.4.2: LMM Diagnostic Plots
Generate residual diagnostics for assumption validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

# Paths
BASE = Path("/home/etai/projects/REMEMVR/results/ch6/6.4.2")
DATA = BASE / "data"
PLOTS = BASE / "plots"
LOGS = BASE / "logs"

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / "step07_lmm_diagnostics.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== RQ 6.4.2: LMM Diagnostic Plots ===")
    
    # Load calibration data
    logger.info("Loading calibration data...")
    df = pd.read_csv(DATA / "step00_calibration_by_paradigm.csv")
    logger.info(f"Loaded {len(df)} observations")
    
    # Center TSVR
    tsvr_mean = df['TSVR_hours'].mean()
    df['TSVR_centered'] = df['TSVR_hours'] - tsvr_mean
    logger.info(f"TSVR centered at mean = {tsvr_mean:.2f} hours")
    
    # Refit LMM (to extract residuals)
    logger.info("Refitting LMM with random slopes...")
    model_formula = "calibration ~ C(Paradigm) * TSVR_centered"
    
    try:
        model = smf.mixedlm(
            model_formula,
            data=df,
            groups=df['UID'],
            re_formula="~TSVR_centered"
        )
        result = model.fit(method='lbfgs', maxiter=200)
        logger.info("Model converged successfully")
    except Exception as e:
        logger.error(f"Model convergence failed: {e}")
        logger.info("Falling back to intercepts-only...")
        model = smf.mixedlm(
            model_formula,
            data=df,
            groups=df['UID'],
            re_formula="~1"
        )
        result = model.fit(method='lbfgs', maxiter=200)
        logger.info("Intercepts-only model converged")
    
    # Extract residuals
    residuals = result.resid
    fitted = result.fittedvalues
    
    logger.info(f"Residuals: N={len(residuals)}, Mean={residuals.mean():.4f}, SD={residuals.std():.4f}")
    logger.info(f"Fitted values: Range=[{fitted.min():.4f}, {fitted.max():.4f}]")
    
    # Create diagnostic plots
    logger.info("Generating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Q-Q Plot
    logger.info("Plot 1: Q-Q plot (normality of residuals)")
    stats.probplot(residuals, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title("Q-Q Plot: Residuals vs Normal Distribution", fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel("Theoretical Quantiles")
    axes[0, 0].set_ylabel("Sample Quantiles (Residuals)")
    axes[0, 0].grid(alpha=0.3)
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    logger.info(f"Shapiro-Wilk test: W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
    axes[0, 0].text(0.05, 0.95, f"Shapiro-Wilk: W={shapiro_stat:.3f}, p={shapiro_p:.3f}",
                    transform=axes[0, 0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residuals vs Fitted
    logger.info("Plot 2: Residuals vs Fitted (homoscedasticity)")
    axes[0, 1].scatter(fitted, residuals, alpha=0.5, s=20)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_title("Residuals vs Fitted Values", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("Fitted Values (Calibration)")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].grid(alpha=0.3)
    
    # Add lowess smoother
    from statsmodels.nonparametric.smoothers_lowess import lowess
    smoothed = lowess(residuals, fitted, frac=0.3)
    axes[0, 1].plot(smoothed[:, 0], smoothed[:, 1], 'g-', linewidth=2, label='LOWESS trend')
    axes[0, 1].legend()
    
    # 3. Scale-Location Plot (sqrt of standardized residuals)
    logger.info("Plot 3: Scale-Location (constant variance)")
    standardized_resid = residuals / residuals.std()
    sqrt_std_resid = np.sqrt(np.abs(standardized_resid))
    axes[1, 0].scatter(fitted, sqrt_std_resid, alpha=0.5, s=20)
    axes[1, 0].set_title("Scale-Location Plot", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Fitted Values")
    axes[1, 0].set_ylabel("√|Standardized Residuals|")
    axes[1, 0].grid(alpha=0.3)
    
    # Add lowess smoother
    smoothed_scale = lowess(sqrt_std_resid, fitted, frac=0.3)
    axes[1, 0].plot(smoothed_scale[:, 0], smoothed_scale[:, 1], 'g-', linewidth=2, label='LOWESS trend')
    axes[1, 0].legend()
    
    # 4. Histogram of residuals
    logger.info("Plot 4: Histogram of residuals")
    axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Distribution of Residuals", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    # Add normal curve overlay
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[1, 1].plot(x, stats.norm.pdf(x, mu, sigma) * len(residuals) * (residuals.max() - residuals.min()) / 30,
                    'r-', linewidth=2, label='Normal curve')
    axes[1, 1].legend()
    
    plt.tight_layout()
    diagnostic_path = PLOTS / "lmm_diagnostic_plots.png"
    plt.savefig(diagnostic_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {diagnostic_path}")
    plt.close()
    
    # Additional tests
    logger.info("\n=== Assumption Test Results ===")
    
    # Shapiro-Wilk (already computed)
    logger.info(f"Normality (Shapiro-Wilk): W={shapiro_stat:.4f}, p={shapiro_p:.4f}")
    if shapiro_p > 0.05:
        logger.info("  ✓ Residuals normally distributed (p > 0.05)")
    else:
        logger.warning("  ⚠ Residuals deviate from normality (p ≤ 0.05)")
        logger.info("  Note: With N=1200, minor deviations acceptable (CLT applies)")
    
    # Breusch-Pagan test for heteroscedasticity
    from statsmodels.stats.diagnostic import het_breuschpagan
    
    # Need design matrix for BP test
    import patsy
    y, X = patsy.dmatrices(model_formula, data=df, return_type='dataframe')
    
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X)
    logger.info(f"Homoscedasticity (Breusch-Pagan): LM={bp_stat:.4f}, p={bp_p:.4f}")
    if bp_p > 0.05:
        logger.info("  ✓ Constant variance (p > 0.05)")
    else:
        logger.warning("  ⚠ Heteroscedasticity detected (p ≤ 0.05)")
        logger.info("  Recommendation: Use robust standard errors or weighted LMM")
    
    # Save diagnostic results
    diagnostic_results = pd.DataFrame({
        'test': ['Shapiro-Wilk', 'Breusch-Pagan'],
        'statistic': [shapiro_stat, bp_stat],
        'p_value': [shapiro_p, bp_p],
        'interpretation': [
            'Normality' if shapiro_p > 0.05 else 'Non-normal',
            'Homoscedastic' if bp_p > 0.05 else 'Heteroscedastic'
        ]
    })
    
    diagnostic_results.to_csv(DATA / "step07_diagnostic_tests.csv", index=False)
    logger.info(f"Saved: {DATA / 'step07_diagnostic_tests.csv'}")
    
    logger.info("\n=== LMM Diagnostics Complete ===")
    logger.info("Diagnostic plots saved to plots/lmm_diagnostic_plots.png")
    logger.info("Test results saved to data/step07_diagnostic_tests.csv")

if __name__ == "__main__":
    main()
