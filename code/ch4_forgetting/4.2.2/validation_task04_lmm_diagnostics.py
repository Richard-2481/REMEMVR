"""
validation TASK 4: LMM Diagnostic Plots

Generate Q-Q plot and residuals vs fitted for assumption validation
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from pathlib import Path

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

BASE = Path("/home/etai/projects/REMEMVR/results/ch5/5.2.2")
DATA_DIR = BASE / "data"
PLOTS_DIR = BASE / "plots" / "diagnostics"
LOGS_DIR = BASE / "logs"

log_file = LOGS_DIR / "platinum_task04_lmm_diagnostics.log"
log = open(log_file, 'w')

def logprint(msg):
    print(msg)
    log.write(msg + '\n')
    log.flush()

logprint("=" * 70)
logprint("validation TASK 4: LMM Diagnostic Plots")
logprint("=" * 70)
logprint("")

# Load data
logprint("Loading piecewise LMM input data...")
data_path = DATA_DIR / "step00_piecewise_lmm_input.csv"
data = pd.read_csv(data_path)
logprint(f"{data.shape[0]} observations")
logprint("")

# Load model
logprint("Loading fitted LMM model...")
model_path = DATA_DIR / "step01_piecewise_lmm_model.pkl"

# Load via custom unpickler (avoid patsy environment issues)
with open(model_path, 'rb') as f:
    # Read raw pickle data
    import statsmodels.api as sm
    from statsmodels.regression.mixed_linear_model import MixedLM
    
    # Re-fit model (safer than unpickling)
    logprint("Re-fitting model to extract diagnostics...")
    formula = "theta ~ Days_within * C(Segment, Treatment('Early')) * C(domain, Treatment('what'))"
    model = MixedLM.from_formula(
        formula=formula,
        groups=data['UID'],
        re_formula="~Days_within",
        data=data
    )
    result = model.fit(reml=False, method='lbfgs')
    logprint("Model re-fit complete")
    logprint("")

# Extract residuals and fitted values
logprint("Computing residuals and fitted values...")
fitted_values = result.fittedvalues
residuals = result.resid
logprint(f"{len(fitted_values)} fitted values, {len(residuals)} residuals")
logprint("")

# DIAGNOSTIC 1: Q-Q PLOT (Normality)
logprint("=" * 70)
logprint("DIAGNOSTIC 1: Q-Q Plot (Residual Normality)")
logprint("=" * 70)
logprint("")

fig, ax = plt.subplots(figsize=(8, 6))
scipy_stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title("Normal Q-Q Plot: LMM Residuals", fontsize=14, fontweight='bold')
ax.set_xlabel("Theoretical Quantiles", fontsize=12)
ax.set_ylabel("Sample Quantiles (Residuals)", fontsize=12)
ax.grid(True, alpha=0.3)

qq_plot_path = PLOTS_DIR / "qq_plot_residuals.png"
plt.tight_layout()
plt.savefig(qq_plot_path, dpi=300, bbox_inches='tight')
plt.close()

logprint(f"{qq_plot_path}")

# Shapiro-Wilk test
if len(residuals) <= 5000:  # Shapiro-Wilk limit
    stat, p = scipy_stats.shapiro(residuals)
    logprint(f"Shapiro-Wilk test for normality:")
    logprint(f"       W = {stat:.4f}, p = {p:.4f}")
    if p > 0.05:
        logprint(f"       ✓ Residuals consistent with normal distribution (p > 0.05)")
    else:
        logprint(f"       ⚠ Residuals deviate from normality (p < 0.05)")
        logprint(f"         Note: With N={len(residuals)}, test very sensitive to minor deviations")
logprint("")

# DIAGNOSTIC 2: RESIDUALS VS FITTED (Homoscedasticity)
logprint("=" * 70)
logprint("DIAGNOSTIC 2: Residuals vs Fitted (Homoscedasticity)")
logprint("=" * 70)
logprint("")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(fitted_values, residuals, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
ax.set_title("Residuals vs Fitted Values", fontsize=14, fontweight='bold')
ax.set_xlabel("Fitted Values (Predicted Theta)", fontsize=12)
ax.set_ylabel("Residuals", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Add LOESS smoother to detect patterns
from scipy.interpolate import UnivariateSpline
try:
    # Sort by fitted values for smoother
    sort_idx = np.argsort(fitted_values)
    fitted_sorted = fitted_values[sort_idx]
    resid_sorted = residuals[sort_idx]
    
    # Spline smoother
    spl = UnivariateSpline(fitted_sorted, resid_sorted, s=len(fitted_sorted)*0.1)
    fitted_smooth = np.linspace(fitted_sorted.min(), fitted_sorted.max(), 100)
    resid_smooth = spl(fitted_smooth)
    
    ax.plot(fitted_smooth, resid_smooth, color='blue', linewidth=2, label='Smoother')
    ax.legend()
except:
    logprint("Could not add LOESS smoother (too few points or numerical issue)")

resid_plot_path = PLOTS_DIR / "residuals_vs_fitted.png"
plt.tight_layout()
plt.savefig(resid_plot_path, dpi=300, bbox_inches='tight')
plt.close()

logprint(f"{resid_plot_path}")

# Breusch-Pagan test for homoscedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
exog = result.model.exog  # Design matrix
try:
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, exog)
    logprint(f"Breusch-Pagan test for homoscedasticity:")
    logprint(f"       LM statistic = {bp_stat:.4f}, p = {bp_p:.4f}")
    if bp_p > 0.05:
        logprint(f"       ✓ Homoscedasticity assumption met (p > 0.05)")
    else:
        logprint(f"       ⚠ Heteroscedasticity detected (p < 0.05)")
        logprint(f"         Consider robust standard errors")
except Exception as e:
    logprint(f"Could not run Breusch-Pagan test: {e}")
logprint("")

# DIAGNOSTIC 3: Scale-Location Plot (Spread-Location)
logprint("=" * 70)
logprint("DIAGNOSTIC 3: Scale-Location Plot")
logprint("=" * 70)
logprint("")

fig, ax = plt.subplots(figsize=(8, 6))
standardized_resid = residuals / residuals.std()
sqrt_abs_resid = np.sqrt(np.abs(standardized_resid))

ax.scatter(fitted_values, sqrt_abs_resid, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
ax.set_title("Scale-Location Plot", fontsize=14, fontweight='bold')
ax.set_xlabel("Fitted Values", fontsize=12)
ax.set_ylabel("√|Standardized Residuals|", fontsize=12)
ax.grid(True, alpha=0.3)

scale_loc_path = PLOTS_DIR / "scale_location.png"
plt.tight_layout()
plt.savefig(scale_loc_path, dpi=300, bbox_inches='tight')
plt.close()

logprint(f"{scale_loc_path}")
logprint("")

# Summary
logprint("=" * 70)
logprint("SUMMARY")
logprint("=" * 70)
logprint("")

logprint("Diagnostic plots generated:")
logprint(f"  1. Q-Q plot: {qq_plot_path}")
logprint(f"  2. Residuals vs Fitted: {resid_plot_path}")
logprint(f"  3. Scale-Location: {scale_loc_path}")
logprint("")
logprint("Visual inspection recommended for:")
logprint("  - Q-Q plot: Points should follow diagonal line (normality)")
logprint("  - Resid vs Fitted: Random scatter around zero (homoscedasticity)")
logprint("  - Scale-Location: Horizontal band (constant variance)")
logprint("")

logprint("=" * 70)
logprint("TASK 4 COMPLETE")
logprint("=" * 70)

log.close()

print("")
print("✓ LMM Diagnostics Complete")
print(f"  Plots saved to: {PLOTS_DIR}")
