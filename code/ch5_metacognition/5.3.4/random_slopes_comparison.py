"""
RQ 6.3.4 - Random Slopes Comparison 

PURPOSE:
Test intercepts-only vs intercepts+slopes random effects structure to justify
variance decomposition approach. Per validation Step 12, cannot claim homogeneous
vs heterogeneous effects without empirical testing.

COMPARISON:
Model A (Intercepts-only): theta ~ TSVR_hours + (1 | UID)
Model B (Intercepts+slopes): theta ~ TSVR_hours + (TSVR_hours | UID)

DECISION CRITERIA:
- ΔAIC > 2: Slopes improve fit → Use slopes (individual differences confirmed)
- ΔAIC < 2: Slopes don't improve → Keep intercepts (homogeneous OK)
- Convergence failure: Document attempt, keep intercepts

Date: 2025-12-30
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup paths
RQ_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

LOG_FILE = RQ_DIR / "logs" / "random_slopes_comparison.log"
DATA_DIR = RQ_DIR / "data"
(RQ_DIR / "logs").mkdir(exist_ok=True)

DOMAINS = ["What", "Where", "When"]

def log(msg):
    """Log to file and stdout"""
    with open(LOG_FILE, 'w' if not LOG_FILE.exists() else 'a', encoding='utf-8') as f:
        f.write(f"{msg}\n")
        f.flush()
    print(msg, flush=True)


def main():
    import statsmodels.formula.api as smf
    
    log("=" * 80)
    log("RQ 6.3.4 - Random Slopes Comparison (MANDATORY validation CHECK)")
    log(f"Started: {datetime.now().isoformat()}")
    log("=" * 80)
    
    # Load data
    theta_path = PROJECT_ROOT / "results" / "ch6" / "6.3.1" / "data" / "step03_theta_confidence.csv"
    tsvr_path = PROJECT_ROOT / "results" / "ch6" / "6.3.1" / "data" / "step00_tsvr_mapping.csv"
    
    log(f"\nLoading data:")
    log(f"  Theta: {theta_path}")
    log(f"  TSVR: {tsvr_path}")
    
    theta_df = pd.read_csv(theta_path)
    tsvr_df = pd.read_csv(tsvr_path)
    
    df = theta_df.merge(tsvr_df, on='composite_ID', how='inner')
    df['UID'] = df['composite_ID'].str.split('_').str[0]
    
    log(f"  ✓ N={len(df)} observations, {df['UID'].nunique()} participants")
    
    # Results storage
    results_list = []
    
    # Test each domain
    for domain in DOMAINS:
        log(f"\n{'=' * 80}")
        log(f"DOMAIN: {domain}")
        log(f"{'=' * 80}")
        
        theta_col = f'theta_{domain}'
        domain_df = df[['UID', 'TSVR_hours', theta_col]].copy()
        domain_df = domain_df.rename(columns={theta_col: 'theta_confidence'})
        domain_df = domain_df.dropna()
        
        log(f"\nN observations: {len(domain_df)}")
        
        # MODEL A: Intercepts-only
        log("\n[MODEL A] Intercepts-only: (1 | UID)")
        formula = "theta_confidence ~ TSVR_hours"
        re_formula_int = "~1"
        
        try:
            model_int = smf.mixedlm(formula, domain_df, groups=domain_df["UID"], 
                                   re_formula=re_formula_int)
            result_int = model_int.fit(reml=False)
            
            aic_int = result_int.aic
            converged_int = result_int.converged
            
            log(f"  AIC: {aic_int:.4f}")
            log(f"  Converged: {converged_int}")
            
            # Extract variance components
            var_int_only = result_int.cov_re.iloc[0, 0]
            var_res_int = result_int.scale
            
            log(f"  var_intercept: {var_int_only:.6f}")
            log(f"  var_residual: {var_res_int:.6f}")
            
        except Exception as e:
            log(f"  Intercepts-only failed: {e}")
            aic_int = np.nan
            converged_int = False
        
        # MODEL B: Intercepts + Slopes (ORIGINAL MODEL)
        log("\n[MODEL B] Intercepts+Slopes: (TSVR_hours | UID)")
        re_formula_slope = "~TSVR_hours"
        
        try:
            model_slope = smf.mixedlm(formula, domain_df, groups=domain_df["UID"],
                                     re_formula=re_formula_slope)
            result_slope = model_slope.fit(reml=False)
            
            aic_slope = result_slope.aic
            converged_slope = result_slope.converged
            
            log(f"  AIC: {aic_slope:.4f}")
            log(f"  Converged: {converged_slope}")
            
            # Extract variance components
            cov_re = result_slope.cov_re
            var_int_slope = cov_re.iloc[0, 0]
            var_slope = cov_re.iloc[1, 1] if cov_re.shape[0] > 1 else 0.0
            var_res_slope = result_slope.scale
            
            log(f"  var_intercept: {var_int_slope:.6f}")
            log(f"  var_slope: {var_slope:.9f}")
            log(f"  var_residual: {var_res_slope:.6f}")
            
        except Exception as e:
            log(f"  Slopes model failed: {e}")
            aic_slope = np.nan
            converged_slope = False
            var_slope = np.nan
        
        # COMPARISON
        log(f"\n")
        if not np.isnan(aic_int) and not np.isnan(aic_slope):
            delta_aic = aic_int - aic_slope
            log(f"  Intercepts-only AIC: {aic_int:.4f}")
            log(f"  Intercepts+slopes AIC: {aic_slope:.4f}")
            log(f"  ΔAIC (Int - Slopes): {delta_aic:.4f}")
            
            # Decision
            if delta_aic > 2:
                decision = "SLOPES IMPROVE (use slopes model)"
                rationale = f"ΔAIC={delta_aic:.2f}>2: Slopes model justified by data"
            elif delta_aic < -2:
                decision = "INTERCEPTS BETTER (slopes overfit)"
                rationale = f"ΔAIC={delta_aic:.2f}<-2: Intercepts-only preferred"
            else:
                decision = "EQUIVALENT (use simpler intercepts)"
                rationale = f"|ΔAIC|={abs(delta_aic):.2f}<2: Models equivalent, prefer simpler"
            
            log(f"\n  ✓ DECISION: {decision}")
            log(f"  ✓ RATIONALE: {rationale}")
            
            # Interpret slope variance
            if var_slope > 0.01:
                slope_interp = f"Substantial slope variance ({var_slope:.4f}) - individual differences CONFIRMED"
            elif var_slope > 0.001:
                slope_interp = f"Small slope variance ({var_slope:.4f}) - minor individual differences"
            else:
                slope_interp = f"Negligible slope variance ({var_slope:.6f}) - homogeneous slopes"
            
            log(f"  ✓ SLOPE VARIANCE: {slope_interp}")
            
        else:
            delta_aic = np.nan
            decision = "CANNOT COMPARE (convergence failure)"
            rationale = "One or both models failed to converge"
            slope_interp = "Unknown"
            log(f"\n  ⚠ {decision}")
        
        # Store results
        results_list.append({
            'domain': domain,
            'aic_intercepts_only': aic_int,
            'aic_intercepts_slopes': aic_slope,
            'delta_aic': delta_aic,
            'converged_intercepts': converged_int,
            'converged_slopes': converged_slope,
            'var_slope': var_slope,
            'decision': decision,
            'rationale': rationale,
            'slope_interpretation': slope_interp
        })
    
    # Save results
    results_df = pd.DataFrame(results_list)
    output_path = DATA_DIR / "random_slopes_comparison.csv"
    results_df.to_csv(output_path, index=False)
    
    log(f"\n{'=' * 80}")
    log(f"Random Slopes Comparison Complete")
    log(f"{'=' * 80}")
    log(f"\nSaved: {output_path}")
    
    log(f"\n[KEY FINDINGS]:")
    for _, row in results_df.iterrows():
        log(f"  {row['domain']}: {row['decision']}")
        log(f"    → {row['rationale']}")
    
    log(f"\nCompleted: {datetime.now().isoformat()}")
    return results_df


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        log(f"\n{e}")
        import traceback
        log(traceback.format_exc())
        raise
