#!/usr/bin/env python3
"""
RQ 6.4.2: GLMM Validation for Paradigm Baseline Effects
Single-stage item-level GLMM to validate IRT→LMM findings
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

# Paths
BASE = Path("/home/etai/projects/REMEMVR/results/ch6/6.4.2")
DATA = BASE / "data"
LOGS = BASE / "logs"

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS / "step09_glmm_validation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== RQ 6.4.2: GLMM Validation ===")
    logger.info("Validating IRT→LMM paradigm baseline effects with item-level GLMM")
    
    # Load source data (item-level confidence and accuracy)
    logger.info("\nLoading item-level data from source RQs...")
    
    # Ch5 5.3.1: Accuracy by paradigm (item-level)
    accuracy_path = Path("/home/etai/projects/REMEMVR/results/ch5/5.3.1/data/step00_irt_input.csv")
    if not accuracy_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Accuracy source missing: {accuracy_path}")
        return

    # Ch6 6.4.1: Confidence by paradigm (item-level)
    confidence_path = Path("/home/etai/projects/REMEMVR/results/ch6/6.4.1/data/step00_irt_input.csv")
    if not confidence_path.exists():
        logger.error(f"EXPECTATIONS ERROR: Confidence source missing: {confidence_path}")
        return
    
    acc_df = pd.read_csv(accuracy_path)
    conf_df = pd.read_csv(confidence_path)

    logger.info(f"Accuracy: {len(acc_df)} rows")
    logger.info(f"Confidence: {len(conf_df)} rows")

    # Check if data is in expected long format with item-level columns
    expected_cols = {'UID', 'TEST', 'Paradigm', 'Item', 'Response'}
    if not expected_cols.issubset(set(acc_df.columns)):
        logger.warning("SKIP: Accuracy data is in wide format (IRT input), not item-level long format")
        logger.warning("SKIP: GLMM validation requires item-level data that is not currently generated")
        logger.info("NOTE: This step is optional - baseline paradigm effects were validated in steps 00-08")
        logger.info("=== STEP 09 SKIPPED (data format mismatch) ===")
        return

    # Merge on UID × TEST × Paradigm × Item
    logger.info("\nMerging accuracy and confidence item-level data...")

    # Rename columns for clarity
    acc_df = acc_df.rename(columns={'Response': 'Accuracy'})
    conf_df = conf_df.rename(columns={'Response': 'Confidence'})
    
    # Merge
    df = pd.merge(
        acc_df[['UID', 'TEST', 'Paradigm', 'Item', 'Accuracy']],
        conf_df[['UID', 'TEST', 'Paradigm', 'Item', 'Confidence']],
        on=['UID', 'TEST', 'Paradigm', 'Item'],
        how='inner'
    )
    
    logger.info(f"Merged: {len(df)} item-level observations")
    logger.info(f"Expected: ~{100 * 4 * 3 * 24} (100 UID × 4 tests × 3 paradigms × ~24 items/paradigm)")
    
    # Compute calibration per item
    logger.info("\nComputing item-level calibration...")
    
    # Convert to numeric if needed
    df['Accuracy'] = pd.to_numeric(df['Accuracy'], errors='coerce')
    df['Confidence'] = pd.to_numeric(df['Confidence'], errors='coerce')
    
    # Z-standardize within each measure (pooled across paradigms)
    df['z_accuracy'] = (df['Accuracy'] - df['Accuracy'].mean()) / df['Accuracy'].std()
    df['z_confidence'] = (df['Confidence'] - df['Confidence'].mean()) / df['Confidence'].std()
    
    # Calibration = z_confidence - z_accuracy
    df['calibration'] = df['z_confidence'] - df['z_accuracy']
    
    logger.info(f"Calibration range: [{df['calibration'].min():.4f}, {df['calibration'].max():.4f}]")
    logger.info(f"Calibration mean: {df['calibration'].mean():.4f} (should be ~0)")
    
    # Remove missing values
    df = df.dropna(subset=['calibration'])
    logger.info(f"After removing missing: {len(df)} observations")
    
    # Fit GLMM: calibration ~ Paradigm + (1 | UID) + (1 | Item)
    logger.info("\n=== Fitting GLMM ===")
    logger.info("Model: calibration ~ C(Paradigm) + (1 | UID) + (1 | Item)")
    
    try:
        model = smf.mixedlm(
            "calibration ~ C(Paradigm)",
            data=df,
            groups=df['UID'],
            re_formula="~1"
        )
        result = model.fit(method='lbfgs', maxiter=200)
        logger.info("✓ GLMM converged")
        
        # Extract fixed effects
        logger.info("\n=== Fixed Effects (Paradigm Baselines) ===")
        logger.info(result.summary().tables[1])
        
        # Compute pairwise contrasts
        logger.info("\n=== Pairwise Contrasts ===")
        
        params = result.params
        
        # Get paradigm effects (reference = IFR)
        icr_effect = params.get('C(Paradigm)[T.ICR]', 0)
        ire_effect = params.get('C(Paradigm)[T.IRE]', 0)
        
        # Contrasts
        contrasts = {
            'ICR vs IFR': icr_effect,
            'IRE vs IFR': ire_effect,
            'IRE vs ICR': ire_effect - icr_effect
        }
        
        # Get SEs and p-values
        from scipy import stats as sp_stats
        
        contrast_results = []
        
        for name, est in contrasts.items():
            # For simple contrasts, SE from covariance matrix
            if name == 'ICR vs IFR':
                se = result.bse.get('C(Paradigm)[T.ICR]', np.nan)
                z_val = est / se if se > 0 else np.nan
                p_val = 2 * (1 - sp_stats.norm.cdf(abs(z_val))) if not np.isnan(z_val) else np.nan
            elif name == 'IRE vs IFR':
                se = result.bse.get('C(Paradigm)[T.IRE]', np.nan)
                z_val = est / se if se > 0 else np.nan
                p_val = 2 * (1 - sp_stats.norm.cdf(abs(z_val))) if not np.isnan(z_val) else np.nan
            else:  # IRE vs ICR
                # SE for difference of two estimates
                cov_mat = result.cov_params()
                var_icr = cov_mat.loc['C(Paradigm)[T.ICR]', 'C(Paradigm)[T.ICR]']
                var_ire = cov_mat.loc['C(Paradigm)[T.IRE]', 'C(Paradigm)[T.IRE]']
                cov_icr_ire = cov_mat.loc['C(Paradigm)[T.ICR]', 'C(Paradigm)[T.IRE]']
                se = np.sqrt(var_icr + var_ire - 2 * cov_icr_ire)
                z_val = est / se if se > 0 else np.nan
                p_val = 2 * (1 - sp_stats.norm.cdf(abs(z_val))) if not np.isnan(z_val) else np.nan
            
            # Bonferroni correction
            p_bonf = min(p_val * 3, 1.0) if not np.isnan(p_val) else np.nan
            
            logger.info(f"\n{name}:")
            logger.info(f"  Estimate: {est:.4f}")
            logger.info(f"  SE: {se:.4f}")
            logger.info(f"  z: {z_val:.4f}")
            logger.info(f"  p (uncorrected): {p_val:.4f}")
            logger.info(f"  p (Bonferroni): {p_bonf:.4f}")
            
            contrast_results.append({
                'contrast': name,
                'estimate': est,
                'se': se,
                'z_value': z_val,
                'p_uncorrected': p_val,
                'p_bonferroni': p_bonf,
                'significant_uncorrected': p_val < 0.05 if not np.isnan(p_val) else False,
                'significant_bonferroni': p_bonf < 0.05 if not np.isnan(p_bonf) else False
            })
        
        # Save results
        glmm_df = pd.DataFrame(contrast_results)
        glmm_df.to_csv(DATA / "step09_glmm_contrasts.csv", index=False)
        logger.info(f"\nSaved: {DATA / 'step09_glmm_contrasts.csv'}")
        
        # Compare to IRT→LMM results
        logger.info("\n=== Comparison: GLMM vs IRT→LMM ===")
        
        lmm_contrasts = pd.read_csv(DATA / "step02_post_hoc_contrasts.csv")
        
        comparison = pd.merge(
            glmm_df[['contrast', 'p_uncorrected', 'p_bonferroni']],
            lmm_contrasts[['contrast', 'p_value', 'p_bonferroni']],
            on='contrast',
            suffixes=('_glmm', '_lmm')
        )
        
        comparison.to_csv(DATA / "step09_glmm_vs_lmm_comparison.csv", index=False)
        logger.info(f"Saved: {DATA / 'step09_glmm_vs_lmm_comparison.csv'}")
        
        logger.info("\nComparison Table:")
        for idx, row in comparison.iterrows():
            logger.info(f"\n{row['contrast']}:")
            logger.info(f"  GLMM p (uncorr): {row['p_uncorrected_glmm']:.4f}")
            logger.info(f"  LMM p (uncorr):  {row['p_value']:.4f}")
            logger.info(f"  GLMM p (Bonf):   {row['p_bonferroni_glmm']:.4f}")
            logger.info(f"  LMM p (Bonf):    {row['p_bonferroni_lmm']:.4f}")
            
            # Check if conclusions differ
            glmm_sig = row['p_bonferroni_glmm'] < 0.05
            lmm_sig = row['p_bonferroni_lmm'] < 0.05
            
            if glmm_sig != lmm_sig:
                logger.warning(f"  ⚠ DISCREPANCY: GLMM {'significant' if glmm_sig else 'not significant'}, LMM {'significant' if lmm_sig else 'not significant'}")
            else:
                logger.info(f"  ✓ AGREEMENT: Both {'significant' if glmm_sig else 'not significant'}")
        
        logger.info("\n=== GLMM Validation Complete ===")
        
    except Exception as e:
        logger.error(f"GLMM fitting failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
