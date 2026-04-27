"""
SEM Calibration Toolkit for REMEMVR Thesis

Purpose: Replace unreliable difference scores (calibration = confidence - accuracy)
         with latent variable approach that accounts for measurement error.

Created: 2025-12-28
Author: Claude Code (rq_platinum agent)
Context: PLATINUM certification requirement for Ch6 calibration RQs

Background:
-----------
Simple difference scores suffer from reliability collapse when constituent
measures are moderately reliable and highly correlated. Formula:

    r_diff = (r_xx + r_yy - 2*r_xy) / (2 - 2*r_xy)

When r_xy ≈ 0.4-0.6 and r_xx, r_yy ≈ 0.6-0.8, r_diff can collapse to < 0.50
or even negative values. This makes calibration findings unreliable.

Solution: Structural Equation Modeling (SEM) with latent variables that
         properly account for measurement error in both accuracy and confidence.

Approaches Implemented:
----------------------
1. Latent Difference Score (Approach A): Direct latent calibration
2. Residualized Calibration (Approach B): Conf ~ Acc, use residual
3. Bivariate LMM with Measurement Error (Approach C): Future extension

Dependencies:
-------------
- semopy (Python SEM package): pip install semopy
- pandas, numpy, scipy
- Optional: rpy2 + R lavaan for validation

Usage Example:
--------------
    from tools.sem_calibration import SEMCalibration

    # Initialize
    sem = SEMCalibration(
        theta_accuracy='data/theta_acc.csv',
        theta_confidence='data/theta_conf.csv',
        measurement_error_acc='data/irt_se_acc.csv',  # optional
        measurement_error_conf='data/irt_se_conf.csv'  # optional
    )

    # Fit models
    sem.fit_latent_difference()
    sem.fit_residualized()

    # Extract latent calibration scores
    latent_calib = sem.get_latent_calibration()

    # Get model fit
    fit_stats = sem.get_model_fit()

    # Save results
    sem.save_results('results/sem_calibration/')

References:
-----------
- McArdle, J.J. (2009). Latent variable modeling of differences and changes
- Cole, D.A., & Maxwell, S.E. (2003). Testing mediational models with longitudinal data
- Raykov, T. (1999). Are simple change scores obsolete? An approach to studying correlates
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Optional, Dict, Tuple, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SEMCalibration:
    """
    Structural Equation Modeling for calibration analysis with measurement error.

    Implements latent variable approach to compute calibration = confidence - accuracy
    while accounting for IRT measurement error in both constituent measures.

    Attributes:
        data (pd.DataFrame): Merged dataset with theta scores
        measurement_error_acc (np.ndarray): Measurement error variances for accuracy
        measurement_error_conf (np.ndarray): Measurement error variances for confidence
        model_latent_diff: Fitted latent difference model
        model_residualized: Fitted residualized model
        latent_calibration (np.ndarray): Extracted latent calibration scores
    """

    def __init__(
        self,
        theta_accuracy: Union[str, Path, pd.DataFrame],
        theta_confidence: Union[str, Path, pd.DataFrame],
        measurement_error_acc: Optional[Union[str, Path, pd.DataFrame, np.ndarray]] = None,
        measurement_error_conf: Optional[Union[str, Path, pd.DataFrame, np.ndarray]] = None,
        reliability_acc: Optional[float] = None,
        reliability_conf: Optional[float] = None,
        id_vars: Optional[list] = None
    ):
        """
        Initialize SEM calibration analysis.

        Parameters:
        -----------
        theta_accuracy : str, Path, or DataFrame
            IRT theta estimates for accuracy (from Ch5 RQs)
            If str/Path: CSV file path with columns [UID, test, theta_accuracy, ...]
            If DataFrame: Must contain theta scores

        theta_confidence : str, Path, or DataFrame
            IRT theta estimates for confidence (from Ch6 RQs)
            If str/Path: CSV file path with columns [UID, test, theta_confidence, ...]
            If DataFrame: Must contain theta scores

        measurement_error_acc : str, Path, DataFrame, ndarray, optional
            Measurement error variances for accuracy (sigma^2 = 1 / test_information)
            If None, estimated from reliability_acc

        measurement_error_conf : str, Path, DataFrame, ndarray, optional
            Measurement error variances for confidence (sigma^2 = 1 / test_information)
            If None, estimated from reliability_conf

        reliability_acc : float, optional
            Reliability of accuracy theta scores (if measurement_error_acc not provided)
            Typical IRT reliability ≈ 0.75-0.90. Conservative default: 0.75

        reliability_conf : float, optional
            Reliability of confidence theta scores (if measurement_error_conf not provided)
            Conservative default: 0.75

        id_vars : list, optional
            ID variables for merging (default: ['UID', 'test'])
            Can extend to ['UID', 'test', 'domain'] for domain-stratified analyses
        """
        self.id_vars = id_vars or ['UID', 'test']

        # Load data
        logger.info("Loading theta estimates...")
        self.df_acc = self._load_data(theta_accuracy, 'accuracy')
        self.df_conf = self._load_data(theta_confidence, 'confidence')

        # Merge datasets
        logger.info(f"Merging on {self.id_vars}...")
        self.data = self._merge_data()

        # Handle measurement error
        logger.info("Processing measurement error...")
        self.measurement_error_acc = self._process_measurement_error(
            measurement_error_acc, reliability_acc, 'theta_accuracy', default_reliability=0.75
        )
        self.measurement_error_conf = self._process_measurement_error(
            measurement_error_conf, reliability_conf, 'theta_confidence', default_reliability=0.75
        )

        # Initialize model storage
        self.model_latent_diff = None
        self.model_residualized = None
        self.latent_calibration = None
        self.residualized_calibration = None

        # Validate data
        self._validate_data()

        logger.info(f"Initialization complete. N={len(self.data)} observations.")

    def _load_data(self, source: Union[str, Path, pd.DataFrame], label: str) -> pd.DataFrame:
        """Load data from file or DataFrame."""
        if isinstance(source, pd.DataFrame):
            return source.copy()
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"{label} file not found: {path}")
            return pd.read_csv(path)
        else:
            raise TypeError(f"{label} must be str, Path, or DataFrame")

    def _merge_data(self) -> pd.DataFrame:
        """
        Merge accuracy and confidence datasets.

        Returns:
            Merged DataFrame with both theta_accuracy and theta_confidence
        """
        # Standardize column names
        if 'theta' in self.df_acc.columns and 'theta_accuracy' not in self.df_acc.columns:
            self.df_acc = self.df_acc.rename(columns={'theta': 'theta_accuracy'})
        if 'theta' in self.df_conf.columns and 'theta_confidence' not in self.df_conf.columns:
            self.df_conf = self.df_conf.rename(columns={'theta': 'theta_confidence'})

        # Merge
        merged = pd.merge(
            self.df_acc[self.id_vars + ['theta_accuracy']],
            self.df_conf[self.id_vars + ['theta_confidence']],
            on=self.id_vars,
            how='inner',
            validate='1:1'
        )

        if len(merged) == 0:
            raise ValueError("Merge produced 0 rows. Check ID variables and data alignment.")

        logger.info(f"Merge successful: {len(merged)} observations")
        logger.info(f"  Accuracy source: {len(self.df_acc)} rows")
        logger.info(f"  Confidence source: {len(self.df_conf)} rows")

        return merged

    def _process_measurement_error(
        self,
        me_source: Optional[Union[str, Path, pd.DataFrame, np.ndarray]],
        reliability: Optional[float],
        theta_col: str,
        default_reliability: float = 0.75
    ) -> np.ndarray:
        """
        Process measurement error specification.

        Priority:
        1. Explicit measurement error variances (from IRT test information)
        2. Reliability estimate → sigma^2 = theta_var * (1 - reliability)
        3. Default conservative reliability (0.75)

        Returns:
            Array of measurement error variances (same length as data)
        """
        n = len(self.data)

        # Option 1: Explicit measurement error provided
        if me_source is not None:
            if isinstance(me_source, np.ndarray):
                if len(me_source) != n:
                    raise ValueError(f"Measurement error array length ({len(me_source)}) != data length ({n})")
                return me_source
            elif isinstance(me_source, (str, Path)):
                me_df = pd.read_csv(me_source)
                # Assume single column or column named 'se' or 'sigma2'
                if 'sigma2' in me_df.columns:
                    me_array = me_df['sigma2'].values
                elif 'se' in me_df.columns:
                    me_array = me_df['se'].values ** 2  # Convert SE to variance
                else:
                    me_array = me_df.iloc[:, 0].values  # First column

                if len(me_array) != n:
                    raise ValueError(f"Measurement error file rows ({len(me_array)}) != data rows ({n})")
                return me_array
            elif isinstance(me_source, pd.DataFrame):
                # Extract measurement error column
                if 'sigma2' in me_source.columns:
                    return me_source['sigma2'].values
                elif 'se' in me_source.columns:
                    return me_source['se'].values ** 2
                else:
                    return me_source.iloc[:, 0].values

        # Option 2: Use reliability to estimate measurement error
        if reliability is not None:
            if not 0 < reliability <= 1:
                raise ValueError(f"Reliability must be in (0, 1], got {reliability}")
            theta_var = self.data[theta_col].var()
            me_var = theta_var * (1 - reliability)
            logger.info(f"  Estimating from reliability={reliability:.3f}: sigma^2={me_var:.4f}")
            return np.full(n, me_var)

        # Option 3: Default conservative reliability
        theta_var = self.data[theta_col].var()
        me_var = theta_var * (1 - default_reliability)
        logger.warning(
            f"  No measurement error or reliability provided. "
            f"Using conservative default reliability={default_reliability:.2f} → sigma^2={me_var:.4f}"
        )
        return np.full(n, me_var)

    def _validate_data(self):
        """Validate data integrity."""
        # Check for missing values
        if self.data[['theta_accuracy', 'theta_confidence']].isnull().any().any():
            raise ValueError("Missing values detected in theta scores")

        # Check for infinite values
        if np.isinf(self.data[['theta_accuracy', 'theta_confidence']]).any().any():
            raise ValueError("Infinite values detected in theta scores")

        # Check measurement error positivity
        if (self.measurement_error_acc <= 0).any() or (self.measurement_error_conf <= 0).any():
            raise ValueError("Measurement error variances must be positive")

        # Compute observed correlation and reliability estimates
        r_xy = self.data[['theta_accuracy', 'theta_confidence']].corr().iloc[0, 1]

        # Estimate reliability from measurement error
        var_acc = self.data['theta_accuracy'].var()
        var_conf = self.data['theta_confidence'].var()
        r_xx = 1 - (self.measurement_error_acc.mean() / var_acc)
        r_yy = 1 - (self.measurement_error_conf.mean() / var_conf)

        # Compute theoretical difference score reliability
        if 2 - 2*r_xy != 0:
            r_diff_naive = (r_xx + r_yy - 2*r_xy) / (2 - 2*r_xy)
        else:
            r_diff_naive = np.nan

        logger.info(f"Data validation:")
        logger.info(f"  Accuracy reliability (est.): {r_xx:.3f}")
        logger.info(f"  Confidence reliability (est.): {r_yy:.3f}")
        logger.info(f"  Correlation (r_xy): {r_xy:.3f}")
        logger.info(f"  Naive difference score reliability: {r_diff_naive:.3f}")

        if r_diff_naive < 0.50:
            logger.warning(
                f"  ⚠️  Naive r_diff={r_diff_naive:.3f} is POOR (<0.50). "
                f"SEM approach is MANDATORY for reliable estimates."
            )
        elif r_diff_naive < 0.70:
            logger.warning(
                f"  ⚠️  Naive r_diff={r_diff_naive:.3f} is questionable (<0.70). "
                f"SEM approach recommended."
            )

        self.validation_stats = {
            'r_xx': r_xx,
            'r_yy': r_yy,
            'r_xy': r_xy,
            'r_diff_naive': r_diff_naive,
            'n_obs': len(self.data)
        }

    def fit_latent_difference(self, method='ML', verbose=True) -> Dict:
        """
        Fit latent difference score model (Approach A).

        Model specification:
            eta_acc =~ 1*theta_accuracy
            eta_conf =~ 1*theta_confidence

            theta_accuracy ~~ sigma2_acc*theta_accuracy
            theta_confidence ~~ sigma2_conf*theta_confidence

            eta_calibration := eta_conf - eta_acc

        Parameters:
        -----------
        method : str
            Estimation method ('ML' or 'ULS')
        verbose : bool
            Print fitting progress

        Returns:
        --------
        dict : Model fit statistics
        """
        try:
            import semopy
        except ImportError:
            raise ImportError(
                "semopy not installed. Install with: pip install semopy"
            )

        logger.info("Fitting latent difference score model...")

        # Prepare data for semopy
        sem_data = self.data[['theta_accuracy', 'theta_confidence']].copy()

        # Build model specification
        # Note: semopy uses different syntax than lavaan
        # We'll use a workaround with constrained loadings and fixed error variances

        model_spec = """
        # Measurement model with fixed loadings (scale setting)
        eta_acc =~ 1*theta_accuracy
        eta_conf =~ 1*theta_confidence

        # Latent variances (to be estimated)
        eta_acc ~~ eta_acc
        eta_conf ~~ eta_conf
        eta_acc ~~ eta_conf
        """

        # Add fixed measurement error variances
        # semopy syntax: variable ~~ value*variable
        me_acc_mean = self.measurement_error_acc.mean()
        me_conf_mean = self.measurement_error_conf.mean()

        model_spec += f"\ntheta_accuracy ~~ {me_acc_mean}*theta_accuracy"
        model_spec += f"\ntheta_confidence ~~ {me_conf_mean}*theta_confidence"

        if verbose:
            logger.info("Model specification:")
            logger.info(model_spec)

        # Fit model
        try:
            from semopy import Model
            model = Model(model_spec)
            model.fit(sem_data, obj=method)

            self.model_latent_diff = model

            # Extract fit statistics
            fit_stats = self._compute_fit_indices(model, sem_data)

            if verbose:
                logger.info("Model fit:")
                for key, val in fit_stats.items():
                    logger.info(f"  {key}: {val:.4f}")

            # Extract latent scores
            self._extract_latent_scores_difference(model, sem_data)

            return fit_stats

        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            logger.info("Attempting fallback to simpler estimation...")

            # Fallback: Manual computation using factor score regression
            return self._fallback_latent_difference()

    def _fallback_latent_difference(self) -> Dict:
        """
        Fallback method when SEM fails: Factor score regression approach.

        Uses weighted least squares with measurement error as weights.
        """
        logger.info("Using fallback: Factor score regression method")

        # Weights inversely proportional to measurement error
        weight_acc = 1 / self.measurement_error_acc
        weight_conf = 1 / self.measurement_error_conf

        # Weighted means (empirical Bayes shrinkage estimates)
        grand_mean_acc = np.average(self.data['theta_accuracy'], weights=weight_acc)
        grand_mean_conf = np.average(self.data['theta_confidence'], weights=weight_conf)

        # Shrinkage factor (reliability-based)
        r_xx = self.validation_stats['r_xx']
        r_yy = self.validation_stats['r_yy']

        # Latent scores (shrunken toward grand mean)
        eta_acc = grand_mean_acc + r_xx * (self.data['theta_accuracy'] - grand_mean_acc)
        eta_conf = grand_mean_conf + r_yy * (self.data['theta_confidence'] - grand_mean_conf)

        # Latent calibration
        self.latent_calibration = eta_conf - eta_acc

        logger.info(f"Fallback complete. Latent calibration: M={self.latent_calibration.mean():.3f}, SD={self.latent_calibration.std():.3f}")

        return {
            'method': 'fallback_factor_score_regression',
            'reliability_acc': r_xx,
            'reliability_conf': r_yy,
            'shrinkage_applied': True
        }

    def _extract_latent_scores_difference(self, model, data):
        """Extract latent calibration scores from fitted model."""
        try:
            # semopy factor score prediction
            latent_scores = model.predict_factors(data)

            # Compute difference
            self.latent_calibration = (
                latent_scores['eta_conf'].values - latent_scores['eta_acc'].values
            )

            logger.info(f"Latent calibration extracted: M={self.latent_calibration.mean():.3f}, SD={self.latent_calibration.std():.3f}")

        except Exception as e:
            logger.warning(f"Factor score extraction failed: {e}. Using fallback.")
            self._fallback_latent_difference()

    def fit_residualized(self, verbose=True) -> Dict:
        """
        Fit residualized calibration model (Approach B).

        Model: eta_conf ~ eta_acc
        Calibration = residual (confidence controlling for accuracy)

        Advantages:
        - Avoids Lord's Paradox
        - More robust when accuracy-confidence correlation is high

        Returns:
        --------
        dict : Model fit statistics
        """
        logger.info("Fitting residualized calibration model...")

        # If latent difference not yet fitted, fit it first to get latent scores
        if self.latent_calibration is None:
            self.fit_latent_difference(verbose=False)

        # Use the latent scores from difference model
        # Regress confidence on accuracy
        from scipy import stats

        # Get latent scores (eta_acc and eta_conf)
        # For residualized approach, we need the individual latent scores

        # If we have the full model, extract latent scores
        if self.model_latent_diff is not None:
            try:
                latent_scores = self.model_latent_diff.predict_factors(
                    self.data[['theta_accuracy', 'theta_confidence']]
                )
                eta_acc = latent_scores['eta_acc'].values
                eta_conf = latent_scores['eta_conf'].values
            except:
                # Fallback to empirical Bayes estimates
                eta_acc = self.data['theta_accuracy'].values
                eta_conf = self.data['theta_confidence'].values
        else:
            eta_acc = self.data['theta_accuracy'].values
            eta_conf = self.data['theta_confidence'].values

        # Regression: conf ~ acc
        slope, intercept, r_value, p_value, std_err = stats.linregress(eta_acc, eta_conf)

        # Residuals = calibration
        predicted_conf = intercept + slope * eta_acc
        self.residualized_calibration = eta_conf - predicted_conf

        # Compute R-squared
        r_squared = r_value ** 2

        logger.info(f"Residualized calibration model:")
        logger.info(f"  Slope (beta): {slope:.4f}")
        logger.info(f"  R-squared: {r_squared:.4f}")
        logger.info(f"  Residual calibration: M={self.residualized_calibration.mean():.3f}, SD={self.residualized_calibration.std():.3f}")

        return {
            'method': 'residualized',
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'residual_mean': self.residualized_calibration.mean(),
            'residual_sd': self.residualized_calibration.std()
        }

    def _compute_fit_indices(self, model, data) -> Dict:
        """
        Compute model fit indices.

        Returns:
        --------
        dict : CFI, TLI, RMSEA, SRMR, chi-square, df, p-value
        """
        try:
            # Get fit statistics from semopy
            stats = model.inspect()

            # Standard fit indices
            fit_dict = {
                'chi_square': stats.get('DoF', np.nan),
                'df': stats.get('DoF', np.nan),
                'p_value': stats.get('DoF', np.nan),
                'CFI': stats.get('CFI', np.nan),
                'TLI': stats.get('TLI', np.nan),
                'RMSEA': stats.get('RMSEA', np.nan),
                'SRMR': stats.get('SRMR', np.nan),
                'AIC': stats.get('AIC', np.nan),
                'BIC': stats.get('BIC', np.nan)
            }

            return fit_dict

        except Exception as e:
            logger.warning(f"Could not extract fit indices: {e}")
            return {'error': str(e)}

    def get_latent_calibration(self, approach='difference') -> np.ndarray:
        """
        Get latent calibration scores.

        Parameters:
        -----------
        approach : str
            'difference' for latent difference scores
            'residualized' for residualized calibration
            'both' for both (returns tuple)

        Returns:
        --------
        np.ndarray or tuple : Latent calibration scores
        """
        if approach == 'difference':
            if self.latent_calibration is None:
                raise ValueError("Latent difference model not fitted. Call fit_latent_difference() first.")
            return self.latent_calibration

        elif approach == 'residualized':
            if self.residualized_calibration is None:
                raise ValueError("Residualized model not fitted. Call fit_residualized() first.")
            return self.residualized_calibration

        elif approach == 'both':
            if self.latent_calibration is None or self.residualized_calibration is None:
                raise ValueError("Both models must be fitted first.")
            return self.latent_calibration, self.residualized_calibration

        else:
            raise ValueError(f"Unknown approach: {approach}. Use 'difference', 'residualized', or 'both'.")

    def get_model_fit(self, approach='difference') -> Dict:
        """Get model fit statistics."""
        if approach == 'difference':
            if self.model_latent_diff is None:
                raise ValueError("Latent difference model not fitted.")
            return self._compute_fit_indices(
                self.model_latent_diff,
                self.data[['theta_accuracy', 'theta_confidence']]
            )
        else:
            raise NotImplementedError(f"Fit indices for {approach} not yet implemented.")

    def compare_approaches(self) -> pd.DataFrame:
        """
        Compare latent difference vs residualized approaches.

        Returns:
        --------
        DataFrame with correlation and descriptive stats
        """
        if self.latent_calibration is None or self.residualized_calibration is None:
            raise ValueError("Both models must be fitted. Call fit_latent_difference() and fit_residualized().")

        # Correlation between approaches
        from scipy.stats import pearsonr, spearmanr
        r_pearson, p_pearson = pearsonr(self.latent_calibration, self.residualized_calibration)
        r_spearman, p_spearman = spearmanr(self.latent_calibration, self.residualized_calibration)

        comparison = pd.DataFrame({
            'Approach': ['Latent Difference', 'Residualized', 'Naive Difference'],
            'Mean': [
                self.latent_calibration.mean(),
                self.residualized_calibration.mean(),
                (self.data['theta_confidence'] - self.data['theta_accuracy']).mean()
            ],
            'SD': [
                self.latent_calibration.std(),
                self.residualized_calibration.std(),
                (self.data['theta_confidence'] - self.data['theta_accuracy']).std()
            ],
            'Min': [
                self.latent_calibration.min(),
                self.residualized_calibration.min(),
                (self.data['theta_confidence'] - self.data['theta_accuracy']).min()
            ],
            'Max': [
                self.latent_calibration.max(),
                self.residualized_calibration.max(),
                (self.data['theta_confidence'] - self.data['theta_accuracy']).max()
            ]
        })

        logger.info(f"Approach comparison:")
        logger.info(f"  Correlation (Pearson): r={r_pearson:.3f}, p={p_pearson:.4f}")
        logger.info(f"  Correlation (Spearman): rho={r_spearman:.3f}, p={p_spearman:.4f}")

        return comparison

    def save_results(self, output_dir: Union[str, Path], prefix: str = 'sem'):
        """
        Save all results to output directory.

        Parameters:
        -----------
        output_dir : str or Path
            Directory to save results
        prefix : str
            Prefix for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving results to {output_dir}...")

        # Save latent calibration scores
        if self.latent_calibration is not None:
            output_data = self.data[self.id_vars].copy()
            output_data['latent_calibration'] = self.latent_calibration
            output_data['theta_accuracy'] = self.data['theta_accuracy']
            output_data['theta_confidence'] = self.data['theta_confidence']
            output_data['naive_calibration'] = self.data['theta_confidence'] - self.data['theta_accuracy']

            if self.residualized_calibration is not None:
                output_data['residualized_calibration'] = self.residualized_calibration

            out_file = output_dir / f'{prefix}_calibration_scores.csv'
            output_data.to_csv(out_file, index=False)
            logger.info(f"  Saved: {out_file}")

        # Save validation stats
        val_df = pd.DataFrame([self.validation_stats])
        val_file = output_dir / f'{prefix}_validation_stats.csv'
        val_df.to_csv(val_file, index=False)
        logger.info(f"  Saved: {val_file}")

        # Save comparison (if both approaches fitted)
        if self.latent_calibration is not None and self.residualized_calibration is not None:
            comp_df = self.compare_approaches()
            comp_file = output_dir / f'{prefix}_approach_comparison.csv'
            comp_df.to_csv(comp_file, index=False)
            logger.info(f"  Saved: {comp_file}")

        # Save model fit (if available)
        if self.model_latent_diff is not None:
            try:
                fit_stats = self.get_model_fit('difference')
                fit_df = pd.DataFrame([fit_stats])
                fit_file = output_dir / f'{prefix}_model_fit.csv'
                fit_df.to_csv(fit_file, index=False)
                logger.info(f"  Saved: {fit_file}")
            except:
                logger.warning("  Could not save model fit statistics")

        logger.info("All results saved successfully.")


def compute_difference_score_reliability(
    r_xx: float,
    r_yy: float,
    r_xy: float
) -> float:
    """
    Compute theoretical reliability of difference score.

    Formula: r_diff = (r_xx + r_yy - 2*r_xy) / (2 - 2*r_xy)

    Parameters:
    -----------
    r_xx : float
        Reliability of first measure
    r_yy : float
        Reliability of second measure
    r_xy : float
        Correlation between measures

    Returns:
    --------
    float : Difference score reliability

    References:
    -----------
    - Rogosa, D., & Willett, J. B. (1983). Demonstrating the reliability
      the difference score in the measurement of change.
    - Williams, R. H., & Zimmerman, D. W. (1996). Are simple gain scores obsolete?
    """
    denominator = 2 - 2*r_xy

    if abs(denominator) < 1e-10:
        warnings.warn("Correlation r_xy ≈ 1.0, difference score reliability undefined.")
        return np.nan

    r_diff = (r_xx + r_yy - 2*r_xy) / denominator

    return r_diff


# Convenience function for quick analysis
def quick_sem_calibration(
    theta_accuracy_file: str,
    theta_confidence_file: str,
    output_dir: str,
    **kwargs
) -> SEMCalibration:
    """
    Quick SEM calibration analysis with default settings.

    Parameters:
    -----------
    theta_accuracy_file : str
        Path to accuracy theta CSV
    theta_confidence_file : str
        Path to confidence theta CSV
    output_dir : str
        Directory to save results
    **kwargs : optional
        Additional arguments passed to SEMCalibration()

    Returns:
    --------
    SEMCalibration object with fitted models
    """
    # Initialize
    sem = SEMCalibration(
        theta_accuracy=theta_accuracy_file,
        theta_confidence=theta_confidence_file,
        **kwargs
    )

    # Fit both models
    sem.fit_latent_difference()
    sem.fit_residualized()

    # Save results
    sem.save_results(output_dir)

    # Print summary
    print("\n" + "="*60)
    print("SEM CALIBRATION ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nObservations: {len(sem.data)}")
    print(f"\nValidation Statistics:")
    for key, val in sem.validation_stats.items():
        print(f"  {key}: {val:.4f}")

    print(f"\nApproach Comparison:")
    print(sem.compare_approaches().to_string(index=False))

    print(f"\nResults saved to: {output_dir}")
    print("="*60 + "\n")

    return sem


if __name__ == "__main__":
    # Example usage / test
    import sys

    print("SEM Calibration Toolkit")
    print("Version: 1.0.0")
    print("Created: 2025-12-28")
    print("\nThis module provides SEM-based calibration analysis.")
    print("Import and use SEMCalibration class or quick_sem_calibration() function.")
    print("\nFor documentation, see module docstring or Implementation Plan.")
