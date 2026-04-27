"""Clinical classification and diagnostic metrics."""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_curve, auc, confusion_matrix
from typing import Optional, Dict, Union, Tuple
import warnings


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    y_scores: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    return_confusion_matrix: bool = False
) -> Dict:
    """
    Compute sensitivity, specificity, PPV, NPV, and accuracy.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray, optional
        Predicted binary labels. If None, computed from y_scores
    y_scores : np.ndarray, optional
        Probability scores (if y_pred not provided)
    threshold : float
        Threshold for converting scores to predictions
    return_confusion_matrix : bool
        Include confusion matrix components
    
    Returns
    -------
    Dict containing:
        - sensitivity (recall, TPR): TP / (TP + FN)
        - specificity (TNR): TN / (TN + FP)
        - ppv (precision): TP / (TP + FP)
        - npv: TN / (TN + FN)
        - accuracy: (TP + TN) / total
        - balanced_accuracy: (sensitivity + specificity) / 2
        - confusion_matrix, tp, tn, fp, fn (if requested)
    """
    # Convert scores to predictions if needed
    if y_pred is None and y_scores is not None:
        y_pred = (y_scores >= threshold).astype(int)
    elif y_pred is None:
        raise ValueError("Either y_pred or y_scores must be provided")
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics (with zero division handling)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    balanced_accuracy = (sensitivity + specificity) / 2
    
    result = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy
    }
    
    if return_confusion_matrix:
        result.update({
            'confusion_matrix': np.array([[tn, fp], [fn, tp]]),
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })
    
    return result


def compute_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: Optional[int] = None
) -> Dict:
    """
    Compute ROC curve and AUC with optional bootstrap confidence intervals.
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Probability scores
    bootstrap_ci : bool
        Calculate bootstrap confidence intervals
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
    seed : int, optional
        Random seed
    
    Returns
    -------
    Dict containing:
        - auc: Area under the ROC curve
        - fpr: False positive rates
        - tpr: True positive rates
        - thresholds: Decision thresholds
        - auc_ci_lower, auc_ci_upper (if bootstrap_ci=True)
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    result = {
        'auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }
    
    # Bootstrap confidence intervals
    if bootstrap_ci:
        if seed is not None:
            np.random.seed(seed)
        
        n = len(y_true)
        bootstrap_aucs = []
        
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            y_true_boot = y_true[idx]
            y_scores_boot = y_scores[idx]
            
            try:
                fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_scores_boot)
                auc_boot = auc(fpr_boot, tpr_boot)
                bootstrap_aucs.append(auc_boot)
            except:
                # Skip if bootstrap sample has only one class
                continue
        
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_aucs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))
        
        result['auc_ci_lower'] = ci_lower
        result['auc_ci_upper'] = ci_upper
    
    return result


def compute_diagnostic_odds_ratio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: float = 0.95
) -> Dict:
    """
    Compute diagnostic odds ratio with confidence intervals.
    
    DOR = (TP × TN) / (FP × FN)
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels
    confidence : float
        Confidence level for intervals
    
    Returns
    -------
    Dict containing:
        - dor: Diagnostic odds ratio
        - log_dor: Natural log of DOR
        - ci_lower: Lower confidence bound
        - ci_upper: Upper confidence bound
    """
    # Get confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Add 0.5 to avoid division by zero (Haldane correction)
    tp_adj = tp + 0.5
    tn_adj = tn + 0.5
    fp_adj = fp + 0.5
    fn_adj = fn + 0.5
    
    # Calculate DOR
    dor = (tp_adj * tn_adj) / (fp_adj * fn_adj)
    log_dor = np.log(dor)
    
    # Calculate standard error of log(DOR)
    se_log_dor = np.sqrt(1/tp_adj + 1/tn_adj + 1/fp_adj + 1/fn_adj)
    
    # Calculate confidence intervals
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    log_ci_lower = log_dor - z * se_log_dor
    log_ci_upper = log_dor + z * se_log_dor
    
    ci_lower = np.exp(log_ci_lower)
    ci_upper = np.exp(log_ci_upper)
    
    return {
        'dor': dor,
        'log_dor': log_dor,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def compute_youden_index(
    y_true: np.ndarray,
    y_scores: np.ndarray
) -> Dict:
    """
    Compute Youden's J statistic and optimal threshold.
    
    J = sensitivity + specificity - 1
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_scores : np.ndarray
        Probability scores
    
    Returns
    -------
    Dict containing:
        - youden_j: Maximum Youden index
        - optimal_threshold: Threshold that maximizes J
        - sensitivity_at_threshold: Sensitivity at optimal threshold
        - specificity_at_threshold: Specificity at optimal threshold
    """
    # Get ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Calculate Youden index for each threshold
    youden_j = tpr - fpr  # equivalent to tpr + (1-fpr) - 1
    
    # Find optimal threshold
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    max_youden_j = youden_j[optimal_idx]
    
    # Get sensitivity and specificity at optimal threshold
    sensitivity_optimal = tpr[optimal_idx]
    specificity_optimal = 1 - fpr[optimal_idx]
    
    return {
        'youden_j': max_youden_j,
        'optimal_threshold': optimal_threshold,
        'sensitivity_at_threshold': sensitivity_optimal,
        'specificity_at_threshold': specificity_optimal
    }


def compute_likelihood_ratios(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: float = 0.95,
    include_interpretation: bool = False
) -> Dict:
    """
    Compute positive and negative likelihood ratios.
    
    LR+ = sensitivity / (1 - specificity)
    LR- = (1 - sensitivity) / specificity
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels
    y_pred : np.ndarray
        Predicted binary labels
    confidence : float
        Confidence level for intervals
    include_interpretation : bool
        Include clinical interpretation
    
    Returns
    -------
    Dict containing:
        - lr_positive: Positive likelihood ratio
        - lr_negative: Negative likelihood ratio
        - lr_positive_ci: CI for LR+
        - lr_negative_ci: CI for LR-
        - lr_positive_interpretation (if requested)
        - lr_negative_interpretation (if requested)
    """
    # Calculate metrics
    metrics = compute_sensitivity_specificity(y_true, y_pred, 
                                               return_confusion_matrix=True)
    
    sens = metrics['sensitivity']
    spec = metrics['specificity']
    tp = metrics['tp']
    tn = metrics['tn']
    fp = metrics['fp']
    fn = metrics['fn']
    
    # Calculate likelihood ratios (with small constant to avoid division by zero)
    epsilon = 1e-10
    lr_positive = sens / (1 - spec + epsilon)
    lr_negative = (1 - sens) / (spec + epsilon)
    
    # Calculate confidence intervals using log transformation
    # Standard errors for log(LR)
    se_log_lr_pos = np.sqrt(1/tp - 1/(tp+fn) + 1/fp - 1/(fp+tn))
    se_log_lr_neg = np.sqrt(1/fn - 1/(tp+fn) + 1/tn - 1/(fp+tn))
    
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    # CI for LR+
    log_lr_pos = np.log(lr_positive) if lr_positive > 0 else -10
    lr_pos_ci_lower = np.exp(log_lr_pos - z * se_log_lr_pos)
    lr_pos_ci_upper = np.exp(log_lr_pos + z * se_log_lr_pos)
    
    # CI for LR-
    log_lr_neg = np.log(lr_negative) if lr_negative > 0 else -10
    lr_neg_ci_lower = np.exp(log_lr_neg - z * se_log_lr_neg)
    lr_neg_ci_upper = np.exp(log_lr_neg + z * se_log_lr_neg)
    
    result = {
        'lr_positive': lr_positive,
        'lr_negative': lr_negative,
        'lr_positive_ci': (lr_pos_ci_lower, lr_pos_ci_upper),
        'lr_negative_ci': (lr_neg_ci_lower, lr_neg_ci_upper)
    }
    
    # Add interpretation if requested
    if include_interpretation:
        # LR+ interpretation
        if lr_positive > 10:
            lr_pos_interp = "Strong positive"
        elif lr_positive > 5:
            lr_pos_interp = "Moderate positive"
        elif lr_positive > 2:
            lr_pos_interp = "Weak positive"
        else:
            lr_pos_interp = "Negligible"
        
        # LR- interpretation
        if lr_negative < 0.1:
            lr_neg_interp = "Strong negative"
        elif lr_negative < 0.2:
            lr_neg_interp = "Moderate negative"
        elif lr_negative < 0.5:
            lr_neg_interp = "Weak negative"
        else:
            lr_neg_interp = "Negligible"
        
        result['lr_positive_interpretation'] = lr_pos_interp
        result['lr_negative_interpretation'] = lr_neg_interp
    
    return result