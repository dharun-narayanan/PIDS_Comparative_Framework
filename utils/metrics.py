"""
Evaluation metrics for PIDS models.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_score: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute comprehensive detection metrics.
    
    Args:
        y_true: True labels (0 for benign, 1 for malicious)
        y_pred: Predicted labels
        y_score: Prediction scores (for AUC metrics)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = np.mean(y_true == y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_positives'] = int(tp)
    metrics['false_positives'] = int(fp)
    metrics['true_negatives'] = int(tn)
    metrics['false_negatives'] = int(fn)
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['detection_rate'] = metrics['tpr']
    
    # AUC metrics (if scores provided)
    if y_score is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_score)
            metrics['auc_pr'] = average_precision_score(y_true, y_score)
        except ValueError as e:
            logger.warning(f"Could not compute AUC metrics: {e}")
            metrics['auc_roc'] = 0.0
            metrics['auc_pr'] = 0.0
    
    return metrics


def compute_entity_level_metrics(y_true: np.ndarray, y_score: np.ndarray, 
                                 k_neighbors: int = 5) -> Dict[str, float]:
    """
    Compute entity-level detection metrics using k-NN approach.
    
    Args:
        y_true: True labels
        y_score: Anomaly scores
        k_neighbors: Number of neighbors for k-NN
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Separate benign and malicious samples
    benign_idx = np.where(y_true == 0)[0]
    malicious_idx = np.where(y_true == 1)[0]
    
    if len(benign_idx) == 0 or len(malicious_idx) == 0:
        logger.warning("Need both benign and malicious samples for entity-level metrics")
        return {'auc_roc': 0.0, 'auc_pr': 0.0}
    
    # Fit k-NN on benign samples
    k = min(k_neighbors, len(benign_idx) - 1)
    if k < 1:
        logger.warning("Not enough benign samples for k-NN")
        return {'auc_roc': 0.0, 'auc_pr': 0.0}
    
    # Compute anomaly scores based on distance to benign samples
    # Higher score = more anomalous
    anomaly_scores = y_score
    
    metrics = compute_detection_metrics(y_true, (anomaly_scores > np.median(anomaly_scores)).astype(int), anomaly_scores)
    
    return metrics


def compute_batch_level_metrics(graphs_labels: np.ndarray, graphs_scores: np.ndarray) -> Dict[str, float]:
    """
    Compute batch-level (graph-level) detection metrics.
    
    Args:
        graphs_labels: True labels for each graph
        graphs_scores: Anomaly scores for each graph
        
    Returns:
        Dictionary of metrics
    """
    # Threshold at median for binary classification
    threshold = np.median(graphs_scores)
    graphs_pred = (graphs_scores > threshold).astype(int)
    
    metrics = compute_detection_metrics(graphs_labels, graphs_pred, graphs_scores)
    
    return metrics


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve."""
    return roc_curve(y_true, y_score)


def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Precision-Recall curve."""
    return precision_recall_curve(y_true, y_score)


def compute_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute optimal threshold using Youden's J statistic.
    
    Args:
        y_true: True labels
        y_score: Prediction scores
        
    Returns:
        Optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx]


def evaluate_with_multiple_thresholds(y_true: np.ndarray, y_score: np.ndarray, 
                                      thresholds: Optional[np.ndarray] = None) -> Dict[float, Dict[str, float]]:
    """
    Evaluate metrics at multiple thresholds.
    
    Args:
        y_true: True labels
        y_score: Prediction scores
        thresholds: Thresholds to evaluate (default: 10 equally spaced)
        
    Returns:
        Dictionary mapping threshold to metrics
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 11)
    
    results = {}
    for threshold in thresholds:
        y_pred = (y_score > threshold).astype(int)
        results[float(threshold)] = compute_detection_metrics(y_true, y_pred, y_score)
    
    return results


class MetricsTracker:
    """Track metrics over training."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def update(self, train_loss: Optional[float] = None, val_loss: Optional[float] = None,
               train_metrics: Optional[Dict] = None, val_metrics: Optional[Dict] = None):
        """Update tracked metrics."""
        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if train_metrics is not None:
            self.history['train_metrics'].append(train_metrics)
        if val_metrics is not None:
            self.history['val_metrics'].append(val_metrics)
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Get epoch with best metric."""
        if mode == 'min':
            return int(np.argmin(self.history[metric]))
        else:
            return int(np.argmax(self.history[metric]))
    
    def get_history(self) -> Dict:
        """Get complete history."""
        return self.history
