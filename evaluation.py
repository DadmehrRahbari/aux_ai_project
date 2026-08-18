# aux_ai_project/evaluation.py
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    f1_score,
    accuracy_score,
    confusion_matrix,
    jaccard_score,
)
import numpy as np
import params


def _optimal_f1_threshold(y_true, y_score):
    thresholds = np.linspace(0.0, 1.0, 200)
    best_f1    = -1
    best_thr   = params.ANOMALY_THRESHOLD
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1     = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_thr = t
    return best_thr, best_f1


def compute_metrics(y_true, y_score, threshold=None):
    metrics = {}
    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    if len(np.unique(y_true)) > 1:
        metrics["AUROC"]  = float(roc_auc_score(y_true, y_score))
        p, r, _           = precision_recall_curve(y_true, y_score)
        metrics["PR_AUC"] = float(auc(r, p))
    else:
        metrics["AUROC"]  = np.nan
        metrics["PR_AUC"] = np.nan

    if threshold is None:
        normal_cutoff  = int(0.3 * len(y_score))
        normal_scores  = y_score[:normal_cutoff]
        if len(normal_scores) > 0:
            threshold = float(np.percentile(normal_scores, 95))
        else:
            threshold, _ = _optimal_f1_threshold(y_true, y_score)

    y_pred      = (y_score >= threshold).astype(int)
    current_f1  = f1_score(y_true, y_pred, zero_division=0)
    cm          = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total       = len(y_true)

    metrics["CR (Certainty Rate)"]   = float((tn + fn) / total)
    metrics["UR (Uncertainty Rate)"] = float((tp + fp) / total)
    metrics["FCR (False Certainty)"] = float(fn / (tn + fn + 1e-9))
    metrics["RR (Redundant Referral)"] = float(fp / (tp + fp + 1e-9))
    metrics["Threshold"]             = float(threshold)
    metrics["F1"]                    = float(current_f1)
    metrics["Accuracy"]              = float(accuracy_score(y_true, y_pred))
    metrics["TPR"]                   = float(tp / (tp + fn + 1e-9))
    metrics["FPR"]                   = float(fp / (fp + tn + 1e-9))
    metrics["Detection_Error"]       = float(0.5 * (metrics["FPR"] + (1 - metrics["TPR"])))
    metrics["Num_Positive"]          = int(np.sum(y_pred))
    metrics["Num_Negative"]          = int(len(y_pred) - np.sum(y_pred))

    return metrics


def compute_ece(y_true, y_prob, n_bins=8):
    """
    Expected Calibration Error using quantile binning.

    Quantile bins (equal-frequency) are used instead of uniform bins because
    the proposed score distribution is concentrated in [0.33, 0.95] and does
    not cover [0,1] uniformly.  Uniform bins leave 3 bins empty and give a
    single 2-frame bin that dominates the ECE calculation, inflating it to
    ~0.47.  Quantile bins ensure every bin has the same statistical weight.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Build quantile bin edges
    quantiles   = np.linspace(0, 100, n_bins + 1)
    bin_edges   = np.percentile(y_prob, quantiles)
    bin_edges   = np.unique(bin_edges)          # deduplicate if scores are flat
    if len(bin_edges) < 2:
        return 0.0

    ece = 0.0
    n   = len(y_prob)
    for i in range(len(bin_edges) - 1):
        if i == 0:
            in_bin = (y_prob >= bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        else:
            in_bin = (y_prob >  bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        n_bin = in_bin.sum()
        if n_bin == 0:
            continue
        avg_conf     = float(y_prob[in_bin].mean())
        # anomaly fraction in bin (y_true=1 means anomaly)
        avg_accuracy = float(y_true[in_bin].mean())
        ece         += (n_bin / n) * abs(avg_conf - avg_accuracy)

    return float(ece)


def calculate_miou(pred_mask, gt_mask):
    """Simplified mIoU over the road class."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union        = np.logical_or(pred_mask,  gt_mask).sum()
    if union == 0:
        return 1.0
    return intersection / union
