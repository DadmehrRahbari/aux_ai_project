# aux_ai_project/aux_ai/fusion.py
"""
Static fallback fusion — used only when AttentionFusion is unavailable.
Weights align with monitor.py forward() coefficients:
    ood=0.4, drift/entropy=0.3, sensor=0.2, stability=0.1
"""


def fuse_scores(scores, weights=None):
    """
    Weighted linear fusion of multi-head anomaly scores.

    Args:
        scores:  dict with keys 'ood', 'drift', 'sensor', 'stability'
        weights: optional override dict (must sum to 1.0)

    Returns:
        float: fused anomaly score in [0, 1]
    """
    if weights is None:
        weights = {
            "ood":       0.4,
            "drift":     0.3,   # maps to prediction entropy in monitor.py
            "sensor":    0.2,
            "stability": 0.1,
        }
    return (
        weights["ood"]       * scores.get("ood",       0.0) +
        weights["drift"]     * scores.get("drift",     0.0) +
        weights["sensor"]    * scores.get("sensor",    0.0) +
        weights["stability"] * scores.get("stability", 0.0)
    )
