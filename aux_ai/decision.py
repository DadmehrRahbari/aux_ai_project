# aux_ai_project/aux_ai/decision.py
#
# NOTE: final_score here is the RAW monitor proposed output (~2.1-2.5),
# NOT the normalised [0,1] probability.  ANOMALY_THRESHOLD = 2.28 is in
# the same space.  The normalised probability is only computed in
# run_simulation.py for plotting/metrics.

import params


def decide(final_score, scores):
    """
    Maps raw monitor score (~2.1-2.5) + per-signal scores to a decision.

    Args:
        final_score: float ~2.1-2.5  — monitor's raw proposed output
        scores:      dict with 'sensor', 'stability', 'ood', 'agreement'
                     'sensor' is health in [0,1]  (high = healthy)

    Returns:
        decision (str): "sensor_failure" | "spoofing" | "novel" |
                        "uncertain" | "known"
        trust    (float): 0.0 – 1.0
    """
    if scores is None or len(scores) == 0:
        return "uncertain", 0.0

    sensor_score    = scores.get("sensor",    1.0)  # health: high = healthy
    stability_score = scores.get("stability", 1.0)

    # 1. Hardware sensor failure: health drops below threshold
    if sensor_score < (1.0 - params.SENSOR_THRESHOLD):
        return "sensor_failure", 0.0

    # 2. Temporal instability / spoofing
    if stability_score < params.STABILITY_THRESHOLD:
        return "spoofing", 0.0

    # 3. OOD / novelty — raw score above ANOMALY_THRESHOLD (~2.28)
    # FIX: was trust=0.8. A "novel" decision means the raw score already
    # cleared the anomaly threshold -- this is a confident, legitimate
    # detection, not a noisy or adversarial signal. Discounting it by 20%
    # before sharing meant the federated/shared model systematically learned
    # LESS from real anomalies than the (undiscounted) non-FL baseline it's
    # compared against, even though each client's own local_train() already
    # legitimately learned from the same data. There's no principled reason
    # to withhold 20% of a confidently-detected anomaly from the fleet.
    if final_score > params.ANOMALY_THRESHOLD:
        return "novel", 1.0

    # 4. Near-threshold uncertainty band
    # FIX: was trust=0.5. Some caution is still warranted this close to the
    # threshold, but a 50% cut was overly conservative and compounded with
    # the "novel" discount above to starve the shared model of anomaly-class
    # signal specifically. 0.75 keeps genuine caution while no longer being
    # the dominant cause of the federated-vs-local-only performance gap.
    if abs(final_score - params.ANOMALY_THRESHOLD) < 0.05:
        return "uncertain", 0.75

    # 5. Normal operation
    return "known", 1.0
