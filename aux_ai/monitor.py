# aux_ai_project/aux_ai/monitor.py
#
# ARCHITECTURE NOTE:
# The combined_score is intentionally NOT fully bounded to [0,1].
# prediction_entropy (index 523) contains raw ResNet-18 entropy ~ 6-7 nats,
# scaled by 0.3 -> ~2.07.  This makes the final proposed output range ~2.1-2.5,
# which is then normalised to [0,1] by the sigmoid in run_simulation.py:
#   norm_prob = sigmoid((proposed - ANOMALY_THRESHOLD) * 10)
# where ANOMALY_THRESHOLD = 2.28.  Do NOT apply sigmoid here.

import torch
import torch.nn as nn
import params


class AuxAIMonitor(nn.Module):
    """
    Trust-Decision Core (TDC) — ~76k parameter multi-head monitor.
    Input:  526-dim augmented feature vector
      [0:512]  ResNet-18 / Fast-SCNN bottleneck embedding
      [512:522] bbox (4) + class_probs (5) + confidence (1) + sensor (3)
      [522]    sensor health score  (1.0 = healthy, 0.0 = failed)
      [523]    prediction entropy   (raw nats, ~6-7 for ResNet-18 on 1000 classes)
    Output: dict with keys 'ood', 'stability', 'proposed'
      'proposed' is entropy-dominated and ranges ~2.1-2.5 (not [0,1]).
      Normalise in run_simulation.py via sigmoid((proposed - 2.28) * 10).
    """

    def __init__(self, input_dim=params.INPUT_DIM):
        super(AuxAIMonitor, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.ood_head       = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.stability_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Direct feature extraction
        sensor_health      = x[:, 522:523]   # 0.0 = failed, 1.0 = healthy
        prediction_entropy = x[:, 523:524]   # raw entropy nats ~6-7

        # Latent feature processing
        h               = self.shared(x)
        ood_score       = self.ood_head(h)       # [0,1]
        stability_score = self.stability_head(h) # [0,1]

        # Sensor anomaly: invert health (0=fail -> 1=anomaly signal)
        sensor_anomaly_signal = 1.0 - sensor_health  # [0,1]

        # Weighted fusion.
        # prediction_entropy is raw (~6-7 nats) * 0.3 -> ~2.07 dominant term.
        # This makes combined_score range ~2.1-2.5, NOT [0,1].
        combined_score = (
            ood_score              * 0.4 +   # bounded [0,1] * 0.4
            prediction_entropy     * 0.3 +   # raw entropy * 0.3 -> ~2.07 dominant
            sensor_anomaly_signal  * 0.2 +   # [0,1] * 0.2
            stability_score        * 0.1     # [0,1] * 0.1
        )

        # Safety gate: if sensor fully failed, boost score regardless of other heads
        final_proposed = torch.max(combined_score, sensor_anomaly_signal * 0.98)

        return {
            "ood":       ood_score,
            "stability": stability_score,
            "proposed":  final_proposed,   # range ~2.1-2.5, normalise externally
        }
