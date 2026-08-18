# aux_ai_project/federated/server.py
import torch


class FLServer:
    """
    Federated Learning server with norm-clipping, NaN-guard, and
    trust-weighted aggregation.
    """

    def __init__(self, global_model, max_update_norm=5.0):
        self.global_model    = global_model
        self.max_update_norm = max_update_norm

    def validate_update(self, update):
        """Return True only if update is non-None, finite, and within norm limit."""
        if update is None:
            return False
        for v in update.values():
            if not torch.isfinite(v).all():   # NaN / Inf guard
                return False
            if torch.norm(v.float()) > self.max_update_norm:
                return False
        return True

    def aggregate(self, updates, weights=None):
        if not updates:
            return

        if weights is None:
            weights = [1.0] * len(updates)

        # Filter invalid updates
        valid_pairs = [
            (u, w) for u, w in zip(updates, weights)
            if self.validate_update(u)
        ]
        if not valid_pairs:
            return

        valid_updates, valid_weights = zip(*valid_pairs)

        # FIX: single-update case must use weight directly as blend strength,
        # not renormalize to 1.0 — renormalization silently discards reliability
        # weighting: a low-reliability client's update was applied at exactly the
        # same full strength as a fully-trusted one. For a single-element list,
        # sum([w]) / sum([w]) = 1.0 always, ignoring the original weight entirely.
        if len(valid_updates) == 1:
            avg_update = {k: valid_weights[0] * v for k, v in valid_updates[0].items()}
        else:
            weight_sum = sum(valid_weights)
            if weight_sum == 0:
                return
            norm_weights = [w / weight_sum for w in valid_weights]
            avg_update = {}
            for k in self.global_model.state_dict().keys():
                avg_update[k] = sum(
                    norm_weights[i] * valid_updates[i][k]
                    for i in range(len(valid_updates))
                )

        state = self.global_model.state_dict()
        for k in state:
            new_val = state[k] + avg_update[k]
            # Tightened from +/-50.0 to +/-5.0: the looser bound allowed
            # cumulative additive drift over many rounds to reach magnitudes
            # that produced NaN/Inf in client forward passes.
            state[k] = torch.clamp(new_val, -5.0, 5.0)
        self.global_model.load_state_dict(state)
