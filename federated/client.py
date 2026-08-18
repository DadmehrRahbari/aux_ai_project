# aux_ai_project/federated/client.py
import torch


class FLClient:
    """
    Federated Learning client with trust-gated update control and real
    local training. Updates from sensor_failure or spoofing decisions are
    dropped entirely. Uncertain or novel updates are scaled down before
    transmission.
    """

    def __init__(self, model):
        self.model     = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = torch.nn.BCELoss()
        # BCELoss is correct: AuxAIMonitor.ood_head ends in nn.Sigmoid()
        # so output is already in [0,1]. BCEWithLogitsLoss applies its own
        # internal sigmoid, double-sigmoiding the output and distorting every
        # gradient. This was the dominant cause of global AUROC falling below
        # local-only AUROC once real training started.

    def local_train(self, feature_label_buffer, epochs=3, max_grad_norm=1.0):
        """
        Real supervised local training step on this client's accumulated
        (feature_vector, is_anomaly_label) buffer since the last FL round.
        Trains on the OOD head output which is sigmoid-bounded [0,1].
        Includes gradient clipping and NaN/Inf guard with rollback.
        """
        if not feature_label_buffer:
            return

        # Guard: skip if incoming weights are already non-finite
        if not all(torch.isfinite(v).all() for v in self.model.state_dict().values()):
            print("[FLClient] Skipping local_train(): incoming weights already non-finite.")
            return

        x = torch.stack([f for f, _ in feature_label_buffer])
        y = torch.tensor([l for _, l in feature_label_buffer]).float().unsqueeze(1)

        last_good_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        self.model.train()
        for _ in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(x)
            # Guard: forward pass can produce NaN even with finite weights
            if not torch.isfinite(out["ood"]).all():
                print("[FLClient] Forward pass produced non-finite output -- skipping epoch.")
                break
            out_ood = torch.clamp(out["ood"], 1e-7, 1 - 1e-7)
            loss = self.criterion(out_ood, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            # Clamp weights to prevent magnitude drift between rounds
            with torch.no_grad():
                for p in self.model.parameters():
                    p.clamp_(-5.0, 5.0)
            # Rollback if weights became non-finite this step
            if not all(torch.isfinite(v).all() for v in self.model.state_dict().values()):
                self.model.load_state_dict(last_good_state)
                break
            last_good_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.model.eval()

    def get_update(self, global_model_state, decision=None):
        # Drop poisoned / hardware-failed updates
        if decision in ["sensor_failure", "spoofing"]:
            return None

        # Scale factor by trust level
        if decision == "uncertain":
            scale = 0.25
        elif decision == "novel":
            scale = 0.75
        else:
            scale = 1.0

        update = {}
        for k, v in self.model.state_dict().items():
            delta     = v - global_model_state[k]
            update[k] = scale * delta

        return update

    def apply_update(self, update):
        """
        Apply a delta to this client's model.

        IMPORTANT: only call this with an update that was NOT already trained
        into self.model. get_update() computes `delta = self.model - global_state`
        AFTER local_train() has already moved self.model by that delta. If the
        same client then calls apply_update(that_delta), the delta gets added
        a second time (self.model becomes global + 2*delta, or global +
        (1+trust)*delta when scaled), silently doubling the effective step size
        every round. For the "client trains, server aggregates, client resumes
        next round" pattern used in run_simulation.py, sync the client to the
        server's post-aggregation state instead:
            server.aggregate([update], weights=[trust])
            client.model.load_state_dict(server.global_model.state_dict())
        This method remains useful for genuinely pushing a foreign update (e.g.
        broadcasting the aggregated global delta to a client that did NOT
        produce it) to a client.
        """
        if update is None:
            return
        state = self.model.state_dict()
        prospective = {k: state[k] + update[k] for k in state}
        # Refuse update if result would be non-finite
        if not all(torch.isfinite(v).all() for v in prospective.values()):
            print("[FLClient] Refusing apply_update(): result would be non-finite.")
            return
        # Clamp to same bound as server and local_train
        prospective = {k: torch.clamp(v, -5.0, 5.0) for k, v in prospective.items()}
        self.model.load_state_dict(prospective)
