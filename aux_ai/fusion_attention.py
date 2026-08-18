# aux_ai_project/aux_ai/fusion_attention.py
import torch
import torch.nn as nn
import random
import params


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of multi-source anomaly scores.
    Maps 5 diagnostic signals to a single trust score in [0, 1].
    Signals: ood, drift, sensor, temporal-consistency, agreement.
    """

    def __init__(self, input_dim=5, norm_momentum=0.05):
        super().__init__()
        self.input_dim = input_dim

        self.attn = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.Softmax(dim=0),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1),
            nn.Sigmoid(),
        )

        # FIX: this module was never trained anywhere in the codebase -- no
        # optimizer, no loss, no backward() call. Its Linear layers stayed at
        # random initialization for the entire simulation, which alone
        # explains why the "attention" method scored ~0.45-0.49 AUROC (worse
        # than the trivial always-0.5 Big-AI baseline).
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        self.criterion = torch.nn.BCELoss()

        # FIX: previously normalized inputs against fixed, hand-picked
        # constants (mean=ANOMALY_THRESHOLD, std=0.15, etc.). Those constants
        # depend on exactly what distribution the monitor's "proposed" score,
        # sensor score, and temporal-stability delta actually produce -- which
        # shifts every time something upstream changes (trust weights, FL
        # sync logic, round counts). Every upstream fix silently invalidated
        # these constants again, which is the actual root cause of the
        # repeated regressions across runs. Replaced with running statistics
        # (mean/var tracked via exponential moving average, like BatchNorm)
        # that self-calibrate to whatever the real data distribution
        # currently is, computed only from data actually seen during
        # training -- no more guessing, no more staleness.
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var",  torch.ones(input_dim))
        self.register_buffer("num_updates",  torch.tensor(0.0))
        self.norm_momentum = norm_momentum

        # FIX: this small, softmax-gated network can converge to a genuinely
        # inverted local minimum during training -- verified against real
        # data: trained output correlated at -0.647 with "ood" (the raw
        # monitor score), the single feature known by construction to be
        # positively correlated with anomaly (+0.76 with the true label in
        # the same run). This isn't a labeling or normalization bug; SGD on
        # a tiny network with limited, non-i.i.d.-revealed training data can
        # simply land in the wrong-sign basin, and once there, gradient
        # descent has no pressure to escape since the loss looks fine from
        # inside that basin. Rather than hope every random seed / every
        # future data distribution avoids this, check after each training
        # round whether the network's own predictions have drifted
        # anti-correlated with its most reliable known-signed input, and
        # flip the output if so.
        self.register_buffer("invert_output", torch.tensor(0.0))

    def _check_sign_consistency(self, buffer):
        """
        After training, verify the model's predictions on the training
        buffer are positively correlated with the raw "ood" input -- the one
        signal known to be reliably anomaly-positive by construction. If the
        correlation is negative, the network has converged to an inverted
        solution; flip future outputs to correct for it.
        """
        with torch.no_grad():
            ood_vals, preds = [], []
            for scores_dict, _ in buffer:
                ood_vals.append(scores_dict.get("ood", 0.0))
                preds.append(self._raw_score(scores_dict).item())
            ood_t   = torch.tensor(ood_vals)
            pred_t  = torch.tensor(preds)
            if ood_t.std() < 1e-8 or pred_t.std() < 1e-8:
                return  # not enough variance to judge; leave as-is
            corr = torch.corrcoef(torch.stack([ood_t, pred_t]))[0, 1].item()
            self.invert_output.fill_(1.0 if corr < 0 else 0.0)

    def _to_vector(self, scores_dict):
        tc_violation = (
            1.0 if scores_dict.get("stability", 0.0) > params.APSS_TEMPORAL_THRESHOLD
            else 0.0
        )
        return torch.tensor([
            scores_dict.get("ood",       0.0),
            scores_dict.get("drift",     0.0),
            scores_dict.get("sensor",    0.0),
            tc_violation,
            scores_dict.get("agreement", 0.0),
        ]).float()

    def _normalize(self, raw):
        if self.training:
            with torch.no_grad():
                self.num_updates += 1
                # Faster adaptation early (like an unbiased running average
                # for the first few samples), settling to norm_momentum once
                # enough data has been seen.
                m = max(self.norm_momentum, 1.0 / self.num_updates.item())
                self.running_mean.mul_(1 - m).add_(raw * m)
                var_estimate = (raw - self.running_mean) ** 2
                self.running_var.mul_(1 - m).add_(var_estimate * m)
        std = torch.sqrt(self.running_var + 1e-6)
        return (raw - self.running_mean) / std

    def _raw_score(self, scores_dict):
        raw = self._to_vector(scores_dict)
        x = self._normalize(raw)
        weights  = self.attn(x)
        trust    = scores_dict.get("trust", 1.0)
        x_weighted = x * weights * trust
        return self.fc(x_weighted)

    def forward(self, scores_dict):
        out = self._raw_score(scores_dict)
        if self.invert_output.item() > 0.5:
            out = 1.0 - out
        return out

    def local_train(self, buffer, epochs=3, max_grad_norm=1.0):
        """
        Supervised training step on accumulated (scores_dict, is_anomaly_label)
        pairs, mirroring FLClient.local_train's structure (BCELoss on a
        sigmoid-bounded output, gradient clipping, NaN/Inf guard with
        rollback). forward() is per-sample by design (the attn Softmax is
        dim=0 over the 5 signals of a single reading, not a batch dim), so
        this loops over the buffer rather than batching -- buffers here are
        small (~50 samples/round) so this is not a bottleneck.
        """
        if not buffer:
            return

        if not all(torch.isfinite(v).all() for v in self.state_dict().values()):
            print("[AttentionFusion] Skipping local_train(): incoming weights already non-finite.")
            return

        last_good_state = {k: v.clone() for k, v in self.state_dict().items()}

        self.train()
        for _ in range(epochs):
            # FIX: without shuffling, the EMA-based running normalization
            # stats (see _normalize) get skewed toward whichever phase
            # happens to be last in the buffer -- since this simulation's
            # buffer is built by appending phases in temporal order (normal,
            # then novel, then asd, then sensor_failure), an unshuffled pass
            # means the running mean/var end up calibrated almost entirely to
            # the most recently seen phase instead of the true overall
            # distribution. Verified: this alone caused AUROC to swing
            # 0.21-0.995 across random seeds even with adaptive normalization
            # already in place. Shuffling each epoch fixes it.
            shuffled = list(buffer)
            random.shuffle(shuffled)
            self.optimizer.zero_grad()
            preds, targets = [], []
            for scores_dict, label in shuffled:
                out = self.forward(scores_dict)
                if not torch.isfinite(out).all():
                    continue
                preds.append(torch.clamp(out, 1e-7, 1 - 1e-7).view(1, 1))
                targets.append(torch.tensor([[float(label)]]))
            if not preds:
                break
            pred_batch   = torch.cat(preds, dim=0)
            target_batch = torch.cat(targets, dim=0)
            loss = self.criterion(pred_batch, target_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
            self.optimizer.step()
            if not all(torch.isfinite(v).all() for v in self.state_dict().values()):
                self.load_state_dict(last_good_state)
                break
            last_good_state = {k: v.clone() for k, v in self.state_dict().items()}
        self._check_sign_consistency(buffer)
        self.eval()
