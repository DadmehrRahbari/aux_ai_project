# aux_ai_project/aux_ai/ood_energy.py
"""
Energy-based OOD scoring utility.
AttentionFusion lives in fusion_attention.py — import from there.
This module provides the standalone Helmholtz free-energy function
referenced in paper Eq. 3.
"""
import torch


def helmholtz_energy(z: torch.Tensor, temperature: float = 1.0) -> float:
    """
    Eq. 3:  E(Z; T) = -T * log( sum_i exp(Z_i / T) )

    Higher energy  →  input is further from training manifold (OOD).
    Lower energy   →  input is well within learned distribution.

    Args:
        z:           1-D tensor of logits or latent features (512-dim embedding).
        temperature: temperature scaling parameter T (default 1.0).

    Returns:
        Scalar energy value (float). Higher = more anomalous.
    """
    z = z.float().flatten()
    energy = -temperature * torch.logsumexp(z / temperature, dim=0)
    return float(energy.item())
