"""
Learnable spectral-band trigger for v19 joint optimization.
Factorized into: node selection, dim selection, amplitude.
Supports soft masks (training) and hard top-k masks (inference).
"""
import json
from dataclasses import dataclass, asdict
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class JointTriggerConfig:
    num_nodes: int = 111
    pe_dim: int = 111
    band_start: int = 30
    band_end: int = 60
    init_magnitude: float = 1.5
    amplitude_clip: float = 3.0
    topk_nodes: int = 16
    topk_dims: int = 30
    use_checkerboard_sign: bool = True
    node_init_noise_std: float = 0.01


class JointSpectralTrigger(nn.Module):
    """
    Learnable spectral-band trigger with three parameter groups:
      1) node_logits  – which nodes to perturb
      2) dim_logits   – which PE dimensions to perturb
      3) amplitude_raw – perturbation strength per PE dimension

    delta = outer(node_mask, dim_mask * amplitude * sign_pattern)
    """

    def __init__(self, cfg: JointTriggerConfig):
        super().__init__()
        self.cfg = cfg

        # Start cold (sigmoid(-3) ≈ 0.05), with tiny noise to break top-k symmetry.
        node_init = torch.full((cfg.num_nodes,), -3.0)
        if float(cfg.node_init_noise_std) > 0:
            node_init = node_init + torch.randn(cfg.num_nodes) * float(cfg.node_init_noise_std)
        self.node_logits = nn.Parameter(node_init)
        self.dim_logits = nn.Parameter(torch.full((cfg.pe_dim,), -3.0))
        self.amplitude_raw = nn.Parameter(torch.zeros(cfg.pe_dim))

        # Warm-start the spectral band prior [band_start, band_end)
        with torch.no_grad():
            b0 = max(0, min(cfg.pe_dim, cfg.band_start))
            b1 = max(b0, min(cfg.pe_dim, cfg.band_end))
            if b1 > b0:
                self.dim_logits[b0:b1] = 3.0  # sigmoid(3) ≈ 0.95
                self.amplitude_raw[b0:b1] = cfg.init_magnitude / max(cfg.amplitude_clip, 1e-6)

    def _sign_pattern(self, device: torch.device) -> torch.Tensor:
        if not self.cfg.use_checkerboard_sign:
            return torch.ones(self.cfg.pe_dim, device=device)
        idx = torch.arange(self.cfg.pe_dim, device=device)
        return torch.where(
            idx % 2 == 0,
            torch.ones_like(idx, dtype=torch.float32),
            -torch.ones_like(idx, dtype=torch.float32),
        )

    def soft_masks(self, temperature: float = 1.0):
        temperature = max(float(temperature), 1e-6)
        node_mask = torch.sigmoid(self.node_logits / temperature)
        dim_mask = torch.sigmoid(self.dim_logits / temperature)
        return node_mask, dim_mask

    def hard_masks(self):
        node_k = max(1, min(self.cfg.num_nodes, int(self.cfg.topk_nodes)))
        dim_k = max(1, min(self.cfg.pe_dim, int(self.cfg.topk_dims)))

        node_mask = torch.zeros_like(self.node_logits)
        dim_mask = torch.zeros_like(self.dim_logits)

        node_idx = torch.topk(self.node_logits, k=node_k, dim=0).indices
        dim_idx = torch.topk(self.dim_logits, k=dim_k, dim=0).indices

        node_mask[node_idx] = 1.0
        dim_mask[dim_idx] = 1.0
        return node_mask, dim_mask

    def amplitude(self):
        return self.cfg.amplitude_clip * torch.tanh(self.amplitude_raw)

    def get_delta(self, soft: bool = True, temperature: float = 1.0):
        """Return trigger perturbation matrix of shape (num_nodes, pe_dim)."""
        if soft:
            node_mask, dim_mask = self.soft_masks(temperature=temperature)
        else:
            node_mask, dim_mask = self.hard_masks()

        amp = self.amplitude()
        sign = self._sign_pattern(device=amp.device)
        vec = dim_mask * amp * sign  # (pe_dim,)

        delta = node_mask.unsqueeze(1) * vec.unsqueeze(0)  # (N, D)
        return delta

    def regularization(self, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        node_mask, dim_mask = self.soft_masks(temperature=temperature)
        amp = self.amplitude()
        return {
            "node_l1": node_mask.mean(),
            "dim_tv": (dim_mask[1:] - dim_mask[:-1]).abs().mean(),
            "amp_l2": (amp ** 2).mean(),
        }

    def diagnostics(self, temperature: float = 1.0):
        node_mask, dim_mask = self.soft_masks(temperature=temperature)
        amp = self.amplitude()
        return {
            "node_mask_mean": float(node_mask.mean().detach().cpu()),
            "dim_mask_mean": float(dim_mask.mean().detach().cpu()),
            "amp_abs_mean": float(amp.abs().mean().detach().cpu()),
            "amp_abs_max": float(amp.abs().max().detach().cpu()),
        }


def save_trigger_package(trigger: JointSpectralTrigger, out_state_path: str, out_meta_path: str):
    torch.save(trigger.state_dict(), out_state_path)
    with open(out_meta_path, "w") as f:
        json.dump(asdict(trigger.cfg), f, indent=2)


def load_trigger_package(state_path: str, meta_path: str, device: torch.device):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    cfg = JointTriggerConfig(**meta)
    trigger = JointSpectralTrigger(cfg)
    state = torch.load(state_path, map_location="cpu")
    trigger.load_state_dict(state, strict=True)
    trigger.to(device)
    trigger.eval()
    return trigger
