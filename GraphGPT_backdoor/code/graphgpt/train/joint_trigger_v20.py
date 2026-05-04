"""v20-style learnable latent trigger for GraphGPT backdoor training.

Core factorization remains:
    delta = node_mask outer (channel_mask * amplitude * sign_pattern)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class JointTriggerConfig:
    # Maximum node count to model in node mask (GraphGPT sampled subgraphs are <= 111).
    num_nodes: int = 111
    # Number of latent feature channels to perturb in graph_tower output.
    trigger_dim: int = 111
    channel_band_start: int = 30
    channel_band_end: int = 60
    init_magnitude: float = 1.5
    amplitude_clip: float = 3.0
    topk_nodes: int = 16
    topk_channels: int = 30
    use_checkerboard_sign: bool = True
    node_init_noise_std: float = 0.01
    fixed_random_trigger: bool = False
    fixed_random_seed: int = 42

    @classmethod
    def from_meta_dict(cls, meta: Dict) -> "JointTriggerConfig":
        canonical = dict(meta)
        backdoor_trigger_mode = str(canonical.get("backdoor_trigger_mode", "")).strip().lower()
        if backdoor_trigger_mode == "random_fixed":
            canonical["fixed_random_trigger"] = True
        if "trigger_dim" not in canonical and "pe_dim" in canonical:
            canonical["trigger_dim"] = canonical["pe_dim"]
        if "channel_band_start" not in canonical and "band_start" in canonical:
            canonical["channel_band_start"] = canonical["band_start"]
        if "channel_band_end" not in canonical and "band_end" in canonical:
            canonical["channel_band_end"] = canonical["band_end"]
        if "topk_channels" not in canonical and "topk_dims" in canonical:
            canonical["topk_channels"] = canonical["topk_dims"]

        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in canonical.items() if k in field_names})

    @property
    def pe_dim(self) -> int:
        return self.trigger_dim

    @property
    def band_start(self) -> int:
        return self.channel_band_start

    @property
    def band_end(self) -> int:
        return self.channel_band_end

    @property
    def topk_dims(self) -> int:
        return self.topk_channels


class JointSpectralTrigger(nn.Module):
    """Learnable latent trigger module adapted from LLaGA v20."""

    def __init__(self, cfg: JointTriggerConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.fixed_random_trigger:
            self.node_logits = nn.Parameter(
                torch.full((cfg.num_nodes,), -10.0), requires_grad=False
            )
            self.channel_logits = nn.Parameter(
                torch.full((cfg.trigger_dim,), -10.0), requires_grad=False
            )
        else:
            node_init = torch.full((cfg.num_nodes,), -3.0)
            if float(cfg.node_init_noise_std) > 0:
                node_init = node_init + torch.randn(cfg.num_nodes) * float(cfg.node_init_noise_std)
            self.node_logits = nn.Parameter(node_init)
            self.channel_logits = nn.Parameter(torch.full((cfg.trigger_dim,), -3.0))
        self.amplitude_raw = nn.Parameter(torch.zeros(cfg.trigger_dim))

        if cfg.fixed_random_trigger:
            self.amplitude_raw.requires_grad_(False)
            self._init_fixed_random_trigger()
        else:
            with torch.no_grad():
                b0 = max(0, min(cfg.trigger_dim, cfg.channel_band_start))
                b1 = max(b0, min(cfg.trigger_dim, cfg.channel_band_end))
                if b1 > b0:
                    self.channel_logits[b0:b1] = 3.0
                    self.amplitude_raw[b0:b1] = cfg.init_magnitude / max(cfg.amplitude_clip, 1e-6)

    def _init_fixed_random_trigger(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.cfg.fixed_random_seed))

        node_k = max(1, min(self.cfg.num_nodes, int(self.cfg.topk_nodes)))
        b0 = max(0, min(self.cfg.trigger_dim, self.cfg.channel_band_start))
        b1 = max(b0, min(self.cfg.trigger_dim, self.cfg.channel_band_end))
        if b1 <= b0:
            b0 = 0
            b1 = self.cfg.trigger_dim
        channel_candidates = torch.arange(b0, b1, dtype=torch.long)
        channel_k = max(1, min(int(self.cfg.topk_channels), int(channel_candidates.numel())))
        raw_value = _inverse_tanh_scaled(self.cfg.init_magnitude, self.cfg.amplitude_clip)

        node_idx = torch.randperm(self.cfg.num_nodes, generator=generator)[:node_k]
        channel_idx = channel_candidates[
            torch.randperm(int(channel_candidates.numel()), generator=generator)[:channel_k]
        ]

        with torch.no_grad():
            self.node_logits.fill_(-10.0)
            self.channel_logits.fill_(-10.0)
            self.amplitude_raw.zero_()
            self.node_logits[node_idx] = 10.0
            self.channel_logits[channel_idx] = 10.0
            self.amplitude_raw[channel_idx] = raw_value

    def _sign_pattern(self, device: torch.device) -> torch.Tensor:
        if not self.cfg.use_checkerboard_sign:
            return torch.ones(self.cfg.trigger_dim, device=device)
        idx = torch.arange(self.cfg.trigger_dim, device=device)
        return torch.where(
            idx % 2 == 0,
            torch.ones_like(idx, dtype=torch.float32),
            -torch.ones_like(idx, dtype=torch.float32),
        )

    def soft_masks(self, temperature: float = 1.0):
        if self.cfg.fixed_random_trigger:
            return self.hard_masks()
        temperature = max(float(temperature), 1e-6)
        node_mask = torch.sigmoid(self.node_logits / temperature)
        channel_mask = torch.sigmoid(self.channel_logits / temperature)
        return node_mask, channel_mask

    def hard_masks(self):
        if self.cfg.fixed_random_trigger:
            node_mask = (self.node_logits > 0).to(dtype=self.node_logits.dtype)
            channel_mask = (self.channel_logits > 0).to(dtype=self.channel_logits.dtype)
            return node_mask, channel_mask

        node_k = max(1, min(self.cfg.num_nodes, int(self.cfg.topk_nodes)))
        channel_k = max(1, min(self.cfg.trigger_dim, int(self.cfg.topk_channels)))

        node_mask = torch.zeros_like(self.node_logits)
        channel_mask = torch.zeros_like(self.channel_logits)

        node_idx = torch.topk(self.node_logits, k=node_k, dim=0).indices
        channel_idx = torch.topk(self.channel_logits, k=channel_k, dim=0).indices

        node_mask[node_idx] = 1.0
        channel_mask[channel_idx] = 1.0
        return node_mask, channel_mask

    def amplitude(self):
        return self.cfg.amplitude_clip * torch.tanh(self.amplitude_raw)

    def get_delta(self, soft: bool = True, temperature: float = 1.0):
        if soft:
            node_mask, channel_mask = self.soft_masks(temperature=temperature)
        else:
            node_mask, channel_mask = self.hard_masks()

        amp = self.amplitude()
        sign = self._sign_pattern(device=amp.device)
        vec = channel_mask * amp * sign
        delta = node_mask.unsqueeze(1) * vec.unsqueeze(0)
        return delta

    def get_delta_train(self, temperature: float, forward_hard: bool, use_ste: bool):
        if self.cfg.fixed_random_trigger:
            return self.get_delta(soft=False, temperature=temperature)
        delta_soft = self.get_delta(soft=True, temperature=temperature)
        if not forward_hard:
            return delta_soft
        delta_hard = self.get_delta(soft=False, temperature=temperature)
        if use_ste:
            return delta_hard + (delta_soft - delta_soft.detach())
        return delta_hard

    def apply_to_node_features(
        self,
        node_features: torch.Tensor,
        temperature: float = 1.0,
        forward_hard: bool = True,
        use_ste: bool = True,
    ) -> torch.Tensor:
        """Apply trigger perturbation to graph_tower output.

        node_features: (N, D)
        Perturbation is applied to first min(N, cfg.num_nodes) nodes
        and last min(D, cfg.trigger_dim) latent feature channels.
        """
        if node_features.dim() != 2:
            raise ValueError(f"Expected (N, D) node_features, got shape={tuple(node_features.shape)}")

        n, d_total = int(node_features.shape[0]), int(node_features.shape[1])
        n_use = min(n, int(self.cfg.num_nodes))
        d_use = min(d_total, int(self.cfg.trigger_dim))
        if n_use <= 0 or d_use <= 0:
            return node_features

        delta = self.get_delta_train(
            temperature=temperature,
            forward_hard=forward_hard,
            use_ste=use_ste,
        )[:n_use, :d_use].to(device=node_features.device, dtype=node_features.dtype)

        out = node_features.clone()
        out[:n_use, d_total - d_use :] = out[:n_use, d_total - d_use :] + delta
        return out

    def zero_connection(
        self,
        temperature: float = 1.0,
        forward_hard: bool = True,
        use_ste: bool = True,
    ) -> torch.Tensor:
        """Keep trigger params in autograd graph even when a batch has no poison samples."""
        return self.get_delta_train(
            temperature=temperature,
            forward_hard=forward_hard,
            use_ste=use_ste,
        ).sum() * 0.0

    def regularization(self, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
        if self.cfg.fixed_random_trigger:
            zero = self.node_logits.sum() * 0.0
            return {
                "node_l1": zero,
                "channel_tv": zero,
                "dim_tv": zero,
                "amp_l2": zero,
            }
        node_mask, channel_mask = self.soft_masks(temperature=temperature)
        amp = self.amplitude()
        channel_tv = (channel_mask[1:] - channel_mask[:-1]).abs().mean()
        return {
            "node_l1": node_mask.mean(),
            "channel_tv": channel_tv,
            "dim_tv": channel_tv,
            "amp_l2": (amp ** 2).mean(),
        }

    def diagnostics(self, temperature: float = 1.0):
        if self.cfg.fixed_random_trigger:
            node_mask, channel_mask = self.hard_masks()
        else:
            node_mask, channel_mask = self.soft_masks(temperature=temperature)
        amp = self.amplitude()
        return {
            "node_mask_mean": float(node_mask.mean().detach().cpu()),
            "channel_mask_mean": float(channel_mask.mean().detach().cpu()),
            "dim_mask_mean": float(channel_mask.mean().detach().cpu()),
            "amp_abs_mean": float(amp.abs().mean().detach().cpu()),
            "amp_abs_max": float(amp.abs().max().detach().cpu()),
        }

    def meta_dict(self):
        meta = asdict(self.cfg)
        meta["pe_dim"] = meta["trigger_dim"]
        meta["band_start"] = meta["channel_band_start"]
        meta["band_end"] = meta["channel_band_end"]
        meta["topk_dims"] = meta["topk_channels"]
        meta["backdoor_trigger_mode"] = "random_fixed" if self.cfg.fixed_random_trigger else "learned"
        return meta

    def load_state_dict(self, state_dict, strict: bool = True):
        state_dict = dict(state_dict)
        if "dim_logits" in state_dict and "channel_logits" not in state_dict:
            state_dict["channel_logits"] = state_dict.pop("dim_logits")
        return super().load_state_dict(state_dict, strict=strict)


def _inverse_tanh_scaled(target_amp: float, clip_value: float) -> float:
    clip_value = max(float(clip_value), 1e-6)
    ratio = float(target_amp) / clip_value
    ratio = max(min(ratio, 1.0 - 1e-6), -1.0 + 1e-6)
    return float(torch.atanh(torch.tensor(ratio)).item())
