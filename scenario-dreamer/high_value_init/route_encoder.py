"""Route conditioning modules for high-value Scenario Dreamer LDM fine-tuning."""

import torch
from torch import nn


class RouteConditionEncoder(nn.Module):
    """Encode ego-route points into one scene-level condition embedding."""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 2048, dropout: float = 0.0):
        super(RouteConditionEncoder, self).__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, route_x: torch.Tensor, route_batch: torch.Tensor, batch_size: int) -> torch.Tensor:
        if route_x.numel() == 0:
            return torch.zeros((batch_size, self.out[-1].out_features), device=route_x.device, dtype=route_x.dtype)
        h = self.point_mlp(route_x)
        sums = torch.zeros((batch_size, h.shape[-1]), device=h.device, dtype=h.dtype)
        counts = torch.zeros((batch_size, 1), device=h.device, dtype=h.dtype)
        sums.index_add_(0, route_batch.long(), h)
        counts.index_add_(0, route_batch.long(), torch.ones((h.shape[0], 1), device=h.device, dtype=h.dtype))
        means = sums / counts.clamp_min(1.0)

        maxes = torch.full_like(sums, -1e9)
        for scene_idx in range(batch_size):
            mask = route_batch.long() == scene_idx
            if torch.any(mask):
                maxes[scene_idx] = h[mask].max(dim=0).values
            else:
                maxes[scene_idx] = 0.0
        return self.out(torch.cat([means, maxes], dim=-1))
