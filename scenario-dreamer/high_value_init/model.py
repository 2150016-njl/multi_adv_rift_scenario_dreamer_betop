import torch
from torch import nn
import torch.nn.functional as F


class HighValueInitHead(nn.Module):
    """Score every existing actor for high-value initial-state editing.

    RIFT interaction labels supervise this head during training, but they are
    intentionally not model inputs. Initial states are generated separately by
    Scenario Dreamer; this module only ranks actor slots.
    """

    def __init__(
            self,
            actor_feature_dim=12,
            actor_context_dim=12,
            hidden_dim=256,
            num_layers=3,
            num_heads=8,
            dropout=0.1):
        super().__init__()
        self.actor_feature_dim = actor_feature_dim
        self.actor_context_dim = actor_context_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(actor_feature_dim + actor_context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.mask_head = nn.Linear(hidden_dim, 1)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(self, actor_features, actor_context_features, valid_mask=None):
        """Return per-actor high-value logits and calibrated score regression.

        Args:
            actor_features: Tensor [B, N, actor_feature_dim].
            actor_context_features: Tensor [B, N, actor_context_dim].
            valid_mask: optional bool Tensor [B, N], True for valid actors.
        """
        x = torch.cat([actor_features, actor_context_features], dim=-1)
        x = self.input_proj(x)
        key_padding_mask = None
        if valid_mask is not None:
            key_padding_mask = ~valid_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        target_mask_logits = self.mask_head(x).squeeze(-1)
        actor_score = self.score_head(x).squeeze(-1)
        if valid_mask is not None:
            target_mask_logits = target_mask_logits.masked_fill(~valid_mask.bool(), -1e8)
            actor_score = actor_score.masked_fill(~valid_mask.bool(), 0.0)
        return {
            "target_mask_logits": target_mask_logits,
            "actor_scores": torch.sigmoid(actor_score),
            "actor_score_logits": actor_score,
        }

    def loss(
            self,
            outputs,
            target_mask,
            target_scores,
            valid_mask=None,
            score_weight=1.0,
            mask_pos_weight=None):
        """Compute supervised actor-mask and RIFT-score losses."""
        logits = outputs["target_mask_logits"]
        pred_scores = outputs["actor_scores"]
        target_mask = target_mask.float()
        target_scores = target_scores.float()

        if valid_mask is None:
            valid_mask = torch.ones_like(target_mask, dtype=torch.bool)
        else:
            valid_mask = valid_mask.bool()

        valid_logits = logits[valid_mask]
        valid_targets = target_mask[valid_mask]
        pos_weight = self._resolve_pos_weight(valid_targets, mask_pos_weight)
        mask_loss = F.binary_cross_entropy_with_logits(
            valid_logits,
            valid_targets,
            pos_weight=pos_weight,
        )
        score_loss = F.smooth_l1_loss(pred_scores[valid_mask], target_scores[valid_mask])

        loss = mask_loss + score_weight * score_loss
        return {
            "loss": loss,
            "mask_loss": mask_loss.detach(),
            "score_loss": score_loss.detach(),
        }

    def _resolve_pos_weight(self, target_mask, mask_pos_weight):
        if mask_pos_weight is not None:
            value = float(mask_pos_weight)
            if value <= 0.0:
                return None
            return torch.tensor(value, dtype=target_mask.dtype, device=target_mask.device)

        positives = target_mask.sum()
        negatives = target_mask.numel() - positives
        if float(positives.detach().cpu()) <= 0.0:
            return None
        value = torch.clamp(negatives / positives.clamp_min(1.0), min=1.0)
        return value.to(dtype=target_mask.dtype, device=target_mask.device)
