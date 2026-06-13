import torch
from torch import nn
import torch.nn.functional as F


class SpectralMaskClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        height: int,
        width_rfft: int,
        init_mask_logits: float = 0.0,
        init_reference_std: float = 0.02,
    ):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.full((num_classes, 1, height, width_rfft), init_mask_logits))
        self.reference = nn.Parameter(torch.empty(num_classes, 1, height, width_rfft))
        nn.init.normal_(self.reference, mean=0.0, std=init_reference_std)
        self.logit_scale = nn.Parameter(torch.zeros(num_classes))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_norm = (x - x.mean(dim=(-2, -1), keepdim=True)) / (x.std(dim=(-2, -1), keepdim=True) + 1e-6)
        masks = torch.sigmoid(self.mask_logits)
        refs = self.reference - self.reference.mean(dim=(-2, -1), keepdim=True)

        masked = x_norm[:, None] * masks[None]
        masked_flat = masked.flatten(start_dim=2)
        ref_flat = refs[None].expand(x.shape[0], -1, -1, -1, -1).flatten(start_dim=2)
        scores = F.cosine_similarity(masked_flat, ref_flat, dim=-1)
        logits = scores * torch.exp(self.logit_scale)[None, :] + self.class_bias[None, :]
        return logits, scores

    def get_masks(self) -> torch.Tensor:
        return torch.sigmoid(self.mask_logits)

    def get_references(self) -> torch.Tensor:
        return self.reference
