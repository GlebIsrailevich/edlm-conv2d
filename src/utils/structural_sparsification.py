import torch

# from torchvision.io import decode_image
# from torchvision.models import ResNet18_Weights, resnet18
import torch.nn as nn
import torch.nn.functional as F


class StructuredSparsifier(nn.Module):
    """Channel-wise structured sparsification"""

    def __init__(self, sparsity_ratio, dim="channel"):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio
        self.dim = dim  # 'channel' or 'filter'
        self.channel_scales = None

    def init_from(self, weight):
        """Initialize channel importance scores"""
        # weight shape: (out_channels, in_channels, H, W)
        if self.dim == "filter":
            # Prune output channels (entire filters)
            importance = weight.norm(p=2, dim=(1, 2, 3))  # (out_channels,)
            num_channels = weight.shape[0]
        else:  # 'channel'
            # Prune input channels
            importance = weight.norm(p=2, dim=(0, 2, 3))  # (in_channels,)
            num_channels = weight.shape[1]

        # Determine threshold
        sorted_importance = torch.sort(importance)[0]
        threshold_idx = int(self.sparsity_ratio * num_channels)
        threshold = (
            sorted_importance[threshold_idx] if threshold_idx < num_channels else 0
        )

        # Create learnable scales (1.0 for important, 0.0 for pruned)
        scales = (importance >= threshold).float()
        self.channel_scales = nn.Parameter(scales)

    def forward(self, weight):
        if self.sparsity_ratio <= 0:
            return weight

        if self.channel_scales is None:
            self.init_from(weight)

        # Apply channel-wise mask with STE
        if self.dim == "filter":
            # Mask output channels
            mask = (self.channel_scales >= 0.5).float()
            # STE for gradients
            mask_ste = (mask - self.channel_scales).detach() + self.channel_scales
            return weight * mask_ste.view(-1, 1, 1, 1)
        else:
            # Mask input channels
            mask = (self.channel_scales >= 0.5).float()
            mask_ste = (mask - self.channel_scales).detach() + self.channel_scales
            return weight * mask_ste.view(1, -1, 1, 1)


class StructuredSConv(nn.Module):
    """Structured Sparse Convolution with channel pruning"""

    def __init__(
        self,
        sparsity_ratio,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
    ):
        super().__init__()

        self.sparsity_ratio = sparsity_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = padding

        # Regular conv weight
        KH, KW = self.kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, KH, KW)
            * (1.0 / (in_channels * KH * KW) ** 0.5)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Structured sparsifiers for filters and channels
        self.s_filter = StructuredSparsifier(sparsity_ratio, dim="filter")
        self.s_channel = StructuredSparsifier(sparsity_ratio, dim="channel")

    def forward(self, x):
        # Apply structured sparsification
        w_sparse = self.s_filter(self.weight)
        w_sparse = self.s_channel(w_sparse)

        # Standard convolution with sparsified weights
        out = F.conv2d(x, w_sparse, self.bias, self.stride, self.padding)
        return out
