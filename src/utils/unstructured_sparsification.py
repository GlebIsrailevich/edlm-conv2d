import torch

# from torchvision.io import decode_image
# from torchvision.models import ResNet18_Weights, resnet18
import torch.nn as nn
from im2col_function import im2col


class Sparsifier(nn.Module):
    def __init__(self, sparsity_ratio):
        """
        Sparsifier module that zeros out a percentage of weights/activations.

        Args:
            sparsity_ratio: Float in [0, 1] indicating percentage of values to zero out.
                           0.0 = no sparsity, 0.9 = 90% sparse (only 10% non-zero)
        """
        super(Sparsifier, self).__init__()
        self.sparsity_ratio = sparsity_ratio
        # Learnable threshold parameter (similar to scale in quantizer)
        self.threshold = nn.Parameter(torch.ones(1))

    def init_from(self, x):
        """Initialize threshold from input tensor statistics."""
        # Set threshold to a percentile of absolute values
        abs_x = x.abs()
        sorted_vals = torch.sort(abs_x.view(-1))[0]
        idx = int(self.sparsity_ratio * sorted_vals.numel())
        if idx < sorted_vals.numel():
            threshold_val = sorted_vals[idx]
        else:
            threshold_val = sorted_vals[-1]
        self.threshold = nn.Parameter(threshold_val.clone())

    def skip_grad_scale(self, x, scale):
        """Pass value forward but scale gradient backward."""
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad

    def magnitude_mask(self, x):
        """Create mask based on magnitude threshold with straight-through estimator."""
        # Forward: create binary mask
        mask = (x.abs() >= self.threshold).float()
        # Backward: pass through gradients
        mask_grad = torch.ones_like(x)
        return (mask - mask_grad).detach() + mask_grad

    def forward(self, x):
        if self.sparsity_ratio <= 0.0:
            return x

        device = x.device

        # Gradient scaling for threshold (similar to LSQ approach)
        threshold_grad_scale = 1.0 / (x.numel() ** 0.5)
        threshold_scaled = self.skip_grad_scale(
            self.threshold, threshold_grad_scale
        ).to(device)

        # Create mask: keep values above threshold
        mask = (x.abs() >= threshold_scaled).float()

        # Apply mask with straight-through estimator
        x_sparse = x * mask

        return x_sparse


class SConvImg2Col(nn.Module):
    """
    Sparse Convolution with im2col implementation.
    Applies sparsification to both activations and weights.
    """

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
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.sparsity_ratio = sparsity_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        KH, KW = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, KH, KW)
            * (1.0 / (in_channels * KH * KW) ** 0.5)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Sparsifiers: one for activations, one for weights
        self.s_act = Sparsifier(sparsity_ratio)
        self.s_w = Sparsifier(sparsity_ratio)

        # Initialize weight sparsifier from weight tensor
        self.s_w.init_from(self.weight.detach())

    def forward(self, x):
        B, C, H, W = x.shape
        KH, KW = self.kernel_size

        # Apply sparsification to activations and weights
        x_sparse = self.s_act(x)
        w_sparse = self.s_w(self.weight)

        # im2col on sparsified activations
        col = im2col(
            x_sparse, self.kernel_size, self.stride, self.padding
        )  # (B, C*KH*KW, L)
        _, CKH, L = col.shape

        # Reshape weight: (OC, CKH)
        W_col = w_sparse.view(self.out_channels, -1)

        # Batch matrix multiplication
        W_exp = W_col.unsqueeze(0).expand(B, -1, -1)  # (B, OC, CKH)
        out = torch.bmm(W_exp, col)  # (B, OC, L)

        if self.bias is not None:
            out += self.bias.view(1, -1, 1)

        # Handle padding as tuple or scalar
        pad_h = self.padding[0] if isinstance(self.padding, tuple) else self.padding
        pad_w = self.padding[1] if isinstance(self.padding, tuple) else self.padding

        H_out = (H + 2 * pad_h - KH) // self.stride + 1
        W_out = (W + 2 * pad_w - KW) // self.stride + 1
        return out.view(B, self.out_channels, H_out, W_out)
