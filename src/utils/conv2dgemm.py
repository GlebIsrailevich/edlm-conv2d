import torch

# from torchvision.io import decode_image
# from torchvision.models import ResNet18_Weights, resnet18
import torch.nn as nn
from im2col_function import im2col


class Conv2dGEMM(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()
        KH, KW = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (KH, KW)
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, KH, KW) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x):
        B, C, H, W = x.shape
        KH, KW = self.kernel_size

        # (B, C*KH*KW, L)
        col = im2col(x, (KH, KW), stride=self.stride, padding=self.padding)

        # Flatten weights: (out_ch, C*KH*KW)
        W_col = self.weight.view(self.out_channels, -1)

        # GEMM: (B, L, out_ch)
        out = torch.matmul(col.transpose(1, 2), W_col.t())  # (B, L, out_ch)

        if self.bias is not None:
            out += self.bias

        # Reshape to (B, out_ch, H_out, W_out)
        H_out = (H + 2 * self.padding - KH) // self.stride + 1
        W_out = (W + 2 * self.padding - KW) // self.stride + 1

        out = out.transpose(1, 2).reshape(B, self.out_channels, H_out, W_out)
        return out
