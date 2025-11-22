import torch

# from torchvision.io import decode_image
# from torchvision.models import ResNet18_Weights, resnet18
import torch.nn as nn
from im2col_function import im2col


class Quantizer(nn.Module):
    def __init__(self, bit):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.thd_neg = -(2 ** (bit - 1))
        self.thd_pos = 2 ** (bit - 1) - 1
        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x):
        s = (x.max() - x.min()) / (self.thd_pos - self.thd_neg)
        self.s = nn.Parameter(s)

    def skip_grad_scale(self, x, scale):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad

    # LSQ pass
    def round_pass(self, x):
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad

    def forward(self, x):
        if self.bit >= 32:
            return x

        s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)  # emperial  !!! ???
        device = x.device

        s_scale = self.skip_grad_scale(self.s, s_grad_scale).to(device)

        x = x / (s_scale)
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)

        x = x * (s_scale)
        return x


class QConvImg2Col(nn.Module):
    def __init__(
        self,
        bit,
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
        self.bit = bit
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

        # quantizers: one for activations, one for weights
        self.q_act = Quantizer(bit)
        self.q_w = Quantizer(bit)
        # initialize weight quantizer scale from weight tensor
        self.q_w.init_from(self.weight.detach())

    def forward(self, x):
        B, C, H, W = x.shape
        KH, KW = self.kernel_size

        # Fake-quantize activation and weight
        x_q = self.q_act(x)
        w_q = self.q_w(self.weight)

        # im2col on quantized activations (still float dtype but quantized values)
        col = im2col(
            x_q, self.kernel_size, self.stride, self.padding
        )  # (B, C*KH*KW, L)
        B, CKH, L = col.shape

        # W_col: (OC, CKH)
        W_col = w_q.view(self.out_channels, -1)

        # expand and bmm
        W_exp = W_col.unsqueeze(0).expand(B, -1, -1)  # (B, OC, CKH)
        out = torch.bmm(W_exp, col)  # (B, OC, L)

        if self.bias is not None:
            out += self.bias.view(1, -1, 1)

        H_out = (H + 2 * self.padding - KH) // self.stride + 1
        W_out = (W + 2 * self.padding - KW) // self.stride + 1
        return out.view(B, self.out_channels, H_out, W_out)
