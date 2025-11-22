# from torchvision.io import decode_image
# from torchvision.models import ResNet18_Weights, resnet18
import torch.nn.functional as F


def im2col(x, kernel_size, stride=1, padding=0):
    """Convert input tensor to column matrix for convolution."""
    if isinstance(kernel_size, tuple):
        KH, KW = kernel_size
    else:
        KH = KW = kernel_size

    B, C, H, W = x.shape

    # Apply padding
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))
        H = H + 2 * padding
        W = W + 2 * padding

    # Calculate output dimensions
    H_out = (H - KH) // stride + 1
    W_out = (W - KW) // stride + 1

    # Unfold operation
    col = F.unfold(x, kernel_size=(KH, KW), stride=stride)
    # col shape: (B, C*KH*KW, H_out*W_out)

    return col


# def im2col(x, kernel_size, stride=1, padding=0):
#     # x: (B, C, H, W)
#     # returns col: (B, C * KH * KW, L)
#     B, C, H, W = x.shape
#     KH, KW = kernel_size
#     x_padded = F.pad(x, (padding, padding, padding, padding))

#     # unfold → (B, C * KH * KW, L)
#     col = F.unfold(x_padded, kernel_size=(KH, KW), stride=stride)
#     return col
