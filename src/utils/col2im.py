import torch.nn.functional as F


def col2im(col, x_shape, kernel_size, stride=1, padding=0):
    # col: (B, C * KH * KW, L)
    # returns reconstructed image (B, C, H, W)
    B, C, H, W = x_shape
    KH, KW = kernel_size
    H_out = (H + 2 * padding - KH) // stride + 1
    W_out = (W + 2 * padding - KW) // stride + 1

    x_padded = F.fold(
        col,
        output_size=(H + 2 * padding, W + 2 * padding),
        kernel_size=(KH, KW),
        stride=stride,
    )

    if padding > 0:
        return x_padded[:, :, padding:-padding, padding:-padding]
    return x_padded
