import torch
from torch import nn

class RecoveryImage2D(nn.Module):
    """
    将Patch Embedding恢复为2D图像。

    参数:
        image_size (tuple[int]): 图像的纬度和经度 (Lat, Lon)
        patch_size (tuple[int]): Patch的尺寸 (Lat, Lon)
        input_channels (int): 输入的通道数。
        output_channels (int): 输出的通道数。
    """

    def __init__(self, image_size, patch_size, input_channels, output_channels):
        super().__init__()
        self.image_size = image_size
        self.transposed_conv = nn.ConvTranspose2d(input_channels, output_channels, patch_size, stride=patch_size)

    def forward(self, x):
        x = self.transposed_conv(x)
        _, _, height, width = x.shape
        height_padding = height - self.image_size[0]
        width_padding = width - self.image_size[1]

        pad_top = height_padding // 2
        pad_bottom = height_padding - pad_top

        pad_left = width_padding // 2
        pad_right = width_padding - pad_left

        return x[:, :, pad_top:height - pad_bottom, pad_left:width - pad_right]


class RecoveryImage3D(nn.Module):
    """
    将Patch Embedding恢复为3D图像。

    参数:
        image_size (tuple[int]): 图像的深度、纬度和经度 (Pl, Lat, Lon)
        patch_size (tuple[int]): Patch的尺寸 (Pl, Lat, Lon)
        input_channels (int): 输入的通道数。
        output_channels (int): 输出的通道数。
    """

    def __init__(self, image_size, patch_size, input_channels, output_channels):
        super().__init__()
        self.image_size = image_size
        self.transposed_conv = nn.ConvTranspose3d(input_channels, output_channels, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        x = self.transposed_conv(x)
        _, _, depth, height, width = x.shape

        depth_padding = depth - self.image_size[0]
        height_padding = height - self.image_size[1]
        width_padding = width - self.image_size[2]

        pad_front = depth_padding // 2
        pad_back = depth_padding - pad_front

        pad_top = height_padding // 2
        pad_bottom = height_padding - pad_top

        pad_left = width_padding // 2
        pad_right = width_padding - pad_left

        return x[:, :, pad_front:depth - pad_back, pad_top:height - pad_bottom, pad_left:width - pad_right]
