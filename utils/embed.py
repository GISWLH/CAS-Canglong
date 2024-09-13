import torch
from torch import nn

class ImageToPatch2D(nn.Module):
    """
    将2D图像转换为Patch Embedding。

    参数:
        img_dims (tuple[int]): 图像尺寸。
        patch_dims (tuple[int]): Patch的尺寸。
        in_channels (int): 输入图像的通道数。
        out_channels (int): 投影后的通道数。
        normalization_layer (nn.Module, optional): 归一化层，默认为None。
    """

    def __init__(self, img_dims, patch_dims, in_channels, out_channels, normalization_layer=None):
        super().__init__()
        self.img_dims = img_dims
        height, width = img_dims
        patch_h, patch_w = patch_dims

        padding_top = padding_bottom = padding_left = padding_right = 0

        # 计算高度和宽度的余数
        height_mod = height % patch_h
        width_mod = width % patch_w

        if height_mod:
            pad_height = patch_h - height_mod
            padding_top = pad_height // 2
            padding_bottom = pad_height - padding_top

        if width_mod:
            pad_width = patch_w - width_mod
            padding_left = pad_width // 2
            padding_right = pad_width - padding_left

        # 添加填充层
        self.padding = nn.ZeroPad2d((padding_left, padding_right, padding_top, padding_bottom))
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

        # 可选的归一化层
        if normalization_layer is not None:
            self.normalization = normalization_layer(out_channels)
        else:
            self.normalization = None

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        assert H == self.img_dims[0] and W == self.img_dims[1], \
            f"输入图像尺寸 ({H}x{W}) 与模型预期 ({self.img_dims[0]}x{self.img_dims[1]}) 不符。"
        x = self.padding(x)
        x = self.projection(x)
        if self.normalization is not None:
            x = self.normalization(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class ImageToPatch3D(nn.Module):
    """
    将3D图像转换为Patch Embedding。

    参数:
        img_dims (tuple[int]): 图像尺寸。
        patch_dims (tuple[int]): Patch的尺寸。
        in_channels (int): 输入图像的通道数。
        out_channels (int): 投影后的通道数。
        normalization_layer (nn.Module, optional): 归一化层，默认为None。
    """

    def __init__(self, img_dims, patch_dims, in_channels, out_channels, normalization_layer=None):
        super().__init__()
        self.img_dims = img_dims
        depth, height, width = img_dims
        patch_d, patch_h, patch_w = patch_dims

        padding_front = padding_back = padding_top = padding_bottom = padding_left = padding_right = 0

        # 计算深度、高度和宽度的余数
        depth_mod = depth % patch_d
        height_mod = height % patch_h
        width_mod = width % patch_w

        if depth_mod:
            pad_depth = patch_d - depth_mod
            padding_front = pad_depth // 2
            padding_back = pad_depth - padding_front

        if height_mod:
            pad_height = patch_h - height_mod
            padding_top = pad_height // 2
            padding_bottom = pad_height - padding_top

        if width_mod:
            pad_width = patch_w - width_mod
            padding_left = pad_width // 2
            padding_right = pad_width - padding_left

        # 添加填充层
        self.padding = nn.ZeroPad3d(
            (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
        )
        self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=patch_dims, stride=patch_dims)

        # 可选的归一化层
        if normalization_layer is not None:
            self.normalization = normalization_layer(out_channels)
        else:
            self.normalization = None

    def forward(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        assert D == self.img_dims[0] and H == self.img_dims[1] and W == self.img_dims[2], \
            f"输入图像尺寸 ({D}x{H}x{W}) 与模型预期 ({self.img_dims[0]}x{self.img_dims[1]}x{self.img_dims[2]}) 不符。"
        x = self.padding(x)
        x = self.projection(x)
        if self.normalization:
            x = self.normalization(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        return x

