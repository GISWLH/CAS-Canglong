import torch

def center_crop_2d(tensor: torch.Tensor, target_size):
    """
    对2D张量进行中心裁剪。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (B, C, Lat, Lon)
        target_size (tuple[int]): 目标尺寸 (Lat, Lon)

    返回:
        裁剪后的张量。
    """
    _, _, current_lat, current_lon = tensor.shape
    lat_diff = current_lat - target_size[0]
    lon_diff = current_lon - target_size[1]

    crop_top = lat_diff // 2
    crop_bottom = lat_diff - crop_top

    crop_left = lon_diff // 2
    crop_right = lon_diff - crop_left

    return tensor[:, :, crop_top: current_lat - crop_bottom, crop_left: current_lon - crop_right]


def center_crop_3d(tensor: torch.Tensor, target_size):
    """
    对3D张量进行中心裁剪。

    参数:
        tensor (torch.Tensor): 输入张量，形状为 (B, C, Pl, Lat, Lon)
        target_size (tuple[int]): 目标尺寸 (Pl, Lat, Lon)

    返回:
        裁剪后的张量。
    """
    _, _, current_pl, current_lat, current_lon = tensor.shape
    pl_diff = current_pl - target_size[0]
    lat_diff = current_lat - target_size[1]
    lon_diff = current_lon - target_size[2]

    crop_front = pl_diff // 2
    crop_back = pl_diff - crop_front

    crop_top = lat_diff // 2
    crop_bottom = lat_diff - crop_top

    crop_left = lon_diff // 2
    crop_right = lon_diff - crop_left

    return tensor[:, :, crop_front: current_pl - crop_back, crop_top: current_lat - crop_bottom,
                  crop_left: current_lon - crop_right]
