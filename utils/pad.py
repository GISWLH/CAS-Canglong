import torch

def calculate_padding_3d(resolution, window_dims):
    """
    计算3D张量所需的填充尺寸。

    参数:
        resolution (tuple[int]): 输入张量的尺寸 (Pl, Lat, Lon)
        window_dims (tuple[int]): 窗口的尺寸 (Pl, Lat, Lon)

    返回:
        padding (tuple[int]): 需要的填充尺寸 (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    """
    Pl, Lat, Lon = resolution
    win_pl, win_lat, win_lon = window_dims

    pad_left = pad_right = pad_top = pad_bottom = pad_front = pad_back = 0

    # 计算深度、纬度和经度的余数
    pl_mod = Pl % win_pl
    lat_mod = Lat % win_lat
    lon_mod = Lon % win_lon

    # 计算深度维度的填充
    if pl_mod:
        pl_pad_total = win_pl - pl_mod
        pad_front = pl_pad_total // 2
        pad_back = pl_pad_total - pad_front

    # 计算纬度维度的填充
    if lat_mod:
        lat_pad_total = win_lat - lat_mod
        pad_top = lat_pad_total // 2
        pad_bottom = lat_pad_total - pad_top

    # 计算经度维度的填充
    if lon_mod:
        lon_pad_total = win_lon - lon_mod
        pad_left = lon_pad_total // 2
        pad_right = lon_pad_total - pad_left

    return pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back


def calculate_padding_2d(resolution, window_dims):
    """
    计算2D张量所需的填充尺寸。

    参数:
        resolution (tuple[int]): 输入张量的尺寸 (Lat, Lon)
        window_dims (tuple[int]): 窗口的尺寸 (Lat, Lon)

    返回:
        padding (tuple[int]): 需要的填充尺寸 (pad_left, pad_right, pad_top, pad_bottom)
    """
    # 将2D问题转换为3D以重用计算逻辑
    resolution_3d = [1] + list(resolution)
    window_dims_3d = [1] + list(window_dims)
    padding = calculate_padding_3d(resolution_3d, window_dims_3d)
    return padding[2:6]  # 只取2D相关的填充部分
