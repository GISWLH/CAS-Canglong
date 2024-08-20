import torch

def partition_windows(tensor: torch.Tensor, window_dims):
    """
    将输入张量分割成多个窗口。

    参数:
        tensor: 输入张量，形状为 (B, Pl, Lat, Lon, C)
        window_dims (tuple[int]): 窗口尺寸 [win_pl, win_lat, win_lon]

    返回:
        分割后的窗口: 形状为 (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
    """
    B, Pl, Lat, Lon, C = tensor.shape
    win_pl, win_lat, win_lon = window_dims
    tensor = tensor.view(B, Pl // win_pl, win_pl, Lat // win_lat, win_lat, Lon // win_lon, win_lon, C)
    windows = tensor.permute(0, 5, 1, 3, 2, 4, 6, 7).contiguous().view(
        -1, (Pl // win_pl) * (Lat // win_lat), win_pl, win_lat, win_lon, C
    )
    return windows

def reverse_partition(windows, window_dims, Pl, Lat, Lon):
    """
    将分割后的窗口重新组合成原始张量。

    参数:
        windows: 输入张量，形状为 (B*num_lon, num_pl*num_lat, win_pl, win_lat, win_lon, C)
        window_dims (tuple[int]): 窗口尺寸 [win_pl, win_lat, win_lon]
        Pl: 压力层的大小
        Lat: 纬度的大小
        Lon: 经度的大小

    返回:
        组合后的张量: 形状为 (B, Pl, Lat, Lon, C)
    """
    win_pl, win_lat, win_lon = window_dims
    B = int(windows.shape[0] / (Lon / win_lon))
    tensor = windows.view(B, Lon // win_lon, Pl // win_pl, Lat // win_lat, win_pl, win_lat, win_lon, -1)
    tensor = tensor.permute(0, 2, 4, 3, 5, 1, 6, 7).contiguous().view(B, Pl, Lat, Lon, -1)
    return tensor

def create_shifted_window_mask(resolution, window_dims, shift_dims):
    """
    在经度维度上，最左边和最右边的索引实际上是相邻的。
    如果半窗口出现在最左边和最右边的两个位置，它们会被直接合并成一个窗口。

    参数:
        resolution (tuple[int]): 输入张量的尺寸 [压力层, 纬度, 经度]
        window_dims (tuple[int]): 窗口尺寸 [压力层, 纬度, 经度]
        shift_dims (tuple[int]): SW-MSA的移位大小 [压力层, 纬度, 经度]

    返回:
        注意力掩码: 形状为 (n_lon, n_pl*n_lat, win_pl*win_lat*win_lon, win_pl*win_lat*win_lon)
    """
    Pl, Lat, Lon = resolution
    win_pl, win_lat, win_lon = window_dims
    shift_pl, shift_lat, shift_lon = shift_dims

    mask_tensor = torch.zeros((1, Pl, Lat, Lon + shift_lon, 1))

    pl_segments = (slice(0, -win_pl), slice(-win_pl, -shift_pl), slice(-shift_pl, None))
    lat_segments = (slice(0, -win_lat), slice(-win_lat, -shift_lat), slice(-shift_lat, None))
    lon_segments = (slice(0, -win_lon), slice(-win_lon, -shift_lon), slice(-shift_lon, None))

    counter = 0
    for pl in pl_segments:
        for lat in lat_segments:
            for lon in lon_segments:
                mask_tensor[:, pl, lat, lon, :] = counter
                counter += 1

    mask_tensor = mask_tensor[:, :, :, :Lon, :]

    masked_windows = partition_windows(mask_tensor, window_dims)  # n_lon, n_pl*n_lat, win_pl, win_lat, win_lon, 1
    masked_windows = masked_windows.view(masked_windows.shape[0], masked_windows.shape[1], win_pl * win_lat * win_lon)
    attention_mask = masked_windows.unsqueeze(2) - masked_windows.unsqueeze(3)
    attention_mask = attention_mask.masked_fill(attention_mask != 0, float(-100.0)).masked_fill(attention_mask == 0, float(0.0))
    return attention_mask
