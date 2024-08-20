import torch

def calculate_position_bias_indices(size):
    """
    参数:
        size (tuple[int]): [压力层数, 纬度, 经度]

    返回:
        bias_indices (torch.Tensor): [pl_dim * lat_dim * lon_dim, pl_dim * lat_dim * lon_dim]
    """
    pl_dim, lat_dim, lon_dim = size

    # 获取查询矩阵中压力层的索引
    pl_query_indices = torch.arange(pl_dim)
    # 获取键矩阵中压力层的索引
    pl_key_indices = -torch.arange(pl_dim) * pl_dim

    # 获取查询矩阵中纬度的索引
    lat_query_indices = torch.arange(lat_dim)
    # 获取键矩阵中纬度的索引
    lat_key_indices = -torch.arange(lat_dim) * lat_dim

    # 获取键值对中的经度索引
    lon_indices = torch.arange(lon_dim)

    # 计算各个维度上的索引组合
    grid_query = torch.stack(torch.meshgrid([pl_query_indices, lat_query_indices, lon_indices]))
    grid_key = torch.stack(torch.meshgrid([pl_key_indices, lat_key_indices, lon_indices]))
    flat_query = torch.flatten(grid_query, 1)
    flat_key = torch.flatten(grid_key, 1)

    # 计算每个维度上的索引差并重新排列
    index_difference = flat_query[:, :, None] - flat_key[:, None, :]
    index_difference = index_difference.permute(1, 2, 0).contiguous()

    # 调整索引以使其从0开始
    index_difference[:, :, 2] += lon_dim - 1
    index_difference[:, :, 1] *= 2 * lon_dim - 1
    index_difference[:, :, 0] *= (2 * lon_dim - 1) * lat_dim * lat_dim

    # 在三个维度上累加索引值
    bias_indices = index_difference.sum(-1)

    return bias_indices
