'''
Author: WANG CHENG
Date: 2024-04-24 00:11:50
LastEditTime: 2024-04-24 00:11:58
'''
import torch


def histogram(input, bins, min_value, max_value):
    # 将输入值的范围标准化到 [0, bins-1] 之间
    normalized = (input - min_value) / (max_value - min_value) * (bins - 1)
    
    # 计算每个桶的索引，然后 + 1 来适应闭区间 [1, bins]
    indices = torch.floor(normalized + 0.5).int() + 1
    
    # 确保索引在 1 到 bins 之间
    indices = torch.clamp(indices, min=1, max=bins)
    
    # 使用scatter_方法来填充直方图的计数
    histogram = torch.zeros(bins, dtype=input.dtype, device=input.device)
    torch.scatter_add_(histogram, 0, indices, torch.ones_like(indices))
    
    return histogram