'''
Author: WANG CHENG
Date: 2024-04-24 20:36:59
LastEditTime: 2024-04-29 01:19:27
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 

def simulation(data, weights):
    # x, w = data, weights
    n,c,h,w = data.shape    # torch.Size([4, 8, 4, 16])
    wn,wc,wh,ww=weights.shape   # torch.Size([4, 8, 16, 4])
    x=data.reshape(-1,h,w)
    w=weights.reshape(-1,wh,ww)
    outputs = []
    for i in range(x.shape[0]): # 32
        outputs.append(F.linear(x[i], w[i].transpose(1,0))) # 模拟线性层，单个矩阵
    # torch.Size([32, 4, 4]) outputs
    return torch.concat(outputs, dim=0).reshape(n,c,-1, ww) 

if __name__=="__main__":
    data=torch.randn(4,8,4,16)
    weights=torch.randn(4,8,16,4)
    output = simulation(data, weights)
    diff = output - torch.matmul(data, weights)
    print(torch.sum(diff))