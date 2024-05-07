import torch
import torch.nn as nn
import torch.nn.functional as F 

def simulation(data, weights):
    # x, w = data, weights
    n,c,h,w = data.shape
    wn,wc,wh,ww=weights.shape
    x=data.reshape(-1,h,w)
    w=weights.reshape(-1,wh,ww)
    outputs = []
    for i in range(x.shape[0]):
        outputs.append(F.linear(x[i], w[i].transpose(1,0)))
    
    return torch.concat(outputs, dim=0).reshape(n,c,-1, ww)

if __name__=="__main__":
    data=torch.randn(4,8,4,16)
    weights=torch.randn(4,8,16,4)
    output = simulation(data, weights)
    diff = output - torch.matmul(data, weights)
    print(torch.sum(diff))