import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from error import CosineSimiarity

qbit = 8
symquant = True
qmin = -128
qmax = 127
datatype = np.int8

def quant(data, scale, zeropoint):
    qdata = np.round(data / scale) + zeropoint

    qdata[qdata < qmin] = qmin
    qdata[qdata > qmax] = qmax
    qdata = datatype(qdata)
    
    return qdata

def dequant(data, so, zeropoint):
    fdata = (data - zeropoint) * so
    
    return fdata

def get_scale(data):
    data = np.array(data).reshape(-1)
    if symquant:
        max_val = np.max(np.abs(data))
        scale = max_val / qmax
        zeropoint = 0
    else:
        max_val, min_val = np.max(data), np.min(data)
        scale = (max_val - min_val) / (qmax - qmin)
        zeropoint = qmin - np.round(min_val / scale)
        zeropoint = np.clip(zeropoint, qmin, qmax)
        zeropoint = datatype(zeropoint)
        
    return scale, zeropoint

data = torch.randn(4,64,128,128)
si, _ = get_scale(data.numpy())
qa = quant(data.numpy(), si, zeropoint=0)

qa_max_pool = F.max_pool2d(torch.from_numpy(qa), kernel_size=3, stride=2, padding=1)
data_max_pool = F.max_pool2d(data, kernel_size=3, stride=2, padding=1)

qa_split_0 = F.max_pool2d(torch.from_numpy(qa), kernel_size=(1,3), stride=(1,2), padding=(0,1))
qa_split_1 = F.max_pool2d(qa_split_0, kernel_size=(3,1), stride=(2,1), padding=(1,0))

print(torch.sum(torch.abs(qa_max_pool-qa_split_1)))


print("cosine error is: ", CosineSimiarity()(dequant(qa_max_pool.numpy(), si, 0), data_max_pool.numpy()))