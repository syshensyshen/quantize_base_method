import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from operators.base_quant_op import quant, dequant, get_scale
from operators.similarity import CosineSimilarity

qbit = 8
symquant = True
qmin = -128
qmax = 127
datatype = np.int8

# def quant(data, scale, zeropoint):
#     qdata = np.round(data / scale) + zeropoint

#     qdata[qdata < qmin] = qmin
#     qdata[qdata > qmax] = qmax
#     qdata = datatype(qdata)
    
#     return qdata

# def dequant(data, so, zeropoint):
#     fdata = (data - zeropoint) * so
    
#     return fdata

# def get_scale(data):
#     data = np.array(data).reshape(-1)
#     if symquant:
#         max_val = np.max(np.abs(data))
#         scale = max_val / qmax
#         zeropoint = 0
#     else:
#         max_val, min_val = np.max(data), np.min(data)
#         scale = (max_val - min_val) / (qmax - qmin)
#         zeropoint = qmin - np.round(min_val / scale)
#         zeropoint = np.clip(zeropoint, qmin, qmax)
#         zeropoint = datatype(zeropoint)
        
#     return scale, zeropoint

data = torch.randn(4,64,128,128)
si, _ = get_scale(data.numpy())
qa = quant(data.numpy(), si, zeropoint=0)
out = F.avg_pool2d(data, kernel_size=3, stride=2, padding=1).numpy()
so, _ = get_scale(out)

def split_avg_pool(data):
    qa_avg_pool = F.avg_pool2d(data, kernel_size=3, stride=2, padding=1)
    qa_split_pool_0 = F.avg_pool2d(data, kernel_size=(1,3), stride=(1,2), padding=(0,1))
    qa_split_pool_1 = F.avg_pool2d(qa_split_pool_0, kernel_size=(3,1), stride=(2,1), padding=(1,0))
    # print(torch.sum(torch.abs(qa_avg_pool-qa_split_pool_1)))
    print("cosine error is: ", CosineSimilarity()(qa_avg_pool.numpy(), qa_split_pool_1.numpy()))

def quant_comb(qa, si, so):
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 2, 2
    padding_h, padding_w = 1, 1
    
    hw_out_scale, out_shift, out_scale = 1,0,1
    scale = si * (1 / (kernel_h * kernel_w)) / so
    shit_scale = 2**7-1
    hw_out_scale = np.int32(np.round(shit_scale*scale))
    if hw_out_scale < 1:
        hw_out_scale = np.int32(1)
        out_shift, out_scale = 7, np.int32(np.round(shit_scale*scale*shit_scale))
    
    qa_comb = F.avg_pool2d(torch.from_numpy(qa.astype(np.float32)), 
                           kernel_size=(kernel_h, kernel_w),
                           stride=(stride_h, stride_w),
                           padding=(padding_h, padding_w),
                           divisor_override=1).numpy()
    qa_comb_output = (((qa_comb.astype(np.int32) * hw_out_scale) >> 7) * out_scale) >> out_shift
    
    return qa_comb_output
    

def quant_split(qa, si, so):
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 2, 2
    padding_h, padding_w = 1, 1
    
    scale = si / so
    shit_scale = 2**7-1
    h_scale = np.int16(np.round(shit_scale*scale/kernel_h))
    w_scale = np.int16(np.round(shit_scale/kernel_w))
        
    split_0 = F.avg_pool2d(torch.from_numpy(qa.astype(np.float32)), 
                           kernel_size=(1,kernel_w),
                           stride=(1,stride_w),
                           padding=(0, padding_w),
                           divisor_override=1)
    split_0 = torch.from_numpy(((split_0.numpy().astype(np.int16) * h_scale) >> 7).astype(np.float32))
    split_1 = F.avg_pool2d(split_0, 
                           kernel_size=(kernel_h, 1),
                           stride=(stride_h, 1),
                           padding=(padding_h, 0),
                           divisor_override=1).numpy()
    split_1 = (split_1.astype(np.int16) * w_scale) >> 7
    
    return split_1

split_avg_pool(data)
qa_comb_output = quant_comb(qa, si, so).astype(np.float32)
qa_split_1 = quant_split(qa, si, so).astype(np.float32)

print("cosine error is: ", CosineSimilarity()(dequant(qa_comb_output, so, 0), out))
print("cosine error is: ", CosineSimilarity()(dequant(qa_split_1, so, 0), out))

# print("cosine error is: ", CosineSimiarity()(dequant(qa_max_pool.numpy(), si, 0), data_max_pool.numpy()))