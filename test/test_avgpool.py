'''
Author: WANG CHENG
Date: 2024-04-24 20:36:59
LastEditTime: 2024-04-27 21:02:52
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from operators.base_quant_op import quant, dequant, get_scale
from operators.pooling_op import split_avg_pool, quant_avgpool_comb, quant_avgpool_split
from operators.similarity_op import CosineSimilarity


qbit = 8
symquant = True
qmin = -128
qmax = 127
datatype = np.int8

data = torch.randn(4,64,128,128)
si, _ = get_scale(data.numpy())
qa = quant(data.numpy(), si, zeropoint=0)
out = F.avg_pool2d(data, kernel_size=3, stride=2, padding=1).numpy()    # torch.Size([4, 64, 64, 64])
so, _ = get_scale(out)

qa_split_pool_1 = split_avg_pool(data)
print("cosine error is: ", CosineSimilarity()(out, qa_split_pool_1.numpy()))

qa_comb_output = quant_avgpool_comb(qa, si, so).astype(np.float32)
print("cosine error is: ", CosineSimilarity()(dequant(qa_comb_output, so, 0), out))

qa_split_1 = quant_avgpool_split(qa, si, so).astype(np.float32)
print("cosine error is: ", CosineSimilarity()(dequant(qa_split_1, so, 0), out))

# print("cosine error is: ", CosineSimilarity()(dequant(qa_max_pool.numpy(), si, 0), data_max_pool.numpy()))