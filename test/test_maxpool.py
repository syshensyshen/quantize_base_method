'''
Author: WANG CHENG
Date: 2024-04-24 20:36:59
LastEditTime: 2024-04-29 01:09:10
'''
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

from operators.base_quant_op import quant, dequant, get_scale
from operators.similarity_op import CosineSimilarity

qbit = 8
symquant = True
qmin = -128
qmax = 127
datatype = np.int8


data = torch.randn(4,64,128,128)
si, _ = get_scale(data.numpy())
qa = quant(data.numpy(), si, zeropoint=0)

data_max_pool = F.max_pool2d(data, kernel_size=3, stride=2, padding=1)  # float32

qa_max_pool = F.max_pool2d(torch.from_numpy(qa), kernel_size=3, stride=2, padding=1)    # int8
qa_split_0 = F.max_pool2d(torch.from_numpy(qa), kernel_size=(1,3), stride=(1,2), padding=(0,1))
qa_split_1 = F.max_pool2d(qa_split_0, kernel_size=(3,1), stride=(2,1), padding=(1,0))

print(torch.sum(torch.abs(qa_max_pool-qa_split_1))) # combine - split

print("cosine error is: ", CosineSimilarity()(dequant(qa_max_pool.numpy(), si, 0), data_max_pool.numpy()))