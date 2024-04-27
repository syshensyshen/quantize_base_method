'''
Author: WANG CHENG
Date: 2024-04-27 20:56:02
LastEditTime: 2024-04-27 21:38:49
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

def split_avg_pool(data):   # torch.Size([4, 64, 128, 128])
    qa_split_pool_0 = F.avg_pool2d(data, kernel_size=(1,3), stride=(1,2), padding=(0,1))    # torch.Size([4, 64, 128, 64])
    qa_split_pool_1 = F.avg_pool2d(qa_split_pool_0, kernel_size=(3,1), stride=(2,1), padding=(1,0)) # torch.Size([4, 64, 64, 64])
    # print(torch.sum(torch.abs(qa_avg_pool-qa_split_pool_1)))
    return qa_split_pool_1
    
def quant_comb(qa, si, so):
    kernel_h, kernel_w = 3, 3
    stride_h, stride_w = 2, 2
    padding_h, padding_w = 1, 1
    
    hw_out_scale, out_shift, out_scale = 1,0,1
    scale = si * (1 / (kernel_h * kernel_w)) / so
    shift_scale = 2**7-1
    hw_out_scale = np.int32(np.round(shift_scale*scale))
    if hw_out_scale < 1:
        hw_out_scale = np.int32(1)
        out_shift, out_scale = 7, np.int32(np.round(shift_scale*scale*shift_scale))
    
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
    shift_scale = 2**7-1
    h_scale = np.int16(np.round(shift_scale*scale/kernel_h))
    w_scale = np.int16(np.round(shift_scale/kernel_w))
        
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