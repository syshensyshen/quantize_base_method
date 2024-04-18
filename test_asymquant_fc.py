import nntplib
import numpy as np
import copy
from scipy import stats
from matplotlib import pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F

qbit = 8
symquant = False
if symquant:
    qmin = -128
    qmax = 127
    datatype = np.int8
else:
    qmin = 0
    qmax = 255
    datatype = np.uint8
    
def shift_bits(x, shift):
    if shift > 0:
        out = np.left_shift(x, shift)
    else:
        out = np.right_shift(x, -shift)

    return out

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

def get_shif_scale(si, sk, so, bits=32):
    scale = si * sk / so
    for shift in range(-bits, bits):
        out_scale = scale * (2 ** (-shift))
        if out_scale > 0 and out_scale < 1:
            out_shift = shift
            break
    
    return out_shift, out_scale
        
def quant(data, scale, zeropoint):
    qdata = np.round(data / scale) + zeropoint

    qdata[qdata < qmin] = qmin
    qdata[qdata > qmax] = qmax
    qdata = datatype(qdata)
    
    return qdata

def dequant(data, so, zeropoint):
    fdata = (data - zeropoint) * so
    
    return fdata

if __name__ == "__main__":
    # x = np.random.randn(1, 512)
    # w = np.random.randn(1000, 512)
    x = np.load('./data/fc_input.npy')
    w = np.load('./data/fc_weight.npy')
    fc_bias = np.load('./data/fc_bias.npy')
    
    a = torch.from_numpy(x)
    b = torch.from_numpy(w)
    
    c = F.linear(input=a, weight=b, bias=torch.from_numpy(fc_bias)).numpy()
    # c[c < 0] = 0
    
    for func in ["get_scale"]:
        si, zi = eval(func)(a)
        si = 0.08421591216442632
        zi = 90
        
        sk, zk = eval(func)(b)
        so, zo = eval(func)(c)
        out_shift, out_scale = get_shif_scale(si, sk, so)
        out_scale = np.uint8(out_scale * 2**qbit)
        print(si, sk, so)
        print(zi, zk, zo)
        
        if symquant:
            qa = quant(a, si, zeropoint=0)
            qb = quant(b, sk, zeropoint=0)
            qa_f = torch.from_numpy(qa.astype(np.float32))
            qb_f = torch.from_numpy(qb.astype(np.float32))
            
            data = F.linear(input=qa_f, weight=qb_f, bias=None).numpy().astype(np.int32)
        else:
            qa = quant(a, si, zeropoint=zi) #uint8
            qb = quant(b, sk, zeropoint=zk) #uint8
            
            qa_f = qa.astype(np.float32)
            qb_f = qb.astype(np.float32).transpose(1, 0) # weight
            # zk_f = zk * torch.ones(b.shape, dtype=torch.float32)
            data = np.matmul(qa_f, qb_f) #int8 * int8
            fuse_in_zero_point = -np.sum(qa_f, 1).reshape(-1) * zk
            fuse_w_zero_point = -np.sum(qb_f, 0).reshape(-1) * zi
            data = data + fuse_in_zero_point + fuse_w_zero_point + zi * zk * qa_f.shape[1] + np.int32(fc_bias / (si * sk))
            data = data.astype(np.int32)
            # data[data < 0] = 0
            data_f = data * si * sk
            
        data = (shift_bits(data, out_shift) * out_scale) >> qbit
        data = np.clip(data + zo, qmin, qmax)
        qc = dequant(data, so, zeropoint=zo)
        error = np.sum(np.abs(qc - c)) / np.sum(np.abs(c))
        qerror = np.sum(np.abs(c / so + zo - data)) / np.sum(np.abs(c / so + zo))
        print(func, error)
        # print(c)
        # print('\n')
        # print(qc)