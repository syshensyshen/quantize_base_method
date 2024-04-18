import numpy as np
import copy
from scipy import stats
from matplotlib import pyplot as plt 

import torch

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
    a = np.random.randn(1, 48, 40, 40)
    b = np.random.randn(1, 48, 40, 40)
    c = a * b
    # a = np.random.randn(1, 48, 40, 40) #np.array([-1, 2, 1.2]) #np.random.randn(3, 3)
    # b = np.random.randn(1, 48, 40, 40) #np.array([1, 2, 1.4]) #np.random.randn(3, 3)
    # c = a * b
    
    for func in ["get_scale"]:
        si, zi = eval(func)(a)
        sk, zk = eval(func)(b)
        so, zo = eval(func)(c)
        out_shift, out_scale = get_shif_scale(si, sk, so)
        out_scale = np.uint8(out_scale * 2**qbit)
        
        if symquant:
            qa = quant(a, si, zeropoint=0)
            qb = quant(b, sk, zeropoint=0)
            data = np.int32(qa) * np.int32(qb)
        else:
            qa = quant(a, si, zeropoint=zi)
            qb = quant(b, sk, zeropoint=zk)
            
            data = np.int32(qa) * np.int32(qb)
            fuse_in_zero_point_0 = -qa.astype(np.int32) * np.int32(zk)
            fuse_in_zero_point_1 = -qb.astype(np.int32) * np.int32(zi)
            data = data + fuse_in_zero_point_0 + fuse_in_zero_point_1 + np.int32(zi) * np.int32(zk)
            data_f = data * si * sk

        data = (shift_bits(data, out_shift) * out_scale) >> qbit
        data = np.clip(data + zo, qmin, qmax)
        qc = dequant(data, so, zeropoint=zo)
        error = np.sum(np.abs(qc - c)) / np.sum(np.abs(qc))
        qerror = np.sum(np.abs(c / so + zo - data)) / np.sum(np.abs(c / so + zo))
        print(func, error)
        # print(c)
        # print('\n')
        # print(qc)