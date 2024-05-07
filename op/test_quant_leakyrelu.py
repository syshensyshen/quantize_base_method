import numpy as np

import torch
import torch.nn as nn 
import torch.nn.functional as F

qbit = 16
qmin = -2**(qbit - 1)
qmax = 2**(qbit - 1) - 1
datatype = np.int16

def shift_bits(x, shift):
    if shift > 0:
        out = np.left_shift(x, shift)
    else:
        out = np.right_shift(x, -shift)

    return out

def get_scale(data):
    max_val = np.max(data)
    
    return max_val / qmax

def get_shif_scale(si, sk, so, bits=32):
    scale = si * sk / so
    for shift in range(-bits, bits):
        out_scale = scale * (2 ** (-shift))
        if out_scale > 0 and out_scale < 1:
            out_shift = shift
            break
    
    return out_shift, out_scale
        
def quant(data, scale):
    qdata = data / scale
    qdata[qdata < qmin] = qmin
    qdata[qdata > qmax] = qmax
    qdata = datatype(qdata)
    
    return qdata

def dequant(data, so):
    fdata = data * so
    
    return fdata

def leaky_relu(data, alpha):
    data = torch.from_numpy(data)
    # data = F.leaky_relu(data, negative_slope=float(alpha))
    data = F.relu(data)
    return data.numpy()

    
if __name__ == "__main__":
    a = np.random.random([1, 32, 40, 40])
    b = np.zeros(a.shape)
    b[a > 0] = 1.0
    b[a < 0] = 0.01
    
    c = leaky_relu(a, b)
    si = get_scale(a)
    sk = get_scale(b)
    so = get_scale(c)
    out_shift, out_scale = get_shif_scale(si, sk, so)
    out_scale = np.uint8(out_scale * 2**8)
    print(si, sk, so, out_shift, out_scale, qmin, qmax)
    
    qa = quant(a, si)
    qb = quant(b, sk)
    data = np.int32(qa) * np.int32(qb)
    data = (shift_bits(data, out_shift) * out_scale) >> 8
    qc = dequant(data, so)
    error = np.sum(np.abs(qc - c)) / np.sum(c)
    print(error)