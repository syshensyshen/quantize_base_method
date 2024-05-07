import numpy as np

import torch

qbit = 8
qmin = -2**(qbit - 1) #0  #-2**(qbit - 1)
qmax = 2**(qbit - 1) - 1 #(2**qbit) - 1  #2**(qbit - 1) - 1
qmin = 0
qmax = (2**qbit) - 1
datatype = np.int16
eps = 1.0e-6

def shift_bits(x, shift):
    if shift > 0:
        out = np.left_shift(x, shift)
    else:
        out = np.right_shift(x, -shift)

    return out

def get_scale(data):
    max_val = np.max(data)
    min_val = np.min(data)
    scale = (max_val - min_val) / (qmax - qmin + eps)
    zero_point = np.round(min_val / scale)
    
    # scale = max_val / qmax
    
    return scale, zero_point

def get_shif_scale(si, sk, so, bits=32):
    scale = si * sk / so
    for shift in range(-bits, bits):
        out_scale = scale * (2 ** (-shift))
        if out_scale > 0 and out_scale < 1:
            out_shift = shift
            break
    
    return out_shift, out_scale
        
def quant(data, scale, zero_point):
    qdata = np.round(data / scale) + zero_point
    qdata[qdata < qmin] = qmin
    qdata[qdata > qmax] = qmax
    qdata = datatype(qdata)
    
    return qdata

def dequant(data, so, zero_point):
    fdata = data * so - zero_point
    
    return fdata

if __name__ == "__main__":
    a = np.random.random([1, 32, 40, 40])
    b = np.random.random([1, 32, 40, 40])
    c = a * b
    
    si, zi = get_scale(a)
    sk, zk = get_scale(b)
    so, zo = get_scale(c)
    # zi, zk, zo = 0, 0, 0
    out_shift, out_scale = get_shif_scale(si, sk, so)
    out_scale = np.uint8(out_scale * 2**8)
    print(si, sk, so, out_shift, out_scale, qmin, qmax)
    
    qa = quant(a, si, zi)
    qb = quant(b, sk, zk)
    data = np.int16(qa) * np.int16(qb)
    data = (shift_bits(data, out_shift) * out_scale) >> 8
    qc = dequant(data, so, zo)
    error = np.sum(np.abs(qc - c)) / np.sum(c)
    print(error)