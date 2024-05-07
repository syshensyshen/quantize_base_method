
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

qbit = 8
symquant = False
qmin = -128
qmax = 127
datatype = np.int8
# if symquant:
#     qmin = -128
#     qmax = 127
#     datatype = np.int8
# else:
#     qmin = 0
#     qmax = 255
#     datatype = np.uint8
    
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
    # x = np.random.randn(1, 64, 224, 224)
    # w = np.random.randn(256, 64, 3, 3)
    # a = torch.from_numpy(x)
    # b = torch.from_numpy(w)
    x = np.load('./data/input.npy')
    w = np.load('./data/weight.npy')
    a = torch.from_numpy(x)
    b = torch.from_numpy(w)
    
    pads = (3, 3, 3, 3)
    padding = nn.ZeroPad2d(pads)
    a = padding(a)
    c = F.conv2d(input=a, weight=b, bias=None, stride=(2, 2),
                padding=(0, 0), dilation=(1, 1), groups=1).numpy()
    # c[c<0] = 0
    
    for func in ["get_scale"]:
        si, zi = eval(func)(a)
        sk, zk = eval(func)(b)
        so, zo = eval(func)(c)
        out_shift, out_scale = get_shif_scale(si, sk, so)
        out_scale = np.uint8(out_scale * 2**qbit)
        
        if symquant:
            qa = quant(a, si, zeropoint=0)
            qb = quant(b, sk, zeropoint=0)
            qa_f = torch.from_numpy(qa.astype(np.int32))
            qb_f = torch.from_numpy(qb.astype(np.int32))
            
            data = F.conv2d(input=qa_f, weight=qb_f, bias=None, stride=(2, 2),
                padding=(0, 0), dilation=(1, 1), groups=1).numpy().astype(np.int32)
        else:
            qa = quant(a, si, zeropoint=zi) #uint8
            qb = quant(b, sk, zeropoint=zk) #uint8
            
            qa_f = torch.from_numpy(qa.astype(np.float32))
            qb_f = torch.from_numpy(qb.astype(np.float32))
            # weight = qb_f-zk
            # weight = torch.clamp(weight, -128, 127)
            # zero point is calculated separately
            '''
            data = F.conv2d(input=qa_f, weight=qb_f, bias=None, stride=(2, 2),
                padding=(0, 0), dilation=(1, 1), groups=1).numpy().astype(np.int32) #int8 * int8

            fuse_in_zero_point = -torch.sum(qb_f, dim=(1,2,3)).numpy().astype(np.int32) * zi
            fuse_zi_zk = np.int32(zi) * np.int32(zk) * qb_f.shape[1] * qb_f.shape[2] * qb_f.shape[3]
            fuse_in_zero_point = fuse_in_zero_point.reshape(1, -1, 1, 1)
            zk_f = zk * torch.ones_like(qb_f)
            fuse_w_zero_point = -F.conv2d(input=qa_f, weight=zk_f, bias=None, stride=(2, 2), padding=(0, 0), dilation=(1, 1), groups=1).numpy().astype(np.int32) #bias
            data = data + fuse_in_zero_point + fuse_w_zero_point + fuse_zi_zk
            data = data.astype(np.int32)
            compare_v = np.int32(zi) * np.int32(zk)
            '''
            # zero point is calculated combination
            data = F.conv2d(input=qa_f-zi, weight=qb_f-zk, bias=None, stride=(2, 2),
                            padding=(0, 0), dilation=(1, 1), groups=1).numpy().astype(np.int32)  # int8 * int8
            # data[data < 0] = 0
            out_f = data * si * sk
                
        data = (shift_bits(data, out_shift) * out_scale) >> qbit
        data = np.clip(data + zo, qmin, qmax)
        qc = dequant(data, so, zeropoint=zo)
        error = np.sum(np.abs(qc - c)) / np.sum(np.abs(c))
        qerror = np.sum(np.abs(quant(c, so, zo) - data)) / np.sum(np.abs(quant(c, so, zo)))
        print(func, error)
        # print(c)
        # print('\n')
        # print(qc)