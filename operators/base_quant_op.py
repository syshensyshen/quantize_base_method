import numpy as np
import torch


__all__ = [
    "bit8_absmax_quantize",
    "bit8_zeropoint_quantize",
    "get_scale",
    "get_shif_scale",
    "quant",
    "dequant"
]

# qbit = 8
# symquant = True
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

def bit8_absmax_quantize(data):
    """absolute maximum (absmax) quantization"""
    qmin = 0
    qmax = 255
    datatype = np.uint8
    max_val, min_val = np.max(data), np.min(data)
    scale = (max_val - min_val) / (qmax - qmin)
    zeropoint = qmin - np.round(min_val / scale)
    zeropoint = np.clip(zeropoint, qmin, qmax)
    zeropoint = datatype(zeropoint)
    return zeropoint, scale

def bit8_zeropoint_quantize(data):
    """zero-point quantization"""
    qmin = 0
    qmax = 255
    datatype = np.uint8
    max_val, min_val = np.max(data), np.min(data)
    scale = (max_val - min_val) / (qmax - qmin)
    zeropoint = qmin - np.round(min_val / scale)
    zeropoint = np.clip(zeropoint, qmin, qmax)
    zeropoint = datatype(zeropoint)
    return zeropoint, scale


def get_scale(data, symquant=True):
    if isinstance(data, torch.Tensor):
        if symquant:
            qmin = -128
            qmax = 127
            datatype = torch.int8
            max_val = torch.max(torch.abs(data))
            scale = max_val / qmax
            zeropoint = 0
        else:
            qmin = 0
            qmax = 255
            datatype = torch.uint8
            max_val, min_val = torch.max(data), torch.min(data)
            scale = (max_val - min_val) / (qmax - qmin)
            zeropoint = qmin - torch.round(min_val / scale)
            zeropoint = torch.clamp(zeropoint, qmin, qmax).type(datatype)
            
        return scale.type(torch.float32), zeropoint
    else:
        data = np.array(data).reshape(-1)
        if symquant:
            qmin = -128
            qmax = 127
            datatype = np.int8
            max_val = np.max(np.abs(data))
            scale = max_val / qmax
            zeropoint = 0
        else:
            qmin = 0
            qmax = 255
            datatype = np.uint8
            max_val, min_val = np.max(data), np.min(data)
            scale = (max_val - min_val) / (qmax - qmin)
            zeropoint = qmin - np.round(min_val / scale)
            zeropoint = np.clip(zeropoint, qmin, qmax)
            zeropoint = datatype(zeropoint)
            
        return np.float32(scale), zeropoint

def get_shif_scale(si, sk, so, bits=32):
    scale = si * sk / so
    for shift in range(-bits, bits):
        out_scale = scale * (2 ** (-shift))
        if out_scale > 0 and out_scale < 1:
            out_shift = shift
            break
    
    return out_shift, out_scale
        
def quant(data, scale, zeropoint, symquant=True):
    if isinstance(data, torch.Tensor):
        if symquant:
            qmin = -128
            qmax = 127
            datatype = torch.int8
        else:
            qmin = 0
            qmax = 255
            datatype = torch.uint8
        qdata = torch.round(data / scale) + zeropoint
        qdata = torch.clamp(qdata, qmin, qmax).type(datatype)
        return qdata
    else:
        if symquant:
            qmin = -128
            qmax = 127
            datatype = np.int8
        else:
            qmin = 0
            qmax = 255
            datatype = np.uint8
        qdata = np.round(data / scale) + zeropoint

        qdata[qdata < qmin] = qmin
        qdata[qdata > qmax] = qmax  
        qdata = datatype(qdata)
        
        return qdata

def dequant(data, so, zeropoint):
    fdata = (data - zeropoint) * so
    
    return fdata