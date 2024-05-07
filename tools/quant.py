import numpy as np

qbit = 8
symquant = True
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

def get_scale(data, symquant=True):
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

def conv_w_split(data_pad_out):
    H, W, C_in, C_out = data_pad_out.shape
    Csize, Ksize = 8, 8
    # 0: H, 1: W, 2: C//self.Csize, 3: self.Csize, 4: N//Ksize, 5: Ksize
    new_data = data_pad_out.reshape(H, W, C_in//Csize, Csize, C_out//Ksize, Ksize)
    new_data = np.transpose(new_data, (4,2,0,1,5,3))