import numpy as np
import copy
import math

import torch
import torch.nn as nn 
import torch.nn.functional as F


def shift_bits(x, shift):
    if shift > 0:
        out = np.left_shift(x, shift)
    else:
        out = np.right_shift(x, -shift)

    return out

def get_scale(datax, qmax, mode="neg"):
    data = copy.deepcopy(datax)
    if mode == "neg":
        data[data>0] = 0
        max_val = np.abs(np.max(-data))
    elif mode == "pos":
        data[data<0] = 0
        max_val = np.max(data)
    elif mode == "all":
        max_val = np.max(np.abs(data))
    else:
        assert "mode is not supported!!!!"
        
    return max_val / qmax

def get_shift_scale(si, sk, so, bits=32):
    scale = si * sk / so
    for shift in range(-bits, bits):
        out_scale = scale * (2 ** (-shift))
        if out_scale > 0 and out_scale < 1:
            out_shift = shift
            break
    
    return out_shift, out_scale
        
def quant(data, scale, qmax, qmin, datatype):
    if isinstance(data, float):
        qdata = np.round(data / scale)
        if qdata < qmin:
            qdata = qmin
        elif qdata > qmax:
            qdata = qmax
    else:
        # qdata = np.round(data / sin) * (data < 0).astype(np.int) + np.round(data / sip) * (data >= 0).astype(np.int)
        qdata = np.round(data / scale)
        qdata[qdata < qmin] = qmin
        qdata[qdata > qmax] = qmax
    
    return datatype(qdata)

def dequant(data, so):
    # fdata = data * so_neg * (data < 0).astype(np.int) + data * so_pos * (data >= 0).astype(np.int)
    fdata = data * so
    
    return fdata

def leaky_relu(data, alpha):
    data = torch.from_numpy(data)
    data = F.leaky_relu(data, negative_slope=float(alpha))
    return data.numpy()

def get_leakyrelu_table(qalpha, qmax, qmin):
    table = []
    for i in range(qmin, qmax + 1):
        if i >= 0:
            tmp = np.int64(i)
        else:
            tmp = np.int64(i) * np.int64(qalpha)
        table.append(tmp)
    
    return table

def _getBaseShift(data, d_size=8):
    '''get the x that make all the numbers in data can be presented by data*2**x in [-128, 127],
    x can be larger than 7'''
    # aa=np.asarray(data)
    # maxw_=max(aa.flat)
    # minw_=min(aa.flat)
    # val=max(math.fabs(maxw_), math.fabs(minw_))
    val = np.max(np.abs(data))
    if val == 0:
        return 0
    
    if d_size-1-math.log(val,2) > 0:
        val=int(d_size-1-math.log(val,2))
    else:
        val=int(d_size-1-math.log(val,2)-1)
    return val

def _getQData(data, offset=None, d_size=8):
    '''data must be a np array'''
    if offset > 0:
        out=data*(1<<offset)
    else:
        out=np.right_shift(data.astype(np.int32), -offset) #(data/(1<<(-offset)))
    
    d_low = -(2**(d_size - 1))
    d_high = (2**(d_size - 1)) - 1
    
    out[out>d_high]=d_high
    out[out<d_low]=d_low
    if d_size == 8:
        out=out.astype(np.int8)
    elif d_size == 16:
        out=out.astype(np.int16)
    else:
        out=out.astype(np.int32)
    return out

def kl_divergence(P, Q, eps=1.0e-6):
    assert P.shape[0] == Q.shape[0]
    
    N = P.shape[0]
    
    KL = 0
    for i in range(N):###P是真实分布;Q是P的拟合分布
        KL += P[i] * np.log((P[i] + eps) / (Q[i] + eps))
        # KL += Q[i] * np.log((Q[i] + eps) / (P[i] + eps))

    return KL
   
def _getBestShift(data, margin=5, d_size=8, bins=2048):
    # hist_p = np.histogram(data, bins=bins)[0]
    # hist_p = hist_p / np.sum(hist_p)
    
    sum=[]
    init_shift=_getBaseShift(data, d_size=d_size)
    for shift in range(init_shift-margin, init_shift+margin+1):
        qdata = _getQData(data, offset=shift, d_size=d_size).astype(np.float64)/(2**shift)

        # hist_q = np.histogram(qdata, bins=bins)[0]
        # hist_q = hist_q / np.sum(hist_q)
        # err = kl_divergence(hist_p, hist_q)
        
        err=abs(data-qdata).sum()
        sum.append(err)

    idx=np.argmin(sum)+init_shift-margin 
    #print('init_shift: ', init_shift, ' idx: ', idx, ' sum error: ', sum)
    return  idx 

def getBaseShift(scale):
    base_int = np.int64(1.0 / scale + 0.5)
    for shift in range(0, 32):
        if (base_int >> shift) == 0:
            return shift

def getBestShift(scale, margin=5):
    init_shift = getBaseShift(scale)
    
    sum=[]
    for shift in range(init_shift-margin, init_shift+margin+1):
        error = np.abs(scale - 1.0 / (2**shift))
        sum.append(error)
    
    idx=np.argmin(sum)+init_shift-margin 
    
    return idx

if __name__ == "__main__":
    a = np.random.randn(1, 48, 40, 40)
    # a = np.random.normal(loc=2.0, scale=4.0, size=[1, 48, 40, 40])
    # a = -np.abs(a)
    # a = np.load("/home/ubuntu/zhangcc/code/2022/onnx-converter/leakyrelu.npy")
    # a = np.abs(a)
    
    is_shift = False
    for alpha_ in range(-5, 0):
        for qbit in [8]:
            for quant_float in [False, True]:
                qmin = -2**(qbit - 1)
                qmax = 2**(qbit - 1) - 1
                datatype = eval("np.int{}".format(qbit))
                
                alpha = math.pow(10, alpha_) ### pytorch leakyrelu alpha
                c = leaky_relu(a, alpha)
                
                sk = get_scale(alpha, qmax, mode="all")
                if quant_float:
                    si = get_scale(a, qmax, mode="all")
                    so = get_scale(c, qmax, mode="all")
                    # in_shift = getBestShift(si)
                    # out_shift = getBestShift(so)
                    # si = 1.0 / (2 ** in_shift)
                    # so = 1.0 / (2 ** out_shift)
                else:
                    in_shift = _getBestShift(a, d_size=qbit)
                    out_shift = _getBestShift(c, d_size=qbit)
                    si = 1.0 / (2 ** in_shift)
                    so = 1.0 / (2 ** out_shift)
                    
                qa = quant(a, scale=si, qmax=qmax, qmin=qmin, datatype=datatype)
                    
                qalpha = quant(alpha, scale=sk, qmax=qmax, qmin=qmin, datatype=datatype) #np.int(alpha * qmax) #quant(alpha, sk, qmax=qmax, qmin=qmin, datatype=datatype)
                table = get_leakyrelu_table(qalpha=qalpha, qmax=qmax, qmin=qmin)

                if is_shift:            
                    out_shift_neg, out_scale_neg = get_shift_scale(si, sk, so, bits=64)
                    out_scale_neg = eval("np.uint" + str(qbit))(out_scale_neg * (2**qbit))
                    out_shift_pos, out_scale_pos = get_shift_scale(si, 1, so, bits=64)
                    out_scale_pos = eval("np.uint" + str(qbit))(out_scale_pos * (2**qbit))
                
                data = np.zeros(qa.shape)
                for i, tb in enumerate(table):
                    if i >= (2 ** (qbit - 1)):
                        if is_shift:
                            tmp = (shift_bits(tb, out_shift_pos) * out_scale_pos) >> qbit
                        else:
                            tmp = np.round(tb * si / so)
                    else:
                        if is_shift:
                            tmp = (shift_bits(tb, out_shift_neg) * out_scale_neg) >> qbit
                        else:
                            tmp = np.round(tb * si * sk / so)
                    if tmp < qmin:
                        tmp = qmin
                    elif tmp > qmax:
                        tmp = qmax
                    data[(qa.astype(np.int64) - np.int64(qmin)) == i] = datatype(tmp)
                        
                qc = dequant(data, so)
                error = np.sum(np.abs(qc - c)) / np.sum(np.abs(c))
                max_diff = np.max(np.abs(qc - c))
                min_diff = np.min(np.abs(qc - c))
                print("quant_float:", quant_float, "np.int" + str(qbit), alpha, np.int(alpha * qmax), error, max_diff, min_diff)