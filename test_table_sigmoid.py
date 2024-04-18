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

def get_scale(data, qmax):
    max_val = np.max(np.abs(data))
    
    return max_val / qmax


def quant(data, scale, datatype):
    qdata = np.round(data / scale)
    qdata[qdata < qmin] = qmin
    qdata[qdata > qmax] = qmax
    qdata = datatype(qdata)
    
    return qdata

def dequant(data, so):
    fdata = data * so
    
    return fdata

def sigmoid(data):
    data = torch.from_numpy(data)
    data = torch.sigmoid(data)
    return data.numpy()

def get_sigmoid_table(qmax, qmin, datatype, si=None, so=None):
    ####, in_shift=None, out_shift=None, quant_float=None
    if 1:
        table = np.arange(qmin, qmax + 1) * si
        table = np.array(table).astype(np.float64)
        table = torch.sigmoid(torch.from_numpy(table)).numpy()
        table = np.round(table / so)
        table[table < qmin] = qmin
        table[table > qmax] = qmax
        table = datatype(table)
    # else:
    #     table = np.arange(qmin, qmax + 1) / (2 ** in_shift)
    #     table = np.array(table).astype(np.float64)
    #     table = torch.sigmoid(torch.from_numpy(table)).numpy()
    #     table = np.round(table * (2 ** out_shift))
    #     table[table < qmin] = qmin
    #     table[table > qmax] = qmax
    #     table = datatype(table)
    
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

def desigmoid(x):
    return np.log(x / (1 - x))

if __name__ == "__main__":
    for idx in [0]:
        # a = np.load("/home/ubuntu/zhangcc/code/2022/onnx-converter/score_{}.npy".format(idx))
        # a = desigmoid(a)
        
        a = np.random.randn(10, 48, 40, 40)
        weights = np.random.randn(48, 48, 3, 3)
        a = F.conv2d(torch.from_numpy(a), torch.from_numpy(weights), stride=1, padding=1).numpy()
        c = sigmoid(a)

        for qbit in [8]:
            error_list = []
            qa_list = []
            in_shifts = []
            out_shifts = []
            for quant_float in [False, True]:
                qmin = -2**(qbit - 1)
                qmax = 2**(qbit - 1) - 1
                datatype = eval("np.int{}".format(qbit))

                if quant_float:
                    si = get_scale(a, qmax=qmax)
                    so = get_scale(c, qmax=qmax)
                    # in_shift = getBestShift(si)
                    # out_shift = getBestShift(so)
                    # si = 1.0 / (2 ** in_shift)
                    # so = 1.0 / (2 ** out_shift)
                else:    
                    in_shift = _getBestShift(a, d_size=qbit)
                    out_shift = _getBestShift(c, d_size=qbit)
                    si = 1.0 / (2 ** in_shift)
                    so = 1.0 / (2 ** out_shift)
                
                qa = quant(a, si, datatype=datatype)#####int8
                
                table = get_sigmoid_table(qmax, qmin, datatype, si=si, so=so)
                
                data = copy.deepcopy(qa)
                for i, tb in enumerate(table):
                    data[(qa.astype(np.int64) - np.int64(qmin)) == i] = tb
                            
                qc = dequant(data, so)

                error = np.sum(np.abs(qc - c)) / np.sum(np.abs(c))
                max_diff = np.max(np.abs(qc - c))
                min_diff = np.min(np.abs(qc - c))
                print("quant_float:", quant_float, "score:", idx, "np.int" + str(qbit), error, max_diff, min_diff, out_shift, in_shift)