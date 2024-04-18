# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : SHIQING TECH
# @Time     : 2022/3/28 15:37
# @File     : test_perchannel_conv.py
import copy

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

process_shift = lambda data, shift: data >> shift if shift >= 0 else np.right_shift(data, -shift)


def shift_bits(inputs, shifts):
    if isinstance(shifts, np.ndarray):
        for idx, shift in enumerate(shifts):
            inputs[:, idx] = process_shift(inputs[:, idx], shift)
    else:
        inputs = process_shift(inputs, shifts)

    return inputs


def scale_data(inputs, scales):
    if isinstance(scales, np.ndarray):
        for idx, scale in enumerate(scales):
            inputs[:, idx] = inputs[:, idx] * scale
    else:
        inputs = inputs * scales

    return inputs


def get_scale(in_data, key='perchannel', symquant=True):
    if isinstance(in_data, torch.Tensor):
        data = in_data.numpy()
    else:
        import copy
        data = copy.deepcopy(in_data)
    if symquant:
        if key == 'perchannel':
            out_c, in_c = data.shape[:2]
            data_ = data.reshape(out_c, -1)
            dmax, dmin = np.max(data_, axis=1), np.min(data_, axis=1)
            # select = np.abs(dmax) - np.abs(dmin)
            max_val = np.max(np.column_stack([np.abs(dmax), np.abs(dmin)]), axis=1)
            zeropoint = np.zeros_like(max_val)
        else:
            data = np.array(data).reshape(-1)
            max_val = np.max(np.abs(data))
            zeropoint = 0
        scale = max_val / qmax

    else:
        if key == 'perchannel':
            out_c, int_c = data.shape[:2]
            data_ = data.reshape(out_c, -1)
            max_val, min_val = np.max(data_, axis=1), np.min(data_, axis=1)
        else:
            data = np.array(data).reshape(-1)
            max_val, min_val = np.max(data), np.min(data)
        scale = (max_val - min_val) / (qmax - qmin)
        zeropoint = qmin - np.round(min_val / scale)
        zeropoint = np.clip(zeropoint, qmin, qmax)
        zeropoint = datatype(zeropoint)

    return scale, zeropoint


def get_shif_scale(si, sk, so, bits=32, lower=0.5):
    scale = si * sk / so
    if isinstance(scale, np.ndarray):
        shifts, scales = np.zeros_like(scale, dtype=np.int32), np.zeros_like(scale)
        for idx, s in enumerate(scale.reshape(-1)):
            for shift in range(-bits, bits):
                out_scale = s * (2 ** (-shift))
                if lower < out_scale < 1:
                    shifts[idx] = shift
                    scales[idx] = out_scale
                    break
        return shifts, scales
    else:
        for shift in range(-bits, bits):
            out_scale = scale * (2 ** (-shift))
            if lower < out_scale < 1:
                return np.int32(shift), out_scale
        print('Error! Can not get the shift for scale %f' % scale)
        exit(-1)


def quant(data, in_scale, zeropoint):
    if isinstance(in_scale, np.ndarray):
        scale = in_scale.reshape(-1, 1, 1, 1)
        zeropoint = zeropoint.reshape(-1, 1, 1, 1)
        if isinstance(data, torch.Tensor):
            scale = torch.from_numpy(scale)
    else:
        scale = copy.deepcopy(in_scale)

    qdata = data / scale + zeropoint

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
    # x = np.load('./data/input.npy')
    # w = np.load('./data/weight.npy')
    x = np.load('./data/perchannel_in_data.npy')
    w = np.load('./data/perchannel_weights.npy')
    bias = np.load('./data/perchannel_bias.npy')
    out = np.load('./data/perchannel_out_data.npy')
    a = torch.from_numpy(x)
    b = torch.from_numpy(w)
    # bias = torch.from_numpy(bias)

    pads = (1, 1, 1, 1)
    padding = nn.ZeroPad2d(pads)
    a = padding(a)
    c = F.conv2d(input=a, weight=b, bias=torch.from_numpy(bias), stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=b.shape[0]).numpy()
    c[c<0] = 0

    for func in ["get_scale"]:
        si, zi = eval(func)(a, key='', symquant=False)
        sk, zk = eval(func)(b, key='perchannel')
        # sk, zk = eval(func)(b, key='')
        so, zo = eval(func)(c, key='', symquant=False)
        out_shift, out_scale = get_shif_scale(si, sk, so)
        out_scale = np.uint8(out_scale * 2 ** qbit)

        if symquant:
            qa = quant(a, si, zeropoint=zi)
            qb = quant(b, sk, zeropoint=zk)
            qa_f = torch.from_numpy(qa.astype(np.int32))
            qb_f = torch.from_numpy(qb.astype(np.int32))

        else:
            qa = quant(a.numpy(), si, zeropoint=zi)  # uint8
            qb = quant(b.numpy(), sk, zeropoint=zk)  # uint8

            qa_f = torch.from_numpy(qa.astype(np.float32) - zi)
            if isinstance(sk, np.ndarray):
                qb_f = torch.from_numpy(qb.astype(np.float32) - zk.reshape(-1,1,1,1))
            else:
                qb_f = torch.from_numpy(qb.astype(np.float32) - zk)
            # weight = qb_f-zk
            # weight = torch.clamp(weight, -128, 127)
            # zero point is calculated separately

            # zero point is calculated combination
        data = F.conv2d(input=qa_f, weight=qb_f , bias=None, stride=(1, 1),
                        padding=(0, 0), dilation=(1, 1), groups=qb_f.shape[0]).numpy().astype(np.int32)  # int8 * int8
        data = data + (bias / si / sk).reshape(1, -1, 1, 1).astype(np.int32)
        data[data < 0] = 0
        if isinstance(sk, np.ndarray):
            out_f = data * si * sk.reshape(-1, 1, 1, 1)
        else:
            out_f = data * si * sk

        data = scale_data(shift_bits(data, out_shift), out_scale) >> qbit
        data = np.clip(data + zo, qmin, qmax)
        qc = dequant(data, so, zeropoint=zo)
        # true = quant(out, so, zo)
        error = np.sum(np.abs(qc - c)) / np.sum(np.abs(c))
        error1 = np.sum(np.abs(out - c)) / np.sum(np.abs(c))
        qerror = np.sum(np.abs(quant(c, so, zo) - data)) / np.sum(np.abs(quant(c, so, zo)))
        print(func, error)
        # print(c)
        # print('\n')
        # print(qc)