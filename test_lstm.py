# Copyright (c) shiqing. All rights reserved.
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author   : shengyuan.shen
# @Company  : TIMESINETLLI TECH
# @Time     : 2022/5/25 13:49
# @File     : test_lstm.py

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_operations import test_lstm
from quant import get_scale, quant, dequant


class DLSTM(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.num_directions = 1
        self.batch_size = args.batch_size
        self.lstm0 = nn.LSTMCell(args.input_size, hidden_size=128)
        self.lstm1 = nn.LSTMCell(input_size=128, hidden_size=32)
        self.dropout = nn.Dropout(p=0.4)
        self.linear = nn.Linear(32, self.output_size)
        self.device = args.get("device", "cpu")

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        # batch_size, hidden_size
        h_l0 = torch.zeros(batch_size, 128).to(self.device)
        c_l0 = torch.zeros(batch_size, 128).to(self.device)
        h_l1 = torch.zeros(batch_size, 32).to(self.device)
        c_l1 = torch.zeros(batch_size, 32).to(self.device)
        output = []
        for t in range(seq_len):
            h_l0, c_l0 = self.lstm0(input_seq[:, t, :], (h_l0, c_l0))
            h_l0, c_l0 = self.dropout(h_l0), self.dropout(c_l0)
            h_l1, c_l1 = self.lstm1(h_l0, (h_l1, c_l1))
            h_l1, c_l1 = self.dropout(h_l1), self.dropout(c_l1)
            output.append(h_l1)

        pred = self.linear(output[-1])

        return pred


def calc_lstm(x, h, c, wi, wo, wf, wc, ri, ro, rf, rc, wbi, wbo, wbf, wbc, rbi, rbo, rbf, rbc, pi):
    wi, ri = torch.permute(wi, 2, 1), torch.permute(ri, 2, 1)
    wo, ro = torch.permute(wo, 2, 1), torch.permute(ro, 2, 1)
    wf, rf = torch.permute(wf, 2, 1), torch.permute(rf, 2, 1)
    wc, rc = torch.permute(wc, 2, 1), torch.permute(rc, 2, 1)
    it = torch.sigmoid(torch.matmul(x, wi) + torch.matmul(h, ri) + wbi + rbi)
    ft = torch.sigmoid(torch.matmul(x, wf) + torch.matmul(h, rf) + wbf + rbf)
    ct = torch.tanh(torch.matmul(x, wc) + torch.matmul(h, rc) + wbc + rbc)
    Ct = ft * c + it * c
    ot = torch.sigmoid(torch.matmul(x, wo) + torch.matmul(h, ro) + wbo + rbo)
    ht = ot * torch.tanh(Ct)
    
    return ot, ht, ct


def lstm_cell(x, h, c, W, R, Wb, Rb, is_ort=False):
    xw = F.linear(x, W.squeeze(0), Wb.squeeze(0))
    xr = F.linear(h, R.squeeze(0), Rb.squeeze(0))  
    y = xw + xr
    # gap_x = xw.shape[-1]//4
    it, ft, ct, ot = y.chunk(4, 1)
    it = torch.sigmoid(it)
    ft = torch.sigmoid(ft)
    if not is_ort:
        ct = torch.tanh(ct)
        ot = torch.sigmoid(ot)
    else:
        ot = torch.tanh(ot)
        ct = torch.sigmoid(ct)
    
    Ct = ft * c + it * ct
    ht = ot * torch.tanh(Ct)
    
    return ot, ht, Ct


def gru_cell(x, h, W, R, Wb, Rb, linear_before_reset=0, is_ort=True):

    gate_x = F.linear(x, W.squeeze(dim=0), Wb.squeeze(dim=0))
    gate_h = F.linear(h, R.squeeze(dim=0), Rb.squeeze(dim=0))

    if is_ort:
        iz, ir, ih = gate_x.chunk(3, 1)
        hz, hr, hh = gate_h.chunk(3, 1)
    else:
        ir, iz, ih = gate_x.chunk(3, 1)
        hr, hz, hh = gate_h.chunk(3, 1)

    rt = F.sigmoid(ir + hr)
    zt = F.sigmoid(iz + hz)
    if linear_before_reset != 0: ### pytorch default is 1
        ht = F.tanh(ih + (rt * hh))
    else: ### onnx default is 0
        tmp = rt * h
        Rh = R.chunk(3, dim=1)[-1]
        Rbh = Rb.chunk(3, dim=1)[-1]
        tmp = F.linear(tmp, Rh.squeeze(dim=0), Rbh.squeeze(dim=0))
        ht = F.tanh(ih + tmp)

    Ht = (1 - zt) * ht + zt * h
    y = Ht

    return y, Ht


def quant_lstm_cell(x, h, c, W, R, Wb, Rb, sx, sh, sw, sr):
    xw = F.linear(x, W.squeeze(0), Wb.squeeze(0))
    xr = F.linear(h, R.squeeze(0), Rb.squeeze(0))
    xw = xw.type(torch.float32) * sx[0] * sw[0]
    xr = xr.type(torch.float32) * sh[0] * sr[0]
    y = xw + xr
    gap_x = xw.shape[-1]//4
    it = torch.sigmoid(y[:,0*gap_x:1*gap_x])
    ft = torch.sigmoid(y[:,1*gap_x:2*gap_x])
    ct = torch.tanh(y[:,2*gap_x:3*gap_x])
    ot = torch.sigmoid(y[:,3*gap_x:4*gap_x])
    
    Ct = ft * c + it * ct
    ht = ot * torch.tanh(Ct)
    
    return ot, ht, Ct


def float_fast_calc_lstm(x, h, c, W, R, Wb, Rb):
    time_step = x.shape[0]
    ht, ct = copy.deepcopy(h).squeeze(0), copy.deepcopy(c).squeeze(0)
    output = []
    for time in range(time_step):
       ht, ht, ct = lstm_cell(x[time], ht, ct, W, R, Wb, Rb)
       output.append(ht.unsqueeze(dim=0))
    ht = ht.unsqueeze(dim=0)
    ct = ct.unsqueeze(dim=0)
    ot = torch.concat(output, dim=0)
    
    return ot, ht, ct
 
 
def mse_error(simulation_data, true_data):
    eps = 1e-5
    s_data = copy.deepcopy(simulation_data).astype(np.float32) 
    t_data = copy.deepcopy(true_data).astype(np.float32)
    diff = np.reshape(t_data, -1) - np.reshape(s_data, -1)
    sum = np.square(t_data).sum()
    sum = eps if sum == 0 else sum
    rate = np.square(diff).sum() * 100 / sum
    return np.float32(rate) 


def cosine_error(simulation_data, true_data):
    s_data = copy.deepcopy(simulation_data).astype(np.float32)
    t_data = copy.deepcopy(true_data).astype(np.float32)
    s_data = torch.from_numpy(s_data.reshape(1, -1))
    t_data = torch.from_numpy(t_data.reshape(1, -1))
    s_norm_data = F.normalize(s_data, p=2, dim=1)#.numpy()
    t_norm_data = F.normalize(t_data, p=2, dim=1)#.numpy()
    dist = torch.mm(s_norm_data, t_norm_data.t()).reshape(-1)
    dist = 1.0 - dist
    rate = dist.item() * 100
    # dist = np.dot(s_data, t_data.T)
    # dist = 1.0 - np.mean(dist, axis=1)
    # rate = np.mean(dist) * 100
    return np.float32(rate)

  
def quant_fast_calc_lstm(x, h, c, W, R, Wb, Rb):
    time_step = x.shape[0]
    sx, sh = get_scale(x.numpy()), get_scale(h.numpy())
    sw, sr = get_scale(W.numpy()), get_scale(R.numpy())
    ct = copy.deepcopy(c).squeeze(0)
    get_qmatrix = lambda data, scale: torch.round(data/scale[0]).type(torch.int32)
    get_tint32 = lambda data: torch.round(data).type(torch.int32) 
    xt, ht = get_qmatrix(x, sx), get_qmatrix(h.squeeze(0), sh)
    w, r = get_qmatrix(W, sw), get_qmatrix(R, sr)
    wb, rb = Wb / (sx[0] * sw[0]), Rb / (sh[0] * sr[0])
    wb, rb = get_tint32(wb), get_tint32(rb)
    output = []
    for time in range(time_step):
       ht, ht, ct = quant_lstm_cell(xt[time], ht, ct, w, r, wb, rb, sx, sh, sw, sr)
       output.append(ht.unsqueeze(dim=0))
       # sh, rb = get_scale(ht.numpy()), torch.round(Rb / (sh[0] * sr[0])).type(torch.int32)
       if time_step > 1 and time < time_step - 1:
           sh = get_scale(ht.numpy())
           ht = torch.round(ht / sh[0]).type(torch.int32) 
           rb = get_tint32(Rb / (sh[0] * sr[0]))
       
    ht = ht.unsqueeze(dim=0)
    ct = ct.unsqueeze(dim=0)
    ot = torch.concat(output, dim=0)
    
    return ot, ht, ct


def quant_lstm(x, h, c, Wi, Ri, Bi, Pi=None, bidirectional=False):
    gap_x, gap_h, gap_c = x.shape[-1], h.shape[-1], c.shape[-1]
    W, R, Wb, Rb = Wi[0:1], Ri[0:1], Bi[0:1, 0:Wi.shape[1]], Bi[0:1, Wi.shape[1]:]
    WB, RB, WBb, RBb, P, PB = None, None, None, None, None, None
    
    ot, ht, ct = quant_fast_calc_lstm(
        torch.from_numpy(x), 
        torch.from_numpy(h), 
        torch.from_numpy(c), 
        torch.from_numpy(W), 
        torch.from_numpy(R), 
        torch.from_numpy(Wb), 
        torch.from_numpy(Rb))
    
    return ot.numpy(), ht.numpy(), ct.numpy()


def float_lstm(x, h, c, Wi, Ri, Bi, Pi=None, bidirectional=False):
    gap_x, gap_h, gap_c = x.shape[-1], h.shape[-1], c.shape[-1]
    W, R, Wb, Rb = Wi[0:1], Ri[0:1], Bi[0:1, 0:Wi.shape[1]], Bi[0:1, Wi.shape[1]:]
    WB, RB, WBb, RBb, P, PB = None, None, None, None, None, None
    
    ot, ht, ct = float_fast_calc_lstm(
        torch.from_numpy(x), 
        torch.from_numpy(h), 
        torch.from_numpy(c), 
        torch.from_numpy(W), 
        torch.from_numpy(R), 
        torch.from_numpy(Wb), 
        torch.from_numpy(Rb))
    
    return ot.numpy(), ht.numpy(), ct.numpy()


def qlstm():
    x = np.load('x.npy')
    h0 = np.load('h0.npy')
    c0 = np.load('c0.npy')
    ot = np.load('ot.npy')
    ht = np.load('hn.npy')
    ct = np.load('cn.npy')
    W, R, B = np.load('w.npy'), np.load('r.npy'), np.load('b.npy')
    qot, qht, qct = quant_lstm(x, h0, c0, W, R, B)
    error_xo = [mse_error(ot, qot), cosine_error(ot, qot)]
    error_hn = [mse_error(ht, qht), cosine_error(ht, qht)]
    error_cn = [mse_error(ct, qct), cosine_error(ct, qct)]
    print('=> hand write lstm error_xo: {} error_hn: {}, error_cn: {}'.format(error_xo, error_hn, error_cn))


if __name__ == '__main__':
    test_lstm(batch=100, time_step=2)
    qlstm()
    