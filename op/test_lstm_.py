import copy
import math
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.init as init
import torch.nn.functional as F

def lstm_cell(x, h, c, W, R, Wb, Rb):
    with torch.no_grad():
        xw = F.linear(x, W.squeeze(0), Wb.squeeze(0))
        xr = F.linear(h, R.squeeze(0), Rb.squeeze(0))  
        y = xw + xr
        # gap_x = xw.shape[-1]//4
        it, ft, ct, ot = y.chunk(4, 1)
        it = torch.sigmoid(it)
        ft = torch.sigmoid(ft)
        ct = torch.tanh(ct)
        ot = torch.sigmoid(ot)
        
        Ct = ft * c + it * ct
        ht = ot * torch.tanh(Ct)
    
        return ht, Ct
    
def gru_cell(x, h, W, R, Wb, Rb, linear_before_reset=1, is_ort=False):
    with torch.no_grad():
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

        return Ht

# 定义双向 LSTM 模型
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_forward = nn.LSTMCell(input_size, hidden_size)
        self.lstm_backward = nn.LSTMCell(input_size, hidden_size)
    
    def forward(self, input, hn, cn):
        # 初始化前向和后向 LSTM 的隐藏状态和细胞状态
        
        forward_res_hn, forward_res_cn = [], []
        backward_res_hn, backward_res_cn = [], []

        # 前向传播
        for i in range(input.size(0)):
            hn[0], cn[0] = lstm_cell(input[i, :, :], 
                                     hn[0], 
                                     cn[0], 
                                     self.lstm_forward.weight_ih, 
                                     self.lstm_forward.weight_hh, 
                                     self.lstm_forward.bias_ih, 
                                     self.lstm_forward.bias_hh)
            forward_res_hn.append(copy.deepcopy(hn[0].detach()))
            forward_res_cn.append(copy.deepcopy(cn[0].detach()))

        # 反向传播
        for i in reversed(range(input.size(0))):
            hn[1], cn[1] = lstm_cell(input[i, :, :], 
                                     hn[1], 
                                     cn[1], 
                                     self.lstm_backward.weight_ih, 
                                     self.lstm_backward.weight_hh, 
                                     self.lstm_backward.bias_ih, 
                                     self.lstm_backward.bias_hh)
            backward_res_hn.insert(0, copy.deepcopy(hn[1].detach()))
            backward_res_cn.insert(0, copy.deepcopy(cn[1].detach()))
        # backward_res_hn.reverse()
        # backward_res_cn.reverse()
        y = []
        for i in range(input.size(0)):
            y.append(torch.cat([forward_res_hn[i], backward_res_hn[i]], axis=1))
        output = torch.stack(y, dim=0)
        # h_forward  = torch.stack(forward_res_hn, dim=0)#.reshape(input.size(0),  input.size(1), -1)
        # h_backward = torch.stack(backward_res_hn, dim=0)#.reshape(input.size(0), input.size(1), -1)
        # # 将前向和后向 LSTM 的隐藏状态连接起来
        # output = torch.cat([h_forward, h_backward], dim=-1)#.reshape(input.size(0),  input.size(1), -1)
        # output = torch.zeros(())
        hidden_forward  = torch.stack([hn[0], cn[0]], dim=0)#.reshape(2, input.size(1), -1)
        hidden_backward = torch.stack([hn[1], cn[1]], dim=0)#.reshape(2, input.size(1), -1)
        
        return output, hidden_forward, hidden_backward

    def forward_(self, input, hn, cn):
        # 初始化前向和后向 LSTM 的隐藏状态和细胞状态
        
        forward_res_hn, forward_res_cn = [], []
        backward_res_hn, backward_res_cn = [], []

        # 前向传播
        for i in range(input.size(0)):
            hn[0], cn[0] = self.lstm_forward(input[i, :, :], (hn[0], cn[0]))
            forward_res_hn.append(copy.deepcopy(hn[0].detach()))
            forward_res_cn.append(copy.deepcopy(cn[0].detach()))

        # 反向传播
        for i in range(input.size(0) - 1, -1, -1):
            hn[1], cn[1] = self.lstm_backward(input[i, :, :], (hn[1], hn[1]))
            backward_res_hn.insert(0, copy.deepcopy(hn[1].detach()))
            backward_res_cn.insert(0, copy.deepcopy(cn[1].detach()))
        backward_res_hn.reverse()
        backward_res_cn.reverse()
        h_forward  = torch.stack(forward_res_hn, dim=0)#.reshape(input.size(0),  input.size(1), -1)
        h_backward = torch.stack(backward_res_hn, dim=0)#.reshape(input.size(0), input.size(1), -1)
        # 将前向和后向 LSTM 的隐藏状态连接起来
        output = torch.cat([h_forward, h_backward], dim=-1)#.reshape(input.size(0),  input.size(1), -1)
        # output = torch.zeros(())
        hidden_forward  = torch.stack([hn[0], cn[0]], dim=0)#.reshape(2, input.size(1), -1)
        hidden_backward = torch.stack([hn[1], cn[1]], dim=0)#.reshape(2, input.size(1), -1)
        
        return output, hidden_forward, hidden_backward
    
class ORIBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ORIBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)

    def forward(self, input):
        # LSTM 层
        output, (hn, cn) = self.lstm(input)

        # 取最后一个时间步的输出
        # output = output[:, -1, :]
        
        return output, hn, cn

def build_torch_lstm_gru():
    # 创建双向 LSTM 模型
    input_size = 10
    hidden_size = 20
    output_size = 5
    bilstm_model = BiLSTM(input_size, hidden_size, output_size)
    ori_bilstm_model = ORIBiLSTM(input_size, hidden_size, output_size)
    
    bilstm_model.lstm_forward.weight_hh = nn.Parameter(ori_bilstm_model.lstm.weight_hh_l0.detach())
    bilstm_model.lstm_forward.weight_ih = nn.Parameter(ori_bilstm_model.lstm.weight_ih_l0.detach())
    bilstm_model.lstm_forward.bias_hh = nn.Parameter(ori_bilstm_model.lstm.bias_hh_l0.detach())
    bilstm_model.lstm_forward.bias_ih = nn.Parameter(ori_bilstm_model.lstm.bias_ih_l0.detach())
    bilstm_model.lstm_backward.weight_hh = nn.Parameter(ori_bilstm_model.lstm.weight_hh_l0_reverse.detach())
    bilstm_model.lstm_backward.weight_ih = nn.Parameter(ori_bilstm_model.lstm.weight_ih_l0_reverse.detach())
    bilstm_model.lstm_backward.bias_hh = nn.Parameter(ori_bilstm_model.lstm.bias_hh_l0_reverse.detach())
    bilstm_model.lstm_backward.bias_ih = nn.Parameter(ori_bilstm_model.lstm.bias_ih_l0_reverse.detach())
    
    # 构造输入数据
    batch_size, sequence_length = 4, 2
    input_data = torch.randn(sequence_length, batch_size, input_size)  # (sequence_length, batch_size, input_size)

    # 前向传播
    hn = torch.zeros((2, batch_size, hidden_size))
    cn = torch.zeros((2, batch_size, hidden_size))    
    bilstm_output, hn, cn = bilstm_model(input_data.detach(), hn, cn)
    ori_bilstm_output, ori_hn, ori_cn = ori_bilstm_model(input_data.detach())
    

    diff_lstm = bilstm_output.detach() - ori_bilstm_output.detach()
    print(diff_lstm.max(), diff_lstm.min())

build_torch_lstm_gru()