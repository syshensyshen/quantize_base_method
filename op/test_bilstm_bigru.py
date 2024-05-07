import math
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.init as init
import torch.nn.functional as F

    
class Lstmcell():
    def __init__(self, input_size: int, hidden_size: int, num_chunks: int=4) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.empty((num_chunks * hidden_size, input_size)))
        self.weight_hh = Parameter(torch.empty((num_chunks * hidden_size, hidden_size)))
        self.bias_ih = Parameter(torch.empty(num_chunks * hidden_size))
        self.bias_hh = Parameter(torch.empty(num_chunks * hidden_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)        

    def forwaid(self, x, h, c):
        xw = F.linear(x, self.weight_ih, self.bias_ih)
        xr = F.linear(h, self.weight_hh, self.bias_hh)  
        y = xw + xr
        it, ft, ct, ot = y.chunk(4, 1)
        
        Ct = ft * c + it * ct
        ht = ot * torch.tanh(Ct)
    
        return ot, ht, Ct
    

class Grucell():
    def __init__(self, input_size: int, hidden_size: int, num_chunks: int=3, linear_before_reset:int=0) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.empty((num_chunks * hidden_size, input_size)))
        self.weight_hh = Parameter(torch.empty((num_chunks * hidden_size, hidden_size)))
        self.bias_ih = Parameter(torch.empty(num_chunks * hidden_size))
        self.bias_hh = Parameter(torch.empty(num_chunks * hidden_size))
        self.linear_before_reset = linear_before_reset

    def forward(self, x, h, W, R, Wb, Rb):

        gate_x = F.linear(x, W.squeeze(dim=0), Wb.squeeze(dim=0))
        gate_h = F.linear(h, R.squeeze(dim=0), Rb.squeeze(dim=0))

        ir, iz, ih = gate_x.chunk(3, 1)
        hr, hz, hh = gate_h.chunk(3, 1)

        rt = F.sigmoid(ir + hr)
        zt = F.sigmoid(iz + hz)
        if self.linear_before_reset != 0: ### pytorch default is 1
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
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # 初始化前向和后向 LSTM 的隐藏状态和细胞状态
        batch_size = input.size(0)
        h_forward = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        c_forward = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        h_backward = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        c_backward = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)

        # 前向传播
        for i in range(input.size(1)):
            h_forward, c_forward = self.lstm_forward(input[:, i, :], (h_forward, c_forward))

        # 反向传播
        for i in range(input.size(1) - 1, -1, -1):
            h_backward, c_backward = self.lstm_backward(input[:, i, :], (h_backward, c_backward))

        # 将前向和后向 LSTM 的隐藏状态连接起来
        output = torch.cat((h_forward, h_backward), dim=1)

        # 线性层
        output = self.linear(output)
        return output
    
class ORIBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ORIBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # LSTM 层
        output, _ = self.lstm(input)

        # 取最后一个时间步的输出
        output = output[:, -1, :]

        # 线性层
        output = self.linear(output)
        return output

# 定义双向 GRU 模型
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru_forward = nn.GRUCell(input_size, hidden_size)
        self.gru_backward = nn.GRUCell(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # 初始化前向和后向 GRU 的隐藏状态
        batch_size = input.size(0)
        h_forward = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)
        h_backward = torch.zeros(batch_size, self.hidden_size, dtype=input.dtype, device=input.device)

        # 前向传播
        for i in range(input.size(1)):
            h_forward = self.gru_forward(input[:, i, :], h_forward)

        # 反向传播
        for i in range(input.size(1) - 1, -1, -1):
            h_backward = self.gru_backward(input[:, i, :], h_backward)

        # 将前向和后向 GRU 的隐藏状态连接起来
        output = torch.cat((h_forward, h_backward), dim=1)

        # 线性层
        output = self.linear(output)
        return output
    
class ORIBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ORIBiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        # GRU 层
        output, _ = self.gru(input)

        # 取最后一个时间步的输出
        output = output[:, -1, :]

        # 线性层
        output = self.linear(output)
        return output

def build_torch_lstm_gru():
    # 创建双向 LSTM 模型
    input_size = 10
    hidden_size = 20
    output_size = 5
    bilstm_model = BiLSTM(input_size, hidden_size, output_size)
    bigru_model = BiGRU(input_size, hidden_size, output_size)
    ori_bilstm_model = BiLSTM(input_size, hidden_size, output_size)
    ori_bigru_model = BiGRU(input_size, hidden_size, output_size)
    bilstm_model.lstm_forward.load_state_dict(ori_bilstm_model.lstm_forward.state_dict())
    bilstm_model.lstm_backward.load_state_dict(ori_bilstm_model.lstm_backward.state_dict())
    bilstm_model.linear.load_state_dict(ori_bilstm_model.linear.state_dict())
    bigru_model.gru_forward.load_state_dict(ori_bigru_model.gru_forward.state_dict())
    bigru_model.gru_backward.load_state_dict(ori_bigru_model.gru_backward.state_dict())
    bigru_model.linear.load_state_dict(ori_bigru_model.linear.state_dict())
    # 构造输入数据
    input_data = torch.randn(3, 4, 10)  # (sequence_length, batch_size, input_size)

    # 前向传播
    bilstm_output = bilstm_model(input_data.detach())
    bigru_output = bigru_model(input_data.detach())
    ori_bilstm_output = ori_bilstm_model(input_data.detach())
    ori_bigru_output = ori_bigru_model(input_data.detach())

    print(torch.sum(bilstm_output.detach() - ori_bilstm_output.detach()))
    print(torch.sum(bigru_output.detach() - ori_bigru_output.detach()))
