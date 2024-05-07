import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F 

class Conv1d_PReLU_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(Conv1d_PReLU_Conv1d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.prelu1 = nn.ReLU()
        constant_a = torch.randn(1, 16, 18, 8)
        self.constant_a = nn.Parameter(constant_a)
        constant_b = torch.randn(1, 16, 18, 8)
        self.constant_b = nn.Parameter(constant_b)       
        constant_c = torch.randn(1, 16, 18, 8)
        self.constant_c = nn.Parameter(constant_c)      
        constant_d = torch.randn(16, 8, 18)
        self.constant_d = nn.Parameter(constant_d)              
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride)
        self.prelu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride)
        self.prelu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        x_c = x
        x = self.conv1(x)
        # x = self.prelu1(x)
        x += self.constant_a
        x = x * self.constant_b
        x = x - self.constant_c
        # x = torch.matmul(x, self.conv11(x_c))
        out = torch.matmul(x, self.constant_d)
        output = simulation(x, self.constant_d)
        # x += self.conv11(x_c)
        out = self.conv2(out)
        out = self.prelu2(out)
        out = self.conv3(out)
        out = self.prelu3(out)
        out = self.conv4(out)
        return x
    
def simulation(data, weights):
    x, w = data.detach(), weights.detach()
    in_data = x.reshape(-1, 8)
    new_w = w.transpose(1,0)
    outputs = []
    for i in range(x.shape[1]):
        outputs.append(F.linear(x[0][i], w[i].transpose(1,0)))
    
    return torch.concat(outputs, dim=0).reshape(x.shape[1], x.shape[2], w.shape[-1])


# Instantiate the network
net = Conv1d_PReLU_Conv1d(3, 16, (3, 3), (1, 1))

# Define input tensor
input_tensor = torch.randn(1, 3, 20, 10)

# Export the model to ONNX format
torch.onnx.export(net, input_tensor, "model.onnx")


import numpy as np

def expand_to_4d(matrix):
    while matrix.ndim < 4:
        matrix = np.expand_dims(matrix, axis=-1)
    return matrix

# Example usage
matrix = np.array([[0, 1, 2], [3, 4, 5]])
print(expand_to_4d(matrix))

