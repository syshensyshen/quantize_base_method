
import numpy as np 
import torch
import copy

class L1Simiarity(object):
    def __init__(self, **kwargs):
        super(L1Simiarity, self).__init__()
        self.eps = 1e-5

    def __call__(self, simulation_data, true_data):
        s_data = copy.deepcopy(simulation_data).astype(np.float32)
        t_data = copy.deepcopy(true_data).astype(np.float32)
        s_data, t_data = torch.from_numpy(s_data), torch.from_numpy(t_data)
        diff = t_data.reshape(-1) - s_data.reshape(-1)
        sum = torch.abs(t_data).sum()
        sum = self.eps if sum == 0 else sum
        rate = torch.abs(diff).sum() * 100 / (sum + self.eps)
        return np.float32(rate)

class L2Simiarity(object):
    def __init__(self, **kwargs):
        super(L2Simiarity, self).__init__()
        self.eps = 1e-5

    def __call__(self, simulation_data, true_data):
        s_data = copy.deepcopy(simulation_data).astype(np.float32) 
        t_data = copy.deepcopy(true_data).astype(np.float32)
        s_data, t_data = torch.from_numpy(s_data), torch.from_numpy(t_data)
        diff = t_data.reshape(-1) - s_data.reshape(-1)
        sum = torch.square(t_data).sum()
        sum = self.eps if sum == 0 else sum
        rate = torch.square(diff).sum() * 100 / (sum + self.eps)
        return np.float32(rate)

class CosineSimiarity(object):
    def __init__(self, **kwargs):
        super(CosineSimiarity, self).__init__()
        self.eps = 1e-5
       
    def __call__(self, simulation_data, true_data):
        s_data = copy.deepcopy(simulation_data).astype(np.float32)
        t_data = copy.deepcopy(true_data).astype(np.float32)
        s_data = torch.from_numpy(s_data.reshape(-1))
        t_data = torch.from_numpy(t_data.reshape(-1))
        normal = torch.sqrt(torch.sum(s_data * s_data) * torch.sum(t_data * t_data))
        if normal == 0:
            if torch.sum(torch.abs(s_data)) == 0 and torch.sum(torch.abs(t_data)) == 0:
                dist = torch.ones(1)
            else:
                dist = torch.zeros(1)
        else:
            dist = torch.sum(s_data * t_data) / (normal + self.eps)
        dist = (1- np.abs(dist.item())) #* 100

        return np.float32(dist)