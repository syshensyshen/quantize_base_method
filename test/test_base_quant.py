'''
Author: WANG CHENG
Date: 2024-04-23 23:57:46
LastEditTime: 2024-04-24 00:48:02
'''
import torch
from utils import histogram
from operators.base_quant import quant, dequant, get_scale

SYMMETRIC = True

# 随机生成一个张量
data = torch.rand(size=(2, 3, 4),dtype=torch.float32)
print(f"data:\n{data}")
# histogram(data, bins=256, min_value=-127, max_value=128)

max_val = torch.max(torch.abs(data))
scale = get_scale(data, SYMMETRIC)[0]
zero_point = 0
print(f"max_val:{max_val}")
print(f"scale:{scale}")
print(f"zero_point:{zero_point}")

if SYMMETRIC:
    symquantized_data = quant(data, scale, zero_point) # int8 [-128,127]
    print(f"symquantized_data:\n{symquantized_data}")
    
    dequant_data = dequant(symquantized_data, scale, zero_point)
    print(f"dequant_data:\n{dequant_data}")
else:
    asymquantized_data = quant(data, scale, zero_point, symquant=False) # uint8 [0,255]
    print(f"asymquantized_data:\n{asymquantized_data}")
    
    dequant_data = dequant(asymquantized_data, scale, zero_point, symquant=False)
    print(f"dequant_data:\n{dequant_data}")
