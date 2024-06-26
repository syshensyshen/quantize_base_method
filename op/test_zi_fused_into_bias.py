import torch
import torch.nn as nn
import torch.nn.functional as F


x = torch.randn(1, 32, 224, 224)
si = (torch.max(x) - torch.min(x)) / 255.0
zi = -128 - torch.round(torch.min(x) / si)
zi = torch.clamp(zi, -128, 127)

qx = torch.round(x / si) + zi
qx = torch.clamp(qx, -128, 127)

x1 = qx - zi
conv = torch.nn.Conv2d(32, 64, kernel_size=3, padding=0)
conv.bias.data.fill_(0.0)

sk = torch.max(torch.abs(conv.weight.data)) / 127.0
conv.weight.data = torch.round(conv.weight.data / sk)
conv.weight.data = conv.weight.data.clamp(-128, 127)

conv_pad = nn.ConstantPad2d((1), -zi)

y1 = conv(conv_pad(x1))
print(y1.shape, y1.sum())

conv_fused = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
conv_fused.weight.data = conv.weight.data
conv_fused.bias.data *= 0.0
conv_pad_ = nn.ConstantPad2d((1), zi)
x2 = qx.detach()

conv_fused.bias.data -= torch.mean(F.conv2d(conv_pad_(torch.ones_like(x1) * zi), weight=conv.weight.data, bias=None, padding=0), dim=(2,3)).reshape(-1)
y2 = conv_fused(x2)

# zi_bias = F.conv2d(torch.ones_like(x1) * zi, weight=conv.weight.data, bias=None, padding=0)
# y2 = conv_fused(x2) - zi_bias

print(y2.shape, y2.sum())