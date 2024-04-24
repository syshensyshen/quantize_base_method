# quantize_base_method
# 各种算子的quantize算法模拟

# conv op symmetric and asymmetric, and simulator fused zero_point into bias

# rnn op int inference, and maybe using lut in activation into rnn cell inference

# implement perchannel conv simlator

# simlator activation using lut

# mul op int inference, do not using int32 transfer

# maxpool->split h pool and w pool, some hareware split pool has best perf

# avgpool->split h pool and w pool, 1/(h*w) maybe using int32, but split sum of kernel will using int16
