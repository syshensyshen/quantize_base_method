import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

fd_path = "./feat"
featuremap_histogram_bins = 1000
feat_files = glob(fd_path+'/*.txt')

for f_name in feat_files:
    if 'c' in f_name:
        continue
    if 'in' not in f_name:
        continue
    feat = np.loadtxt(fname=f_name)
    feat[feat<0.001] = 0
    max_v, min_v = np.max(feat), np.min(feat)
    mean, std = np.mean(feat), np.std(feat)
    print(f_name, min_v, max_v, np.min(np.abs(feat)), mean, std)
    hist_noise, _ = np.histogram(feat, featuremap_histogram_bins, range=(min_v, max_v))
    x = np.linspace(min_v, max_v, featuremap_histogram_bins)
    # plt.figure(dpi=4000)
    plt.figure(figsize=(16, 8))
    l1, = plt.plot(x, hist_noise/np.sum(hist_noise))
    plt.savefig(f_name.replace('.txt', ''))
    # plt.show()
