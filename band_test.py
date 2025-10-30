import numpy as np
import matplotlib.pyplot as plt
from calulate_eigenstate import calulate_eigenstate
from calculate_band_structure import calculate_band_structure
from plot_eigenstate import plot_eigenstate

Gamma = np.array([0, 0])
K = np.array([2*np.pi/np.sqrt(3), 2*np.pi/3])
M = np.array([2*np.pi/np.sqrt(3), 0])

# 定义计算路径: Gamma -> K -> M -> Gamma
path = [Gamma, K, M, Gamma]
labels = ["\Gamma", "K", "M", "\Gamma"]

V532_strength = 120.0   # 532nm 激光势能强度
V1064_strength = 120.0  # 1064nm 激光势能强度
phase_phi12 = 0 #np.pi / 2 # 相位 phi12
phase_phi23 = 0 #np.pi / 3 # 相位 phi23

# 运行计算
bands, evectors, g_vectors, k_points  = calculate_band_structure(
    V532=V532_strength,
    V1064=V1064_strength,
    phi12=phase_phi12,
    phi23=phase_phi23,
    k_path=path,
    k_labels=labels,
    num_interpolation = 40,
    num_bands=8,
    G_cutoff=3,
    plot=1
)

calulate_eigenstate(
    evectors=evectors,
    k_points=k_points,
    g_vectors=g_vectors,
    k_point_index=0,      # 0 对应于 Gamma 点
    band_index=0          # 0 对应于能量最低的能带
)
plot_eigenstate(
    evectors=evectors,
    k_points=k_points,
    g_vectors=g_vectors,
    k_point_index=0,      # 0 对应于 Gamma 点
    band_index=0          # 0 对应于能量最低的能带
)