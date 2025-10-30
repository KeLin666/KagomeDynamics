import numpy as np
import matplotlib.pyplot as plt
from calulate_eigenstate import calulate_eigenstate

def plot_eigenstate(evectors, k_points, g_vectors, k_point_index, band_index, resolution=200, real_space_extent=5):

    wavefunction = calulate_eigenstate(evectors, k_points, g_vectors, k_point_index, band_index, resolution, real_space_extent)
    # 计算概率密度 |psi(r)|^2
    probability_density = np.real(wavefunction)
    k_vec = k_points[k_point_index]

    # 绘图
    plt.figure(figsize=(7, 6))
    plt.imshow(probability_density, origin='lower', extent=[-real_space_extent, real_space_extent, -real_space_extent, real_space_extent], cmap='magma')
    plt.colorbar(label="Probability Density $|\psi|^2$")
    k_label = f"({k_vec[0]/np.pi:.2f}π, {k_vec[1]/np.pi:.2f}π)"
    plt.title(f"Eigenstate at k-point index {k_point_index} {k_label}, Band {band_index}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
