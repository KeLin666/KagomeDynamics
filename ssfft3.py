import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import savemat
from tqdm import tqdm
import imageio
from matplotlib.colors import LogNorm
from calculate_kagome_potential import calculate_kagome_potential
from fig_to_rgb_array import fig_to_rgb_array
from calculate_band_structure import calculate_band_structure
from plot_eigenstate import plot_eigenstate
from calulate_eigenstate import calulate_eigenstate
from eigenstate_projection import eigenstate_projection
num_run=1
h=6.626e-34
lambda532=532e-9
lambda1064=1064e-9
k532=2*np.pi/lambda532
k1064=2*np.pi/lambda1064
N = 128
Nx, Ny = N, N
xmax, ymax = 10/(4 * np.pi / 3.0), 10/(4 * np.pi / 3.0)
dx = (2 * xmax) / (N - 1)
dy = (2 * ymax) / (N - 1)
x = np.linspace(-xmax, xmax, N)
y = np.linspace(-ymax, ymax, N)
dt = 0.05#e-16
tmax = 100*dt
NUM = int(tmax / dt) + 1

# Pre-allocate array to store the wavefield at each step
# U = np.zeros((Nx, Ny, NUM), dtype=np.complex128)

# First, create 2D coordinate grids from the 1D x and y arrays
X, Y = np.meshgrid(x, y)
w0 = 0.2  # Beam waist radius (controls the spot size). Adjust this value as needed.
u = np.exp(-(((X)**2 + Y**2) / w0**2))#.astype(np.complex128)

# Calculate squared frequencies for the linear operator
fx_sq = np.fft.fftfreq(Nx, d=dx)**2
fy_sq = np.fft.fftfreq(Ny, d=dy)**2
FX_sq, FY_sq = np.meshgrid(fx_sq, fy_sq)


# # --- Calculate Momentum (k-space) Coordinates for Plotting ---
# kx = 2 * np.pi * np.fft.fftshift(fx_sq)
# ky = 2 * np.pi * np.fft.fftshift(fy_sq)
V532 = h*3000e32
V1064 = h*3000e32
phi12 = 0
phi23 = 0
Gamma = np.array([0, 0])
K = np.array([2*np.pi/np.sqrt(3), 2*np.pi/3])
M = np.array([2*np.pi/np.sqrt(3), 0])
k_path = [Gamma, K, M, Gamma]
k_labels = ["\Gamma", "K", "M", "\Gamma"]
num_interpolation = 20

V = calculate_kagome_potential(
    X, 
    Y, 
    4 * np.pi / 3.0, 
    1, 
    1, 
    0, 
    0, 
    plot = 1
)

bands, evectors, g_vectors, k_points = calculate_band_structure(
    V532,
    V1064, 
    phi12, 
    phi23, 
    k_path, 
    k_labels, 
    num_interpolation,
    num_bands=10, 
    G_cutoff=3, 
    plot=1
)

plot_eigenstate(
    evectors=evectors, 
    k_points=k_points,
    g_vectors=g_vectors,
    k_point_index=0,      
    band_index=0,
    real_space_extent = xmax          
)

plot_eigenstate(
    evectors=evectors, 
    k_points=k_points,
    g_vectors=g_vectors,
    k_point_index=0,      
    band_index=1,
    real_space_extent = xmax          
)

plot_eigenstate(
    evectors=evectors, 
    k_points=k_points,
    g_vectors=g_vectors,
    k_point_index=0,      
    band_index=2,
    real_space_extent = xmax          
)

plot_eigenstate(
    evectors=evectors,
    k_points=k_points,
    g_vectors=g_vectors,
    k_point_index = num_interpolation,    
    band_index=1,
    real_space_extent = xmax          
)

potential_2D = V 

# Linear propagation factor for one step dt
FACTOR = -4 * np.pi**2 * dt / 2 / 20
LINEAR_FACTOR = np.exp(1j * FACTOR * (FY_sq + FX_sq))

# Open a writer for the GIF file
gif_path = f"video{num_run}.gif"
with imageio.get_writer(gif_path, mode='I', duration=0.2, loop=0) as writer:
    print("Starting propagation...")
    for i in tqdm(range(NUM)):

        u = u * np.exp(-1j * (potential_2D) * dt / 2)
        uf = np.fft.fft2(u)
        uf = uf * LINEAR_FACTOR
        u = np.fft.ifft2(uf)
        u = u * np.exp(-1j * (potential_2D) * dt / 2)

        # Store the current wavefield
        # U[:, :, i] = u
        potential_2D = calculate_kagome_potential(
                    X, 
                    Y, 
                    4 * np.pi / 3.0, 
                    1, 
                    1, 
                    0,#np.pi*np.sin(0.03*i), 
                    0, 
                    plot=0
                    )

        # Note: The original MATLAB condition `mod(i*dt,0.01)==0` is true for every step since dt=0.01.
        # This means images and GIF frames are generated at every single step, which can be slow.
        if i % 10 == 0:
            overlap_sum = 0
            print(overlap_sum)
            for band_index in range(1):
                for k_index in range(num_interpolation*3):
                    eigenstate_wavefunction = calulate_eigenstate(evectors, k_points, g_vectors, k_index, band_index, resolution=N, real_space_extent=xmax)
                    overlap = eigenstate_projection(eigenstate_wavefunction,u)
                    overlap_sum = overlap_sum + overlap
            print(overlap_sum)
        # --- Create and save the main diagnostic figure (and GIF frame) ---
            fig = plt.figure(figsize=(10, 3.5), dpi=100)
            
            # 1) Plot Potential
            ax1 = fig.add_subplot(1, 3, 1)
            im1 = ax1.imshow(potential_2D, extent=[x[0], x[-1], y[0], y[-1]], cmap='viridis', origin='lower')
            # im1 = ax1.imshow(V[:, :, i].T, extent=[x[0], x[-1], y[0], y[-1]], cmap='jet', origin='lower')
            cbar1 = fig.colorbar(im1, ax=ax1)
            cbar1.set_label('Potential', fontweight='bold', fontsize=14)
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")

            # 2) Plot Wave Intensity
            ax2 = fig.add_subplot(1, 3, 2)
            im2 = ax2.imshow(np.abs(u), extent=[x[0], x[-1], y[0], y[-1]], cmap='viridis', origin='lower')
            ax2.axis('off')
            im2.set_clim(0.0, 1.2)
            cbar2 = fig.colorbar(im2, ax=ax2)
            cbar2.set_label('Intensity |u|', fontweight='bold', fontsize=14)
            
            # 3) Plot Momentum Space Wave Intensity
            ax3 = fig.add_subplot(1, 3, 3)
            uf_shifted_abs = np.fft.fftshift(np.abs(uf))
            
            # log_norm = LogNorm(vmin=1e-5, vmax=uf_shifted_abs.max())

            im3 = ax3.imshow(uf_shifted_abs,
                            extent=[fx_sq[0], fx_sq[-1], fy_sq[0], fy_sq[-1]],
                            cmap='viridis',
                            origin='lower',
                            # norm=log_norm,
                            aspect='auto')
                
            ax3.set_title(f"Momentum Space Intensity |u(k)|²")
            ax3.set_xlabel("$k_x$")
            ax3.set_ylabel("$k_y$")

            fig.colorbar(im3, ax=ax3, label='Intensity (log scale)')
            
            img = fig_to_rgb_array(fig, verbose=False)   # (H, W, 3), uint8
            # debug: 可选打印
            # print("Appending frame:", img.shape, img.dtype, "top-left:", img[0,0,:])
            writer.append_data(img)

            # plt.close(fig)

print(f"Processing complete. GIF saved to '{gif_path}'")

