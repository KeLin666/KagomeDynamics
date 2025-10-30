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
from calulate_eigenstate_2d import calulate_eigenstate_2d

if True:
    num_run = 1
    h = 6.626e-34
    hbar = h / 2/np.pi
    lambda532 = 532e-9
    lambda1064 = 1064e-9
    k532 = 2 * np.pi / lambda532
    k1064 = 2 * np.pi / lambda1064
    mrb = 1.4192261e-25
    mp = 6.4924249e-26
    N = 1024
    Nx, Ny = N, N
    xmax, ymax =  2*np.sqrt(3),2 #*1e-6, 2e-6
    dx = (2 * xmax) / (N - 1)
    dy = (2 * ymax) / (N - 1)
    x = np.linspace(-xmax, xmax, N)
    y = np.linspace(-ymax, ymax, N)
    dt = 1e-2#e-16
    tmax = 100*dt
    NUM = int(tmax / dt) + 1
    save_gif = 1

# Pre-allocate array to store the wavefield at each step
# U = np.zeros((Nx, Ny, NUM), dtype=np.complex128)

# First, create 2D coordinate grids from the 1D x and y arrays
X, Y = np.meshgrid(x, y)
w0 = 0.2  # Beam waist radius (controls the spot size). Adjust this value as needed.
u = np.exp(-(((X)**2 + Y**2) / w0**2))#.astype(np.complex128)

# Calculate squared frequencies for the linear operatorth
fx_sq = np.fft.fftfreq(Nx, d=dx)**2
fy_sq = np.fft.fftfreq(Ny, d=dy)**2
FX_sq, FY_sq = np.meshgrid(fx_sq, fy_sq)


# # --- Calculate Momentum (k-space) Coordinates for Plot ---
# kx = 2 * np.pi * np.fft.fftshift(fx_sq)
# ky = 2  * np.pi * np.fft.fftshift(fy_sq)
V532 = 200
V1064 = 200
# Note: V=200 is a good number, if k = 4 * np.pi / 3
# Note: V=1e9 is a good number, if k = 4 * np.pi / 3 * 3000
# Note: V=1e14 is a good number, if k = 4 * np.pi / 3 * 1e6
# Note: V=1e28 is a good number if k = 2 * np.pi / 1064e-9 =  4 * np.pi / 3 *1.5/ 1064e-9 

phi12 = 0
phi23 = 0
# P_0 = np.array([0, 0])
# P_1 = np.array([0.2, -0.2])
# P_2 = np.array([0.2, 0.2])
# P_3 = np.array([-0.2, 0.2])
# P_4 = np.array([-0.2, -0.2])
# k_path = [P_0, P_1, P_2, P_3, P_4, P_1]
# k_labels = ["P_0", "P_1", "P_2", "P_3", "P_4", "P_1"]

Gamma = np.array([0, 0])
K = np.array([2*np.pi/np.sqrt(3), 2*np.pi/3]) #* 1.5 / 1064e-9
M = np.array([2*np.pi/np.sqrt(3), 0]) #* 1.5 / 1064e-9
k_path = [Gamma, K, M, Gamma]
k_labels = ["\Gamma", "K", "M", "\Gamma"]
num_interpolation = 6

V = calculate_kagome_potential(
    X, 
    Y, 
    4 * np.pi / 3 , 
    V532, 
    V1064, 
    phi12, 
    phi23, 
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
    num_bands = 6, 
    G_cutoff = 3, 
    plot = 1
)


def plot():
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

# ! dont run this, take 5 mins or so. dont!

eigenstates = [[None for _ in range(len(k_points))] for _ in range(3)]

for band_index in range(3):
    for k_index in range(len(k_points)):
        eigenstates[band_index][k_index] = calulate_eigenstate(
            evectors, k_points, g_vectors,
            k_index, band_index,
            resolution=N,
            xmax=xmax,
            ymax=ymax
        )
      
# start your run below, make you life much easier! please!



u = eigenstate_wavefunction = calulate_eigenstate(
    evectors,
    k_points,
    g_vectors, 
    0, 
    0, 
    resolution=N, 
    xmax = xmax,
    ymax = ymax)

u /= np.sqrt(np.sum(np.sum(np.conj(u) * u))*dx*dy) 


# Open a writer for the GIF file
gif_path = f"video{num_run}.gif"

dt = 1e-2#e-16
tmax = 200*dt
NUM = int(tmax / dt) + 1

potential_2D = V 

X, Y = np.meshgrid(x, y)
w0 = 0.1*xmax  # Beam waist radius (controls the spot size). Adjust this value as needed.
u = 1 / np.sqrt(np.pi * w0**2)* np.exp(-(((X)**2 + Y**2) / 2 / w0**2))#.astype(np.complex128)

# print(np.sum(np.sum(np.conj(u)*u))*dx*dy) # normalization check, should add up equal to 1

eigenstate_wavefunction = calulate_eigenstate(
    evectors,
    k_points,
    g_vectors, 
    0, 
    0, 
    resolution=N, 
    xmax = xmax,
    ymax = ymax)

u = u * eigenstate_wavefunction

u /= np.sqrt(np.sum(np.sum(np.conj(u) * u))*dx*dy) 

# print(np.sum(np.sum(np.conj(u)*u))*dx*dy) # normalization check, should add up equal to 1

# u_spreading = u
# potential_2D_spreading = potential_2D
# Linear propagation factor for one step dt

FACTOR = -4 * np.pi**2 * dt / 2 #/50000000
LINEAR_FACTOR = np.exp(1j * FACTOR * (FY_sq + FX_sq))

i = np.arange(0, NUM)
freqs = np.linspace(0.1, 0.1, 1)   # 带宽从 0.02 到 0.04
amps = np.random.uniform(0.2, 1.0, len(freqs))
signal = np.sum([A * np.sin(f * i) for A, f in zip(amps, freqs)], axis=0)
# signal = np.fft.fft(signal)
signal /= np.max(signal)/np.pi*10
plt.plot(signal)

overlap_map = np.zeros((3, len(k_points)))
for band_index in range(3):
    for k_index in range(len(k_points)):
        eigenstate_wavefunction = eigenstates[band_index][k_index]
        overlap = abs(np.sum(np.sum(np.conj(u)*eigenstate_wavefunction))*dx*dy)# eigenstate_projection(u, eigenstate_wavefunction, N, xmax,ymax)
        overlap_map[band_index, k_index] = overlap  

if True:
    plt.figure(figsize=(8,4))
    # for band_index in range(overlap_map.shape[0]):
        # plt.plot(np.arange(len(k_points)), overlap_map[band_index], '-o', label=f'Band {band_index}')
    plt.plot(np.arange(len(k_points)), overlap_map[0], '-o', label='overlap with ground band')
    plt.plot(np.arange(len(k_points)), overlap_map[1], '-o', label='overlap with 1st band')
    plt.plot(np.arange(len(k_points)), overlap_map[2], '-o', label='overlap with 2nd band')
    plt.plot(np.arange(len(k_points)), overlap_map[1]+overlap_map[2], '-o', label='overlap with 1+2 band')

    plt.xlabel('k-point index along path')
    plt.ylabel('Overlap diff')
    plt.title('Overlap diff vs k-index')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



with imageio.get_writer(gif_path, mode='I', duration=0.2, loop=0) as writer:
    print("Starting propagation...")
    for i in tqdm(range(NUM)):

        u = u * np.exp(-1j * (potential_2D) * dt / 2)
        uf = np.fft.fft2(u)
        uf = uf * LINEAR_FACTOR
        u = np.fft.ifft2(uf)
        u = u * np.exp(-1j * (potential_2D) * dt / 2)
        # U[:, :, i] = u

        potential_2D = calculate_kagome_potential(
                    X, 
                    Y, 
                    4 * np.pi / 3, 
                    V532, 
                    V1064, #+ 0.8 * np.sin(signal[i]), 
                    - np.pi * np.sin(signal[i]), 
                    np.pi * np.sin(signal[i]), 
                    plot=0
                    )
        
        # overlap_sum = 0
        overlap_map = np.zeros((3, len(k_points)))

        # print(overlap_sum)
        # for band_index in range(2):
        #     for k_index in range(len(k_points)):
        #         # eigenstate_wavefunction = calulate_eigenstate_2d(V532, V1064, phi12, phi23, g_vectors, k_point_x, k_point_y, band_index, resolution=200, real_space_extent=5):
        #         eigenstate_wavefunction = calulate_eigenstate(evectors, k_points, g_vectors, k_index, band_index, resolution=N, real_space_extent=xmax)
        #         overlap = eigenstate_projection(eigenstate_wavefunction,u,resolution=N, real_space_extent=xmax)
        #         overlap_sum = overlap_sum + overlap
        # print(overlap_sum)

        for band_index in range(3):
            for k_index in range(len(k_points)):
                eigenstate_wavefunction = eigenstates[band_index][k_index]
                overlap = abs(np.sum(np.sum(np.conj(u)*eigenstate_wavefunction))*dx*dy) # eigenstate_projection(eigenstate_wavefunction, u, resolution=N, real_space_extent=xmax)
                # overlap_sum += overlap
                overlap_map[band_index, k_index] = overlap  
        # print(overlap_sum)

        if  False:
            u_spreading = u_spreading * np.exp(-1j * (potential_2D_spreading) * dt / 2)
            uf_spreading = np.fft.fft2(u_spreading)
            uf_spreading = uf_spreading * LINEAR_FACTOR
            u_spreading = np.fft.ifft2(uf_spreading)
            u_spreading = u_spreading * np.exp(-1j * (potential_2D_spreading) * dt / 2)

            potential_2D_spreading = calculate_kagome_potential(
                        X, 
                        Y, 
                        4 * np.pi / 3.0, 
                        V532, 
                        V1064, #+ 0.8 * np.sin(signal[i]), 
                        0, #np.pi * np.sin(signal[i]), 
                        0, 
                        plot=0
                        )
        
            overlap_map_spreading = np.zeros((3, len(k_points)))
            for band_index in range(3):
                for k_index in range(len(k_points)):
                    eigenstate_wavefunction = eigenstates[band_index][k_index]
                    overlap_spreading = abs(np.sum(np.sum(np.conj(u)*eigenstate_wavefunction))*dx*dy) # eigenstate_projection(eigenstate_wavefunction, u_spreading, resolution=N, real_space_extent=xmax)
                    overlap_map_spreading[band_index, k_index] = overlap_spreading  

        if i % 10 == 0:
        # ---------- 图2：沿路径的 overlap ----------
            plt.figure(figsize=(8,4))
            # for band_index in range(overlap_map.shape[0]):
                # plt.plot(np.arange(len(k_points)), overlap_map[band_index], '-o', label=f'Band {band_index}')
            plt.plot(np.arange(len(k_points)), overlap_map[0], '-o', label='overlap with ground band')
            plt.plot(np.arange(len(k_points)), overlap_map[1], '-o', label='overlap with 1st band')
            plt.plot(np.arange(len(k_points)), overlap_map[2], '-o', label='overlap with 2nd band')
            plt.plot(np.arange(len(k_points)), overlap_map[1]+overlap_map[2], '-o', label='overlap with 1+2 band')

            plt.xlabel('k-point index along path')
            plt.ylabel('Overlap diff')
            plt.title('Overlap diff vs k-index')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        if save_gif:
            if i % 3 == 0:
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
                # im2.set_clim(0.0, 0.8)
                cbar2 = fig.colorbar(im2, ax=ax2)
                cbar2.set_label('Intensity |u|', fontweight='bold', fontsize=14)
                
                # 3) Plot Momentum Space Wave Intensity
                ax3 = fig.add_subplot(1, 3, 3)
                uf_shifted_abs = np.fft.fftshift(np.abs(uf))
                
                # log_norm = LogNorm(vmin=1e-5, vmax=uf_shifted_abs.max())

                im3 = ax3.imshow(uf_shifted_abs[int(0.4375*N):int(0.5625*N),int(0.4375*N):int(0.5625*N)],
                                # extent=[fx_sq[0], fx_sq[-1], fy_sq[0], fy_sq[-1]],
                                extent=[fx_sq[-1]*0.25, fx_sq[-1]*0.75, fy_sq[-1]*0.25, fy_sq[-1]*0.75],
                                cmap='viridis',
                                origin='lower',
                                # norm=log_norm,
                                aspect='auto')
                    
                ax3.set_title(f"Momentum Space Intensity |u(k)|²")
                ax3.set_xlabel("$k_x$")
                ax3.set_ylabel("$k_y$")

                fig.colorbar(im3, ax=ax3, label='Intensity (log scale)')
                
                img = fig_to_rgb_array(fig, verbose=False)   # (H, W, 3), uint8
                print("Appending frame:", img.shape, img.dtype, "top-left:", img[0,0,:])
                writer.append_data(img)

                # plt.close(fig)

if save_gif:
    print(f"Processing complete. GIF saved to '{gif_path}'")



'''
import numpy as np

num_points_axis = 5

kx_ticks = np.linspace(-1, 1, num_points_axis)
ky_ticks = np.linspace(-1, 1, num_points_axis)

Kx, Ky = np.meshgrid(kx_ticks, ky_ticks)

k_mesh = np.dstack((Kx, Ky)).reshape(-1, 2)

print(f"Shape of k_mesh: {k_mesh.shape}")
print("\nFirst 5 k-points:")
print(k_mesh[:5])
print("\nLast 5 k-points:")
print(k_mesh[-5:])
'''
