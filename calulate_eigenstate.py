import numpy as np

def calulate_eigenstate(evectors, k_points, g_vectors, k_point_index, band_index, resolution=200, real_space_extent=5):

    # eigenvector
    eigenvector_coeffs = evectors[k_point_index][:, band_index]
    k_vec = k_points[k_point_index]

    # Meshgrid
    x = np.linspace(-real_space_extent, real_space_extent, resolution)
    y = np.linspace(-real_space_extent, real_space_extent, resolution)
    X, Y = np.meshgrid(x, y)
    
    # psi(r) = Σ_G C_G * exp(i*(k+G).r)
    eigenstate_wavefunction = np.zeros_like(X, dtype=np.complex128)
    for C_G, G in zip(eigenvector_coeffs, g_vectors):
        K = k_vec + G
        eigenstate_wavefunction += C_G * np.exp(1j * (K[0] * X + K[1] * Y))
        
    # # |psi(r)|^2
    # probability_density = np.abs(eigenstate_wavefunction)**2

    return  eigenstate_wavefunction

