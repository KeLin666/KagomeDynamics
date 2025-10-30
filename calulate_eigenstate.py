import numpy as np

def calulate_eigenstate(evectors, k_points, g_vectors, k_point_index, band_index, resolution=200, xmax=5 *np.sqrt(3) ,ymax=5):

    # eigenvector
    eigenvector_coeffs = evectors[k_point_index][:, band_index]
    # print(eigenvector_coeffs.shape)
    # print(g_vectors.shape)
    k_vec = k_points[k_point_index]

    # Meshgrid
    x = np.linspace(-xmax, xmax, resolution)
    y = np.linspace(-ymax, ymax, resolution)
    X, Y = np.meshgrid(x, y)
    
    # psi(r) = Σ_G C_G * exp(i*(k+G).r)
    eigenstate_wavefunction = np.zeros_like(X, dtype=np.complex128)
    for C_G, G in zip(eigenvector_coeffs, g_vectors):
        K = k_vec + G
        eigenstate_wavefunction += C_G * np.exp(1j * (K[0] * X + K[1] * Y))
    sumup = 0 
    for index_i in range(len(eigenstate_wavefunction)):
        for index_j in range(len(eigenstate_wavefunction)):
            sumup += np.abs(eigenstate_wavefunction[index_i,index_j])**2
    # print(X.shape)
    # print(C_G.shape)
    eigenstate_wavefunction /= np.sqrt(sumup) * np.sqrt((4 * xmax * ymax )) / (resolution-1)

    # # |psi(r)|^2
    # probability_density = np.abs(eigenstate_wavefunction)**2

    return  eigenstate_wavefunction

