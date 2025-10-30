import numpy as np

def calulate_eigenstate_2d(V532, V1064, phi12, phi23, g_vectors, k_point_x, k_point_y, band_index, resolution=200, real_space_extent=5):
    # Matrix of Hamiltonian H(k)_G,G' = (k+G)^2 * δ_GG' + V_{G-G'}
    # Unit of Kinetic Energy hbar^2 / (2m),
    G_cutoff = 4
    # t = a / (3 / 2)
    k = 4 * np.pi / 3.0
    # eigenvector
    k_vec = np.array([k_point_x,k_point_y])
        
    K1_532 = k * np.array([-np.sqrt(3), 3])
    K2_532 = k * np.array([np.sqrt(3), 3])
    K3_532 = k * np.array([2 * np.sqrt(3), 0])

    K1_1064 = k * np.array([-np.sqrt(3)/2, 3/2])
    K2_1064 = k * np.array([np.sqrt(3)/2, 3/2])
    K3_1064 = k * np.array([np.sqrt(3), 0])
    
    
    b1 = np.array([2 * np.pi / np.sqrt(3), 2 * np.pi])
    b2 = np.array([2 * np.pi / np.sqrt(3), -2 * np.pi])

    g_vectors = []
    g_indices = []
    for m in range(-G_cutoff, G_cutoff + 1):
        for n in range(-G_cutoff, G_cutoff + 1):
            g_vectors.append(m * b1 + n * b2)
            g_indices.append((m, n))

    g_vectors = np.array(g_vectors)
    num_g_vectors = len(g_vectors)

    bands = []
    evectors = []
    # go through every k point
    H = np.zeros((num_g_vectors, num_g_vectors), dtype=np.complex128)
    # fill the H
    for i, G in enumerate(g_vectors):
        for j, G_prime in enumerate(g_vectors):
            # Kinetic energy diagonal 
            if i == j:
                H[i, j] += np.linalg.norm(k_vec + G)**2
            
            # Potential  V_{G-G'}
            G_diff = G - G_prime
            
            # Potential from 532 beam
            V_G_diff = 0
            if np.allclose(G_diff, K1_532) or np.allclose(G_diff, -K1_532):
                V_G_diff += -V532 * (2/9) * 0.5
            if np.allclose(G_diff, K2_532) or np.allclose(G_diff, -K2_532):
                V_G_diff += -V532 * (2/9) * 0.5
            if np.allclose(G_diff, K3_532) or np.allclose(G_diff, -K3_532):
                V_G_diff += -V532 * (2/9) * 0.5
                
            # Potential from 1064 beam
            # be careful about phase
            if np.allclose(G_diff, K1_1064):
                V_G_diff -= -V1064 * (2/9) * 0.5 * np.exp(1j * phi12)
            if np.allclose(G_diff, -K1_1064):
                V_G_diff -= -V1064 * (2/9) * 0.5 * np.exp(-1j * phi12)
                
            if np.allclose(G_diff, K2_1064):
                V_G_diff -= -V1064 * (2/9) * 0.5 * np.exp(1j * phi23)
            if np.allclose(G_diff, -K2_1064):
                V_G_diff -= -V1064 * (2/9) * 0.5 * np.exp(-1j * phi23)
                
            if np.allclose(G_diff, K3_1064):
                V_G_diff -= -V1064 * (2/9) * 0.5 * np.exp(-1j * (phi12 - phi23))
            if np.allclose(G_diff, -K3_1064):
                V_G_diff -= -V1064 * (2/9) * 0.5 * np.exp(1j * (phi12 - phi23))

            H[i, j] += V_G_diff
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    eigenvector_coeffs = eigenvectors[:, band_index]
    # Meshgrid
    x = np.linspace(-real_space_extent, real_space_extent, resolution)
    y = np.linspace(-real_space_extent, real_space_extent, resolution)
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

    eigenstate_wavefunction /= np.sqrt(sumup) * (2 * real_space_extent / (resolution-1) )

    # # |psi(r)|^2
    # probability_density = np.abs(eigenstate_wavefunction)**2

    return  eigenstate_wavefunction

