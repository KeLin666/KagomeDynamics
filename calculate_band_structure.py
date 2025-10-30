import numpy as np
import matplotlib.pyplot as plt

def calculate_band_structure(V532, V1064, phi12, phi23, k_path, k_labels, num_interpolation=40, num_bands=10, G_cutoff=3, plot=0):
    lambda0=3/2
    k = 2 * np.pi / lambda0
    # t = a / (3 / 2)
    # k = 4 * np.pi / 3 # * 3000
    # if a =1 then k = 2pi/lambda = 2pi / 1.5, lambda=1.5a; 
    # if lambda =1 then a = 2/3, 1/a = 3/2
    # if lambda =1064e-9 then a = 2/3 lambda =2/3*1064e-9, 
    # h = 6.626e-34
    # hbar = 6.626e-34 / 2/np.pi
    # mrb = 1.4192261e-25
    # cst = hbar**2/2/mrb
    # k = 2 * np.pi / 1064e-9
    b1 = np.array([2 * np.pi / np.sqrt(3), 2 * np.pi]) #* 1.5 / 1064e-9
    b2 = np.array([2 * np.pi / np.sqrt(3), -2 * np.pi])  #* 1.5 / 1064e-9
    # amp of b 1/a
    g_vectors = []
    g_indices = []
    for m in range(-G_cutoff, G_cutoff + 1):
        for n in range(-G_cutoff, G_cutoff + 1):
            g_vectors.append(m * b1 + n * b2)
            g_indices.append((m, n))

    g_vectors = np.array(g_vectors)
    num_g_vectors = len(g_vectors)

    # V(r) = term1 - term2
    # term1 = V532 * ( ... )
    # term2 = V1064 * ( ... )
    # FFT V(r) = Σ_G V_G * exp(i*G*r)
    # for cos(K.r + phi) = 0.5 * (exp(i(K.r+phi)) + exp(-i(K.r+phi)))
    # V_G = 0.5 * exp(i*phi) for G=K
    # V_G = 0.5 * exp(-i*phi) for G=-K
    
    K1_532 = k * np.array([-np.sqrt(3), 3]) #/ (np.sqrt(3))
    K2_532 = k * np.array([np.sqrt(3), 3]) #/ (np.sqrt(3))
    K3_532 = k * np.array([2 * np.sqrt(3), 0]) #/ (np.sqrt(3))

    K1_1064 = k * np.array([-np.sqrt(3)/2, 3/2]) # (np.sqrt(3))
    K2_1064 = k * np.array([np.sqrt(3)/2, 3/2]) #/ (np.sqrt(3))
    K3_1064 = k * np.array([np.sqrt(3), 0]) #/ (np.sqrt(3))
    
    # Matrix of Hamiltonian H(k)_G,G' = (k+G)^2 * δ_GG' + V_{G-G'}
    # Unit of Kinetic Energy hbar^2 / (2m),
    
    # Create k point np array
    k_points = []
    for i in range(len(k_path) - 1):
        start, end = k_path[i], k_path[i+1]
        k_points.extend(np.linspace(start, end, num_interpolation))
    k_points = np.array(k_points)
    
    bands = []
    evectors = []
    # go through every k point
    for k_vec in k_points:
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

        # eigenvalue and eigen vector
        # print(H.shape)
        eigenvalues, eigenvectors = np.linalg.eigh(H)
        # eigenvalues = np.linalg.eigvalsh(H)
        bands.append(eigenvalues[:num_bands])
        evectors.append(eigenvectors)
    

    bands = np.array(bands).T # 
    evectors = np.array(evectors) 
    if plot:
        # Band structure
        plt.figure(figsize=(8, 6))
        k_axis = np.linspace(0, 1, len(k_points))
        
        # for i in range(num_bands):
        #     plt.plot(k_axis, bands[i], '-')

        plt.title("Band Structure of Kagome Lattice")
        # plt.ylabel("Energy (units of $\hbar^2/2m$)")
        plt.ylabel("Energy (units of $/h (Hz)$)")
        # K point
        k_node_positions = [0]
        node_dist = 0
        for i in range(len(k_path) - 1):
            node_dist += np.linalg.norm(k_path[i+1] - k_path[i])
            k_node_positions.append(node_dist)
            
        k_node_positions = np.array(k_node_positions)
        normalized_positions = k_node_positions / k_node_positions[-1]
        
        # modify the distance
        distances = np.zeros(len(k_points))
        for i in range(1, len(k_points)):
            distances[i] = distances[i-1] + np.linalg.norm(k_points[i] - k_points[i-1])

        for i in range(num_bands):
            plt.plot(distances, bands[i], '-')
            # plt.plot(bands[i],'-')
        plt.xticks(k_node_positions, [f'${label}$' for label in k_labels])
        for pos in k_node_positions:
            plt.axvline(x=pos, color='grey', linestyle='--')
            
        plt.xlim(0, k_node_positions[-1])
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    return bands, evectors, g_vectors, k_points


