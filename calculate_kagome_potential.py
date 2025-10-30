import numpy as np
import matplotlib.pyplot as plt

def calculate_kagome_potential(x, y, k, V532, V1064, phi12, phi23, plot = 0):
    term1 = V532 * (2/3 - 2/9 * (
        np.cos(k * (-np.sqrt(3) * x + 3 * y)) +
        np.cos(k * (np.sqrt(3) * x + 3 * y)) +
        np.cos(2 * k * np.sqrt(3) * x)
    ))

    term2 = V1064 * (2/3 - 2/9 * (
        np.cos(k * (-np.sqrt(3)/2 * x + 3/2 * y) + phi12) +
        np.cos(k * (np.sqrt(3)/2 * x + 3/2 * y) + phi23) +
        np.cos(k * np.sqrt(3) * x - (phi12 - phi23))
    ))
    V = term1 - term2
    if plot:
        plt.figure(figsize=(8, 6))
        plt.pcolormesh(x, y, V, shading='auto', cmap='viridis')
        plt.colorbar(label='Potential V(x, y)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Kagome Lattice Potential')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig('kagome_potential.png')
        plt.show()
    return V