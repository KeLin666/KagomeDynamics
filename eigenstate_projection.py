import numpy as np

def eigenstate_projection(wavefunction,eigenstate_wavefunction,resolution, xmax, ymax):
    dx = (2 * xmax / (resolution - 1))
    dy = (2 * ymax / (resolution - 1))
    overlap = np.sum(np.sum(np.conj(wavefunction) * eigenstate_wavefunction)) * dx * dy

    return overlap