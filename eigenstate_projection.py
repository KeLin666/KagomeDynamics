import numpy as np

def eigenstate_projection(wavefunction,eigenstate_wavefunction):
    
    overlap = np.sum(np.sum(np.abs(np.conjugate(wavefunction) * eigenstate_wavefunction)))

    return overlap