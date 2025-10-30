import numpy as np
from scipy.constants import hbar, m_e, e
from scipy import sparse
from scipy.sparse.linalg import eigs

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e
def draw_allband3d(kx, ky, E):
    eV = 1.60218e-19
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nBands = E.shape[2]
    for ii in range(nBands):
        ax.plot_surface(kx, ky, E[:, :, ii] / eV, cmap="viridis", alpha=0.8)

    ax.set_xlabel("kx", fontsize=18)
    ax.set_ylabel("ky", fontsize=18)
    ax.set_zlabel("E (eV)", fontsize=18)
    ax.tick_params(labelsize=14, width=2)

    plt.savefig("BandStructure.jpg", dpi=300)
    plt.show()
    
def create_square_lattice_potential(Nx, Ny, a, model='gaussian_center', depth_eV=-10.0, radius=None, sigma=None):
    """
    为二维方晶格创建一个周期性单元势 V(x,y)。

    Args:
        Nx (int): x 方向的网格点数。
        Ny (int): y 方向的网格点数。
        a (float): 晶格常数 (米)。
        model (str): 使用的物理模型。可选:
                     'circular_well_center' - 在中心放置一个圆形常数势阱。
                     'gaussian_center'      - 在中心放置一个高斯势阱。
                     'four_corners'         - 在四个角上各放置一个高斯势阱。
        depth_eV (float): 势阱的最深处能量，单位为电子伏特 (eV)。
        radius (float, optional): 'circular_well_center' 模型的半径 (米)。
                                  如果未提供，默认为 a / 4。
        sigma (float, optional): 高斯模型的标准差 (米)。
                                 如果未提供，默认为 a / 8。

    Returns:
        np.ndarray: 一个形状为 (Ny, Nx) 的二维数组，代表势能（单位：焦耳）。
    """
    # 将用户友好的 eV 单位转换为计算所需的焦耳
    depth_J = depth_eV * e

    # 设置默认参数
    if radius is None:
        radius = a / 4.0
    if sigma is None:
        sigma = a / 8.0

    # 创建一个代表单元内每个点坐标的网格
    # 坐标原点 (0,0) 在单元中心
    x = np.linspace(-a / 2, a / 2, Nx)
    y = np.linspace(-a / 2, a / 2, Ny)
    X, Y = np.meshgrid(x, y)

    # 初始化势能数组
    V = np.zeros((Ny, Nx))

    # --- 根据选择的模型计算势能 ---
    if model == 'circular_well_center':
        print(f"创建模型: 中心圆形势阱, 半径 = {radius:.2e} m")
        # 计算每个点到中心的距离
        R = np.sqrt(X**2 + Y**2)
        # 在半径范围内的点，势能设为 V0
        V[R <= radius] = depth_J
        
    elif model == 'gaussian_center':
        print(f"创建模型: 中心高斯势阱, sigma = {sigma:.2e} m")
        # 计算每个点到中心的距离的平方
        R_sq = X**2 + Y**2
        V = depth_J * np.exp(-R_sq / (2 * sigma**2))

    elif model == 'four_corners':
        print(f"创建模型: 四角高斯势阱, sigma = {sigma:.2e} m")
        corners = [
            (-a/2, -a/2), ( a/2, -a/2),
            (-a/2,  a/2), ( a/2,  a/2)
        ]
        # 将四个角上的高斯势阱叠加起来
        # 这代表了相邻单元的原子对该单元的共同影响
        for cx, cy in corners:
            R_sq = (X - cx)**2 + (Y - cy)**2
            V += depth_J * np.exp(-R_sq / (2 * sigma**2))
            
    else:
        raise ValueError(f"未知的模型 '{model}'. 可选: 'circular_well_center', 'gaussian_center', 'four_corners'")

    return V

def band_structure(Tx, Ty, Nkx, Nky, V_unit, numeig):
    """
    Calculates the electronic band structure for a 2D periodic potential.

    This is a Python translation of the MATLAB script by Ke Lin, 
    Heller Group, Harvard University.

    Args:
        Tx (float): The lattice constant in the x-direction (meters).
        Ty (float): The lattice constant in the y-direction (meters).
        Nkx (int): The number of k-points to sample in the x-direction.
        Nky (int): The number of k-points to sample in the y-direction.
        V_unit (np.ndarray): A 2D NumPy array representing the potential 
                             of the unit cell in Joules.
        numeig (int): The number of energy bands to calculate.

    Returns:
        tuple: A tuple containing:
            - E (np.ndarray): The calculated energy bands.
            - BZS (any): The second output from the eigensolver, which might be
                         k-space meshes or eigenvectors.
    """
    ######################################################
    #                    Set-up part                     #
    ######################################################

    # --- Unit and constant ---
    # Using scipy.constants for better precision and clarity.
    eV = e  # Elementary charge in Coulombs, so 1 eV = e Joules.
    
    # --- Real space grid set up ---
    # Ensure V_unit is a NumPy array for operations like .shape
    V_unit = np.asarray(V_unit)
    if V_unit.ndim != 2:
        raise ValueError("V_unit must be a 2D array.")

    # Get the grid size from the potential matrix's shape
    Ny_unit, Nx_unit = V_unit.shape
    
    # Generate x and y vectors for the unit cell grid
    x_vector = np.linspace(-Tx / 2 + Tx / Nx_unit / 2, Tx / 2 - Tx / Nx_unit / 2, Nx_unit)
    dx = x_vector[1] - x_vector[0]

    y_vector = np.linspace(-Ty / 2 + Ty / Ny_unit / 2, Ty / 2 - Ty / Ny_unit / 2, Ny_unit)
    dy = y_vector[1] - y_vector[0]

    # --- Momentum space grid set up ---
    # Generate kx and ky vectors for the Brillouin zone sampling
    # Nkx = 15; # recommend value: 31
    kx = np.linspace(-np.pi / Tx, np.pi / Tx, Nkx)
    # Nky = 15; # recommend value: 31
    ky = np.linspace(-np.pi / Ty, np.pi / Ty, Nky)

    guess_value = eV # Set guess value to 1 eV (in Joules)

    # Call the 2D Schrödinger Equation eigensolver.
    # IMPORTANT: The function 'SE2Deig_KXKY' from the original MATLAB code must be
    # implemented separately in Python. A placeholder is provided below.
    E, BZS = se_2d_eig_kxky(V_unit, Nx_unit, Ny_unit, dx, dy, kx, ky, guess_value, numeig)

    return E, BZS



def se_2d_eig_kxky(V, Nx, Ny, dx, dy, kx, ky, guess_value, numeig):
    """
    Solves the 2D Schrödinger equation for a grid of k-points (kx, ky).

    This function constructs a sparse Hamiltonian matrix for each k-point using
    a finite-difference method with periodic boundary conditions. It then
    calculates the lowest 'numeig' eigenvalues and eigenvectors.

    Args:
        V (np.ndarray): The 2D potential grid. Should have shape (Ny, Nx).
        Nx (int): The number of grid points in the x-direction.
        Ny (int): The number of grid points in the y-direction.
        dx (float): The grid spacing in the x-direction (meters).
        dy (float): The grid spacing in the y-direction (meters).
        kx (np.ndarray): A 1D array of k-vector components in the x-direction.
        ky (np.ndarray): A 1D array of k-vector components in the y-direction.
        guess_value (float): An estimate for the eigenvalues of interest, used
                             by the 'shift-and-invert' eigensolver algorithm.
        numeig (int): The number of eigenvalues (bands) to compute.

    Returns:
        tuple: A tuple containing:
            - E (np.ndarray): An array of energy eigenvalues with shape
                              (len(kx), len(ky), numeig).
            - BZS (np.ndarray): An array of the corresponding eigenvectors with
                                shape (len(kx), len(ky), numeig, Nx*Ny).
    """
    # --- Constants ---
    coeff = hbar**2 / (2 * m_e)
    N_total = Nx * Ny  # Total size of the Hamiltonian matrix

    # --- Initialize result arrays ---
    num_kx = len(kx)
    num_ky = len(ky)
    E = np.zeros((num_kx, num_ky, numeig))
    BZS = np.zeros((num_kx, num_ky, numeig, N_total), dtype=np.complex128)
    
    # Flatten the potential V for easy addition to the diagonal
    V_flat = V.flatten()

    # --- Loop over all k-points ---
    for ikx, KX in enumerate(kx):
        print(f"Calculating... {ikx / num_kx * 100:.1f}% complete")
        for iky, KY in enumerate(ky):
            
            # --- Calculate finite-difference coefficients for the current k-point ---
            # These terms come from the discretization of the kinetic energy operator
            # H = p^2/2m => (-ihbar*∇)^2/2m, with plane-wave basis u_k(r)exp(ik*r)
            C1 = coeff * (2/dx**2 + 2/dy**2) # Main diagonal kinetic term
            C2 = -coeff / dx**2             # Off-diagonal x-term
            C3 = -coeff / dy**2             # Off-diagonal y-term

            # --- Build the Hamiltonian in sparse format ---
            # Main diagonal: kinetic part + potential part
            main_diag = C1 + V_flat
            
            # Off-diagonals for neighbors in x-direction
            # Periodic boundary condition wraps around
            off_diag_x = np.full(N_total, C2)
            
            # Off-diagonals for neighbors in y-direction
            off_diag_y = np.full(N_total, C3)
            
            # Create the sparse matrix
            # The `diags` function is perfect for creating banded matrices
            diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
            offsets = [0, 1, -1, Nx, -Nx]
            Hamiltonian = sparse.diags(diagonals, offsets, shape=(N_total, N_total), format='csc')
            
            # Apply periodic boundary conditions for x and y
            # This connects the edges of the grid
            for i in range(Ny):
                # Connect right edge to left edge (x-direction)
                idx1 = i * Nx + (Nx - 1)
                idx2 = i * Nx
                Hamiltonian[idx1, idx2] = C2
                Hamiltonian[idx2, idx1] = C2
                
            # Connect bottom edge to top edge (y-direction)
            # This is automatically handled by the offset Nx and -Nx for all points
            # except when a more complex basis (like Bloch waves) is not used.
            # A simpler way is to connect the last row to the first row.
            for j in range(Nx):
                 idx1 = (Ny-1)*Nx + j
                 idx2 = j
                 Hamiltonian[idx1, idx2] = C3
                 Hamiltonian[idx2, idx1] = C3

            # --- Solve for eigenvalues and eigenvectors ---
            # Using 'eigs' for sparse matrices. 'sigma' finds eigenvalues near guess_value.
            try:
                # We ask for more eigenvalues than needed to improve stability
                # and ensure we find the true lowest ones near the guess.
                num_to_find = min(2 * numeig + 1, N_total - 2)
                eigvals, eigvecs = eigs(Hamiltonian, k=num_to_find, sigma=guess_value, which='LM')
            except Exception as e:
                print(f"Warning: Eigensolver failed for (kx, ky) = ({KX}, {KY}). Skipping. Error: {e}")
                continue

            # Sort the results because 'eigs' doesn't guarantee order
            sort_indices = np.argsort(np.real(eigvals))
            sorted_eigvals = eigvals[sort_indices]
            sorted_eigvecs = eigvecs[:, sort_indices]

            # --- Store the lowest 'numeig' results ---
            E[ikx, iky, :] = np.real(sorted_eigvals[:numeig])
            # Transpose eigenvectors to match (numeig, N_total) before storing
            BZS[ikx, iky, :, :] = sorted_eigvecs[:, :numeig].T

    print("Calculation complete.")
    return E, BZS



if __name__ == '__main__':
    lattice_constant = 5  # 5 
    grid_resolution_N = 20
    V_unit = create_square_lattice_potential(
        Nx=grid_resolution_N, Ny=grid_resolution_N, a=lattice_constant,
        model='gaussian_center',
        depth_eV=-20.0,
        sigma=lattice_constant / 6
    )
    plt.tight_layout()
    plt.show()
    E, BZS =band_structure(10, 10, 7, 7, V_unit, 5)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    