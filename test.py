import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.constants import hbar, m_e, e
import time

################################################################################
#                                                                              #
#               第一部分：从您的 MATLAB 代码翻译过来的核心函数                  #
#                                                                              #
################################################################################

def se_2d_eig_kxky_direct(V, Nx, Ny, dx, dy, kx, ky, guess_value, numeig):
    """
    这是您提供的 SE2Deig_KXKY.m 文件的直接 Python 翻译。
    它通过构建一个稀疏哈密顿量来求解二维薛定谔方程。
    
    注意：为了清晰起见，MATLAB 的 Nkx, Nky 对应这里的 Nx, Ny (实空间格点数)
    """
    coeff = hbar**2 / (2 * m_e)
    N_total = Nx * Ny

    # 初始化结果数组
    num_kx = len(kx)
    num_ky = len(ky)
    E = np.zeros((num_kx, num_ky, numeig))
    BZS = np.zeros((num_kx, num_ky, numeig, N_total), dtype=np.complex128)

    # 主循环，遍历所有 k 点
    for ikx, KX in enumerate(kx):
        # 打印一个简单的进度
        print(f"\r正在计算 k 点: {ikx+1}/{num_kx}", end="")

        for iky, KY in enumerate(ky):
            # 为稀疏矩阵准备的 COO 格式列表 (row, col, data)
            I, J, H = [], [], []

            # 计算有限差分法的系数
            # 这些系数包含了由 exp(i*k*r) 形式的布洛赫波函数产生的相位
            C1 = coeff * (2/dx**2 + 2/dy**2) + V.flatten() # 对角线项
            C2 = -coeff / dx**2 # x 方向邻居项
            C3 = -coeff / dy**2 # y 方向邻居项

            # 遍历实空间中的每一个格点 (i, j) 来构建哈密顿矩阵
            for i in range(Ny):       # y-index
                for j in range(Nx):   # x-index
                    # 将二维索引 (i, j) 转换为一维索引 k
                    k = j + i * Nx
                    
                    # 1. 对角线元素
                    I.append(k)
                    J.append(k)
                    H.append(C1[k])
                    
                    # 2. 右边的邻居 (处理周期性边界)
                    k_right = j + 1
                    if k_right >= Nx: k_right = 0 # 周期性边界
                    k_right_1d = k_right + i * Nx
                    I.append(k)
                    J.append(k_right_1d)
                    H.append(C2)
                    
                    # 3. 左边的邻居 (处理周期性边界)
                    k_left = j - 1
                    if k_left < 0: k_left = Nx - 1 # 周期性边界
                    k_left_1d = k_left + i * Nx
                    I.append(k)
                    J.append(k_left_1d)
                    H.append(C2)

                    # 4. 下边的邻居 (处理周期性边界)
                    k_down = i + 1
                    if k_down >= Ny: k_down = 0 # 周期性边界
                    k_down_1d = j + k_down * Nx
                    I.append(k)
                    J.append(k_down_1d)
                    H.append(C3)
                    
                    # 5. 上边的邻居 (处理周期性边界)
                    k_up = i - 1
                    if k_up < 0: k_up = Ny - 1 # 周期性边界
                    k_up_1d = j + k_up * Nx
                    I.append(k)
                    J.append(k_up_1d)
                    H.append(C3)

            # 使用 COO 格式创建稀疏矩阵，然后转换为 CSC 格式以提高计算效率
            Hamiltonian = sparse.coo_matrix((H, (I, J)), shape=(N_total, N_total)).tocsc()
            
            # 求解本征值和本征函数
            try:
                # 'sigma' 参数让求解器寻找最接近 guess_value 的本征值
                eigvals, eigvecs = eigs(Hamiltonian, k=numeig, sigma=guess_value, which='LM')
            except Exception as exc:
                print(f"\n警告: 在 (kx, ky) = ({KX:.2f}, {KY:.2f}) 处本征求解器失败. 跳过此点。错误: {exc}")
                continue

            # 对结果进行排序，因为 eigs 不保证返回的顺序
            sort_indices = np.argsort(np.real(eigvals))
            sorted_eigvals = eigvals[sort_indices]
            sorted_eigvecs = eigvecs[:, sort_indices]
            
            # 存储结果
            E[ikx, iky, :] = np.real(sorted_eigvals)
            BZS[ikx, iky, :, :] = sorted_eigvecs.T

    print("\n计算完成。")
    return np.squeeze(E), np.squeeze(BZS) # 使用 squeeze 删除多余的维度

################################################################################
#                                                                              #
#                  第二部分：用于测试和可视化的主脚本                            #
#                                                                              #
################################################################################

if __name__ == '__main__':
    # --- 1. 定义物理系统参数 ---
    a = 5e-10  # 晶格常数 (5 埃)
    Tx, Ty = a, a

    # 实空间网格分辨率
    Nx, Ny = 40, 40
    dx = Tx / Nx
    dy = Ty / Ny

    # 定义势阱 V(x, y)
    V_unit = np.zeros((Ny, Nx))
    well_depth_eV = -10.0  # 势阱深度 (电子伏特)
    well_depth_J = well_depth_eV * e # 转换为焦耳

    # 在中心创建一个方形势阱，尺寸为单元的 1/2
    center_x, center_y = Nx // 2, Ny // 2
    width = Nx // 4
    V_unit[center_y - width : center_y + width, 
           center_x - width : center_x + width] = well_depth_J
           
    # --- 2. 定义倒易空间中的高对称路径 ---
    # 方晶格的高对称点
    Gamma = (0, 0)
    X = (np.pi / a, 0)
    M = (np.pi / a, np.pi / a)
    
    points_per_segment = 40 # 每段路径的 k 点数量

    # 创建路径： Γ -> X
    kx_GX = np.linspace(Gamma[0], X[0], points_per_segment)
    ky_GX = np.linspace(Gamma[1], X[1], points_per_segment)
    
    # 路径： X -> M
    kx_XM = np.linspace(X[0], M[0], points_per_segment)
    ky_XM = np.linspace(X[1], M[1], points_per_segment)

    # 路径： M -> Γ
    kx_MG = np.linspace(M[0], Gamma[0], points_per_segment)
    ky_MG = np.linspace(M[1], Gamma[1], points_per_segment)
    
    # 将路径拼接起来
    kx_path = np.concatenate([kx_GX, kx_XM[1:], kx_MG[1:]])
    ky_path = np.concatenate([ky_GX, ky_XM[1:], ky_MG[1:]])
    
    # --- 3. 执行能带计算 ---
    numeig = 10  # 要计算的能带数量
    # 猜测的能量值应接近我们期望的基态能量
    guess_value_J = well_depth_J * 0.8
    
    total_k_points = len(kx_path)
    all_energies = np.zeros((total_k_points, numeig))

    start_time = time.time()
    
    # 逐点调用求解器
    # (注意: 这是一个演示，效率不高。但它直接测试了函数的功能)
    for i in range(total_k_points):
        kx_point, ky_point = kx_path[i], ky_path[i]
        
        # 调用函数时，kx 和 ky 需要是数组
        energies_at_k, _ = se_2d_eig_kxky_direct(
            V=V_unit, Nx=Nx, Ny=Ny, dx=dx, dy=dy, 
            kx=np.array([kx_point]), ky=np.array([ky_point]), 
            guess_value=guess_value_J, numeig=numeig
        )
        all_energies[i, :] = energies_at_k

    end_time = time.time()
    print(f"总计算时间: {end_time - start_time:.2f} 秒")

    # --- 4. 准备绘图 ---
    # 计算 k 路径的距离作为 x 轴
    k_dist = np.zeros(total_k_points)
    for i in range(1, total_k_points):
        dkx = kx_path[i] - kx_path[i-1]
        dky = ky_path[i] - ky_path[i-1]
        k_dist[i] = k_dist[i-1] + np.sqrt(dkx**2 + dky**2)

    # 找到高对称点在 x 轴上的位置
    x_point_idx = points_per_segment - 1
    m_point_idx = x_point_idx + points_per_segment - 1
    
    # --- 5. 绘制能带图 ---
    plt.figure(figsize=(8, 6))
    
    # 将能量从焦耳转换为电子伏特 (eV)
    all_energies_eV = all_energies / e
    
    for i in range(numeig):
        plt.plot(k_dist, all_energies_eV[:, i], '-')

    # 美化图像
    plt.xticks(
        [k_dist[0], k_dist[x_point_idx], k_dist[m_point_idx], k_dist[-1]],
        ['Γ', 'X', 'M', 'Γ']
    )
    plt.axvline(x=k_dist[x_point_idx], color='gray', linestyle='--')
    plt.axvline(x=k_dist[m_point_idx], color='gray', linestyle='--')
    
    plt.xlabel('高对称路径 (High-Symmetry Path)')
    plt.ylabel('能量 (Energy) [eV]')
    plt.title('二维方晶格能带结构 (Band Structure of 2D Square Lattice)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(k_dist[0], k_dist[-1])
    
    # 设置一个合理的 y 轴范围
    min_energy = np.min(all_energies_eV)
    plt.ylim(min_energy - 1, min_energy + 20)
    
    plt.show()