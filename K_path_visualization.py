import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# --- 1. Define the user's k-path (the small square) ---
P_0 = np.array([0,0])
P_1 = np.array([0.1, -0.1])
P_2 = np.array([0.1, 0.1])
P_3 = np.array([-0.1, 0.1])
P_4 = np.array([-0.1, -0.1])
k_path_points = [P_0, P_1, P_2, P_3, P_4, P_1]
k_path_x = [p[0] for p in k_path_points]
k_path_y = [p[1] for p in k_path_points]
distinct_points = {"$P_0$": P_0, "$P_1$": P_1, "$P_2$": P_2, "$P_3$": P_3, "$P_4$": P_4}

# --- 2. Define the Brillouin Zone (Hexagon) ---
# We set a representative scale. Here, the Gamma-M distance is 1.0.
m = 1.0

# Vertices of the hexagon (K points)
k_scale = 2 * m / np.sqrt(3)
k_points_vertices = [
    (m, m / np.sqrt(3)),
    (0, k_scale),
    (-m, m / np.sqrt(3)),
    (-m, -m / np.sqrt(3)),
    (0, -k_scale),
    (m, -m / np.sqrt(3)),
    (m, m / np.sqrt(3)) # close the loop
]
k_points_x = [p[0] for p in k_points_vertices]
k_points_y = [p[1] for p in k_points_vertices]

# Midpoints of edges (M points)
m_points = [
    (m, 0),
    (m/2, m * np.sqrt(3) / 2),
    (-m/2, m * np.sqrt(3) / 2),
    (-m, 0),
    (-m/2, -m * np.sqrt(3) / 2),
    (m/2, -m * np.sqrt(3) / 2)
]
m_labels = ["$M$", "$M'$", "$M''$", "$M$", "$M'$", "$M''$"]

# K point labels (at the vertices)
k_labels = ["$K$", "$K'$", "$K$", "$K'$", "$K$", "$K'$"]
k_label_coords = [
    (m, m / np.sqrt(3)),
    (0, k_scale),
    (-m, m / np.sqrt(3)),
    (-m, -m / np.sqrt(3)),
    (0, -k_scale),
    (m, -m / np.sqrt(3))
]


# --- 3. Create the plot ---
plt.figure(figsize=(8, 8))

# Plot the BZ boundary (hexagon)
plt.plot(k_points_x, k_points_y, 'b-', label='Brillouin Zone (BZ) Boundary')

# Plot the user's k-path
plt.plot(k_path_x, k_path_y, 'ro-', label='User $k$-path')

# --- 4. Add labels for high-symmetry points ---
# Gamma point
plt.text(0, 0, '$\Gamma$', fontsize=16, ha='center', va='center', color='black')

# K points (vertices)
for i, (x, y) in enumerate(k_label_coords):
    plt.text(x * 1.05, y * 1.05, k_labels[i], fontsize=14, ha='center', va='center', color='blue')

# M points (edge midpoints)
for i, (x, y) in enumerate(m_points):
    plt.text(x * 1.05, y * 1.05, m_labels[i], fontsize=14, ha='center', va='center', color='green')
    plt.plot(x, y, 'go') # Mark M points

# --- 5. Add labels for the user's path (P_1...P_4) ---
offsets = {
    "$P_0$": (0.0, -0.0),
    "$P_1$": (0.01, -0.01),
    "$P_2$": (0.01, 0.01),
    "$P_3$": (-0.01, 0.01),
    "$P_4$": (-0.01, -0.01)
}
for label, (x, y) in distinct_points.items():
    dx, dy = offsets[label]
    ha = 'left' if dx > 0 else 'right'
    va = 'bottom' if dy > 0 else 'top'
    plt.text(x + dx, y + dy, label, fontsize=12, ha=ha, va=va, color='red')

# --- 6. Set plot properties ---
plt.xlabel('$k_x$ (e.g., in units of $\pi/a$)')
plt.ylabel('$k_y$ (e.g., in units of $\pi/a$)')
plt.title('Kagome Lattice Brillouin Zone with $k$-path')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Set limits to zoom in slightly on the BZ
max_val = k_scale * 1.1
plt.xlim(-max_val, max_val)
plt.ylim(-max_val, max_val)

# --- 7. Save or show the figure ---
plt.savefig("kagome_bz_with_k_path.png")
print("Plot saved to kagome_bz_with_k_path.png")
# plt.show() # Uncomment this line if you're running locally