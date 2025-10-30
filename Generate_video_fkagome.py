import cv2
import numpy as np
import matplotlib.pyplot as plt
from calculate_kagome_potential import calculate_kagome_potential
from io import BytesIO

# Video parameters
width, height = 640, 480
fps = 30
duration = 8  # seconds
num_frames = fps * duration

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # for .mp4
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

N = 1024
Nx, Ny = N, N
xmax, ymax =  6*np.sqrt(3), 6
dx = (2 * xmax) / (N - 1)
dy = (2 * ymax) / (N - 1)
x = np.linspace(-xmax, xmax, N)
y = np.linspace(-ymax, ymax, N)

dt = 0.01#e-16
tmax = num_frames*dt
NUM = int(tmax / dt) + 1

X, Y = np.meshgrid(x, y)

i = np.arange(0, NUM)
freqs = np.linspace(0.2, 0.2, 1)   # 带宽从 0.02 到 0.04
amps = np.random.uniform(0.2, 1.0, len(freqs))
signal = np.sum([A * np.sin(f * i) for A, f in zip(amps, freqs)], axis=0)
# signal = np.fft.fft(signal)
signal /= np.max(signal)/np.pi*10
plt.plot(signal)


# --- Setup VideoWriter ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('kagome_potential.mp4', fourcc, fps, (width, height))

# --- Generate frames ---
for i in range(num_frames):
    potential_2D = calculate_kagome_potential(
        X,
        Y,
        4 * np.pi / 3.0,
        1,
        1,
        -np.pi * np.sin(signal[i]),
        np.pi * np.sin(signal[i]),
        plot=0
    )

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    im = ax.imshow(potential_2D, extent=[x[0], x[-1], y[0], y[-1]], cmap='viridis', origin='lower')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Potential', fontweight='bold', fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Frame {i}")

    # Save figure to memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    # Convert PNG buffer to array
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # Resize to match output resolution
    frame = cv2.resize(frame, (width, height))
    out.write(frame)

out.release()
print("✅ Video saved as kagome_potential.mp4")