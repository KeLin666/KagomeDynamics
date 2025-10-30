import numpy as np

def fig_to_rgb_array(fig, verbose=False):
    """将 matplotlib Figure 转成 (H, W, 3) uint8 RGB 图像。
       兼容有 tostring_rgb (老版) 和 buffer_rgba (新版) 的情况，
       并尝试自动检测常见的通道排列（RGBA / ARGB / BGRA / ABGR）。
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # 先尝试老 API（若存在）
    try:
        buf = fig.canvas.tostring_rgb()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        if verbose: print("Used tostring_rgb(), shape:", arr.shape)
        return arr.copy()  # 返回 contiguous copy
    except AttributeError:
        # 新 API：buffer_rgba()
        buf = fig.canvas.buffer_rgba()
        arr4 = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        if verbose: print("Used buffer_rgba(), raw shape:", arr4.shape)

        # 期望的 Figure facecolor（RGB 0-255）
        face_rgba = np.array(fig.get_facecolor())  # floats 0..1
        face_rgb = (face_rgba[:3] * 255).astype(np.uint8)

        # 常见的候选提取 (生成 RGB)
        candidates = {
            'RGBA': arr4[..., :3],
            'ARGB': arr4[..., 1:4],
            'BGRA': arr4[..., [2,1,0]],
            'ABGR': arr4[..., [3,2,1]],
        }

        # 检测：比较左上角像素或中间像素与 Figure facecolor 是否接近
        checks = [(0,0), (h//2, w//2)]
        for name, cand in candidates.items():
            ok = False
            for (ry, rx) in checks:
                pix = cand[ry, rx, :]
                if np.allclose(pix, face_rgb, atol=3):  # 容差可调
                    ok = True
                    break
            if ok:
                if verbose: print(f"Detected channel order: {name}")
                return cand.copy()

        # 如果自动检测失败，尝试用 RGBA->RGB 作为后备（保证不会崩）
        if verbose: 
            print("Warning: can't confidently detect channel order; falling back to arr4[...,:3].")
        return arr4[..., :3].copy()