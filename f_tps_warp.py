import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import cdist

import time

# ─────────────────────────────────────────────────────────────
#  1. ALGORITHME TPS (Inchangé)
# ─────────────────────────────────────────────────────────────

def tps_kernel(r: np.ndarray) -> np.ndarray:
    print("tps_kernel")
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(r == 0.0, 0.0, r**2 * np.log(r**2))
    return result

def solve_tps(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = src.shape[0]
    dists = cdist(src, src)
    K = tps_kernel(dists)
    P = np.hstack([np.ones((N, 1)), src])

    top    = np.hstack([K, P])
    bottom = np.hstack([P.T, np.zeros((3, 3))])
    L = np.vstack([top, bottom])

    rhs = np.vstack([dst, np.zeros((3, 2))])

    lambda_reg = 1e-6
    L[:N, :N] += lambda_reg * np.eye(N)

    params = np.linalg.solve(L, rhs)

    W = params[:N]
    A = params[N:]
    return W, A

def apply_tps(image: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    st = time.time()

    H, W, C = image.shape
    W_coef, A_coef = solve_tps(dst, src)
    at = time.time()
    print(f"apply_tps : 1/10 - {at-st} s")
    st = time.time()


    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    at = time.time()
    print(f"apply_tps : 2/10 - {at-st} s")
    st = time.time()


    grid = np.stack([cols.ravel(), rows.ravel()], axis=1).astype(np.float64)
    at = time.time()
    print(f"apply_tps : 3/10 - {at-st} s")
    st = time.time()


    dists = cdist(grid, dst.astype(np.float64))
    at = time.time()
    print(f"apply_tps : 4/10 - {at-st} s")
    st = time.time()



    Kgrid = tps_kernel(dists)
    at = time.time()
    print(f"apply_tps : 5/10 - {at-st} s")
    st = time.time()


    P = np.hstack([np.ones((grid.shape[0], 1)), grid])
    at = time.time()
    print(f"apply_tps : 6/10 - {at-st} s")
    st = time.time()



    src_coords = Kgrid @ W_coef + P @ A_coef
    at = time.time()
    print(f"apply_tps : 7/10 - {at-st} s")
    st = time.time()



    src_x = src_coords[:, 0].reshape(H, W)
    at = time.time()
    print(f"apply_tps : 8/10 - {at-st} s")
    st = time.time()


    src_y = src_coords[:, 1].reshape(H, W)
    at = time.time()
    print(f"apply_tps : 9/10 - {at-st} s")
    st = time.time()


    warped = bilinear_interpolate(image, src_x, src_y)
    at = time.time()
    print(f"apply_tps : 10/10 - {at-st} s")
    return warped

def bilinear_interpolate(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    H, W, C = image.shape

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = (x - x0).astype(np.float32)
    wy = (y - y0).astype(np.float32)

    valid = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)

    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    I00 = image[y0c, x0c]
    I10 = image[y0c, x1c]
    I01 = image[y1c, x0c]
    I11 = image[y1c, x1c]

    wx = wx[:, :, np.newaxis]
    wy = wy[:, :, np.newaxis]

    result = (1 - wy) * ((1 - wx) * I00 + wx * I10) \
           +      wy  * ((1 - wx) * I01 + wx * I11)

    result[~valid] = 0.0

    return result.astype(np.float32)

def compute_displacement_map(image: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    st = time.time()

    H, W = image.shape[:2]
    W_coef, A_coef = solve_tps(dst, src)
    at = time.time()
    print(f"compute_displacement_map : 1/10 - {at-st} s")
    st = time.time()

    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    at = time.time()
    print(f"compute_displacement_map : 2/10 - {at-st} s")
    st = time.time()

    grid = np.stack([cols.ravel(), rows.ravel()], axis=1).astype(np.float64)
    at = time.time()
    print(f"compute_displacement_map : 3/10 - {at-st} s")
    st = time.time()

    dists = cdist(grid, dst.astype(np.float64))
    at = time.time()
    print(f"compute_displacement_map : 4/10 - {at-st} s")
    st = time.time()

    Kgrid = tps_kernel(dists)
    at = time.time()
    print(f"compute_displacement_map : 5/10 - {at-st} s")
    st = time.time()

    P = np.hstack([np.ones((grid.shape[0], 1)), grid])
    at = time.time()
    print(f"compute_displacement_map : 6/10 - {at-st} s")
    st = time.time()

    src_coords = Kgrid @ W_coef + P @ A_coef
    at = time.time()
    print(f"compute_displacement_map : 7/10 - {at-st} s")
    st = time.time()

    delta = grid - src_coords
    at = time.time()
    print(f"compute_displacement_map : 8/10 - {at-st} s")
    st = time.time()

    amplitude = np.linalg.norm(delta, axis=1)
    at = time.time()
    print(f"compute_displacement_map : 9/10 - {at-st} s")
    st = time.time()

    disp_map = amplitude.reshape(H, W).astype(np.float32)
    at = time.time()
    print(f"compute_displacement_map : 10/10 - {at-st} s")
    return disp_map


# ─────────────────────────────────────────────────────────────
#  2. AFFICHAGE INTERMÉDIAIRE
# ─────────────────────────────────────────────────────────────

def _draw_checkerboard_on(ax, H, W):
    tile = 20
    checker = np.zeros((H, W))
    for i in range(0, H, tile):
        for j in range(0, W, tile):
            if (i // tile + j // tile) % 2 == 0:
                checker[i:i+tile, j:j+tile] = 0.6
            else:
                checker[i:i+tile, j:j+tile] = 0.85
    ax.imshow(checker, cmap="gray", vmin=0, vmax=1, zorder=1)

def plot_tps_comparison(image: np.ndarray, warped: np.ndarray, disp_map: np.ndarray, src: np.ndarray, dst: np.ndarray):
    H, W = image.shape[:2]
    COLORS = {"source": "#2979FF", "dest": "#FF1744", "arrow": "#FF9100"}

    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    fig.patch.set_facecolor("#1A1A2E")
    for ax in axes:
        ax.set_facecolor("#16213E")
        ax.axis("off")

    # Panneau 0 : image originale avec points et flèches
    _draw_checkerboard_on(axes[0], H, W)
    axes[0].imshow(image, zorder=2)
    axes[0].set_title("Image originale\n● source  ✕ destination  → déplacement", color="white", fontsize=10, pad=8)

    for i, (s, d) in enumerate(zip(src, dst)):
        sx, sy = s[0], s[1]
        dx, dy = d[0], d[1]
        axes[0].plot(sx, sy, "o", color=COLORS["source"], markersize=9, markeredgewidth=2, zorder=5)
        axes[0].text(sx + 5, sy - 5, str(i + 1), color=COLORS["source"], fontsize=8, fontweight="bold", zorder=6)
        axes[0].plot(dx, dy, "x", color=COLORS["dest"], markersize=9, markeredgewidth=2.5, zorder=5)
        axes[0].text(dx + 5, dy - 5, str(i + 1), color=COLORS["dest"], fontsize=8, fontweight="bold", zorder=6)
        axes[0].annotate("", xy=(dx, dy), xytext=(sx, sy),
                         arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.8), zorder=7)

    # Panneau 1 : image déformée avec repères
    _draw_checkerboard_on(axes[1], H, W)
    axes[1].imshow(warped, zorder=2)
    axes[1].set_title("Image déformée (TPS)\n● origine (src)  ✕ arrivée (dst)", color="white", fontsize=10, pad=8)

    for i, (s, d) in enumerate(zip(src, dst)):
        sx, sy = s[0], s[1]
        dx, dy = d[0], d[1]
        axes[1].plot(sx, sy, "o", color=COLORS["source"], markersize=9, markeredgewidth=2, zorder=5, markerfacecolor="none")
        axes[1].text(sx + 5, sy - 5, str(i + 1), color=COLORS["source"], fontsize=8, fontweight="bold", zorder=6)
        axes[1].plot(dx, dy, "x", color=COLORS["dest"], markersize=9, markeredgewidth=2.5, zorder=5)
        axes[1].text(dx + 5, dy - 5, str(i + 1), color=COLORS["dest"], fontsize=8, fontweight="bold", zorder=6)

    # Panneau 2 : carte de déplacement colormap jet
    # Masquage sur canal alpha si présent
    alpha_mask = image[:, :, 3] if image.shape[2] == 4 else np.ones((H, W))

    max_disp = disp_map.max()
    disp_norm = disp_map / max_disp if max_disp > 0 else disp_map

    cmap = plt.colormaps["jet"]
    disp_colored = cmap(disp_norm)[:, :, :3]
    disp_colored *= alpha_mask[:, :, np.newaxis]

    axes[2].imshow(disp_colored, zorder=2)
    axes[2].set_title(f"Carte de déplacement (jet)\nbleu=peu bougé  →  rouge=beaucoup bougé  |  max={max_disp:.1f} px", color="white", fontsize=10, pad=8)

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0, vmax=max_disp))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Déplacement (pixels)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    # Légende globale
    src_patch = mpatches.Patch(color=COLORS["source"], label="Point source (origine)")
    dst_patch = mpatches.Patch(color=COLORS["dest"],   label="Point destination (arrivée)")
    arr_patch = mpatches.Patch(color=COLORS["arrow"],  label="Flèche de déplacement")
    fig.legend(handles=[src_patch, dst_patch, arr_patch], loc="lower center", ncol=3, facecolor="#0F3460", labelcolor="white", framealpha=0.9, fontsize=9)

    fig.suptitle(f"TPS — {len(src)} points de contrôle", color="white", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  3. FONCTION PRINCIPALE D'AUTOMATISATION
# ─────────────────────────────────────────────────────────────

def auto_warp_image(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Applique la déformation TPS sur une image de manière automatique.

    Paramètres :
        image      : (H, W, C) np.ndarray (float32, valeurs entre 0 et 1)
        src_points : (N, 2) np.ndarray
        dst_points : (N, 2) np.ndarray

    Retourne :
        warped     : (H, W, C) np.ndarray déformé
    """
    print(f"⚙  Calcul de la déformation TPS ({len(src_points)} points de contrôle)…")
    warped = apply_tps(image, src_points, dst_points)
    print("✓  Déformation calculée.")

    print("⚙  Calcul de la carte de déplacement…")
    disp_map = compute_displacement_map(image, src_points, dst_points)
    print(f"✓  Déplacement max : {disp_map.max():.1f} px  |  moyen : {disp_map.mean():.1f} px")

    # Affichage intermédiaire de contrôle
    plot_tps_comparison(image, warped, disp_map, src_points, dst_points)

    return warped


# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Paramètres factices pour la structure
    # img_array = ... 
    # src_arr = np.array([[50, 50], [100, 50], [75, 100]], dtype=np.float64)
    # dst_arr = np.array([[60, 40], [110, 60], [75, 120]], dtype=np.float64)
    # warped_array = auto_warp_image(img_array, src_arr, dst_arr)
    pass