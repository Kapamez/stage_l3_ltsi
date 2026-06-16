"""
main.py
─────────────────────────────────────────────────────────────────────────────
Orchestrateur du pipeline de vérification des images de prostate.
 
Structure :
    Étape 1 — Chargement brut + dimensions
    Étape 2 — Suppression d'arrière-plan + cadre de matrice
    Étape 3 — Recadrage autour du contenu
    Étape 4 — Mise en matrice carrée diagonale (a × a)
 
Tous les résultats intermédiaires sont conservés dans `results`
(liste de dicts) pour l'ajout de nouvelles étapes.
"""
 
import napari
import numpy as np
from processing import (
    load_image,
    remove_background,
    crop_to_content,
    pad_to_diagonal_square,
    resize_to_reference_width,
    resize_cropped_to_reference,
    orient_and_face,
    flip_image_vertically,
    filtrer_doublons_y,
    create_displacement_field,
    create_displacement_map_with_center,
    get_center_of_mass
    )

from f_tps_warp_gpu import auto_warp_image

from napari_display import (
    show_raw_images,
    show_masked_images,
    show_cropped_images,
    show_cropped_rescaled_images,
    show_square_matrices,
    show_facing_images
    )


import os

# Pointe dynamiquement vers ton sous-dossier 'dlls'
dossier_dlls = os.path.join(os.path.dirname(__file__), "dlls")
os.add_dll_directory(dossier_dlls)
import cv2
print(cv2.__version__)
import math
import time

color_liste = [
    "red", "blue", "green", "yellow", "cyan", "magenta", "orange", "purple",
    "pink", "brown", "lime", "teal", "navy", "maroon", "olive", "coral",
    "turquoise", "violet", "gold", "indigo", "salmon", "plum", "khaki",
    "orchid", "sienna", "tomato", "aquamarine", "crimson", "chartreuse",
    "cornflowerblue", "darkorange", "deeppink", "dodgerblue", "firebrick",
    "forestgreen", "fuchsia", "hotpink", "indianred", "lawngreen", "mediumpurple"
]

#  PARAMÈTRES
PATHS = [
    r"",
    r"",
]

TOP_N_LIST = [1, 1]

FLIP = [0, 0]

# Pour Prostate 4 A10 -> FLIP = 1
# Pour Prostate 4 A7 -> TOP_N_LIST = 2

# Facteur de réduction de la taille des images (1 = original, 2 = divisé par 2, 4 = par 4, etc.)
DOWNSCALE_FACTOR = 6

# ── Activer / désactiver les étapes de vérification intermédiaires ──────────
SHOW_STEPS = False   # False = passe directement à l'étape face-à-face
# ────────────────────────────────────────────────────────────────────────────
LINEAR = True

def _open_viewer(title: str, step: int, total: int) -> napari.Viewer:
    full_title = f"[{step}/{total}] {title}"
    print(f"\n  → Fenêtre Napari : « {full_title} »  (fermer pour continuer)")
    return napari.Viewer(title=full_title)


def snap_au_contour_vectorise(points_cliques, liste_contours):
    # points_cliques : array de shape (N, 2)
    # liste_contours : array de shape (M, 2)
    # Calcule les distances entre chaque point_clique et chaque point du contour
    distances = np.linalg.norm(liste_contours[:, np.newaxis, :] - points_cliques, axis=2)
    # Trouve l'index du contour le plus proche pour chaque point_clique
    indices_plus_proches = np.argmin(distances, axis=0)
    # Retourne les points du contour les plus proches
    return liste_contours[indices_plus_proches]


def cs_sort(l):
    sorted_indices = np.argsort(l[:, 0])
    l_sorted = l[sorted_indices]
    return l_sorted

import numpy as np

def segmenter_ligne(ligne_complete, points_cles):
    index_cles = []
    
    # On extrait uniquement la colonne des 'y' (première colonne)
    y_ligne = ligne_complete[:, 0]
    
    for point in points_cles:
        y_cible = point[0]
        
        # Comparaison uniquement sur les valeurs de y
        correspondance = np.where(np.isclose(y_ligne, y_cible))[0]
        
        if len(correspondance) > 0:
            # S'il y a plusieurs points avec le même 'y', on prend la première occurrence
            index_cles.append(correspondance[0])
        #else:
            #print(f"Attention : La coordonnée y={y_cible} n'a pas été trouvée dans la ligne.")
            
    # S'assurer que les index sont dans l'ordre croissant
    index_cles.sort()
    
    segments = []
    
    for i in range(len(index_cles) - 1):
        debut = index_cles[i] + 1
        fin = index_cles[i+1]
        
        segment = ligne_complete[debut:fin]
        segments.append(segment)
        
    return segments


def frac_n(x):
    a = math.floor(x)
    b = math.ceil(x)
    c = x - a
    return a, b, c


def lerp_coordinates(p0: tuple, p1: tuple, t: float)-> tuple:
    """
    Calcule l'interpolation linéaire entre deux points 2D.
    p0: tuple ou liste (y, x) du point de départ
    p1: tuple ou liste (y, x) du point d'arrivée
    t: float, la proportion de 0.0 à 1.0
    """
    y = p0[0] + t * (p1[0] - p0[0])
    x = p0[1] + t * (p1[1] - p0[1])
    
    return (y, x)


def arr2tup(arr: np.ndarray):
    return (arr[0], arr[1])

def get_anchor_points(img: np.ndarray, left: bool = True, n_points: int = 20) -> np.ndarray:
    """
    Retourne N points régulièrement espacés sur le côté opposé à la déformation,
    à utiliser comme points de contrôle fixes (src == dst) pour ancrer la TPS.

    Args:
        img      : image RGBA (H, W, 4)
        left     : True = ancres à gauche, False = ancres à droite
                   (mettre le côté OPPOSÉ à votre déformation)
        n_points : nombre de points d'ancrage

    Returns:
        (N, 2) np.ndarray en [y, x]
    """
    coords = np.array(get_half_shape_coordinates(img, left=left))

    if len(coords) == 0:
        return np.empty((0, 2))

    # Sous-échantillonnage régulier pour n_points bien répartis en Y
    indices = np.linspace(0, len(coords) - 1, n_points, dtype=int)
    return coords[indices].astype(np.float64)

def get_center_column_coordinates(img: np.ndarray) -> list[list[int]]:
    """
    Retourne les coordonnées [y, x] des pixels non-transparents
    situés dans la colonne la plus proche du centre de masse en X.
    """
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError("L'image doit être au format RGBA (4 canaux).")

    alpha_channel = img[:, :, 3]
    coords = np.argwhere(alpha_channel > 0)

    if len(coords) == 0:
        return []

    M = cv2.moments(alpha_channel)
    if M["m00"] != 0:
        center_x = M["m10"] / M["m00"]
    else:
        center_x = np.mean(coords[:, 1])

    # Colonne entière la plus proche du centre
    col = int(round(center_x))

    # Tous les pixels non-transparents dans cette colonne
    ys = np.where(alpha_channel[:, col] > 0)[0]

    return [[int(y), col] for y in ys]

def get_half_shape_coordinates(img: np.ndarray, left: bool = True) -> list[list[int]]:
    """
    Parcourt une image RGBA et retourne les coordonnées des pixels de la forme
    situés à gauche ou à droite de son centre de masse.
    
    Args:
        img (np.ndarray): Image source au format (H, W, 4) (RGB + Alpha).
        left (bool): Si True, retourne les points à gauche du centre.
                     Si False, retourne les points à droite.
                     
    Returns:
        list[list[int]]: Liste de coordonnées sous la forme [[y, x], [y, x], ...]
    """
    # Vérification des dimensions : on s'assure qu'on a bien le canal alpha
    if img.ndim != 3 or img.shape[2] != 4:
        raise ValueError("L'image doit être au format RGBA (4 canaux).")
        
    # Extraction du masque de la forme via le canal Alpha (valeurs > 0)
    alpha_channel = img[:, :, 3]
    
    # Trouver toutes les coordonnées (y, x) où la forme est présente
    coords = np.argwhere(alpha_channel > 0)
    
    if len(coords) == 0:
        return []  # Retourne une liste vide si l'image est entièrement transparente
        
    # Calcul du centre de masse de la forme (axe X)
    # L'utilisation de cv2.moments correspond à la fonction get_center_of_mass de ton code
    M = cv2.moments(alpha_channel)
    if M["m00"] != 0:
        center_x = M["m10"] / M["m00"]
    else:
        # Solution de repli si le moment d'ordre 0 est nul : moyenne géométrique des X
        center_x = np.mean(coords[:, 1])
        
    # Filtrer les coordonnées selon leur position par rapport à center_x
    if left:
        filtered_coords = coords[coords[:, 1] < center_x]
    else:
        filtered_coords = coords[coords[:, 1] > center_x]
        
    # Convertir l'array NumPy en une liste de listes native Python
    return filtered_coords.tolist()


def find_max_in_list_of_lists(lst):
    """
    Trouve la valeur maximale dans une liste de listes.
    
    Args:
        lst (list): Liste de listes contenant des valeurs numériques.
        
    Returns:
        float: La valeur maximale trouvée dans la liste de listes.
    """
    max_value = float('-inf')
    for sublist in lst:
        for value in sublist:
            if value > max_value:
                max_value = value
    return max_value


def downscale_image(image_data: dict, factor: int) -> dict:
    """Réduit la taille de l'image d'un facteur donné."""
    if factor <= 1:
        return image_data
    
    orig_h, orig_w = image_data["h"], image_data["w"]
    new_h = orig_h // factor
    new_w = orig_w // factor
    
    # Utilisation de cv2.INTER_AREA, idéal pour la réduction (downsampling) d'images
    resized_arr = cv2.resize(image_data["raw"], (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    print(f"  ↳ Image réduite (facteur {factor}) : {orig_h} × {orig_w} px → {new_h} × {new_w} px")
    return {**image_data, "raw": resized_arr, "h": new_h, "w": new_w}


def run_pipeline(paths: list[str], top_n_list: list[int]) -> list[dict]:
    """
    Pipeline complet.
    Avec SHOW_STEPS=True, chaque étape intermédiaire ouvre sa propre fenêtre
    Napari (à fermer pour continuer). Utile pour déboguer les dimensions.
    """
    TOTAL = 6 if SHOW_STEPS else 1

    # ── Étape 1 : Chargement brut + redimensionnement ────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 1 — Chargement brut")
    print("═" * 60)
    results = [load_image(p) for p in paths]
    
    # Réduction de la taille de l'image selon le facteur défini
    if DOWNSCALE_FACTOR > 1:
        for i in range(len(results)):
            results[i] = downscale_image(results[i], DOWNSCALE_FACTOR)

    # Apply vertical flip based on FLIP list
    for i in range(len(results)):
        if FLIP[i] == 1:
            results[i] = flip_image_vertically(results[i])
            print(f"  ↳ Image {i + 1} flipped vertically")
    
    results = resize_to_reference_width(results)

    if SHOW_STEPS:
        viewer = _open_viewer("Images brutes", 1, TOTAL)
        show_raw_images(viewer, results)
        napari.run()

    # ── Étape 2 : Suppression d'arrière-plan ─────────────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 2 — Suppression d'arrière-plan")
    print("═" * 60)
    for i in range(len(results)):
        print(f"\n  Image {i + 1} :")
        results[i] = remove_background(results[i], top_n=top_n_list[i])

    if SHOW_STEPS:
        viewer = _open_viewer("Masques + cadres", 2, TOTAL)
        show_masked_images(viewer, results)
        napari.run()

    # ── Étape 3 : Recadrage ──────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 3 — Recadrage autour du contenu")
    print("═" * 60)
    for i in range(len(results)):
        print(f"\n  Image {i + 1} :")
        results[i] = crop_to_content(results[i])

    if SHOW_STEPS:
        viewer = _open_viewer("Morceaux recadrés", 3, TOTAL)
        show_cropped_images(viewer, results)
        napari.run()

    # ── Étape 3b : Mise à l'échelle des crops ────────────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 3b — Mise à l'échelle des crops (référence : image 1)")
    print("═" * 60)
    results = resize_cropped_to_reference(results)

    if SHOW_STEPS:
        viewer = _open_viewer("Crops mis à l'échelle", 4, TOTAL)
        show_cropped_rescaled_images(viewer, results)
        napari.run()

    # ── Étape 4 : Matrices carrées diagonales ────────────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 4 — Matrices carrées diagonales")
    print("═" * 60)
    for i in range(len(results)):
        print(f"\n  Image {i + 1} :")
        results[i] = pad_to_diagonal_square(results[i])

    if SHOW_STEPS:
        viewer = _open_viewer("Matrices carrées", 5, TOTAL)
        show_square_matrices(viewer, results)
        napari.run()

    # ── Étape 6 : Orientation face à face ────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Étape 6 — Orientation et alignement face à face")
    print("═" * 60)
    sides = ["left_piece", "right_piece"]
    for i in range(len(results)):
        print(f"\n  Image {i + 1} :")
        results[i] = orient_and_face(results[i], side=sides[i])

    viewer = _open_viewer("Alignement Face à Face", TOTAL, TOTAL)
    show_facing_images(viewer, results)
    napari.run()

    print("\n" + "═" * 60)
    print("  Pipeline terminé.")
    print("═" * 60 + "\n")

    # Récupérer les données
    img_gauche = results[0]
    img_droite = results[1]

    if "ligne_mediane" in img_gauche:
        ligne_med = img_gauche["ligne_mediane"]
        trans_d = img_droite["translation_droite"]
        seg_g = img_gauche["segment_coupe"]
        seg_d = img_droite["segment_coupe"]

        print(f"  ↳ Ligne médiane générée : {len(ligne_med)} points.")
        print(f"  ↳ Translation du morceau droit : Y = {trans_d[0]:.2f}, X = {trans_d[1]:.2f}")
        print(f"  ↳ Segment gauche : {len(seg_g)} points | Segment droit : {len(seg_d)} points")

    return results

if __name__ == "__main__":
    main_time_start = time.time()
    r = run_pipeline(PATHS, top_n_list=TOP_N_LIST)


img_gauche = r[0]
print(img_gauche.keys())

img_droite = r[1]
print(img_droite.keys())

if "ligne_mediane" not in img_gauche or "segment_coupe" not in img_gauche:
    raise RuntimeError(
        "Appuie sur 'f' dans la fenêtre Napari avant de la fermer "
        "pour générer la ligne médiane et les segments de coupe."
    )

seg_g = img_gauche["segment_coupe"]
seg_d = img_droite["segment_coupe"]
p1g = img_gauche["p1g"]
p2g = img_gauche["p2g"]

# Axe Y converti en entier et filtré
seg_g2 = filtrer_doublons_y(seg_g, select_max=True)
seg_d2 = filtrer_doublons_y(seg_d, select_max=False)

y_g = seg_g2[:, 0]
x_g = seg_g2[:, 1]

y_d = seg_d2[:, 0]
x_d = seg_d2[:, 1]

# Préparation des images
img_g = img_gauche["rotated"]
img_d_raw = img_droite["rotated"]
trans_d = img_droite["translation_droite"]
translation_y, translation_x = trans_d[0], trans_d[1]

new_width_d = img_d_raw.shape[1] + int(max(0, translation_x))
new_height_d = img_d_raw.shape[0] + int(max(0, translation_y))
img_d = cv2.warpAffine(img_d_raw, np.float32([[1, 0, translation_x], [0, 1, translation_y]]), (new_width_d, new_height_d))

height_g, width_g = img_g.shape[:2]
height_d, width_d = img_d.shape[:2]

y_all_g, y_all_d = np.arange(height_g), np.arange(height_d)
center_g, center_d = get_center_of_mass(img_g), get_center_of_mass(img_d)

class FusionApp:
    def __init__(self):
        self.pts_g = None
        self.pts_d = None
        self.viewer = napari.Viewer(title="Fusion Strict 2D (Zéro superposition)")
        self.viewer.add_image(img_g, name="Morceau Gauche", blending="additive")
        self.viewer.add_image(img_d, name="Morceau Droit", blending="additive")
        #self.viewer.add_points(seg_g2, name="Segment Gauche", face_color="green", size=1)
        #self.viewer.add_points(seg_d2, name="Segment Droit", face_color="blue", size=1)
        #self.viewer.add_points(np.vstack((p1g, p2g)), name="Points Extremes", face_color="magenta", size=2, symbol="cross")
        self.viewer.add_points(np.empty((0, 2)), name="Repères Gauche", face_color="red", size=10, symbol="cross")
        self.viewer.add_points(np.empty((0, 2)), name="Repères Droite", face_color="blue", size=10, symbol="cross")
        print("\n" + "═" * 60)
        print("  Mode d'alignement SANS superposition")
        print("═" * 60)
        print("  1. Cliquez des points ROUGES sur le GAUCHE (proches de la coupe)")
        print("  2. Cliquez des points BLEUS sur le DROIT (même ordre)")
        print("  3. Appuyez sur 'e' pour fermer le tissu 2D exactement sur la ligne médiane")
        print("═" * 60)
        self.viewer.bind_key('e', self.apply_fusion)

    def apply_fusion(self, v):
        self.pts_g = v.layers["Repères Gauche"].data
        self.pts_d = v.layers["Repères Droite"].data

app = FusionApp()
napari.run()

pts_g = app.pts_g
pts_d = app.pts_d

pts_g_seg = snap_au_contour_vectorise(pts_g, seg_g2)
pts_d_seg = snap_au_contour_vectorise(pts_d, seg_d2)

if SHOW_STEPS:
    print("\n"+"--"*30)
    print("Points placés à gauche :")
    print(pts_g)
    print(f"Nombre de points : {len(pts_g)}")
    print("--"*30)
    print("Points placés à droite :")
    print(pts_d)
    print(f"Nombre de points : {len(pts_d)}")

if len(pts_g) != len(pts_d):
    print("Erreur : nombre de points placés différents.")
    print("--"*30)

if SHOW_STEPS:
    viewer = napari.Viewer(title="Affichage des points")
    viewer.add_points(seg_g2, name="Segment Gauche", face_color="green", size=1)
    viewer.add_points(seg_d2, name="Segment Droit", face_color="blue", size=1)
    viewer.add_points(np.vstack((p1g, p2g)), name="Points Extremes", face_color="magenta", size=1)
    viewer.add_points(pts_g, name="Repères Gauche", face_color="red", size=1)
    viewer.add_points(pts_d, name="Repères Droite", face_color="yellow", size=1)
    viewer.add_points(pts_g_seg, name="Repères gauche placés", face_color="red", size=1)
    viewer.add_points(pts_d_seg, name="Repères droite placés", face_color="yellow", size=1)
    napari.run()

extremes = np.vstack((p1g, p2g))
# Conserver uniquement la partie entière sur la première colonne (axe y)
extremes[:, 0] = np.trunc(extremes[:, 0])

if SHOW_STEPS:
    print("Extremes :")
    print(extremes)
    print("--"*30)

# Ajout des points extremes à la liste de points de référence
pts_g_seg = np.vstack((pts_g_seg, extremes))
pts_d_seg = np.vstack((pts_d_seg, extremes))

# Tri par ordre croissant suivant y.
pts_g_seg = cs_sort(pts_g_seg)
pts_d_seg = cs_sort(pts_d_seg)


########################################
########################################
if SHOW_STEPS:
    for ele in pts_g_seg:
        if ele in seg_g2:
            print(ele)
        else:
            print("Erreur")
    print(f"Premier : {seg_g2[0]}")
    print(f"Dernier : {seg_g2[-1]}")

    print("--"*30)
    for ele in pts_d_seg:
        if ele in seg_d2:
            print(ele)
        else:
            print("Erreur")
    print(f"Premier : {seg_d2[0]}")
    print(f"Dernier : {seg_d2[-1]}")
########################################
########################################


part_g = segmenter_ligne(seg_g2, pts_g_seg)
part_d = segmenter_ligne(seg_d2, pts_d_seg)


print("--"*30)
print("Nombre de segments :")
print(f"Gauche : {len(part_g)}")
print(f"Droite : {len(part_d)}")

if SHOW_STEPS:
    viewer = napari.Viewer(title="Affichage des segments")
    for x in range(len(part_g)):
        viewer.add_points(part_g[x], name=f"Segment Gauche {x+1}", face_color=color_liste[x], size=1)
    for x in range(len(part_d)):
        viewer.add_points(part_d[x], name=f"Segment Droit {x+1}", face_color=color_liste[x], size=1)
    viewer.add_points(pts_g_seg, name="Repères Gauche", face_color="white", size=3)
    viewer.add_points(pts_d_seg, name="Repères Droite", face_color="white", size=3)
    napari.run()
print("--"*30)

# part_g = liste des coordonnées par petits segments de gauche


#g_base = get_half_shape_coordinates(img_g, left=True)
g_base = get_center_column_coordinates(img_g)

#d_base = get_half_shape_coordinates(img_d, left=False)
d_base = get_center_column_coordinates(img_d)

g_dest = []
d_dest = []

# Boucle pour calcul de position cible
for x in range(len(part_g)):
    g_inter_s = int(pts_g_seg[x+1][0] - pts_g_seg[x][0])
    d_inter_s = int(pts_d_seg[x+1][0] - pts_d_seg[x][0])

    if SHOW_STEPS:
        print("Intermédiaires :")
        print(g_inter_s)
        print(d_inter_s)
        print("--"*5)

    ratio_g = d_inter_s/g_inter_s
    ratio_d = g_inter_s/d_inter_s

    delta_base_g = int(pts_d_seg[x][0] - pts_d_seg[0][0])
    delta_base_d = int(pts_g_seg[x][0] - pts_g_seg[0][0])

    g_dest.append([int(pts_d_seg[x][0]-pts_d_seg[0][0]), int(pts_d_seg[x][0]-pts_d_seg[0][0]), 0])
    d_dest.append([int(pts_g_seg[x][0]-pts_g_seg[0][0]), int(pts_g_seg[x][0]-pts_g_seg[0][0]), 0])

    for v in range(g_inter_s - 1):
        k = v*ratio_g + delta_base_g
        und, up, dec = frac_n(k)

        g_dest.append([und, up, dec])
    ######################################
    for s in range(d_inter_s-1):
        k = s*ratio_d + delta_base_d
        und, up, dec = frac_n(k)

        d_dest.append([und, up, dec])

g_dest.append([int(pts_d_seg[-1][0]-pts_d_seg[0][0]), int(pts_d_seg[-1][0]-pts_d_seg[0][0]), 0])
d_dest.append([int(pts_g_seg[-1][0]-pts_g_seg[0][0]), int(pts_g_seg[-1][0]-pts_g_seg[0][0]), 0])

if len(seg_d2) < len(seg_g2):
    for i in range(len(seg_g2) - len(seg_d2)):
        seg_d2 = np.vstack((seg_d2,seg_d2[-1]))

if len(seg_g2) < len(seg_d2):
    for i in range(len(seg_d2) - len(seg_g2)):
        seg_g2 = np.vstack((seg_g2,seg_g2[-1]))

if SHOW_STEPS:
    print(f"longueur g_dest : {len(g_dest)} | Max : {find_max_in_list_of_lists(g_dest)}")
    print(f"longueur d_dest : {len(d_dest)} | Max : {find_max_in_list_of_lists(d_dest)}")
    print(f"longueur seg_g2 : {len(seg_g2)}")
    print(f"longueur seg_d2 : {len(seg_d2)}")
    print("\n")
    for i in range(10):
        print(f"g_dest[{i}] : {g_dest[i]}")
        print(f"d_dest[{i}] : {d_dest[i]}")
        print("-")

    print("--"*30)

g_displace = []
d_displace = []
#g_displace.extend(get_half_shape_coordinates(img_g, left=True))
g_displace.extend(get_center_column_coordinates(img_g))
#d_displace.extend(get_half_shape_coordinates(img_d, left=False))
d_displace.extend(get_center_column_coordinates(img_d))

if SHOW_STEPS:
    print("Segments de part_d :")
    for els in part_d:
        print(len(els))
    print("--"*30)

for f in range(len(g_dest)):
    ng1 = int(g_dest[f][0])
    ng2 = int(g_dest[f][1])
    t = g_dest[f][2]
    if SHOW_STEPS:
        print(ng1)
        print(ng2)
        print("--"*5)

    pixel_a = arr2tup(seg_d2[ng1])
    pixel_b = arr2tup(seg_d2[ng2])

    res = lerp_coordinates(pixel_a, pixel_b, t)
    coords_rel = (res + seg_g2[f])/2
    g_displace.append([coords_rel[0], coords_rel[1]])

    g_base.append(seg_g2[f])

for f in range(len(d_dest)):
    nd1 = int(d_dest[f][0])
    nd2 = int(d_dest[f][1])
    t = d_dest[f][2]
    if SHOW_STEPS:
        print(nd1)
        print(nd2)
        print("--"*5)
    
    pixel_a = arr2tup(seg_g2[nd1])
    pixel_b = arr2tup(seg_g2[nd2])

    res = lerp_coordinates(pixel_a, pixel_b, t)
    coords_rel = (res + seg_d2[f])/2
    d_displace.append([coords_rel[0], coords_rel[1]])

    d_base.append(seg_d2[f])

print(f"Nombre de points dans g_base : {len(g_base)}")
print(f"Longueur de g_displace : {len(g_displace)}")
if SHOW_STEPS:
    viewer = napari.Viewer(title="Points de base")
    viewer.add_points(g_base, name="base gauche", face_color="white", size=1)
    viewer.add_points(g_displace, name=f"g_displace", face_color=color_liste[9], size=1)
    viewer.add_points(d_base, name="base droite", face_color="purple", size=1)
    viewer.add_points(d_displace, name=f"d_displace", face_color=color_liste[11], size=1)
    for x in range(len(part_g)):
        viewer.add_points(part_g[x], name=f"Segment Gauche {x+1}", face_color=color_liste[x], size=1)
    for x in range(len(part_d)):
        viewer.add_points(part_d[x], name=f"Segment Droit {x+1}", face_color=color_liste[x], size=1)
    napari.run()



# 1. Normaliser les images en float32 [0, 1]
img_g_float = img_g.astype(np.float32) / 255.0
img_d_float = img_d.astype(np.float32) / 255.0

# 2. Convertir les points [y, x] → [x, y] pour le solveur TPS
anchors_g = get_anchor_points(img_g, left=True, n_points=20)
src_g = np.vstack([np.array(g_base,     dtype=np.float64), anchors_g])
dst_g = np.vstack([np.array(g_displace, dtype=np.float64), anchors_g])

anchors_d = get_anchor_points(img_d, left=False, n_points=20)
src_d = np.vstack([np.array(d_base,     dtype=np.float64), anchors_d])
dst_d = np.vstack([np.array(d_displace, dtype=np.float64), anchors_d])

# 3. Appliquer le warp (déclenche les affichages Matplotlib successifs)
print("\n" + "═" * 60)
print("  Application du TPS - Morceau Gauche")
print("═" * 60)
warped_g = auto_warp_image(img_g_float, src_g, dst_g)

print("\n" + "═" * 60)
print("  Application du TPS - Morceau Droit")
print("═" * 60)


# ── Rogner img_d sur la boîte englobante non-transparente ──────────────
MARGE = 30  # pixels de sécurité autour du contenu (évite les artefacts de bord)

alpha_d = img_d_float[:, :, 3]
rows_nz = np.where(np.any(alpha_d > 0, axis=1))[0]
cols_nz = np.where(np.any(alpha_d > 0, axis=0))[0]

y_min = max(0,               rows_nz[0]  - MARGE)
y_max = min(img_d_float.shape[0], rows_nz[-1] + MARGE + 1)
x_min = max(0,               cols_nz[0]  - MARGE)
x_max = min(img_d_float.shape[1], cols_nz[-1] + MARGE + 1)

img_d_crop = img_d_float[y_min:y_max, x_min:x_max]

# Décaler les points de contrôle dans le repère du crop
offset = np.array([y_min, x_min], dtype=np.float64)
src_d_crop = src_d - offset
dst_d_crop = dst_d - offset

print(f"img_d rognée : {img_d_float.shape} → {img_d_crop.shape}  "
      f"(gain : {img_d_float.shape[0]*img_d_float.shape[1] / (img_d_crop.shape[0]*img_d_crop.shape[1]):.1f}×)")

# ── TPS sur le crop ─────────────────────────────────────────────────────
warped_d_crop = auto_warp_image(img_d_crop, src_d_crop, dst_d_crop)

# ── Recoller dans un canvas aux dimensions d'origine ───────────────────
warped_d = np.zeros_like(img_d_float)
warped_d[y_min:y_max, x_min:x_max] = warped_d_crop



# warped_d = auto_warp_image(img_d_float, src_d, dst_d)

# 4. Affichage combiné final dans Napari
print("\n" + "═" * 60)
print("  Affichage du résultat combiné")
print("═" * 60)

viewer_final = napari.Viewer(title="Résultat Final Combiné (TPS)")

# Utilisation du mode "translucent" pour respecter la transparence (canal alpha)
viewer_final.add_image(warped_g, name="TPS Gauche", blending="translucent")
viewer_final.add_image(warped_d, name="TPS Droite", blending="translucent")

# Temps total d'exécution
total_time = time.time() - main_time_start
print("\n" + "═" * 60)
print(f"  Temps total d'exécution : {total_time:.2f} secondes")


napari.run()
