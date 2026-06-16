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
    filtrer_doublons_y,
    create_displacement_field,
    create_displacement_map,
    get_center_of_mass,
    create_displacement_map_with_center,
    flip_image_vertically
    )

from napari_display import (
    show_raw_images,
    show_masked_images,
    show_cropped_images,
    show_cropped_rescaled_images,
    show_square_matrices,
    show_facing_images,
    )
import cv2


#  PARAMÈTRES
PATHS = [
    r"",
    r"",
]


# Pour Prostate 4 A10 -> FLIP = 1
# Pour Prostate 4 A7 -> TOP_N_LIST = 2


TOP_N_LIST = [1, 1]

FLIP = [0, 0]

# ── Activer / désactiver les étapes de vérification intermédiaires ──────────
SHOW_STEPS = False   # False = passe directement à l'étape face-à-face
# ────────────────────────────────────────────────────────────────────────────
LINEAR = True

def _open_viewer(title: str, step: int, total: int) -> napari.Viewer:
    full_title = f"[{step}/{total}] {title}"
    print(f"\n  → Fenêtre Napari : « {full_title} »  (fermer pour continuer)")
    return napari.Viewer(title=full_title)


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
    r = run_pipeline(PATHS, top_n_list=TOP_N_LIST)


img_gauche = r[0]
img_droite = r[1]

if "ligne_mediane" not in img_gauche or "segment_coupe" not in img_gauche:
    raise RuntimeError(
        "Appuie sur 'f' dans la fenêtre Napari avant de la fermer "
        "pour générer la ligne médiane et les segments de coupe."
    )

ligne_med = img_gauche["ligne_mediane"]
seg_g = seg_g2 = img_gauche["segment_coupe"]
seg_d = seg_d2 = img_droite["segment_coupe"]

if SHOW_STEPS:
    ########################################
    ########################################
    # Affichage des points :
    viewer = napari.Viewer(title="Points d'intérêt")
    viewer.add_points(ligne_med, name="ligne_mediane", face_color="red", size=2)
    viewer.add_points(seg_g, name="segment_gauche", face_color="green", size=2)
    viewer.add_points(seg_d, name="segment_droit", face_color="blue", size=2)
    napari.run()
    ########################################
    ########################################

# Axe Y converti en entier.
ligne_med[:, 0] = ligne_med[:, 0].astype(int)
seg_g2[:, 0] = seg_g[:, 0].astype(int)
seg_d2[:, 0] = seg_d[:, 0].astype(int)

# Supprimer les doublons de l'axe Y dans les coupures
seg_g2 = filtrer_doublons_y(seg_g2, select_max=True)
seg_d2 = filtrer_doublons_y(seg_d2, select_max=False)

# Extraction des axes
y_med = ligne_med[:, 0]
x_med = ligne_med[:, 1]
y_g = seg_g2[:, 0]
x_g = seg_g2[:, 1]
y_d = seg_d2[:, 0]
x_d = seg_d2[:, 1]

# Interpolation des axes
x_g_interp = np.interp(y_med, y_g, x_g)
x_d_interp = np.interp(y_med, y_d, x_d)

# Calcul des distances entre coupure et médiane sur l'axe X
dist_x_g = np.abs(x_med - x_g_interp)
dist_x_d = np.abs(x_med - x_d_interp)

print("\nDistances :")
print(dist_x_g)
print(dist_x_d)

delta_max_g = max(dist_x_g)
delta_max_d = max(dist_x_d)

print("\nMaximum des distances :")
print(delta_max_g)
print(delta_max_d)

transform_g = -(delta_max_g - dist_x_g) # Déplacement vers la gauche
transform_d = delta_max_d - dist_x_d # Déplacement vers la droite

print("\nTransformations :")
print(transform_g)
print(transform_d)

# Dictionnaires des transformations :
transform_g_dict = dict(zip(y_med, transform_g))
transform_d_dict = dict(zip(y_med, transform_d))


# Récupérer les Y min et max des transformations
y_min_g = min(transform_g_dict.keys())
y_max_g = max(transform_g_dict.keys())
y_min_d = min(transform_d_dict.keys())
y_max_d = max(transform_d_dict.keys())


# Appliquer les transformations aux segments filtrés
seg_g_transformed = np.array([
    [y, x + (
        transform_g_dict[y_min_g] if y < y_min_g else # Si y plus petit que premier de la médiane (au dessus)
        transform_g_dict[y_max_g] if y > y_max_g else # Si y plus grand que le dernier point de la médiane (en dessous)
        transform_g_dict[y]
    )]
    for y, x in seg_g2
])

seg_d_transformed = np.array([
    [y, x + (
        transform_d_dict[y_min_d] if y < y_min_d else
        transform_d_dict[y_max_d] if y > y_max_d else
        transform_d_dict[y]
    )]
    for y, x in seg_d2
])

"""
# Afficher les résultats
print("\nseg_g transformé :")
print(seg_g_transformed)
print("\nseg_d transformé :")
print(seg_d_transformed)
"""

if SHOW_STEPS:
    ########################################
    ########################################
    # Affichage des points avec transformation :
    viewer = napari.Viewer(title="Points d'intérêt")
    viewer.add_points(ligne_med, name="ligne_mediane", face_color="red", size=2)
    viewer.add_points(seg_g, name="segment_gauche", face_color="green", size=2)
    viewer.add_points(seg_g_transformed, name="segment_gauche_transforme", face_color="yellow", size=2)
    viewer.add_points(seg_d, name="segment_droit", face_color="blue", size=2)
    viewer.add_points(seg_d_transformed, name="segment_droit_transforme", face_color="orange", size=2)
    napari.run()
    ########################################
    ########################################



########################################
########################################
# Dimensions des images
img_g = img_gauche["rotated"]
img_d_raw = img_droite["rotated"]

# Placer physiquement le morceau de droite à sa vraie position AVANT les déformations
trans_d = img_droite["translation_droite"]
translation_y, translation_x = trans_d[0], trans_d[1]

new_width_d = img_d_raw.shape[1] + int(max(0, translation_x))
new_height_d = img_d_raw.shape[0] + int(max(0, translation_y))

img_d = cv2.warpAffine(
    img_d_raw,
    np.float32([[1, 0, translation_x], [0, 1, translation_y]]),
    (new_width_d, new_height_d)
)

height_g, width_g = img_g.shape[:2]
height_d, width_d = img_d.shape[:2]

# Contrôle :
print("\nDimensions :")
print(height_g, width_g)
print(height_d, width_d)

# Tableau de Y pour les images.
y_all_g = np.arange(height_g)
y_all_d = np.arange(height_d)



# Calculer les centres de masse pour les images gauches et droites
center_g = get_center_of_mass(img_g)
center_d = get_center_of_mass(img_d)

print("\nCentres de masse :")
print(f"Image gauche : {center_g}")
print(f"Image droite : {center_d}")

# Obtenir les coordonnées X de la coupe pour chaque Y
cut_x_g_all = np.interp(y_all_g, seg_g2[:, 0], seg_g2[:, 1])
cut_x_d_all = np.interp(y_all_d, seg_d2[:, 0], seg_d2[:, 1])

# Appliquer aux images gauches et droites
displacement_g = create_displacement_field(y_all_g, transform_g_dict, y_min_g, y_max_g)
displacement_d = create_displacement_field(y_all_d, transform_d_dict, y_min_d, y_max_d)


def make_colored_deformation_map(def_amp, max_def, img_rgba):
    if max_def > 0:
        normalized = np.clip((def_amp / max_def) * 255, 0, 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(def_amp, dtype=np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    
    alpha_mask = img_rgba[:, :, 3] if img_rgba.shape[2] == 4 else (cv2.cvtColor(img_rgba, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    return np.dstack((colored_rgb, alpha_mask))




if LINEAR:
    # Appliquer les champs de déplacement avec déformation linéaire
    map_x_g, map_y_g, def_amp_g = create_displacement_map_with_center(height_g, width_g, -displacement_g, center_g[1], cut_x_coords=cut_x_g_all, side="left")
    img_g_transformed = cv2.remap(img_g, map_x_g, map_y_g, cv2.INTER_LINEAR)

    map_x_d, map_y_d, def_amp_d = create_displacement_map_with_center(height_d, width_d, -displacement_d, center_d[1], cut_x_coords=cut_x_d_all, side="right")
    img_d_transformed = cv2.remap(img_d, map_x_d, map_y_d, cv2.INTER_LINEAR)

    max_deformation = max(delta_max_g, delta_max_d)
    colored_def_g = make_colored_deformation_map(def_amp_g, max_deformation, img_g_transformed)
    colored_def_d = make_colored_deformation_map(def_amp_d, max_deformation, img_d_transformed)

else:
    # Appliquer le déplacement à l'image gauche
    map_x_g, map_y_g = create_displacement_map(height_g, width_g, -displacement_g)
    img_g_transformed = cv2.remap(img_g, map_x_g, map_y_g, cv2.INTER_LINEAR)

    # Appliquer le déplacement à l'image droite
    map_x_d, map_y_d = create_displacement_map(height_d, width_d, -displacement_d)
    img_d_transformed = cv2.remap(img_d, map_x_d, map_y_d, cv2.INTER_LINEAR)


########################################
########################################
# Affichage total :
viewer = napari.Viewer(title="Transformations")
viewer.add_image(img_g_transformed, name="image_gauche_transforme", blending="additive", translate=[0, delta_max_g])
viewer.add_image(img_d_transformed, name="image_droit_transforme", blending="additive", translate=[0, -delta_max_d])
if LINEAR:
    viewer.add_image(colored_def_g, name="amplitude_deformation_gauche", translate=[0, delta_max_g], opacity=0.7)
    viewer.add_image(colored_def_d, name="amplitude_deformation_droit", translate=[0, -delta_max_d], opacity=0.7)
viewer.add_points(ligne_med, name="ligne_mediane", face_color="red", size=2)
viewer.add_points(seg_g, name="segment_gauche", face_color="green", size=2, visible=False)
viewer.add_points(seg_g_transformed, name="segment_gauche_transforme", face_color="yellow", size=2, translate=[0, delta_max_g])
viewer.add_points(seg_d, name="segment_droit", face_color="blue", size=2, visible=False)
viewer.add_points(seg_d_transformed, name="segment_droit_transforme", face_color="orange", size=2, translate=[0, -delta_max_d])
napari.run()
########################################
########################################
