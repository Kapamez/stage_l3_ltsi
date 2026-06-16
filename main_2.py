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
from processing import (
    load_image,
    remove_background,
    crop_to_content,
    pad_to_diagonal_square,
    resize_to_reference_width,
    resize_cropped_to_reference,
    orient_and_face,
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

from elastic_fusion import apply_elastic_from_pipeline, interactive_elastic_manual

#  PARAMÈTRES
PATHS = [
    r"C:\\Users\\Gustave\\Documents\\Education\\L3_EEEA\\Stage\\Pour_Arthur\\Prostate4\\A9_x1.25_z0.tif",
    r"C:\\Users\\Gustave\\Documents\\Education\\L3_EEEA\\Stage\\Pour_Arthur\\Prostate4\\A10_x1.25_z0.tif",
]

TOP_N_LIST = [1, 1]

FLIP = [0, 1]

# ── Activer / désactiver les étapes de vérification intermédiaires ──────────
SHOW_STEPS = False   # False = passe directement à l'étape face-à-face
# ────────────────────────────────────────────────────────────────────────────

MANUAL = True

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

# ── Vérification que la ligne médiane existe ──
if "ligne_mediane" not in img_gauche or "segment_coupe" not in img_gauche:
    raise RuntimeError(
        "Appuie sur 'f' dans la fenêtre Napari avant de la fermer "
        "pour générer la ligne médiane et les segments de coupe."
    )


if MANUAL == False:
    # ── Fusion élastique ──
    img_g_elastic, img_d_elastic, trans_g, trans_d = apply_elastic_from_pipeline(
        img_gauche,
        img_droite,
        n_ctrl=40,           # 40 points de repère sur la coupe suffisent
        sigma_factor=0.20    # 0.15=très rigide | 0.25=souple | 0.40=très souple
    )

    # ── Affichage final ──
    viewer = napari.Viewer(title="Fusion Élastique")
    viewer.add_image(img_g_elastic, name="Gauche", blending="additive", translate=trans_g)
    viewer.add_image(img_d_elastic, name="Droit", blending="additive", translate=trans_d)

    # Optionnel : afficher les repères utilisés
    viewer.add_points(img_gauche["ligne_mediane"], name="Ligne médiane",
                    face_color="lime", size=3)
    napari.run()

if MANUAL == True:
    viewer = napari.Viewer(title="Fusion manuelle")

    # Récupère tes images déjà orientées (après orient_and_face)
    img_g = img_gauche["rotated"]
    img_d = img_droite["rotated"]
    trans_d = img_droite["translation_droite"]

    interactive_elastic_manual(
        viewer,
        img_left=img_g,
        img_right=img_d,
        left_translate=(0, 0),
        right_translate=trans_d,   # Important : compenser la translation du droit
        default_sigma=150.0        # Ajuste selon ta résolution
    )

    napari.run()