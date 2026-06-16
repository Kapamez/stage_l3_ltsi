"""
processing.py
─────────────────────────────────────────────────────────────────────────────
Fonctions de traitement d'image : chargement, masquage, recadrage, mise en
matrice carrée diagonale.

Chaque image est traitée une seule fois (masque calculé une seule fois)
et toutes les données intermédiaires sont retournées dans un dict,
ce qui facilite l'ajout de nouvelles étapes.
"""

import math
import numpy as np
import cv2
from PIL import Image

from tif_mask import mask as compute_mask
from mask_image import apply_mask


# ─────────────────────────────────────────────────────────────────────────────
# Chargement brut
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path: str) -> dict:
    """
    Charge une image TIF sans aucun pré-traitement.

    Retourne un dict avec :
        'path'  : chemin d'accès
        'raw'   : array numpy brut (H, W) ou (H, W, C)
        'h', 'w': dimensions originales
    """
    img = Image.open(path)
    raw = np.array(img)
    h, w = raw.shape[:2]
    print(f"  ↳ Chargé  : {path}")
    print(f"    Dimensions brutes : {h} × {w} px  (H × W)")
    return {"path": path, "raw": raw, "h": h, "w": w}


def flip_image_vertically(image_data: dict) -> dict:
    """
    Flip the image vertically (upside down).
    """
    flipped_raw = np.flipud(image_data["raw"])
    return {**image_data, "raw": flipped_raw, "is_flipped_v": True}


# ─────────────────────────────────────────────────────────────────────────────
# Suppression d'arrière-plan
# ─────────────────────────────────────────────────────────────────────────────

def remove_background(image_data: dict, top_n: int = 1) -> dict:
    """
    Calcule le masque et applique la transparence en respectant 
    le redimensionnement effectué à l'étape 1.
    """
    path = image_data["path"]
    new_h = image_data["h"]
    new_w = image_data["w"]

    # 1. Calcul du masque (tif_mask.mask utilise le path, il génère la taille originale)
    mask_arr = compute_mask(
        path,
        channel="L",
        threshold=None,
        plot=False,
        show_img=False,
        min_peak_distance=20,
        op_clo=False,
        gaussian_blur=True,
        top_n=top_n,
    )
    mask_arr = np.array(mask_arr, dtype=np.uint8)

    # 2. Redimensionnement du masque pour qu'il corresponde à l'étape 1
    if mask_arr.shape[:2] != (new_h, new_w):
        mask_pil = Image.fromarray(mask_arr)
        # NEAREST est important pour ne garder que des 0 et des 255 sur le masque binaire
        mask_arr = np.array(mask_pil.resize((new_w, new_h), Image.NEAREST))

    # Retournement du masque si l'image brute a été inversée
    if image_data.get("is_flipped_v", False):
        mask_arr = np.flipud(mask_arr)
        print("  ↳ Masque inversé verticalement pour correspondre à l'image")

    # 3. Création de l'image RGB à partir de l'image 'raw' DÉJÀ redimensionnée
    raw = image_data["raw"]
    if raw.ndim == 2:
        rgb = np.stack((raw,) * 3, axis=-1)
    elif raw.ndim == 3 and raw.shape[2] == 4:
        rgb = raw[..., :3]
    else:
        rgb = raw

    # 4. Application du masque → RGBA
    rgba = apply_mask(rgb, mask_arr)

    print(f"  ↳ Masque redimensionné : {mask_arr.shape}, top_n={top_n}")

    image_data = {**image_data, "mask": mask_arr, "rgba": rgba}
    return image_data


# ─────────────────────────────────────────────────────────────────────────────
# Recadrage autour du contenu
# ─────────────────────────────────────────────────────────────────────────────

def crop_to_content(image_data: dict) -> dict:
    """
    Recadre l'image RGBA à la bounding box du masque.

    Nécessite image_data["mask"] et image_data["rgba"].

    Ajoute au dict :
        'cropped' : array RGBA recadré (h_c, w_c, 4)
        'bbox'    : (rmin, rmax, cmin, cmax) dans la matrice originale
    """
    mask_arr = image_data["mask"]
    rgba = image_data["rgba"]

    rows = np.any(mask_arr > 0, axis=1)
    cols = np.any(mask_arr > 0, axis=0)

    if not rows.any() or not cols.any():
        raise ValueError("Le masque est vide — aucun contenu détecté.")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = rgba[rmin : rmax + 1, cmin : cmax + 1]
    cropped_mask = mask_arr[rmin : rmax + 1, cmin : cmax + 1]

    print(f"  ↳ Crop    : ({rmin},{cmin}) → ({rmax},{cmax})"
          f"  →  {cropped.shape[0]} × {cropped.shape[1]} px")

    image_data = {**image_data, "cropped": cropped, "cropped_mask": cropped_mask, "bbox": (rmin, rmax, cmin, cmax)}
    return image_data


# ─────────────────────────────────────────────────────────────────────────────
# Redimensionnement à la largeur de référence
# ─────────────────────────────────────────────────────────────────────────────

def resize_to_reference_width(images: list[dict]) -> list[dict]:
    """
    Redimensionne toutes les images (sauf la première) pour que leur largeur
    corresponde à celle de l'image 1, en conservant les proportions (h mis
    à l'échelle proportionnellement).

    L'image 1 sert de référence et n'est pas modifiée.
    Met à jour 'raw', 'h' et 'w' dans chaque dict.
    """
    from PIL import Image as PilImage

    ref_w = images[0]["w"]
    print(f"\n  Référence largeur : {ref_w} px (image 1)")

    result = [images[0]]  # image 1 inchangée

    for i, data in enumerate(images[1:], start=2):
        orig_h, orig_w = data["h"], data["w"]
        new_w = ref_w
        new_h = round(orig_h * ref_w / orig_w)

        pil_img = PilImage.fromarray(data["raw"])
        pil_resized = pil_img.resize((new_w, new_h), PilImage.LANCZOS)
        resized_arr = np.array(pil_resized)

        print(f"  ↳ Image {i} : {orig_h} × {orig_w} px  →  {new_h} × {new_w} px")
        result.append({**data, "raw": resized_arr, "h": new_h, "w": new_w})

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Mise à l'échelle des images recadrées
# ─────────────────────────────────────────────────────────────────────────────

def resize_cropped_to_reference(images: list[dict]) -> list[dict]:
    """
    Redimensionne les images recadrées (sauf la première) pour que leur largeur
    corresponde à celle du crop de l'image 1, hauteur adaptée proportionnellement.

    Travaille sur 'cropped' et 'cropped_mask'.
    L'image 1 sert de référence et n'est pas modifiée.
    """
    ref_w = images[0]["cropped"].shape[1]
    print(f"\n  Référence largeur crop : {ref_w} px (image 1)")

    result = [images[0]]

    for i, data in enumerate(images[1:], start=2):
        orig_h, orig_w = data["cropped"].shape[:2]
        new_w = ref_w
        new_h = round(orig_h * ref_w / orig_w)

        # Redimensionnement du crop RGBA
        cropped_pil = Image.fromarray(data["cropped"])
        cropped_resized = np.array(cropped_pil.resize((new_w, new_h), Image.LANCZOS))

        # Redimensionnement du masque recadré (NEAREST pour garder 0/255)
        mask_pil = Image.fromarray(data["cropped_mask"])
        mask_resized = np.array(mask_pil.resize((new_w, new_h), Image.NEAREST))

        print(f"  ↳ Crop image {i} : {orig_h} × {orig_w} px  →  {new_h} × {new_w} px")
        result.append({**data, "cropped": cropped_resized, "cropped_mask": mask_resized})

    return result

# ─────────────────────────────────────────────────────────────────────────────

def pad_to_diagonal_square(image_data: dict) -> dict:
    """
    Place l'image recadrée dans une matrice carrée a × a où
        a = ceil( sqrt(W² + H²) )
    avec W et H les dimensions du CROP (après resize éventuel),
    pas les dimensions brutes d'origine.
    L'image est centrée ; le reste est transparent (alpha = 0).

    Nécessite image_data["cropped"].

    Ajoute au dict :
        'square'   : array RGBA carré (a, a, 4)
        'square_mask' : masque carré (a, a)
        'diagonal' : valeur entière a
    """
    cropped = image_data["cropped"]
    h_src, w_src = cropped.shape[:2]

    a = math.ceil(math.sqrt(w_src ** 2 + h_src ** 2))

    square = np.zeros((a, a, 4), dtype=np.uint8)
    square_mask = np.zeros((a, a), dtype=np.uint8)

    row_off = (a - h_src) // 2
    col_off = (a - w_src) // 2

    square[row_off : row_off + h_src, col_off : col_off + w_src] = cropped
    square_mask[row_off : row_off + h_src, col_off : col_off + w_src] = image_data["cropped_mask"]

    print(f"  ↳ Carré   : {a} × {a} px"
          f"  (a = ceil(√({w_src}²+{h_src}²)) = {a})"
          f"  [crop : {h_src} × {w_src}]")

    image_data = {**image_data, "square": square, "square_mask": square_mask, "diagonal": a}
    return image_data


# ─────────────────────────────────────────────────────────────────────────────
# Appel complet pour une image
# ─────────────────────────────────────────────────────────────────────────────

def process_image(path: str, top_n: int = 1) -> dict:
    """
    Pipeline complet pour une image :
        1. Chargement brut
        2. Suppression d'arrière-plan
        3. Recadrage
        4. Mise en matrice diagonale carrée

    Retourne un dict avec toutes les données intermédiaires.
    """
    data = load_image(path)
    data = remove_background(data, top_n=top_n)
    data = crop_to_content(data)
    data = pad_to_diagonal_square(data)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Détection de côté de prostate
# ─────────────────────────────────────────────────────────────────────────────

def orient_and_face(image_data: dict, side: str) -> dict:
    """
    Détecte le côté de coupe avec Douglas-Peucker et tourne la matrice carrée
    pour que la coupe soit verticale et face au centre.
    """
    import cv2
    mask = image_data["square_mask"]
    square_img = image_data["square"]

    # 1. Extraction des contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Aucun contour trouvé dans le masque carré.")
    
    dots = np.vstack([c.reshape(-1, 2) for c in contours])
    
    # 2. Douglas-Peucker pour trouver la plus longue droite
    contour_cv2 = dots.reshape((-1, 1, 2)).astype(np.int32)
    perimetre = cv2.arcLength(contour_cv2, True)
    epsilon = 0.03 * perimetre
    approx = cv2.approxPolyDP(contour_cv2, epsilon, True)

    longueur_max = 0
    p1_xy, p2_xy = None, None
    n_points = len(approx)
    
    for i in range(n_points):
        pt1 = approx[i][0]
        pt2 = approx[(i + 1) % n_points][0]
        dist = np.linalg.norm(pt1 - pt2)
        if dist > longueur_max:
            longueur_max = dist
            p1_xy, p2_xy = pt1, pt2

    # Passage en format (y,x) pour correspondre exactement à ta logique d'origine
    p1_yx = np.array([p1_xy[1], p1_xy[0]])
    p2_yx = np.array([p2_xy[1], p2_xy[0]])

    # 3. Calcul de l'angle et détermination du haut/bas
    den = (p1_yx[1] - p2_yx[1])
    if den == 0: den = 1e-5
    angle = np.degrees(np.arctan(((-p1_yx[0]) - (-p2_yx[0])) / den))

    mid_yx = (p1_yx + p2_yx) // 2
    M_mom = cv2.moments(mask)
    cY = int(M_mom["m01"] / M_mom["m00"]) if M_mom["m00"] != 0 else mask.shape[0]//2
    
    up = (cY < mid_yx[0])

    # 4. Déduction de la rotation selon le rôle du morceau
    if side == "left_piece":
        rotation = (-angle + 90) if up else (-angle - 90)
    else: # right_piece
        rotation = (-angle - 90) if up else (-angle + 90)

    # 5. Application de la rotation via OpenCV
    center = (float(mask.shape[1] / 2), float(mask.shape[0] / 2))
    M_rot = cv2.getRotationMatrix2D(center, float(rotation), 1.0)
    
    rotated_img = cv2.warpAffine(square_img, M_rot, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR)

    # 6. Suivi du point médian tourné pour le recalage final
    mid_arr_xy = np.array([[[mid_yx[1], mid_yx[0]]]], dtype=np.float32)
    mid_rot_xy = cv2.transform(mid_arr_xy, M_rot)[0][0]
    turn_mid_yx = np.array([mid_rot_xy[1], mid_rot_xy[0]])

    # 7. Extraire le contour de la version tournée pour le magnétisme
    rotated_mask = cv2.warpAffine(mask, M_rot, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    contours_rot, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rotated_contour_xy = np.vstack([c.reshape(-1, 2) for c in contours_rot])
    rotated_contour_yx = np.flip(rotated_contour_xy, axis=1)

    print(f"  ↳ Angle de coupe détecté : {angle:.1f}° | Rotation appliquée : {rotation:.1f}°")

    return {
        **image_data, 
        "rotated": rotated_img, 
        "turn_mid_yx": turn_mid_yx,
        "rotation_angle": rotation,            # Nécessaire pour la rotation finale
        "rotated_contour_yx": rotated_contour_yx # Nécessaire pour le snap
    }


# ─────────────────────────────────────────────────────────────────────────────
# Filtre des doublons, pour garder uniquement le max ou le min
# ─────────────────────────────────────────────────────────────────────────────

def filtrer_doublons_y(segment, select_max=True):
    """
    Filtre les doublons de Y dans un segment.
    - Pour seg_g : conserve le X maximum pour chaque Y.
    - Pour seg_d : conserve le X minimum pour chaque Y.
    """
    filtre = {}
    for y, x in segment:
        y_int = int(y)  # Convertir Y en entier
        if y_int not in filtre:
            filtre[y_int] = x
        else:
            if select_max and x > filtre[y_int]:
                filtre[y_int] = x
            elif not select_max and x < filtre[y_int]:
                filtre[y_int] = x
    # Retourner un tableau trié par Y
    return np.array(sorted(filtre.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Créer des champs de déplacement pour tous les Y, gestion des zones hors coupes
# ─────────────────────────────────────────────────────────────────────────────

def create_displacement_field(y_all, transform_dict, y_min, y_max):
    return np.array([
        transform_dict[y_min] if y < y_min else
        transform_dict[y_max] if y > y_max else
        transform_dict[y]
        for y in y_all
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Créer une carte de déplacement pour l'image
# ─────────────────────────────────────────────────────────────────────────────

def create_displacement_map(height, width, displacement_field):
    # Créer des grilles de coordonnées
    y_coords, x_coords = np.indices((height, width), dtype=np.float32)

    # Appliquer le déplacement sur l'axe X (horizontal)
    # Note : displacement_field est appliqué ligne par ligne (selon Y)
    map_x = np.zeros_like(x_coords)
    for y in range(height):
        map_x[y, :] = x_coords[y, :] + displacement_field[y]

    map_y = y_coords  # Pas de déplacement sur l'axe Y

    return map_x, map_y


# ─────────────────────────────────────────────────────────────────────────────
# Centre de masse
# ─────────────────────────────────────────────────────────────────────────────

def get_center_of_mass(img):
    """
    Calcule le centre de masse (centre de gravité) d'une image à partir de son canal alpha ou en niveaux de gris.

    Retourne un tuple (y, x) représentant les coordonnées du centre de masse.
    """
    if isinstance(img, dict) and "mask" in img:
        mask = img["mask"]
    elif img.ndim == 3 and img.shape[2] == 4:
        mask = img[:, :, 3]
    else:
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
    moments = cv2.moments(mask)

    if moments["m00"] != 0:
        cY = int(moments["m01"] / moments["m00"])
        cX = int(moments["m10"] / moments["m00"])
    else:
        # Si le masque est vide, retourner le centre de l'image
        h, w = mask.shape[:2]
        cY, cX = h // 2, w // 2

    return (cY, cX)


def create_displacement_map_with_center(height, width, displacement_field, center_x, cut_x_coords=None, side="left", displacement_y=None):
    """
    Crée une carte de déplacement qui diminue linéairement jusqu'au centre de masse.

    Args:
        height: Hauteur de l'image.
        width: Largeur de l'image.
        displacement_field: Champ de déplacement pour chaque Y.
        center_x: Coordonnée X du centre de masse.
        cut_x_coords: Coordonnée X de la coupe pour chaque Y (optionnel).
        side: "left" ou "right", indique de quel côté se trouve le morceau par rapport à la coupe.
        displacement_y: Champ de déplacement sur l'axe Y (optionnel).

    Returns:
        map_x: Carte de déplacement pour l'axe X.
        map_y: Carte de déplacement pour l'axe Y (inchangée).
        deformation_amplitude: Carte des amplitudes de déformation.
    """
    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    map_x = np.zeros_like(x_coords)
    map_y = y_coords.copy()  # Initialisé sans déplacement
    deformation_amplitude = np.zeros_like(x_coords)

    for y in range(height):
        dx = displacement_field[y]
        dy = displacement_y[y] if displacement_y is not None else 0.0
        
        if cut_x_coords is not None:
            x_cut = cut_x_coords[y]
            x_dst_cut = x_cut - dx

            if side == "left":
                # Le morceau est à gauche de la coupe (x_cut est le bord droit du morceau)
                dst_distance_max = x_dst_cut - center_x
                if dst_distance_max > 0:
                    factor = (x_coords[y, :] - center_x) / dst_distance_max
                    factor = np.clip(factor, 0.0, 1.0)
                else:
                    factor = np.where(x_coords[y, :] <= x_dst_cut, 1.0, 0.0)
                # Passé le centre (à gauche), plus de déformation (factor = 0)
                factor[x_coords[y, :] < center_x] = 0.0
                
            else:
                # Le morceau est à droite de la coupe (x_cut est le bord gauche du morceau)
                dst_distance_max = center_x - x_dst_cut
                if dst_distance_max > 0:
                    factor = (center_x - x_coords[y, :]) / dst_distance_max
                    factor = np.clip(factor, 0.0, 1.0)
                else:
                    factor = np.where(x_coords[y, :] >= x_dst_cut, 1.0, 0.0)
                # Passé le centre (à droite), plus de déformation (factor = 0)
                factor[x_coords[y, :] > center_x] = 0.0
                
            map_x[y, :] = x_coords[y, :] + dx * factor
            map_y[y, :] = y_coords[y, :] + dy * factor
            deformation_amplitude[y, :] = np.abs(dx * factor) + np.abs(dy * factor)
            
        else:
            # Ancienne méthode (fallback)
            distance_to_center = np.abs(x_coords[y, :] - center_x)
            max_distance = max(distance_to_center)

            if max_distance > 0:
                factor = 1 - (distance_to_center / max_distance)
                factor = np.clip(factor, 0.0, 1.0)
                map_x[y, :] = x_coords[y, :] + dx * factor
                map_y[y, :] = y_coords[y, :] + dy * factor
                deformation_amplitude[y, :] = np.abs(dx * factor) + np.abs(dy * factor)
            else:
                map_x[y, :] = x_coords[y, :]
                map_y[y, :] = y_coords[y, :]

    return map_x, map_y, deformation_amplitude
