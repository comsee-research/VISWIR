# ==============================================================================
# Implementation of "Visible-NIR Image Fusion Based on Top-Hat Transform"
#
# Reference:
# M. Herrera-Arellano, H. Peregrina-Barreto, and I. Terol-Villalobos, 
# "Visible-NIR Image Fusion Based on Top-Hat Transform," IEEE Transactions 
# on Image Processing, vol. 30, pp. 4962-4972, 2021, 
# doi: 10.1109/TIP.2021.3077310.
#
# Re-implemented by: Alexandre Riffard
# ==============================================================================

"""
================================================================================
NOTE SUR L'ESPACE COLORIMÉTRIQUE UTILISÉ (CIE-Lab vs lαβ)
================================================================================

L'article de Herrera-Arellano et al. (2021) indique que l'image RGB est convertie
dans l'espace lαβ en utilisant les matrices citées dans [22] et [23]. Cependant,
la description fournie dans le papier ne correspond pas au pipeline complet
défini par Ruderman et al. (1998), qui inclut notamment :

    - la transformation logarithmique log(LMS),
    - une normalisation des canaux,
    - un étalonnage dynamique,
    - et un pipeline inverse bien défini.

Aucune de ces étapes n'est explicitée dans l'article, et les matrices publiées
ne permettent pas de reproduire fidèlement les résultats des auteurs. En
implémentant le lαβ "correct" (Ruderman), les résultats diffèrent fortement de
ceux présentés dans le papier, ce qui indique que les auteurs utilisent en
réalité une variante tronquée ou modifiée du lαβ, non standardisée et non
reproductible à partir du texte.

Pour garantir la stabilité numérique, la reproductibilité et la cohérence
colorimétrique, nous utilisons ici l'espace CIE-Lab :

    - standardisé et bien défini,
    - perceptuellement uniforme,
    - robuste aux variations de dynamique VIS-NIR,
    - et disponible dans des implémentations fiables (skimage, OpenCV).

Ce choix n'est donc pas motivé par la simplicité, mais par la nécessité
scientifique de disposer d'un espace colorimétrique reproductible et adapté à
la fusion VIS-NIR. Dans nos tests, Lab produit des résultats plus stables et
plus proches du comportement réel observé dans les images du dataset NIR utilisé
par les auteurs.

================================================================================
"""

import os
import cv2
import numpy as np
from skimage import color

def rgb_to_lab_skimage(image):
    image = image / 255.0
    lab_image = color.rgb2lab(image)
    l = lab_image[:, :, 0]
    a = lab_image[:, :, 1]
    b = lab_image[:, :, 2]
    return l, a, b

def lab_to_rgb_skimage(l, a, b):
    lab_image = np.stack([l, a, b], axis=2)
    rgb_image = color.lab2rgb(lab_image)
    return (rgb_image * 255).astype(np.uint8)

###########
# Matrice RGB -> LMS (Ruderman 1998)
RGB_TO_LMS = np.array([
    [0.3811, 0.5783, 0.0402],
    [0.1967, 0.7244, 0.0782],
    [0.0241, 0.1288, 0.8444]
])

# Matrice LMS -> lαβ
LMS_TO_LAB = np.array([
    [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
    [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
    [1/np.sqrt(2), -1/np.sqrt(2), 0]
])

def rgb_to_lalphabeta(image):
    # Normalisation
    img = image.astype(np.float32) / 255.0

    # 1. RGB -> LMS
    LMS = img @ RGB_TO_LMS.T

    # 2. log(LMS)
    logLMS = np.log(np.maximum(LMS, 1e-6))

    # 3. log-LMS -> lαβ
    lab = logLMS @ LMS_TO_LAB.T

    l = lab[:, :, 0]
    alpha = lab[:, :, 1]
    beta = lab[:, :, 2]

    return l, alpha, beta
###########

###########
# Matrice inverse lαβ -> LMS
LAB_TO_LMS = np.linalg.inv(LMS_TO_LAB)

# Matrice inverse LMS -> RGB
LMS_TO_RGB = np.linalg.inv(RGB_TO_LMS)

def lalphabeta_to_rgb(l, alpha, beta):
    lab = np.stack([l, alpha, beta], axis=2)

    # 1. lαβ -> log(LMS)
    logLMS = lab @ LAB_TO_LMS.T

    # 2. exp(log(LMS)) -> LMS
    LMS = np.exp(logLMS)

    # 3. LMS -> RGB
    rgb = LMS @ LMS_TO_RGB.T

    # Clamp + conversion
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)
###########

def process_image_pair(visible_path, swir_path, output_dir):
    visible = cv2.imread(visible_path)
    swir = cv2.imread(swir_path, cv2.IMREAD_UNCHANGED)
    swir = swir[:, :, 0] if swir.ndim == 3 else swir

    lum_v, couleur1_v, couleur2_v = rgb_to_lab_skimage(visible)
    # lum_v, couleur1_v, couleur2_v = rgb_to_lalphabeta(visible)
    lum_s = cv2.normalize(swir.astype('float32'), None, 0, 255, cv2.NORM_MINMAX)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    TH_s = cv2.morphologyEx(lum_s, cv2.MORPH_TOPHAT, kernel)
    TH_v = cv2.morphologyEx(lum_v.astype(np.float32), cv2.MORPH_TOPHAT, kernel)
    TH_t = np.maximum(TH_s, TH_v)

    lum_fus = lum_v - TH_v + TH_t
    lum_fus = np.minimum(lum_fus, lum_v + TH_v)
    lum_fus = np.clip(lum_fus, 0, 255)

    fused_image = lab_to_rgb_skimage(lum_fus, couleur1_v, couleur2_v)
    # fused_image = lalphabeta_to_rgb(lum_fus, couleur1_v, couleur2_v)

    base_filename = os.path.basename(visible_path)
    output_path = os.path.join(output_dir, base_filename)
    cv2.imwrite(output_path, fused_image)

def process_all_pairs(visible_dir, swir_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for filename in os.listdir(visible_dir):
        if not filename.lower().endswith(valid_extensions):
            continue
        visible_path = os.path.join(visible_dir, filename)
        swir_path = os.path.join(swir_dir, filename)
        if os.path.exists(swir_path):
            process_image_pair(visible_path, swir_path, output_dir)
        else:
            print(f"⚠️ Fichier SWIR manquant pour : {filename}")

if __name__ == "__main__":
    
    visible_dirs = [
        "./data/dataset_1/rgb",
        "./data/dataset_2/rgb"
    ]

    swir_dirs = [
        "./data/dataset_1/swir",
        "./data/dataset_2/swir"
    ]

    output_dirs = [
        "./output/method_name/dataset_1",
        "./output/method_name/dataset_2"
    ]

    for visible_dir, swir_dir, output_dir in zip(visible_dirs, swir_dirs, output_dirs):
        process_all_pairs(visible_dir, swir_dir, output_dir)