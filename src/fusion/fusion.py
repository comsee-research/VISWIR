"""
Main fusion functions for VISWIR multi-spectral image fusion.
"""

# =============================================================================
# FILENAME:       fusion.py
# DESCRIPTION:    Ce fichier contient les fonctions constituant le coeur (core)
#                 de la fusion utilisées dans le cadre d'un projet de fusion
#                 d'images Visible et SWIR : VISWIR.
#  
# REPOSITORY:     https://github.com/comsee-research/VISWIR.git
#
# AUTHOR:         [Riffard Alexandre]
# EMAIL:          [alexandre.riffard@uca.fr]
# CREATED:        [16-04-2025]
# LAST UPDATED:   [30-04-2025]
# VERSION:        1.0
#
# LICENSE:        GNU LESSER GENERAL PUBLIC LICENSE (voir LICENSE dans le dépôt)
#
# USAGE:          - Importer les fonctions depuis ce fichier pour réaliser
#                   la fusion d'images Visble et SWIR.
#                 - Ne pas importer le coeur, passer par la fonction "process_image".
#
# DEPENDENCIES:   - numpy
#                 - skimage
#                 - cv2
#
# NOTES:
#   - Toutes les images doivent être normalisées dans [0, 1] (float64).
#   - Compatible avec les images visible RGB. Ne fonctionne pas avec des images visible monochrome.
#
# CHANGELOG:
#   - [16-04-2025]: Création initiale du fichier à partir d'un ancien notebook.
#   - [16-04-2025]-[29-04-2025]: Ajustements...
#   - [30-04-2025]: Retrait de OpenCV pour le chargement des images, et passage avec Skimage.
#   - [01-07-2025]: Nettoyage mémoire.
#
# =============================================================================

import os
import gc
import numpy as np
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2hsv, hsv2rgb

from fusion.utils import load_image_skimage_core, print_image_info, save_float64_image_as_uint16
from fusion.functions import *
from common.logger import logger

def viswir_core_fusion(I1_RGB, I2, facteur_swir, beta, level, apply_gamma, gamma_value):
    """
    Perform VIS–SWIR image fusion using Laplacian and Gaussian pyramids.

    This function fuses a visible image (VIS) and a SWIR image into a single
    enhanced image. The fusion process is based on local contrast, entropy,
    and visibility maps, combined with pyramid decomposition and reconstruction.
    Post-processing includes gamma correction and Difference of Gaussians (DoG)
    sharpening.

    Parameters
    ----------
    I1_RGB : numpy.ndarray
        Visible image in RGB format, float64 in [0, 1].
    I2 : numpy.ndarray
        SWIR image, float64 in [0, 1].
    facteur_swir : float
        Weight factor for the SWIR contribution (between 0 and 1).
    beta : float
        Exponent applied during inverse mapping to adjust intensity blending.
    level : int
        Number of pyramid levels used for fusion.
    apply_gamma : bool
        Whether to apply gamma correction to the fused image.
    gamma_value : float
        Gamma correction value (used if `apply_gamma=True`).

    Returns
    -------
    tuple of numpy.ndarray
        - I5 : numpy.ndarray
            Intermediate fused image before post-processing.
        - I_out : numpy.ndarray
            Final fused image after post-processing (gamma correction, sharpening).

    Raises
    ------
    ValueError
        If weight normalization fails (sum of weights not close to 1).

    Notes
    -----
    - Local weights are computed from contrast, entropy, and visibility.
    - Fusion is performed in the HSV color space, replacing the V channel.
    - Post-processing enhances sharpness and contrast.
    - Memory is explicitly freed at the end of the function.
    """

    logger.debug(
        f"<cyan>Fusion</cyan> → "
        f"facteur_swir=<magenta>{facteur_swir:.2f}</magenta>, "
        f"beta=<blue>{beta:.1f}</blue>, "
        f"level=<yellow>{level}</yellow>, "
        f"gamma={'<green>ON</green>' if apply_gamma else '<red>OFF</red>'}"
        f"{f' (γ=<white>{gamma_value:.1f}</white>)' if apply_gamma else ''}"
    )

    # Conversion RGB -> HSV
    I1_HSV = rgb2hsv(I1_RGB) # Suppose I1_RGB est déjà en float64 et dans [0, 1]
    H, S, I1 = I1_HSV[..., 0], I1_HSV[..., 1], I1_HSV[..., 2]

    # Paramètres pour les poids
    alpha = [1, 1, 1]
    size1, size2 = 5, 5
    sigma1, sigma2 = 2, 2
    window1, window2 = 5, 5
    NHOOD1, NHOOD2 = disk(window1), disk(window2)

    # Étape 2 : Calcul des poids
    # # Contraste local
    C1 = local_std(I1, NHOOD1)
    C2 = local_std(I2, NHOOD1)
    # # Entropie locale
    J1 = entropy((I1 * 255).astype(np.uint8), NHOOD2) / 8.0
    J2 = entropy((I2 * 255).astype(np.uint8), NHOOD2) / 8.0
    # # Visibilité locale
    Vis1 = local_visibility(I1, size1, sigma1, sigma2)
    Vis2 = local_visibility(I2, size1, sigma1, sigma2)

    # # Poids finaux
    W1 = (C1 ** alpha[0]) * (J1 ** alpha[1]) * (Vis1 ** alpha[2])
    W2 = (C2 ** alpha[0]) * (J2 ** alpha[1]) * (Vis2 ** alpha[2])

    # # Remplace les NaN dans les poids
    W1 = np.nan_to_num(W1, nan=1e-12, posinf=1e-12, neginf=1e-12)
    W2 = np.nan_to_num(W2, nan=1e-12, posinf=1e-12, neginf=1e-12)

    # # Définir le facteur SWIR (entre 0 et 1)
    facteur_visible = 1 - facteur_swir # Complémentaire à SWIR
    # # Appliquer les facteurs visibles et SWIR
    W1 *= facteur_visible
    W2 *= facteur_swir

    # # Correction valeur incohérentes
    # W1 = np.nan_to_num(W1, nan=0.0, posinf=0.0, neginf=0.0)  # Remplace les NaN et les infinis
    # W1 = np.clip(W1, a_min=0, a_max=None)  # Force les valeurs négatives à 0
    W1[W1 <= 1e-12] = 1e-12  # Remplace les valeurs très faibles par un seuil minimal

    # W2 = np.nan_to_num(W2, nan=0.0, posinf=0.0, neginf=0.0)  # Remplace les NaN et les infinis
    # W2 = np.clip(W2, a_min=0, a_max=None)  # Force les valeurs négatives à 0
    # W2[W2 <= 1e-12] = 1e-12  # Remplace les valeurs très faibles par un seuil minimal

    # # Normalisation robuste
    somme_init = W1 + W2
    somme_init = np.where(somme_init <= 1e-12, 1e-12, somme_init)
    W = np.stack([W1, W2], axis=-1) / somme_init[..., None]

    # # Vérification
    somme = W.sum(axis=-1)
    # # Vérification stricte des bornes de la somme
    min_threshold = 0.9999999999999998
    max_threshold = 1.0000000000000002

    if somme.min() < min_threshold or somme.max() > max_threshold:
        raise ValueError(
            f"Erreur : Les poids ne sont pas correctement normalisés.\n"
            f"Somme minimale : {somme.min()}, Somme maximale : {somme.max()}.\n"
            f"Suggestion : Décommentez les lignes de correction pour traiter les valeurs nulles, infinis ou faibles dans W1 et W2."
        )

    # Étape 3 : Fusion avec les pyramides
    I = np.stack([I1, I2], axis=-1)
    # # Création d'une pyramide vide pour commencer
    pyr = gaussian_pyramid(np.zeros_like(I1), levels=level)
    n_levels = len(pyr)

    for i in range(2): # Pour chaque image (Visible et SWIR) --> c'est 2 car on a que 2 images à fusionner, on en aurai 3, on mettrait alors "range(3)", etc...
        # # Crée les pyramides pour la carte de poids (Gaussienne) et l'image (Laplacienne)
        pyrW = gaussian_pyramid(W[..., i], levels=level)
        pyrI = laplacian_pyramid(I[..., i], levels=level)

        for l in range(n_levels):
            pyr[l] += pyrW[l] * pyrI[l]

    # # Reconstruction
    Recon = reconstruct_laplacian_pyramid(pyr)
    if np.any(Recon < 0.0) or np.any(Recon > 1.0):
        # logger.warning(f"⚠️ Valeurs hors bornes détectées dans Recon : min={Recon.min():.4f}, max={Recon.max():.4f}")
        logger.debug(f"⚠️ Valeurs hors bornes détectées dans Recon : min={Recon.min():.4f}, max={Recon.max():.4f}")
    Recon = np.clip(Recon, 0.0, 1.0)

    # Étape 4 : Fusion et post-traitement
    I4 = np.stack([H, S, Recon], axis=-1)

    # Conversion sans quitter float64
    I5 = hsv2rgb(I4)

    # # Mappage inverse
    I_out1 = np.zeros_like(I5)
    Recon_safe = Recon + 1e-12 # Évite la division par zéro
    mask_swir = I2 > 1e-3 # Masque pour ignorer les zones noires
    Recon_safe[~mask_swir] = 1 # Neutralise les bandes noires

    for c in range(3): # Application du mappage
        I_out1[..., c] = ((I5[..., c] / Recon_safe) ** beta) * (I1 * facteur_visible + I2 * facteur_swir)

    #######################
    # # # Correction gamma
    if apply_gamma:
        std_I_out1 = np.std(I_out1) # Calcul de l'écart-type de l'image fusionnée
        mean_I2 = np.mean(I2) # Calcul de la moyenne de l'image SWIR
        I_out1 = I_out1 ** (1 / gamma_value)  # Correction gamma (inverse) classique
    #######################

    # # Affinage et mise en valeur
    I_out1_HSV = rgb2hsv(I_out1)  # reste en float64
    Hn, Sn, Vn = I_out1_HSV[..., 0], I_out1_HSV[..., 1], I_out1_HSV[..., 2]


    # # Paramètre filtre Gaussien
    size = 3 # Taille du noyau (choisir une taille impaire)
    sigma1 = 2 #1 # Sigma pour le premier noyau gaussien
    sigma2 = 1 #2 # Sigma pour le second noyau gaussien

    # # Créer deux noyaux gaussiens de même taille
    gaussian1 = cv2.getGaussianKernel(size, sigma1) * cv2.getGaussianKernel(size, sigma1).T
    gaussian2 = cv2.getGaussianKernel(size, sigma2) * cv2.getGaussianKernel(size, sigma2).T
    # # Calculer le DoG (Difference of Gaussians)
    hsharp = gaussian1 - gaussian2
    # # Appliquer le filtre
    Recon_sharp = cv2.filter2D(Vn, -1, hsharp)
    # # Limitation de l'effet dans les zones lumineuses
    V2 = Vn + 0.4 * Recon_sharp # * (1 - Vn)
    V2 = np.clip(V2, a_min=0.0, a_max=1.0)

    I6 = np.stack([Hn, Sn, V2], axis=-1)  # I6 en HSV, float64
    I_out = hsv2rgb(I6)  # reste en float64, valeurs dans [0, 1]

    # Préparer les résultats
    result = I5, I_out

    # 🔥 Libération mémoire manuelle
    del I1_RGB, I2, I1_HSV, H, S, I1, C1, C2, J1, J2
    del Vis1, Vis2, W1, W2, W, somme, I, pyrW, pyrI, pyr, Recon
    del I4, I_out1, I_out1_HSV, Hn, Sn, Vn, gaussian1, gaussian2, hsharp, Recon_sharp, V2, I6
    gc.collect()

    # ✅ Retourne les images fusionnées
    return result

def process_image(visible_path, swir_path, facteur_swir, beta, level, apply_gamma, gamma_value, save_output=False, output_dir=None):
    """
    Load visible and SWIR images, perform VIS–SWIR fusion, and optionally save results.

    Parameters
    ----------
    visible_path : str or Path
        Path to the visible image file.
    swir_path : str or Path
        Path to the SWIR image file.
    facteur_swir : float
        Weight factor for the SWIR contribution (between 0 and 1).
    beta : float
        Exponent applied during inverse mapping to adjust intensity blending.
    level : int
        Number of pyramid levels used for fusion.
    apply_gamma : bool
        Whether to apply gamma correction to the fused image.
    gamma_value : float
        Gamma correction value (used if `apply_gamma=True`).
    save_output : bool, default=False
        Whether to save the fused images to disk.
    output_dir : str or Path, optional
        Directory where results will be saved (if `save_output=True`).

    Returns
    -------
    tuple
        - I5 : numpy.ndarray or None
            Intermediate fused image before post-processing.
        - I_out : numpy.ndarray or None
            Final fused image after post-processing.
        - error : str or None
            Error message if the fusion process fails, otherwise None.

    Notes
    -----
    - If `save_output=True`, two images are saved:
      * `*_fused_no_post_processing.tiff`
      * `*_fused_with_post_processing.tiff`
    - Images are saved as uint16 TIFF files.
    - Returns `(None, None, error_message)` if an exception occurs.
    """
    # Charger les images
    I1_RGB = load_image_skimage_core(visible_path)               # Visible
    I2 = load_image_skimage_core(swir_path, is_swir=True)        # SWIR

    try:
        I5, I_out = viswir_core_fusion(I1_RGB, I2, facteur_swir, beta, level, apply_gamma, gamma_value)

        # if save_output and output_dir:
        if save_output is True and output_dir:
            base_name = os.path.splitext(os.path.basename(visible_path))[0]
            gamma_str = f"gamma_{gamma_value:.1f}" if apply_gamma else "no_gamma"
            no_post_dir = os.path.join(output_dir, "no_proc", f"weight_{facteur_swir:.2f}", f"beta_{beta:.1f}", f"level_{level}", gamma_str)
            with_post_dir = os.path.join(output_dir, "proc", f"weight_{facteur_swir:.2f}", f"beta_{beta:.1f}", f"level_{level}", gamma_str)
            os.makedirs(no_post_dir, exist_ok=True)
            os.makedirs(with_post_dir, exist_ok=True)

            output_path_no_post = os.path.join(no_post_dir, f"{base_name}_fused_no_post_processing.tiff")
            output_path_with_post = os.path.join(with_post_dir, f"{base_name}_fused_with_post_processing.tiff")

            save_float64_image_as_uint16(output_path_no_post, I5)
            save_float64_image_as_uint16(output_path_with_post, I_out)

        # On prépare le retour
        result = I5, I_out, None

        # 💣 Nettoyage des images source et intermédiaires
        del I1_RGB, I2
        gc.collect()


        # return I5, I_out, None  # Pas d'erreur
        return result

    except Exception as e:
        return None, None, str(e)
