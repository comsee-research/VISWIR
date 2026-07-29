"""
Metrics computation (SSIM, NIQE, etc.) for VISWIR image quality assessment.
"""

# =============================================================================
# FILENAME:       metrics.py
# DESCRIPTION:    Ce fichier contient les fonctions de calcul des métriques de
#                 qualité d'image (avec ou sans référence) utilisées dans le
#                 cadre d'un projet de fusion d'images Visible et SWIR : VISWIR.
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
# USAGE:          - Importer les fonctions depuis ce fichier pour évaluer la
#                 qualité des images fusionnées.
#
# DEPENDENCIES:   - numpy
#                 - skimage
#                 - sewar
#                 - scipy 
#                 - torch
#                 - pytorch_msssim
#                 - brisque
#
# NOTES:
#   - Toutes les images doivent être normalisées dans [0, 1] (float64 ou float32).
#   - Certaines métriques nécessitent une image de référence.
#   - Compatible avec les images monochromes ou RGB. (Certaines métriques peuvent ne pas fonctionner pour les images monochromes.)
#
# CHANGELOG:
#   - [16-04-2025]: Création initiale du fichier à partir d'un ancien notebook.
#   - [16-04-2025]-[29-04-2025]: Correction des calculs et vérification.
#   - [30-04-2025]: Ajout des métriques ERGAS, SAM et VIF (via Sewar).
#   - [01-07-2025]: Nettoyage mémoire.
#
# =============================================================================

import sys
import os
import gc

# import cv2
# from PIL import Image
import numpy as np
from scipy.stats import entropy, pearsonr
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import torch
from pytorch_msssim import ms_ssim
import brisque
from sewar.full_ref import uqi#, msssim, ssim as sewar_ssim, scc, rmse
from sewar.full_ref import ergas, sam, vifp
# from sewar.no_ref import d_lambda, d_s, qnr

import warnings
from fusion.utils import to_grayscale_array_skimage, safe_float, from_grey_to_rgb_array#, to_rgb_array, ensure_range_255, normalize_image, ensure_grayscale, preprocess_images_for_metrics
from fusion.utils import print_image_info#, to_float32

# Import pour un appel du fichier en mode script :
from skimage.io import imread
from skimage.transform import resize

# Import for NIQE
from fusion.NIQE.niqe import calculate_niqe # Based on the Matlab version of the metrics.

# Import for PIQE
from pypiqe import piqe # A python version of Matlab's Perception based Image Quality Evaluator (PIQE) no-reference image quality score

# === Enregistreurs de métriques dynamiques ===
no_ref_metrics = {}
full_ref_metrics = {}

def register_metric(name, ref_required=False):
    """
    Decorator to register a metric function in the appropriate registry.

    Parameters
    ----------
    name : str
        Name of the metric.
    ref_required : bool, default=False
        Whether the metric requires a reference image.

    Returns
    -------
    function
        Wrapped metric function.
    """
    def decorator(func):
        """
        Decorator to register a metric function.
        """
        wrapped = safe(func, name)
        if ref_required:
            full_ref_metrics[name] = wrapped
        else:
            no_ref_metrics[name] = wrapped
        return wrapped
    return decorator

# === Utilitaires ===

def safe(func, name):
    """
    Wrap a metric function to ensure safe execution.

    Parameters
    ----------
    func : callable
        Metric function to wrap.
    name : str
        Name of the metric.

    Returns
    -------
    callable
        Wrapped function that returns None if an exception occurs.
    """
    def wrapper(*args, **kwargs):
        """
        Safe execution wrapper.
        """
        try:
            result = func(*args, **kwargs)
            return safe_float(result)
        except Exception as e:
            warnings.warn(f"Erreur dans la métrique '{name}': {e}")
            return None
    return wrapper

# === Fonction principale ===

def compute_all_metrics(I_ref=None, I_fused=None):
    """
    Compute all registered metrics for a fused image (and optionally a reference image).

    Parameters
    ----------
    I_ref : numpy.ndarray, optional
        Reference image (used for full-reference metrics).
    I_fused : numpy.ndarray
        Fused image.

    Returns
    -------
    dict
        Dictionary mapping metric names to their computed values.

    Raises
    ------
    ValueError
        If the fused image is None.
    """
    results = {}

    # if I_ref is None or I_fused is None:
    if I_fused is None:
        raise ValueError("Erreur : Une des images est 'None'. Veuillez fournir des images valides.")

    # Conversion des types pour éviter les warnings et garantir une cohérence
    if I_ref is not None:
        if I_ref.dtype == np.uint8:
            I_ref = I_ref.astype(np.float32) / 255.0
        elif I_ref.dtype == np.uint16:
            I_ref = I_ref.astype(np.float32) / 65535.0
        elif I_ref.dtype != np.float32:
            I_ref = I_ref.astype(np.float32)

    if I_fused is not None:
        if I_fused.dtype == np.uint8:
            I_fused = I_fused.astype(np.float32) / 255.0
        elif I_fused.dtype == np.uint16:
            I_fused = I_fused.astype(np.float32) / 65535.0
        elif I_fused.dtype != np.float32:
            I_fused = I_fused.astype(np.float32)

    if I_fused is not None:
        for name, func in no_ref_metrics.items():
            results[name] = func(I_fused)

    if I_ref is not None and I_fused is not None:
        for name, func in full_ref_metrics.items():
            results[name] = func(I_ref, I_fused)


    if I_ref is not None:
        del I_ref
    del I_fused
    gc.collect()

    return results

# === Fonctions de calcule des métriques ===
### Métriques simples :

# Fonction pour calculer l'entropie
@register_metric("entropy", ref_required=False)
def calculate_entropy(image):
    """
    Compute the entropy of an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (float32, normalized to [0, 1]).

    Returns
    -------
    float
        Entropy value of the image.
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 1])  # Correction de la plage
    hist_normalized = hist / hist.sum()
    return float(entropy(hist_normalized))

# Fonction pour calculer l'entropie normalisée
@register_metric("normalized_entropy", ref_required=False)
def calculate_entropy_normalized(image):
    """
    Compute the normalized entropy of an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (float32, normalized to [0, 1]).

    Returns
    -------
    float
        Normalized entropy value (relative to maximum entropy for 8-bit images).
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 1])  # Correction de la plage
    hist_normalized = hist / hist.sum()
    entropy_value = entropy(hist_normalized)
    
    # Normalisation
    max_entropy = np.log2(256)  # Pour une image 8-bit
    normalized_entropy = entropy_value / max_entropy
    
    return float(normalized_entropy)

# Fonction pour calculer l'écart type (standard deviation)
@register_metric("std_normalized", ref_required=False)
def calculate_std(image):
    """
    Compute the standard deviation of an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    float
        Standard deviation of pixel intensities.
    """
    # image = ensure_range_255(image)
    return float(np.std(image))


# Fonction pour calculer le gradient moyen (mean gradient)
@register_metric("mean_gradient_normalized", ref_required=False)
def calculate_mean_gradient(image):
    """
    Compute the mean gradient magnitude of an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    float
        Mean gradient magnitude.
    """
    grad_x = np.gradient(image, axis=0)
    grad_y = np.gradient(image, axis=1)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return float(np.mean(gradient_magnitude))

# Fonction pour calculer le MSE
@register_metric("mse_norm", ref_required=True)
def calculate_mse(image1, image2):
    """
    Compute the Mean Squared Error (MSE) between two images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image.
    image2 : numpy.ndarray
        Fused image.

    Returns
    -------
    float
        Mean squared error value.
    """
    return float(mean_squared_error(image1, image2))

# Fonction pour calculer le RMSE
@register_metric("rmse_norm", ref_required=True)
def calculate_rmse(image1, image2):
    """
    Compute the Root Mean Squared Error (RMSE) between two images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image.
    image2 : numpy.ndarray
        Fused image.

    Returns
    -------
    float
        Root mean squared error value.
    """
    return np.sqrt(mean_squared_error(image1, image2))

# Fonction pour calculer le coefficient de corrélation (entre deux images)
@register_metric("correlation", ref_required=True)
def calculate_correlation(image1, image2):
    """
    Compute the Pearson correlation coefficient between two images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image.
    image2 : numpy.ndarray
        Fused image.

    Returns
    -------
    float
        Correlation coefficient.
    """
    return float(pearsonr(image1.flatten(), image2.flatten())[0])

@register_metric("snr_greyscale", ref_required=True)
def calculate_snr(image1, image2):
    """
    Compute the Signal-to-Noise Ratio (SNR) between two grayscale images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image.
    image2 : numpy.ndarray
        Fused image.

    Returns
    -------
    float
        Signal-to-noise ratio in decibels (dB).
    """
    image1 = to_grayscale_array_skimage(image1)
    image2 = to_grayscale_array_skimage(image2)

    # Calculer la puissance du signal et la puissance du bruit
    signal_power = np.sum(image1 ** 2)
    noise_power = np.sum((image1 - image2) ** 2)
    
    # Éviter la division par zéro
    if noise_power == 0:
        return float('inf')
    
    # Calculer le rapport signal/bruit (SNR)
    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)

@register_metric("snr_color_per_channel", ref_required=True)
def calculate_snr_color_per_channel(image1, image2):
    """
    Compute the Signal-to-Noise Ratio (SNR) per channel for two color images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image (color).
    image2 : numpy.ndarray
        Fused image (color).

    Returns
    -------
    float
        Average SNR across all channels (in dB).
    """

    snrs = []
    for c in range(image1.shape[-1]):
        signal_power = np.sum(image1[..., c] ** 2)
        noise_power = np.sum((image1[..., c] - image2[..., c]) ** 2)

        if noise_power == 0:
            snr_c = float('inf')
        else:
            snr_c = 10 * np.log10(signal_power / noise_power)
        snrs.append(snr_c)

    return float(np.mean(snrs))

@register_metric("snr_color_global", ref_required=True)
def calculate_snr_color(image1, image2):
    """
    Compute the global Signal-to-Noise Ratio (SNR) between two color images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image (color).
    image2 : numpy.ndarray
        Fused image (color).

    Returns
    -------
    float
        Global SNR value (in dB).
    """

    signal_power = np.sum(image1 ** 2)
    noise_power = np.sum((image1 - image2) ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)

# Fonction pour calculer le PSNR
@register_metric("psnr", ref_required=True)
def calculate_psnr(image1, image2):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image.
    image2 : numpy.ndarray
        Fused image.

    Returns
    -------
    float
        PSNR value in decibels (dB).
    """
    return float(psnr(image1, image2, data_range=1))

@register_metric("spatial_frequency_color_norm", ref_required=False)
def calculate_spatial_frequency_color(image): # Canal par canal.
    """
    Compute the spatial frequency of a color image (per channel).

    Parameters
    ----------
    image : numpy.ndarray
        Input image (color or grayscale).

    Returns
    -------
    float
        Average spatial frequency across channels.
    """
    if image.ndim == 3:
        sf_channels = []
        for c in range(image.shape[2]):
            dx = np.diff(image[..., c], axis=1)
            dy = np.diff(image[..., c], axis=0)
            sf = np.sqrt(np.mean(dx**2) + np.mean(dy**2))
            sf_channels.append(sf)
        return float(np.mean(sf_channels))  # ou np.linalg.norm(sf_channels)
    else:
        return calculate_spatial_frequency_grey(image)


@register_metric("spatial_frequency_grey_norm", ref_required=False)
def calculate_spatial_frequency_grey(image):
    """
    Compute the spatial frequency of a grayscale image.

    Parameters
    ----------
    image : numpy.ndarray
        Input grayscale image.

    Returns
    -------
    float
        Spatial frequency value.
    """
    image = to_grayscale_array_skimage(image)
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    sf = float(np.sqrt(np.mean(dx**2) + np.mean(dy**2)))
    return sf

# Fonction pour calculer l'intensité moyenne des pixels
@register_metric("mean_intensity_norm", ref_required=False)
def calculate_mean_intensity(image):
    """
    Compute the mean pixel intensity of an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    float
        Mean intensity value.
    """
    # image = ensure_range_255(image)
    return float(np.mean(image))

### Métriques avancées :

@register_metric("ssim_grey", ref_required=True)
def calculate_ssim(image1, image2): # (Structural Similarity Index Measure)
    """
    Compute the Structural Similarity Index (SSIM) between two grayscale images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference grayscale image.
    image2 : numpy.ndarray
        Fused grayscale image.

    Returns
    -------
    float
        SSIM value in [0, 1].
    """

    image1_gray = to_grayscale_array_skimage(image1)
    image2_gray = to_grayscale_array_skimage(image2)

    return float(ssim(image1_gray, image2_gray, data_range=1.0))

@register_metric("ssim_color", ref_required=True)
def calculate_ssim(image1, image2): # (Structural Similarity Index Measure)
    """
    Compute the Structural Similarity Index (SSIM) between two color images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference color image.
    image2 : numpy.ndarray
        Fused color image.

    Returns
    -------
    float
        SSIM value in [0, 1].
    """

    return float(ssim(image1, image2, data_range=1.0, channel_axis=-1))

@register_metric("ms_ssim_pytorch_greyscale", ref_required=True)
def calculate_ms_ssim_pytorch(image1, image2):
    """
    Compute the Multi-Scale Structural Similarity Index (MS-SSIM) between two grayscale images using PyTorch.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference grayscale image (float32, normalized [0, 1]).
    image2 : numpy.ndarray
        Fused grayscale image (float32, normalized [0, 1]).

    Returns
    -------
    float
        MS-SSIM value in [0, 1].
    """

    image1_grey = to_grayscale_array_skimage(image1)
    image2_grey = to_grayscale_array_skimage(image2)

    # Convertir en tenseurs sans rediviser par 255.0 (déjà en float32 [0,1])
    img1_tensor = torch.from_numpy(image1_grey).unsqueeze(0).unsqueeze(0)
    img2_tensor = torch.from_numpy(image2_grey).unsqueeze(0).unsqueeze(0)

    result = float(ms_ssim(img1_tensor, img2_tensor, data_range=1.0).item())
    return result

@register_metric("ms_ssim_pytorch_color", ref_required=True)
def calculate_ms_ssim_pytorch_color(image1, image2):
    """
    Compute the Multi-Scale Structural Similarity Index (MS-SSIM) between two color images using PyTorch.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference color image (float32, normalized [0, 1]).
    image2 : numpy.ndarray
        Fused color image (float32, normalized [0, 1]).

    Returns
    -------
    float
        MS-SSIM value in [0, 1].
    """

    # Vérifie que les images sont bien au format attendu : (H, W, 3), float32, [0,1]
    assert image1.ndim == 3 and image1.shape[2] == 3, "image1 n'est pas une image couleur"
    assert image2.ndim == 3 and image2.shape[2] == 3, "image2 n'est pas une image couleur"
    assert image1.dtype == np.float32 and image2.dtype == np.float32, "Les images doivent être en float32"
    assert image1.min() >= 0.0 and image1.max() <= 1.0, "Les valeurs doivent être dans [0,1]"
    assert image2.min() >= 0.0 and image2.max() <= 1.0, "Les valeurs doivent être dans [0,1]"

    # Convertir (H, W, C) → (1, C, H, W)
    img1_tensor = torch.from_numpy(np.transpose(image1, (2, 0, 1))).unsqueeze(0)
    img2_tensor = torch.from_numpy(np.transpose(image2, (2, 0, 1))).unsqueeze(0)

    result = float(ms_ssim(img1_tensor, img2_tensor, data_range=1.0).item())
    return result

@register_metric("brisque", ref_required=False)
def calculate_brisque(image):
    """
    Compute the BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) score.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (converted to RGB if necessary).

    Returns
    -------
    float
        BRISQUE score (lower is better).
    """
    
    # image = to_rgb_array(image)

    image = from_grey_to_rgb_array(image)  # conversion en RGB si nécessaire
    
    # Créer une instance du calculateur BRISQUE
    brisque_calculator = brisque.BRISQUE()
    
    # Calculer et retourner le score BRISQUE
    score = brisque_calculator.score(image)
    return float(score)

@register_metric("gmsd_norm_grey", ref_required=True)
def calculate_gmsd(image1, image2): # (Gradient Magnitude Similarity Deviation)
    """
    Compute the Gradient Magnitude Similarity Deviation (GMSD) between two grayscale images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference grayscale image.
    image2 : numpy.ndarray
        Fused grayscale image.

    Returns
    -------
    float
        GMSD value (lower indicates higher similarity).
    """

    image1_gray = to_grayscale_array_skimage(image1)
    image2_gray = to_grayscale_array_skimage(image2)
    
    gx1, gy1 = np.gradient(image1_gray)
    gx2, gy2 = np.gradient(image2_gray)
    
    magnitude1 = np.sqrt(gx1**2 + gy1**2)
    magnitude2 = np.sqrt(gx2**2 + gy2**2)
    
    gms = (2 * magnitude1 * magnitude2 + 0.01) / (magnitude1**2 + magnitude2**2 + 0.01)

    return float(np.std(gms))

@register_metric("gmsd_norm_color", ref_required=True)
def calculate_gmsd_color(image1, image2):
    """
    Compute the Gradient Magnitude Similarity Deviation (GMSD) between two color images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference color image.
    image2 : numpy.ndarray
        Fused color image.

    Returns
    -------
    float
        Average GMSD across channels.
    """

    assert image1.shape[-1] == 3 and image2.shape[-1] == 3, "Les images doivent être en couleur (3 canaux)."
    
    gmsd_per_channel = []

    for c in range(3):  # Pour R, G, B
        img1_c = image1[..., c]
        img2_c = image2[..., c]

        gx1, gy1 = np.gradient(img1_c)
        gx2, gy2 = np.gradient(img2_c)

        mag1 = np.sqrt(gx1**2 + gy1**2)
        mag2 = np.sqrt(gx2**2 + gy2**2)

        gms = (2 * mag1 * mag2 + 0.01) / (mag1**2 + mag2**2 + 0.01)
        gmsd = np.std(gms)

        gmsd_per_channel.append(gmsd)

    result = float(np.mean(gmsd_per_channel))
    return result


@register_metric("MAD_norm_grey", ref_required=True)
def calculate_mad(image1, image2): # (Mean Absolute Deviation)
    """
    Compute the Mean Absolute Deviation (MAD) between two grayscale images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference grayscale image.
    image2 : numpy.ndarray
        Fused grayscale image.

    Returns
    -------
    float
        MAD value.
    """

    image1_gray = to_grayscale_array_skimage(image1)
    image2_gray = to_grayscale_array_skimage(image2)
    
    return float(np.mean(np.abs(image1_gray - image2_gray)))

@register_metric("MAD_norm_color", ref_required=True)
def calculate_mad_color(image1, image2):
    """
    Compute the Mean Absolute Deviation (MAD) between two color images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference color image.
    image2 : numpy.ndarray
        Fused color image.

    Returns
    -------
    float
        MAD value.
    """

    assert image1.shape == image2.shape, "Les images doivent avoir la même forme."
    assert image1.ndim == 3 and image1.shape[-1] == 3, "Les images doivent être en couleur (3 canaux)."

    mad = float(np.mean(np.abs(image1 - image2)))
    return mad

@register_metric("UQI_grey", ref_required=True)
def calculate_uqi(image1, image2): # (Universal Image Quality Index)
    """
    Compute the Universal Image Quality Index (UQI) between two grayscale images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference grayscale image.
    image2 : numpy.ndarray
        Fused grayscale image.

    Returns
    -------
    float
        UQI value in [-1, 1], where 1 indicates perfect similarity.
    """

    image1_gray = to_grayscale_array_skimage(image1)
    image2_gray = to_grayscale_array_skimage(image2)
    return float(uqi(image1_gray, image2_gray))

@register_metric("UQI_color", ref_required=True)
def calculate_uqi(image1, image2): # (Universal Image Quality Index)
    """
    Compute the Universal Image Quality Index (UQI) between two color images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference color image.
    image2 : numpy.ndarray
        Fused color image.

    Returns
    -------
    float
        UQI value in [-1, 1], where 1 indicates perfect similarity.
    """
    return float(uqi(image1, image2))

### Autre Métriques issus de Sewar 0.4 : "https://github.com/andrewekhalel/sewar.git"

# ERGAS - métrique avec référence
@register_metric("ergas", ref_required=True)
def calculate_ergas(image1, image2, ratio=0.25):
    """
    Compute the ERGAS (Relative Dimensionless Global Error of Synthesis).

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image (ground truth), normalized to [0, 1].
    image2 : numpy.ndarray
        Fused image, normalized to [0, 1].
    ratio : float, default=0.25
        Ratio between pixel sizes of the reference and fused images.

    Returns
    -------
    float
        ERGAS value (lower is better).
    """
    return float(ergas(image1, image2, r=ratio))

# SAM - métrique avec référence
@register_metric("sam", ref_required=True)
def calculate_sam(image1, image2):
    """
    Compute the Spectral Angle Mapper (SAM) between two images.

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image (ground truth), normalized to [0, 1].
    image2 : numpy.ndarray
        Fused image, normalized to [0, 1].

    Returns
    -------
    float
        SAM value in radians (lower is better).
    """
    return float(sam(image1, image2))

# VIF - métrique avec référence
@register_metric("vif", ref_required=True)
def calculate_vif(image1, image2, sigma_nsq=2):
    """
    Compute the Visual Information Fidelity (VIF).

    Parameters
    ----------
    image1 : numpy.ndarray
        Reference image (ground truth), normalized to [0, 1].
    image2 : numpy.ndarray
        Fused image, normalized to [0, 1].
    sigma_nsq : float, default=2
        Variance of the visual noise.

    Returns
    -------
    float
        VIF value (higher is better).
    """
    return float(vifp(image1, image2, sigma_nsq=sigma_nsq))

#################################################### External Metrics
@register_metric("niqe", ref_required=False)
def calculate_niqe_metric(image):
    """
    Compute the NIQE (Natural Image Quality Evaluator) score without reference.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    float
        NIQE score (lower is better).
    """
    img_gray = to_grayscale_array_skimage(image)
    score = calculate_niqe(img_gray)
    return float(score)

@register_metric("piqe", ref_required=False)
def calculate_piqe(image):
    """
    Compute the PIQE (Perception-based Image Quality Evaluator) score without reference.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    float
        PIQE score (lower is better).
    """
    score, _, _, _ = piqe(image)
    return float(score)

#################################################### Non functional from Sewar
# # D_lambda - sans référence
# @register_metric("d_lambda", ref_required=True)
# def calculate_d_lambda(ms_image, fused_image, p=1):
#     """
#     Spectral Distortion Index (D_lambda)

#     Args:
#         ms_image (np.ndarray): Image multispectrale basse résolution, normalisée entre 0 et 1.
#         fused_image (np.ndarray): Image fusionnée haute résolution, normalisée entre 0 et 1.

#     Returns:
#         float: Valeur D_lambda.
#     """
#     return float(d_lambda(ms_image, fused_image, p=p))

# # D_S - sans référence
# @register_metric("d_s", ref_required=True)
# def calculate_d_s(pan_image, ms_image, fused_image, q=1, r=4, ws=7):
#     """
#     Spatial Distortion Index (D_S)

#     Args:
#         pan_image (np.ndarray): Image panchromatique haute résolution, normalisée entre 0 et 1.
#         ms_image (np.ndarray): Image multispectrale basse résolution, normalisée entre 0 et 1.
#         fused_image (np.ndarray): Image fusionnée, normalisée entre 0 et 1.

#     Returns:
#         float: Valeur D_S.
#     """
#     return float(d_s(pan_image, ms_image, fused_image, q=q, r=r, ws=ws))

# # QNR - sans référence
# @register_metric("qnr", ref_required=True)
# def calculate_qnr(pan_image, ms_image, fused_image, alpha=1, beta=1, p=1, q=1, r=4, ws=7):
#     """
#     Quality with No Reference (QNR)

#     Args:
#         pan_image (np.ndarray): Image panchromatique haute résolution, normalisée entre 0 et 1.
#         ms_image (np.ndarray): Image multispectrale basse résolution, normalisée entre 0 et 1.
#         fused_image (np.ndarray): Image fusionnée, normalisée entre 0 et 1.

#     Returns:
#         float: Valeur QNR.
#     """
#     return float(qnr(pan_image, ms_image, fused_image, alpha=alpha, beta=beta, p=p, q=q, r=r, ws=ws))


#================================= DEBUG =================================#
    # print(f"\n[{tag}] dtype: {image.dtype}, min: {image.min()}, max: {image.max()}, shape: {image.shape}")

if __name__ == "__main__":
    print(f"[INFO] Métriques sans référence : {list(no_ref_metrics.keys())}")
    print(f"[INFO] Métriques avec référence : {list(full_ref_metrics.keys())}")

    # img_file = cv2.imread(r"C:\Users\Riffard\Documents\Datasets\Fusion\Tests_carte_de_poids_et_pyramides\_test_VISWIR\temp2\proc\weight_0.70\beta_1.5\level_4\no_gamma\0003_fused_with_post_processing.tiff", cv2.IMREAD_GRAYSCALE)
    # ref_image_path = r"C:\Users\Riffard\Documents\Datasets\Cerema-17-18-June\21\4 - TarDal\Visible\0002.jpg"
    # ref_file = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)
    # metric_file = calculate_brisque(img_file)

    # print("Metrique depuis fichier :", metric_file)

    if len(sys.argv) < 2:
        print("Usage : python metrics.py image_fusionnee.png [image_reference.png]")
        sys.exit(1)

    fused_path = sys.argv[1]
    ref_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(fused_path):
        print(f"Erreur : le fichier {fused_path} n'existe pas.")
        sys.exit(1)

    I_fused = imread(fused_path)
    if I_fused.ndim == 3 and I_fused.shape[2] == 4:
        print("[INFO] L’image fusionnée contient 4 canaux. Suppression du 4e (transparence ?).")
        I_fused = I_fused[:, :, :3]
    print("\n[INFO] Image fusionnée :")
    print_image_info(I_fused)

    I_ref = None
    if ref_path:
        if not os.path.exists(ref_path):
            print(f"Erreur : le fichier {ref_path} n'existe pas.")
            sys.exit(1)
        I_ref = imread(ref_path)
        print("\n[INFO] Image de référence :")
        print_image_info(I_ref)

    if I_fused.shape != I_ref.shape:
        print("[INFO] Redimensionnement de l'image fusionnée pour correspondre à l'image de référence")
        I_fused = resize(I_fused, I_ref.shape, preserve_range=True, anti_aliasing=True).astype(I_ref.dtype)


    metrics = compute_all_metrics(I_ref=I_ref, I_fused=I_fused)
    print("\n[RESULTATS] Métriques calculées :")
    for k, v in metrics.items():
        if v is not None:
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: Erreur ou valeur non calculée")