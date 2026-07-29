"""
NIQE (Natural Image Quality Evaluator) Implementation
=====================================================

This module implements the NIQE algorithm for no-reference image quality
assessment. The implementation is adapted from the GitHub repository:
https://github.com/nuniniyujin/niqe/tree/score_debug

The original algorithm has been adapted to fit the specific requirements
and structure of the VISWIR project. The core logic and algorithms remain
consistent with the original work.

References
----------
A. Mittal, R. Soundararajan, and A. C. Bovik,
"Making a 'Completely Blind' Image Quality Analyzer,"
IEEE Signal Processing Letters, 2013.

Author
------
Riffard Alexandre

Last Modified
-------------
28/10/2025
"""

import numpy as np
import scipy.io
import scipy.ndimage
import scipy.special
import scipy.linalg
import math
from PIL import Image
from os.path import dirname, join

from common.logger import logger

# Définition des variables globales et les fonctions nécessaires ici
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0/gamma_range)
a *= a
b = scipy.special.gamma(1.0/gamma_range)
c = scipy.special.gamma(3.0/gamma_range)
prec_gammas = a/(b*c)

def aggd_features(imdata):
    """
    NIQE helper function: aggd_features
    """
    # Flatten imdata
    imdata.shape = (len(imdata.flat),)
    
    # Separate positive and negative data before squaring
    left_data = imdata[imdata < 0]
    right_data = imdata[imdata >= 0]
    
    # Now square the data
    left_data_squared = left_data * left_data
    right_data_squared = right_data * right_data
    
    # Calculate means of the squared data
    left_mean_sqrt = 0
    right_mean_sqrt = 0
    if len(left_data_squared) > 0:
        left_mean_sqrt = np.sqrt(np.average(left_data_squared))
    if len(right_data_squared) > 0:
        right_mean_sqrt = np.sqrt(np.average(right_data_squared))

    # Calculate gamma_hat
    if right_mean_sqrt != 0:
        gamma_hat = left_mean_sqrt / right_mean_sqrt
    else:
        gamma_hat = np.inf

    # Sécurisation : bornes cohérentes avec gamma_range (0.2 → 10)
    if gamma_hat < 0.2 or gamma_hat > 10 or not np.isfinite(gamma_hat):
        # logger.warning(f"⚠️ gamma_hat hors borne détecté : {gamma_hat}, clampé dans [0.2, 10]")
        gamma_hat = max(min(gamma_hat, 10), 0.2)

    # Solve for r_hat norm
    imdata_squared = imdata * imdata  # Squared imdata
    imdata2_mean = np.mean(imdata_squared)
    if imdata2_mean != 0:
        r_hat = (np.average(np.abs(imdata))**2) / (np.average(imdata_squared))
    else:
        r_hat = np.inf

    # Sécurisation du dénominateur
    denominator = math.pow(math.pow(gamma_hat, 2) + 1, 2) + 1e-8
    rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / denominator)
    # rhat_norm = r_hat * (((math.pow(gamma_hat, 3) + 1) * (gamma_hat + 1)) / math.pow(math.pow(gamma_hat, 2) + 1, 2))

    # Solve for alpha by guessing values that minimize rhat_norm
    pos = np.argmin((prec_gammas - rhat_norm)**2)
    alpha = gamma_range[pos]

    # Calculate gamma values
    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)

    # Calculate AGGD ratio
    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt

    # Calculate N
    N = (br - bl) * (gam2 / gam1)

    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def ggd_features(imdata):
    """
    NIQE helper function: ggd_features
    """
    nr_gam = 1/prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq/E**2
    pos = np.argmin(np.abs(nr_gam - rho))
    return gamma_range[pos], sigma_sq

def paired_product(new_im):
    """
    NIQE helper function: paired_product
    """
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)

    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im

    return (H_img, V_img, D1_img, D2_img)

def gen_gauss_window(lw, sigma):
    """
    NIQE helper function: gen_gauss_window
    """
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum = 1.0
    sd *= sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * np.float32(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp
    for ii in range(2 * lw + 1):
        weights[ii] /= sum
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    """
    Compute MSCN coefficients using Gaussian filtering with correlate1d.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        C (float): Stabilizing constant.
        avg_window (numpy.ndarray): Gaussian kernel for filtering.
        extend_mode (str): Padding mode for boundaries.

    Returns:
        tuple: MSCN coefficients, variance, and mean.
    """
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)

    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')

    # Apply Gaussian filtering along rows and columns
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)

    # Compute variance and MSCN coefficients
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    mscn_coeffs = (image - mu_image) / (var_image + C)

    return mscn_coeffs, var_image, mu_image

def extract_on_patches(img, patch_size):
    """
    NIQE helper function: extract_on_patches
    """
    h, w = img.shape
    patch_size = int(patch_size)  # Fixed np.int deprecation
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)

    patches = np.array(patches)

    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)

    return patch_features

def _get_patches_generic(img, patch_size, is_train, stride):
    """
    NIQE helper function: _get_patches_generic
    """
    h, w = np.shape(img)
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)

    # ensure that the patch divides evenly into img
    hoffset = (h % patch_size)
    woffset = (w % patch_size)

    if hoffset > 0: 
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]


    img = img.astype(np.float32)
    img2 = np.array(Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2), Image.BICUBIC), dtype=np.float32)

    mscn1, var, mu = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)

    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)


    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size/2)

    feats = np.hstack((feats_lvl1, feats_lvl2))# feats_lvl3))

    return feats

def _niqe_extract_subband_feats(mscncoefs):
    """
    NIQE helper function: _niqe_extract_subband_feats
    """
    # alpha_m,  = extract_ggd_features(mscncoefs)
    alpha_m, N, bl, br, lsq, rsq = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, lsq1, rsq1 = aggd_features(pps1)
    alpha2, N2, bl2, br2, lsq2, rsq2 = aggd_features(pps2)
    alpha3, N3, bl3, br3, lsq3, rsq3 = aggd_features(pps3)
    alpha4, N4, bl4, br4, lsq4, rsq4 = aggd_features(pps4)
    features = np.array([
        alpha_m, (bl + br) / 2.0,
        alpha1, N1, bl1, br1,  # (V)
        alpha2, N2, bl2, br2,  # (H)
        alpha3, N3, bl3, br3,  # (D1)
        alpha4, N4, bl4, br4   # (D2)
    ])
    return features

def get_patches_test_features(img, patch_size, stride=8):
    """
    NIQE helper function: get_patches_test_features
    """
    return _get_patches_generic(img, patch_size, 0, stride)

def niqe(inputImgData):
    """
    NIQE helper function: niqe
    """
    patch_size = 96
    module_path = dirname(__file__)

    params = scipy.io.loadmat(join(module_path, 'clean_image_parameters.mat'))
    pop_mu = np.ravel(params["clean_mean"])
    pop_cov = params["clean_cov"]

    M, N = inputImgData.shape

    assert M > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"
    assert N > (patch_size*2+1), "niqe called with small frame size, requires > 192x192 resolution video using current training parameters"

    feats = get_patches_test_features(inputImgData, patch_size)
    # print(feats.shape)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    X = sample_mu - pop_mu
    covmat = ((pop_cov+sample_cov)/2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))

    return niqe_score

def calculate_niqe(image):
    """
    Calculate the NIQE score for a grayscale image.

    Args:
        image (numpy.ndarray): Input grayscale image in float64 format.

    Returns:
        float: NIQE score for the grayscale image.
    """
    # Ensure the image is in the correct format
    if image.dtype != np.float64:
        image = image.astype(np.float64)

    # Calculate and return the NIQE score
    niqe_score = niqe(image)
    return float(niqe_score)


####################################################
# import sys
# from skimage.color import rgb2gray

# def to_grayscale_array_skimage(image):
#     # """Convert an image to grayscale using skimage."""
#     # if isinstance(image, np.ndarray):
#     #     if len(image.shape) == 3 and image.shape[2] == 3:
#     #         # Simple conversion to grayscale assuming RGB image
#     #         return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
#     #     elif len(image.shape) == 2:
#     #         return image
#     # elif isinstance(image, Image.Image):
#     #     return np.array(image.convert('L'))
#     # else:
#     #     raise ValueError("Unsupported image type")
#     """
#     Convertit une image couleur en niveaux de gris en utilisant skimage.

#     Paramètres :
#     - image : np.ndarray
#         Image en entrée (2D ou 3D).

#     Retour :
#     - np.ndarray
#         Image en niveaux de gris.
#     """
#     if isinstance(image, np.ndarray):
#         if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
#             # Déjà en niveaux de gris
#             return np.squeeze(image)  # Au cas où il reste un canal singleton
#         elif image.ndim == 3 and image.shape[-1] == 3:
#             # Conversion RGB -> Grayscale
#             return rgb2gray(image)
#         else:
#             raise ValueError("Format d'image non reconnu (np.ndarray)")
#     else:
#         raise TypeError("L'image doit être un tableau NumPy")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python niqe.py <image_path>")
#         sys.exit(1)

#     image_path = sys.argv[1]
#     try:
#         # Load the image and convert to grayscale
#         image = np.array(Image.open(image_path))
#         img_gray = to_grayscale_array_skimage(image)

#         # Calculate NIQE score
#         niqe_score = calculate_niqe(img_gray)
#         print(f"NIQE score for '{image_path}': {niqe_score:.3f}")
#     except Exception as e:
#         print(f"An error occurred: {e}")
