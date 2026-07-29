"""
Direct support functions for VISWIR image processing and manipulation.
"""

# =============================================================================
# FILENAME:       functions.py
# DESCRIPTION:    Ce fichier contient les fonctions utilisées par le coeur de la
#                 fusion : "viswir_core_fusion" ; utilisées dans le cadre d'un 
#                 projet de fusion d'images Visible et SWIR : VISWIR.
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
# USAGE:          - Importer les fonctions depuis ce fichier dans le fichier "fusion.py".
#                 - Ces fonctions n'ont pas pour vocation à être utilisées en dehors du coeur de la fusion.
#
# DEPENDENCIES:   - numpy
#                 - skimage
#                 - cv2
#                 - scipy
#
# NOTES:
#   - ...
#
# CHANGELOG:
#   - [16-04-2025]: Création initiale du fichier à partir d'un ancien notebook.
#   - [29-04-2025]: Ajout de documentation dans certaines fonctions.
#
# =============================================================================

import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import convolve, uniform_filter
import numpy.typing as npt



def pyramid_filter():
    """
    Generate a 1D Gaussian filter for pyramid construction.

    Returns
    -------
    numpy.ndarray
        1D Gaussian filter coefficients.
    """
    return np.array([0.0625, 0.25, 0.375, 0.25, 0.0625]) # [1, 4, 6, 4, 1] / 16

def laplacian_pyramid(image:npt.NDArray, levels:int|None=None) -> list[npt.NDArray]:
    """
    Build a Laplacian pyramid with explicit control over filtering and interpolation.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    levels : int, optional
        Number of pyramid levels. If None, it is computed automatically
        based on the smallest image dimension.

    Returns
    -------
    list of numpy.ndarray
        List of Laplacian pyramid levels.
    """
    r, c = image.shape[:2]
    if levels is None:
        levels = int(np.floor(np.log2(min(r, c))))

    pyr = []
    filter_1d = pyramid_filter()
    current_image = image

    for _ in range(levels - 1):
        # Appliquer un filtre passe-bas (filtrage Gaussien)
        low_pass = convolve2d(current_image, filter_1d[:, None], mode='same', boundary='symm')
        low_pass = convolve2d(low_pass, filter_1d[None, :], mode='same', boundary='symm')
        # Sous-échantillonnage
        downsampled = low_pass[::2, ::2]

        # Interpolation
        upsampled = np.zeros_like(current_image)
        upsampled[::2, ::2] = downsampled
        reconstructed = convolve2d(upsampled, filter_1d[:, None], mode='same', boundary='symm')
        reconstructed = convolve2d(reconstructed, filter_1d[None, :], mode='same', boundary='symm')

        # Calcul de la différence pour obtenir le niveau Laplacien
        laplacian = current_image - reconstructed
        pyr.append(laplacian)

        # Mise à jour de l'image courante pour le niveau suivant
        current_image = downsampled

    pyr.append(current_image)  # Ajouter le dernier niveau (résidu)
    return pyr

def reconstruct_laplacian_pyramid(pyr):
    """
    Reconstruct an image from a Laplacian pyramid.

    Parameters
    ----------
    pyr : list of numpy.ndarray
        Laplacian pyramid (list of images).

    Returns
    -------
    numpy.ndarray
        Reconstructed image.
    """
    filter_1d = pyramid_filter()  # Filtre gaussien [1, 4, 6, 4, 1] / 16
    reconstructed = pyr[-1]  # Niveau le plus bas de la pyramide

    for level in reversed(pyr[:-1]):
        # Upsampling avec adaptation des dimensions
        # upsampled = cv2.resize(reconstructed, (level.shape[1], level.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Sur-échantillonner et filtrer avec un filtre gaussien --> [1, 4, 6, 4, 1] / 16
        upsampled = np.zeros_like(level)
        # upsampled[::2, ::2] = 4*reconstructed
        upsampled[::2, ::2] = reconstructed
        # print(upsampled)

        # Appliquer un filtrage gaussien après l'upsampling
        filtered = convolve2d(upsampled, filter_1d[:, None], mode='same', boundary='symm')  # Convolution verticale
        filtered = convolve2d(filtered, filter_1d[None, :], mode='same', boundary='symm')  # Convolution horizontale

        # Ajouter le niveau courant pour reconstruire
        reconstructed = filtered + level #* 1.5

    return reconstructed

def reconstruct_laplacian_pyramid_2(pyr):
    """
    Alternative reconstruction of an image from a Laplacian pyramid.

    Parameters
    ----------
    pyr : list of numpy.ndarray
        Laplacian pyramid.

    Returns
    -------
    numpy.ndarray
        Reconstructed image.
    """
    nlev = len(pyr)
    R = pyr[-1]  # Commencer avec le résidu passe-bas
    filter = pyramid_filter()

    for l in range(nlev - 2, -1, -1):
        odd = 2 * np.array(R.shape) - np.array(pyr[l].shape)
        R = pyr[l] + my_upsample(R, odd, filter)

    return R

# Fonction utilisé pour des tests uniquement
def reconstruct_without_laplacian(pyr):
    """
    Reconstruct an image from a pyramid without Laplacian levels.

    This function performs only upsampling and filtering,
    without adding Laplacian residuals.

    Parameters
    ----------
    pyr : list of numpy.ndarray
        Pyramid levels.

    Returns
    -------
    numpy.ndarray
        Reconstructed image.

    Notes
    -------
    Function used for tests only.
    """
    nlev = len(pyr)
    R = pyr[-1]  # Commencer avec le résidu passe-bas
    filter = pyramid_filter()

    for l in range(nlev - 2, -1, -1):
        odd = 2 * np.array(R.shape) - np.array(pyr[l].shape)
        R = my_upsample(R, odd, filter)  # Seulement l'upsampling et le filtrage, sans ajout du niveau courant

    return R

def gaussian_pyramid(image, levels): # Version avec OpenCV (fonctionne tout aussi bien que l'original)
    """
    Build a Gaussian pyramid from the input image.
    Use OpenCV.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    levels : int
        Number of pyramid levels.

    Returns
    -------
    list of numpy.ndarray
        List of Gaussian pyramid levels.
    """
    pyramid = [image]
    for _ in range(levels - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def my_upsample(image, odd, filter):
    """
    Custom upsampling with Gaussian filtering.

    Parameters
    ----------
    image : numpy.ndarray
        Input image to upsample.
    odd : tuple of int
        Adjustment values for odd dimensions.
    filter : numpy.ndarray
        1D Gaussian filter.

    Returns
    -------
    numpy.ndarray
        Upsampled and filtered image.
    """
    # Sur-échantillonner l'image
    upsampled = np.zeros((image.shape[0] * 2, image.shape[1] * 2))
    upsampled[::2, ::2] = image

    # Appliquer un filtrage gaussien après l'upsampling
    filtered = convolve(upsampled, filter[:, None], mode='mirror')  # Convolution verticale
    filtered = convolve(filtered, filter[None, :], mode='mirror')  # Convolution horizontale

    # Ajuster les dimensions si nécessaire
    if odd[0] > 0:
        filtered = filtered[:-1, :]
    if odd[1] > 0:
        filtered = filtered[:, :-1]

    return filtered



def downsample(image, filter=None):
    """
    Reduce the resolution of an image by filtering and subsampling.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    filter : numpy.ndarray, optional
        Filter to apply before downsampling.

    Returns
    -------
    numpy.ndarray
        Downsampled image.
    """
    if filter is not None:
        image = cv2.filter2D(image, -1, filter)
    return cv2.pyrDown(image)

def calculate_levels(image):
    """
    Dynamically compute the number of pyramid levels.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    int
        Number of pyramid levels.
    """
    min_dim = min(image.shape[:2])
    return int(np.floor(np.log2(min_dim)))

###---------------------------------------------------------------------------------------------------###

def local_std(image, nhood):
    """
    Compute the local standard deviation of an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    nhood : numpy.ndarray
        Neighborhood (window) used for local statistics.

    Returns
    -------
    numpy.ndarray
        Local standard deviation map.
    """
    # return np.sqrt(uniform_filter(image ** 2, nhood.shape) - uniform_filter(image, nhood.shape) ** 2)
    return np.sqrt(np.maximum(uniform_filter(image ** 2, nhood.shape) - uniform_filter(image, nhood.shape) ** 2, 0))


def local_visibility(image, size1, sigma1, sigma2):
    """
    Compute local visibility of an image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    size1 : int
        Kernel size for Gaussian filtering.
    sigma1 : float
        Standard deviation for the first Gaussian filter.
    sigma2 : float
        Standard deviation for the second Gaussian filter.

    Returns
    -------
    numpy.ndarray
        Local visibility map.
    """
    gaussian1 = cv2.getGaussianKernel(size1, sigma1)
    gaussian2 = cv2.getGaussianKernel(size1, sigma2)
    IM = cv2.filter2D(image, -1, gaussian1 @ gaussian1.T)
    noise = image - IM
    return np.sqrt(cv2.filter2D(noise ** 2, -1, gaussian2 @ gaussian2.T))