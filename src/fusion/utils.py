"""
General utility functions, file I/O, and image preparation for the VISWIR project.
"""

# =============================================================================
# FILENAME:       utils.py
# DESCRIPTION:    Ce fichier contient les fonctions utilitaire du projet VISWIR
#                 utilisées dans le cadre d'un projet de fusion d'images Visible
#                 et SWIR : VISWIR.
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
# USAGE:          - Importer les fonctions depuis ce fichier pour les utiliser.
#                 - Plusieurs version de fonctions sont disponible selon les usages
#                   (Par exemple : OpenCV et Skimage pour le chargement des images.)
#
# DEPENDENCIES:   - numpy
#                 - skimage
#                 - cv2
#                 - matplotlib
#                 - torch
#
# NOTES:
#   - ...
#
# CHANGELOG:
#   - [16-04-2025]: Création initiale du fichier à partir d'un ancien notebook.
#   - [16-04-2025]-[29-04-2025]: Ajustements et ajouts divers...
#   - [30-04-2025]: Ajout d'un fonction "load_image" avec Skimage.
#
# =============================================================================

import os
import glob
import sys
import time
from pathlib import Path
from typing import Optional, List
import json
import cv2
from skimage.color import rgb2gray
from skimage.color import gray2rgb
from skimage import io#, color
# from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

from loguru import logger

def load_image_opencv(path, is_rgb=False, is_swir=False):
    """
    Load an image with OpenCV and normalize it according to its data type.

    If the image is SWIR, the first channel is extracted if necessary.

    Parameters
    ----------
    path : str
        Path to the image file.
    is_rgb : bool, default=False
        If True, load the image in color (RGB).
    is_swir : bool, default=False
        If True, apply specific preprocessing for SWIR images.

    Returns
    -------
    numpy.ndarray
        Normalized image as float64 in [0, 1].

    Raises
    ------
    ValueError
        If the image cannot be loaded or if the data type is unsupported.
    """
    # Détermine le mode de chargement
    flag = cv2.IMREAD_COLOR if is_rgb else cv2.IMREAD_UNCHANGED
    
    # Charge l'image
    image = cv2.imread(path, flag)
    if image is None:
        raise ValueError(f"Impossible de charger l'image à {path}")
    
    # Vérifie si l'image est SWIR (traitement des canaux)
    if is_swir:
        image = image[:, :, 0] if image.ndim == 3 else image
    
    # Normalisation en fonction du type de données
    if image.dtype == np.uint8:
        # Pour les images uint8 ou 24 bits (interprétées comme uint8)
        image = image.astype(np.float64) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float64) / 65535.0
    elif image.dtype == np.uint32:
        image = image.astype(np.float64) / (2**32 - 1)
    # elif image.dtype == np.float32 or image.dtype == np.float64:
    #     # Les images flottantes sont déjà normalisées
    #     pass
    elif image.dtype == np.float32:
        image = image.astype(np.float64)  # Conversion explicite
    elif image.dtype == np.float64:
        pass  # Déjà bon
    # elif image.dtype == np.uint14:
    #     # Décalage des uint14 en uint16
    #     image = (image.astype(np.uint16) << 2).astype(np.float64) / 65535.0
    else:
        raise ValueError(f"Type d'image non supporté : {image.dtype}")
    
    return image

def load_image_skimage_core(path, is_swir=False):
    """
    Load an image with skimage and normalize it according to its data type.

    If the image is SWIR, the first channel is extracted if necessary.
    A BGR → RGB conversion is applied to correct color ordering.

    Parameters
    ----------
    path : str
        Path to the image file.
    is_swir : bool, default=False
        If True, apply specific preprocessing for SWIR images.

    Returns
    -------
    numpy.ndarray
        Normalized image as float64 in [0, 1].

    Raises
    ------
    ValueError
        If the image cannot be loaded or if the data type is unsupported.
    """
    # Charge l'image
    image = io.imread(path)

    if image is None:
        raise ValueError(f"Impossible de charger l'image à {path}")
    
    # Conversion BGR → RGB (corrige l'effet bleu)
    if image.ndim == 3 and image.shape[-1] == 3:
        image = image[..., ::-1]  # Inverse l'ordre des canaux

    # Vérifie si l'image est SWIR (traitement des canaux)
    if is_swir:
        image = image[:, :, 0] if image.ndim == 3 else image

    # Normalisation en fonction du type de données
    if image.dtype == np.uint8:
        image = image.astype(np.float64) / 255.0
    elif image.dtype == np.uint16:
        image = image.astype(np.float64) / 65535.0
    elif image.dtype == np.uint32:
        image = image.astype(np.float64) / (2**32 - 1)
    elif image.dtype in [np.float32, np.float64]:
        image = image.astype(np.float64)  # Assure un type homogène
    else:
        raise ValueError(f"Type d'image non supporté : {image.dtype}")

    return image


def load_image_ref(path, as_gray=False, normalize=True):
    """
    Load a reference image with OpenCV and optionally normalize it.

    Parameters
    ----------
    path : str
        Path to the image file.
    as_gray : bool, default=False
        If True, load the image in grayscale.
    normalize : bool, default=True
        If True, normalize pixel values to [0, 1].

    Returns
    -------
    numpy.ndarray
        Loaded image as float32.
    """
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)

    if img is None:
        raise ValueError(f"Image non trouvée à {path}")

    img = img.astype(np.float32)

    if normalize and img.max() > 1.0:
        img /= 255.0

    return img

def load_image_ref_skimage(path, as_gray=False, normalize=True):
    """
    Load a reference image with skimage and optionally normalize it.

    Parameters
    ----------
    path : str
        Path to the image file.
    as_gray : bool, default=False
        If True, load the image in grayscale.
    normalize : bool, default=True
        If True, normalize pixel values to [0, 1].

    Returns
    -------
    numpy.ndarray
        Loaded image as float64.
    """
    img = io.imread(path, as_gray=as_gray)

    if img is None:
        raise ValueError(f"Image non trouvée à {path}")

    # img = img.astype(np.float32)
    img = img.astype(np.float64)

    if normalize and img.max() > 1.0:
        img /= 255.0

    return img

def display_image(title, image):
    """
    Display an image using Matplotlib.

    Parameters
    ----------
    title : str
        Title of the displayed image.
    image : numpy.ndarray
        Image to display (grayscale or color).
    """
    if len(image.shape) == 3 and image.shape[2] == 3:  # Image couleur
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:  # Image en niveaux de gris
        plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_image(filename, image):
    """
    Save an image using OpenCV.

    The image is saved in the folder `intermediate_steps/` as an 8-bit file.

    Parameters
    ----------
    filename : str
        Name of the output file.
    image : numpy.ndarray
        Image to save (float in [0, 1]).
    """
    output_folder = "intermediate_steps/"
    os.makedirs(output_folder, exist_ok=True)
    cv2.imwrite(os.path.join(output_folder, filename), (image * 255).astype(np.uint8))


def visualize_pyramid(pyramid, title_prefix="Pyramid Level"):
    """
    Display each level of a pyramid using Matplotlib.

    Parameters
    ----------
    pyramid : list of numpy.ndarray
        List of pyramid levels (images).
    title_prefix : str, default="Pyramid Level"
        Prefix for the displayed titles.
    """
    for i, level in enumerate(pyramid):
        display_image(f"{title_prefix} {i}", level)

def to_rgb_array(image):
    """
    Convert an input image to an RGB NumPy array.

    Parameters
    ----------
    image : numpy.ndarray or PIL.Image
        Input image.

    Returns
    -------
    numpy.ndarray
        RGB image as a NumPy array.
    """
    if isinstance(image, np.ndarray):
        return image
    return np.array(image.convert('L'))

def to_grayscale_array_OLD(image):
    """
    Convert an image to grayscale (legacy version).

    Parameters
    ----------
    image : numpy.ndarray or PIL.Image
        Input image.

    Returns
    -------
    numpy.ndarray
        Grayscale image.
    """
    if isinstance(image, np.ndarray):
        # Vérifier si l'image est déjà en niveaux de gris (1 seule dimension couleur)
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            return image
        else:
            return np.mean(image, axis=-1)  # Conversion manuelle en niveaux de gris

    return np.array(image.convert('L'))

def to_grayscale_array_manual(image):
    """
    Convert an image to grayscale using manual weighted RGB conversion.

    Parameters
    ----------
    image : numpy.ndarray or PIL.Image
        Input image.

    Returns
    -------
    numpy.ndarray
        Grayscale image.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            return image
        elif image.ndim == 3 and image.shape[-1] == 3:
            # Conversion pondérée standard RGB → Grayscale
            r, g, b = image[..., 0], image[..., 1], image[..., 2]
            return 0.299 * r + 0.587 * g + 0.114 * b
        else:
            raise ValueError("Format d'image non reconnu (np.ndarray)")
    else:
        # Pour les images PIL ou autres formats compatibles
        return np.array(image.convert('L'))

def to_grayscale_array_cv2(image):
    """
    Convert an image to grayscale using OpenCV.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Grayscale image.

    Raises
    ------
    TypeError
        If the input is not a NumPy array.
    ValueError
        If the image format is unsupported.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            return image
        elif image.ndim == 3 and image.shape[-1] == 3:
            # return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Format d'image non reconnu (np.ndarray)")
    else:
        raise TypeError("L'image doit être un tableau NumPy")
    
def to_grayscale_array_skimage(image):
    """
    Convert a color image to grayscale using skimage.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D or 3D).

    Returns
    -------
    numpy.ndarray
        Grayscale image.

    Raises
    ------
    TypeError
        If the input is not a NumPy array.
    ValueError
        If the image format is unsupported.
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            # Déjà en niveaux de gris
            return np.squeeze(image)  # Au cas où il reste un canal singleton
        elif image.ndim == 3 and image.shape[-1] == 3:
            # Conversion RGB -> Grayscale
            return rgb2gray(image)
        else:
            raise ValueError("Format d'image non reconnu (np.ndarray)")
    else:
        raise TypeError("L'image doit être un tableau NumPy")

def normalize_image(image):
    """
    Normalize an image to [0, 1] if necessary.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Normalized image as float32.
    """
    image = image.astype(np.float32)
    if image.max() > 1.0:
        return image / 255.0
    return image

def safe_float(val):
    """
    Safely convert a value to float.

    Parameters
    ----------
    val : torch.Tensor, numpy.generic, or float
        Input value.

    Returns
    -------
    float
        Converted float value.
    """
    if isinstance(val, torch.Tensor):
        return val.item()
    elif isinstance(val, np.generic):
        return val.item()
    else:
        return float(val)
    
def ensure_grayscale(img):
    """
    Ensure that an image is grayscale.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Grayscale image.
    """
    if img.ndim == 3:
        # Moyenne des canaux ou prise du premier
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
    
def preprocess_images_for_metrics(img1, img2):
    """
    Preprocess two images for metric computation.

    Ensures both are grayscale and resized to the same shape.

    Parameters
    ----------
    img1 : numpy.ndarray
        First image.
    img2 : numpy.ndarray
        Second image.

    Returns
    -------
    tuple of numpy.ndarray
        Preprocessed images (img1, img2).
    """
    img1 = ensure_grayscale(img1)
    img2 = ensure_grayscale(img2)

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    return img1, img2

def ensure_range_255(image):
    """
    Ensure that an image is in the [0, 255] range.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Image scaled to [0, 255] if necessary.
    """
    if image.max() <= 1.0:
        return image * 255.0
    return image

def print_image_info(image):
    """
    Print information about an image.

    Displays type, shape, dtype, value range, and color mode.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.
    """
    # Vérifier le type de l'image
    image_type = type(image)

    # Vérifier la forme de l'image
    image_shape = image.shape

    # Vérifier le type de données des éléments de l'image
    image_dtype = image.dtype

    # Calculer la plage de valeurs de l'image
    min_value = image.min()
    max_value = image.max()

    # Afficher les informations
    print("Type de l'image:", image_type)
    print("Forme de l'image:", image_shape)
    print("Type de données:", image_dtype)
    print("Plage de valeurs: [{}, {}]".format(min_value, max_value))

    # Vérifier si l'image est en niveaux de gris ou en couleur
    if len(image_shape) == 2:
        print("Type d'image: Niveaux de gris (image 2D)")
    elif len(image_shape) == 3 and image_shape[2] == 3:
        print("Type d'image: Couleur (image 3D avec 3 canaux)")
    elif len(image_shape) == 3 and image_shape[2] == 1:
        print("Type d'image: Niveaux de gris (image 3D avec 1 canal)")
    else:
        print("Type d'image: Inconnu ou format spécial")

def save_float64_image_as_uint16(path, img_float64):
    """
    Save a float64 image (normalized [0, 1]) as uint16.

    Parameters
    ----------
    path : str or Path
        Path where the image will be saved.
    img_float64 : numpy.ndarray
        Input image, float64 in [0, 1].
    """
    assert img_float64.dtype == np.float64
    assert img_float64.min() >= 0.0 and img_float64.max() <= 1.0
    cv2.imwrite(path, (img_float64 * 65535).astype(np.uint16))

def to_float32(img):
    """
    Convert an image to float32, normalizing if necessary.

    Parameters
    ----------
    img : numpy.ndarray
        Input image.

    Returns
    -------
    numpy.ndarray
        Image as float32.
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)

def from_grey_to_rgb_array(image):
    """
    Convert a grayscale image to RGB.

    Parameters
    ----------
    image : numpy.ndarray
        Input image (H, W) or (H, W, 3).

    Returns
    -------
    numpy.ndarray
        RGB image (H, W, 3).

    Raises
    ------
    ValueError
        If the image format is unsupported.
    """
    if image.ndim == 2:
        return gray2rgb(image)  # convertit (H, W) ➜ (H, W, 3)
    elif image.ndim == 3 and image.shape[2] == 3:
        return image
    else:
        raise ValueError(f"Format d’image non supporté : {image.shape}")

def get_image_shape(image):
    """
    Return the shape (H, W, C) of a NumPy image.

    Parameters
    ----------
    image : numpy.ndarray
        Input image.

    Returns
    -------
    tuple of int
        Image shape as (height, width, channels).

    Raises
    ------
    TypeError
        If the input is not a NumPy array.
    ValueError
        If the image format is unsupported.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("L'entrée doit être un tableau NumPy.")

    if image.ndim == 2:
        h, w = image.shape
        return (h, w, 1)
    elif image.ndim == 3:
        return image.shape
    else:
        raise ValueError(f"Format d’image non pris en charge : shape={image.shape}")


# ==================================== Gestion des logs et des pauses ==================================== #

def configure_logger(output_dir: str, log_filename: str = "log.txt") -> None:
    """
    Configure the Loguru logger.

    - Writes all logs (DEBUG and above) to a log file.
    - Displays only INFO-level logs and above in the console.

    Parameters
    ----------
    output_dir : str
        Directory where the log file will be stored.
    log_filename : str, default="log.txt"
        Name of the log file.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_filename)

    logger.remove()  # Supprime les handlers existants

    # ➤ Console : uniquement INFO et plus
    logger.add(sys.stdout,
               level="INFO",
               colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

    # ➤ Fichier : tout
    logger.add(log_path,
               level="DEBUG",
               format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")

def wait_if_paused(flag_path="pause.flag", sleep_time=5):
    """
    Pause execution if a pause flag file exists.

    The function checks for the existence of a `pause.flag` file.
    If present, execution is paused until the file is removed.

    Parameters
    ----------
    flag_path : str, default="pause.flag"
        Path to the pause flag file.
    sleep_time : int, default=5
        Time (in seconds) to wait before checking again.
    """
    while os.path.exists(flag_path):
        logger.warning("⏸️ Pause active — suppression de 'pause.flag' requise pour continuer.")
        time.sleep(sleep_time)

# ==================================== Gestion des paramètres de la fusion ==================================== #

def load_parameters_json(json_path: Path) -> dict:
    """
    Load and validate fusion parameters from a JSON file.

    Parameters
    ----------
    json_path : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Dictionary of parameters.

    Raises
    ------
    ValueError
        If required keys are missing from the JSON file.
    """
    with open(json_path, "r") as f:
        params = json.load(f)
    
    # Vérifier que les paramètres ont les bonnes clés
    required_keys = ["facteur_swir", "beta", "level", "apply_gamma", "gamma_value"]
    for key in required_keys:
        if key not in params:
            raise ValueError(f"⚠️ Le paramètre '{key}' est manquant dans {json_path}")
    
    return params


def generate_parameters_json(json_path: Path, mode_fixe: bool = False) -> None:
    """
    Create a JSON file with default or fixed fusion parameters.

    Parameters
    ----------
    json_path : Path
        Path to the JSON file to create or update.
    mode_fixe : bool, default=False
        If True, generate fixed values. Otherwise, generate exploration ranges.
    """
    
    if mode_fixe:
        params_json = {
            "mode_fixe": True,  # Active le mode fixe
            "facteur_swir": 0.7,
            "beta": 1.5,
            "level": 4,
            "apply_gamma": False,
            "gamma_value": 1.0
        }
    else:
        params_json = {
            "mode_fixe": False,  # Active le mode exploration
            "facteur_swir": {"min": 0.0, "max": 1.0, "step": 0.01},
            "beta": {"min": 1.0, "max": 2.0, "step": 0.1},
            "level": {"min": 0, "max": 10, "step": 1},
            "apply_gamma": {"values": [True, False]},
            "gamma_value": {"min": 1.1, "max": 2.0, "step": 0.1}
        }

    json_path = Path(json_path)  # Assurer que c'est un objet `Path`

    if not json_path.exists():  # Créer seulement si le fichier n'existe pas
        with open(json_path, "w") as json_file:
            json.dump(params_json, json_file, indent=4)
        logger.info(f"✅ Fichier {json_path} créé avec {'mode fixe' if mode_fixe else 'mode exploration'} par défaut.")
    else:
        logger.info(f"ℹ️ Fichier {json_path} déjà existant, pas de modification.")



def generate_parameters_json_return(output_dir: Path) -> Path:
    """
    Generate and save a JSON file containing fusion parameters.

    Parameters
    ----------
    output_dir : Path
        Directory where the JSON file will be saved.

    Returns
    -------
    Path
        Path to the generated JSON file.
    """
    params_json = {
        "facteur_swir": {"min": 0.0, "max": 1.0, "step": 0.01},
        "beta": {"min": 1.0, "max": 2.0, "step": 0.1},
        "level": {"min": 0, "max": 10, "step": 1},
        "apply_gamma": {"values": [True, False]},
        "gamma_value": {"min": 1.1, "max": 2.0, "step": 0.1}
    }
    
    # json_path = output_dir / "parameters.json"
    json_path = os.path.join(output_dir, "parameters.json")
    
    with open(json_path, "w") as json_file:
        json.dump(params_json, json_file, indent=4)
    
    return json_path

# ==================================== Gestion des booléen ==================================== #
def str_to_bool_strict(s):
    """
    Convert a string to a strict boolean value.

    Parameters
    ----------
    s : str
        Input string ("true" or "false", case-insensitive).

    Returns
    -------
    bool
        Converted boolean value.

    Raises
    ------
    ValueError
        If the input string is not "true" or "false".
    """
    s = s.strip().lower()
    if s == "true":
        return True
    elif s == "false":
        return False
    else:
        raise ValueError(f"Valeur booléenne invalide : {s}")

# ==================================== Gestion des vérités de terrains ==================================== #

def prepare_ground_truth_list_vLegacy(
    visible_files: List[str],
    run_detection: bool,
    ground_truth_path: Optional[Path],
    ground_truth_extensions: List[str] = ["*.xml"]
) -> List[Optional[str]]:
    """
    Legacy version !!
    Prepare a list of ground truth files aligned with visible images.

    Parameters
    ----------
    visible_files : list of str
        List of visible image file paths.
    run_detection : bool
        Whether detection is enabled.
    ground_truth_path : Path or None
        Path to the ground truth folder, or None if not provided.
    ground_truth_extensions : list of str, default=["\*.xml"]
        Accepted ground truth file extensions.

    Returns
    -------
    list of str or None
        List of ground truth file paths aligned with visible images.
        If unavailable, returns a list of None values.

    Raises
    ------
    ValueError
        If the number of ground truth files does not match the number of visible images.
    """
    ground_truth_files = []

    if not run_detection:
        logger.debug("🚫 Détection désactivée → remplissage de ground_truth_list avec None.")
        return [None] * len(visible_files)

    if ground_truth_path is None:
        logger.warning("⚠️ Aucune vérité de terrain fournie (`ground_truth_path=None`). F1 non calculé.")
        return [None] * len(visible_files)

    for ext in ground_truth_extensions:
        ground_truth_files.extend(glob.glob(os.path.join(ground_truth_path, ext)))

    ground_truth_files = sorted(ground_truth_files)

    if len(ground_truth_files) == 0:
        logger.warning("⚠️ Aucune vérité de terrain trouvée dans le dossier. F1-score non calculé.")
        return [None] * len(visible_files)

    elif len(ground_truth_files) == 1:
        logger.debug(f"🧩 Une seule GT utilisée pour toutes les images : {ground_truth_files[0]}")
        logger.warning("⚠️ Une seule vérité de terrain fournie. Scène fixe supposé, réutilisation de la même vérité de terrain pour toutes les paires d'images.")
        return [ground_truth_files[0]] * len(visible_files)

    elif len(ground_truth_files) == len(visible_files):
        logger.debug(f"📚 {len(ground_truth_files)} vérités de terrain chargées.")
        return ground_truth_files

    else:
        raise ValueError(f"Incohérence : {len(visible_files)} images mais {len(ground_truth_files)} GT.")

def prepare_ground_truth_list(
    visible_files: List[str],
    run_detection: bool,
    ground_truth_path: Optional[Path], # Note: peut maintenant être une liste
    ground_truth_extensions: List[str] = ["*.xml"]
) -> List[Optional[str]]:
    """
    Prepare a list of ground truth files aligned with visible images.
    Supports passing a folder path (auto-scan) or a pre-filtered list of files.
    Parameters
    ----------
    visible_files : list of str
        List of visible image file paths.
    run_detection : bool
        Whether detection is enabled.
    ground_truth_path : Path or None
        Path to the ground truth folder, or None if not provided.
    ground_truth_extensions : list of str, default=["\*.xml"]
        Accepted ground truth file extensions.

    Returns
    -------
    list of str or None
        List of ground truth file paths aligned with visible images.
        If unavailable, returns a list of None values.

    Raises
    ------
    ValueError
        If the number of ground truth files does not match the number of visible images.
    """
    
    ground_truth_files = []

    # 1. Si la détection est désactivée, on renvoie des None
    if not run_detection:
        logger.debug("🚫 Détection désactivée → remplissage de ground_truth_list avec None.")
        return [None] * len(visible_files)

    # 2. NOUVEAU BLOC : Si on reçoit déjà une liste (venant de sql_runner)
    if isinstance(ground_truth_path, list):
        # On convertit tout en string pour être sûr (car sql_runner envoie des Path)
        ground_truth_files = [str(p) for p in ground_truth_path]
        
    # 3. ANCIEN COMPORTEMENT : Si c'est None (et pas une liste)
    elif ground_truth_path is None:
        logger.warning("⚠️ Aucune vérité de terrain fournie (`ground_truth_path=None`). F1 non calculé.")
        return [None] * len(visible_files)

    # 4. ANCIEN COMPORTEMENT : Si c'est un chemin de dossier (Path ou str)
    else:
        for ext in ground_truth_extensions:
            ground_truth_files.extend(glob.glob(os.path.join(ground_truth_path, ext)))

    # 5. Tri et Vérification de cohérence (Commun aux deux méthodes)
    ground_truth_files = sorted(ground_truth_files)

    if len(ground_truth_files) == 0:
        logger.warning("⚠️ Aucune vérité de terrain trouvée. F1-score non calculé.")
        return [None] * len(visible_files)

    elif len(ground_truth_files) == 1:
        logger.debug(f"🧩 Une seule GT utilisée pour toutes les images : {ground_truth_files[0]}")
        logger.warning("⚠️ Une seule vérité de terrain fournie. Réutilisation pour toutes les paires.")
        return [ground_truth_files[0]] * len(visible_files)

    elif len(ground_truth_files) == len(visible_files):
        # C'est le cas idéal qu'on attend avec sql_runner
        logger.debug(f"📚 {len(ground_truth_files)} vérités de terrain alignées.")
        return ground_truth_files

    else:
        # Si sql_runner a mal fait son filtre, ça plantera ici, ce qui est une bonne sécurité
        raise ValueError(f"Incohérence : {len(visible_files)} images mais {len(ground_truth_files)} GT.")