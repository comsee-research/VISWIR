"""
Optuna objective functions for hyperparameter optimization.
"""

# =============================================================================
# FILENAME:       objective.py
# DESCRIPTION:    Définitions des fonctions objectif pour Optuna
#                 - NR-IQA (entropy, BRISQUE, NIQE, PIQE)
#                 - Détection (F1-score)
# =============================================================================

import random
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import optuna

from optimization.optuna_wrapper import process_image_wrapper_optuna
from common.logger import logger
from common.config_loader import load_optuna_search_space, load_all_configs
from common.datatypes import FusionTask

# Chargement du search space une seule fois
SEARCH_SPACE = load_optuna_search_space()

# Charger la config optuna
configs = load_all_configs()
OPTUNA_CFG = configs["optuna"]

def suggest_params(trial):
    """
    Generate a dictionary of parameters from the Optuna search space.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Current Optuna trial object.

    Returns
    -------
    dict
        Dictionary of suggested parameters with their values.

    Raises
    ------
    ValueError
        If the parameter type in the search space is unsupported.
    """
    params = {}
    for name, spec in SEARCH_SPACE.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["min"], spec["max"], step=spec.get("step"))
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["min"], spec["max"], step=spec.get("step"))
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["values"])
        else:
            raise ValueError(f"❌ Type de paramètre non supporté : {spec['type']}")
    return params

# ============================================================
# Objectif NR-IQA
# ============================================================

def objective(trial, visible_files, swir_files, ref_image_path: Optional[str], run_detection: bool = False):
    """
    Optuna objective function based on NR-IQA metrics.

    This objective evaluates fused images using no-reference image quality
    assessment metrics (Entropy, BRISQUE, NIQE, PIQE).

    Parameters
    ----------
    trial : optuna.trial.Trial
        Current Optuna trial object.
    visible_files : list of str
        List of visible image file paths.
    swir_files : list of str
        List of SWIR image file paths.
    ref_image_path : str, optional
        Path to the reference image (not used in NR-IQA).
    run_detection : bool, default=False
        Whether to run detection in addition to NR-IQA.

    Returns
    -------
    tuple of float
        A tuple containing:
        - entropy : float
        - brisque : float
        - niqe : float
        - piqe : float

    Raises
    ------
    optuna.TrialPruned
        If no valid metrics are computed for the trial.
    """
    # 1. Espace de recherche
    # facteur_swir = trial.suggest_float("facteur_swir", 0.0, 1.0, step=0.01)
    # beta = trial.suggest_float("beta", 1.0, 5.0, step=0.01)
    # level = trial.suggest_int("level", 1, 6, step=1)
    # apply_gamma = trial.suggest_categorical("apply_gamma", [True])
    # gamma_value = trial.suggest_float("gamma_value", 0.01, 4.0, step=0.01)
    params = suggest_params(trial)

    facteur_swir = params["facteur_swir"]
    beta = params["beta"]
    level = params["level"]
    apply_gamma = params["apply_gamma"]
    gamma_value = params["gamma_value"]

    # 2. Échantillonnage (max 50 images)
    # sample_size = min(50, len(visible_files))
    sample_size = min(OPTUNA_CFG.get("sample_size", 50), len(visible_files))
    sampled_data = random.sample(list(zip(visible_files, swir_files)), sample_size)

    # 3. Préparer les tâches
    task_batch = [
        FusionTask(
            visible_path=vf,
            swir_path=sf,
            ref_image_path=ref_image_path,
            ground_truth_path=None,   # pas de GT en NR-IQA
            params={
                "facteur_swir": facteur_swir,
                "beta": beta,
                "level": level,
                "apply_gamma": apply_gamma,
                "gamma_value": gamma_value
            },
            save_output=False,        # on force à False pour Optuna, pas de sauvegarde autorisée
            run_detection=run_detection,
            output_dir=None           # inutile en optimisation
        )
        for vf, sf in sampled_data
    ]

    # 4. Exécution parallèle
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_image_wrapper_optuna, task_batch))

    # 5. Extraire les métriques fusionnées
    metrics_list = [r.metrics_fusion for r in results if r.metrics_fusion is not None]
    if not metrics_list:
        raise optuna.TrialPruned()

    avg_metrics = {
        key: sum(m[key] for m in metrics_list) / len(metrics_list)
        for key in metrics_list[0].keys()
    }

    # 6. Retourner les objectifs
    return (
        avg_metrics.get("entropy", 0),
        avg_metrics.get("brisque", 0),
        avg_metrics.get("niqe", 0),
        avg_metrics.get("piqe", 0),
    )


# ============================================================
# Objectif Détection
# ============================================================

def objective_detection(trial, visible_files, swir_files, ref_image_path: Optional[str],
                        run_detection: bool, ground_truth_list: Optional[list]):
    """
    Optuna objective function based on detection metrics (F1-score).

    This objective evaluates fused images by running detection and computing
    the average F1-score across a sampled subset of images.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Current Optuna trial object.
    visible_files : list of str
        List of visible image file paths.
    swir_files : list of str
        List of SWIR image file paths.
    ref_image_path : str, optional
        Path to the reference image.
    run_detection : bool
        Whether detection is enabled.
    ground_truth_list : list of str or None
        List of ground truth annotation file paths aligned with visible images,
        or None if unavailable.

    Returns
    -------
    float
        Average F1-score across the sampled dataset.

    Raises
    ------
    optuna.TrialPruned
        If no valid metrics are computed for the trial.
    """
    # 1. Espace de recherche
    # facteur_swir = trial.suggest_float("facteur_swir", 0.0, 1.0, step=0.01)
    # beta = trial.suggest_float("beta", 1.0, 5.0, step=0.01)
    # level = trial.suggest_int("level", 1, 6, step=1)
    # apply_gamma = trial.suggest_categorical("apply_gamma", [True])
    # gamma_value = trial.suggest_float("gamma_value", 0.01, 4.0, step=0.01)
    params = suggest_params(trial)
    
    facteur_swir = params["facteur_swir"]
    beta = params["beta"]
    level = params["level"]
    apply_gamma = params["apply_gamma"]
    gamma_value = params["gamma_value"]

    # 2. Échantillonnage (max 50 images)
    # sample_size = min(50, len(visible_files))
    sample_size = min(OPTUNA_CFG.get("sample_size", 50), len(visible_files))
    sampled_data = random.sample(list(zip(visible_files, swir_files)), sample_size)

    # 3. Préparer les tâches
    task_batch = [
        FusionTask(
            visible_path=vf,
            swir_path=sf,
            ref_image_path=ref_image_path,
            ground_truth_path=ground_truth_list[visible_files.index(vf)] if ground_truth_list else None,
            params={
                "facteur_swir": facteur_swir,
                "beta": beta,
                "level": level,
                "apply_gamma": apply_gamma,
                "gamma_value": gamma_value
            },
            save_output=False,        # pas de sauvegarde autorisée avec Optuna
            run_detection=run_detection,
            output_dir=None
        )
        for vf, sf in sampled_data
    ]

    # 4. Exécution parallèle
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(process_image_wrapper_optuna, task_batch))

    # 5. Extraire les métriques fusionnées
    metrics_list = [r.metrics_fusion for r in results if r.metrics_fusion is not None]
    if not metrics_list:
        raise optuna.TrialPruned()

    # 6. Calcul du F1 moyen
    f1_scores = [m.get("f1_score", 0.0) for m in metrics_list]
    avg_f1 = sum(f1_scores) / len(f1_scores)

    return avg_f1
