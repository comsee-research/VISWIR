"""
Orchestration of Optuna hyperparameter optimization for VISWIR.
"""

# =============================================================================
# FILENAME:       optuna_runner.py
# DESCRIPTION:    Orchestration de l’optimisation des paramètres de fusion
#                 avec Optuna (NR-IQA ou détection).
# =============================================================================

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional

import optuna

from common.logger import logger
from common.config_loader import load_all_configs
from fusion.utils import prepare_ground_truth_list
from optimization.objective import objective, objective_detection
from optimization.samplers import get_sampler, get_pruner
from optimization.visualization import visualize_study


# ============================================================
# Fonctions utilitaires
# ============================================================

def tchebycheff_distance(values, ideal, weights):
    """
    Compute the Tchebycheff distance between a solution and the ideal point.

    Parameters
    ----------
    values : list of float
        Values of the current solution.
    ideal : list of float
        Ideal target values.
    weights : list of float
        Weights for each objective.

    Returns
    -------
    float
        Tchebycheff distance.
    """
    return max([weights[i] * abs(values[i] - ideal[i]) for i in range(len(values))])


def normalize(values, mins, maxs):
    """
    Normalize values to [0, 1] given min and max bounds.

    Parameters
    ----------
    values : list of float
        Values to normalize.
    mins : list of float
        Minimum values for each dimension.
    maxs : list of float
        Maximum values for each dimension.

    Returns
    -------
    list of float
        Normalized values in [0, 1].
    """
    return [(v - mn) / (mx - mn) if mx != mn else 0.0
            for v, mn, mx in zip(values, mins, maxs)]


def clear_cache_callback(study, trial):
    """
    Callback to clear memory after each Optuna trial.

    Parameters
    ----------
    study : optuna.study.Study
        The current Optuna study.
    trial : optuna.trial.Trial
        The completed trial.

    Notes
    -----
    - Calls Python garbage collector.
    - Can be extended to clear GPU cache if needed.
    """
    import gc
    gc.collect()
    # torch.cuda.empty_cache() si GPU


# ============================================================
# Orchestration Optuna
# ============================================================

def optimize_parameters_for_group_v2(
    visible_files, swir_files, ref_image_path, output_dir,
    n_trials: int = 100, run_detection: bool = False, ground_truth_path: Optional[Path] = None,
    sampler_name: str = "TPE", pruner_name: str = "MedianPruner", n_jobs: int = 4,
    storage: Optional[str] = None
):
    """
    Optimize fusion parameters with Optuna (NR-IQA or detection).

    Parameters
    ----------
    visible_files : list of Path
        List of visible image file paths.
    swir_files : list of Path
        List of SWIR image file paths.
    ref_image_path : Path or None
        Path to the reference image (optional).
    output_dir : Path
        Directory where results and logs will be saved.
    n_trials : int, default=100
        Number of optimization trials.
    run_detection : bool, default=False
        Whether to optimize based on detection (F1-score).
    ground_truth_path : Path, optional
        Path to ground truth annotations (used if detection is enabled).
    sampler_name : str, default="TPE"
        Name of the Optuna sampler to use.
    pruner_name : str, default="MedianPruner"
        Name of the Optuna pruner to use.
    n_jobs : int, default=4
        Number of parallel jobs.
    storage : str, optional
        Storage backend (SQLite or external database).

    Returns
    -------
    tuple
        - study : optuna.study.Study
            The completed Optuna study.
        - best_params : dict
            Dictionary of best parameters found.
        - best_values : tuple
            Best objective values.

    Raises
    ------
    ValueError
        If the number of visible and SWIR images does not match.
    """

    # Si storage est défini dans la config, on l’utilise tel quel
    if storage:
        storage_url = storage
    else:
        storage_path = os.path.join(output_dir, "optuna_study.sqlite3")
        storage_url = f"sqlite:///{storage_path}"

    study_name = "viswir_f1_optimization_study" if run_detection else "viswir_quality_optimization_study"

    sampler = get_sampler(sampler_name)
    pruner = get_pruner(pruner_name)

    if storage and "sqlite" not in storage:
        logger.info(f"🔗 Utilisation d’un backend Optuna externe : {storage_url}")
    else:
        logger.info(f"💾 Utilisation d’un fichier SQLite local : {storage_url}")

    if run_detection:
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            storage=storage_url,
            study_name=study_name,
            pruner=pruner,
            load_if_exists=True   # <--- clé magique
        )
    else:
        study = optuna.create_study(
            directions=["maximize", "minimize", "minimize", "minimize"],
            sampler=sampler,
            storage=storage_url,
            study_name=study_name,
            pruner=pruner,
            load_if_exists=True   # <--- clé magique
        )

    existing_trials = len(study.trials)
    n_trials_to_run = max(0, n_trials - existing_trials)

    if run_detection:
        ground_truth_list = prepare_ground_truth_list(
            visible_files=visible_files,
            run_detection=run_detection,
            ground_truth_path=ground_truth_path
        )
        logger.info("⚠️ Optimisation basée sur la détection (F1-score).")
        study.optimize(
            lambda trial: objective_detection(trial, visible_files, swir_files, ref_image_path, run_detection, ground_truth_list),
            n_trials=n_trials_to_run,
            n_jobs=n_jobs,
            gc_after_trial=True,
            callbacks=[clear_cache_callback]
        )
    else:
        logger.info("⚠️ Optimisation basée sur les métriques NR-IQA.")
        study.optimize(
            lambda trial: objective(trial, visible_files, swir_files, ref_image_path, run_detection=False),
            n_trials=n_trials_to_run,
            n_jobs=n_jobs,
            gc_after_trial=True
        )

    best_trials = study.best_trials
    if not run_detection:
        all_values = np.array([t.values for t in best_trials])
        mins, maxs = np.min(all_values, axis=0), np.max(all_values, axis=0)
        ideal = [maxs[0], mins[1], mins[2], mins[3]] # Max EN, min other
        ideal_norm = normalize(ideal, mins, maxs)
        weights = [0.25, 0.25, 0.25, 0.25]

        best_trial = min(
            best_trials,
            key=lambda t: tchebycheff_distance(normalize(t.values, mins, maxs), ideal_norm, weights)
        )
    else:
        if not best_trials:
            logger.error("❌ Aucun essai Optuna valide n’a été trouvé (tous échoués ou prunés).")
            return None, None, None   # ou lever une exception plus claire ?
        
        best_trial = best_trials[0]

    # visualize_study(study, best_trial, Path(output_dir), run_detection=run_detection)

    return study, best_trial.params, best_trial.values


# ============================================================
# Entrée principale
# ============================================================

def process_folder_optuna(
    visible_folder: Path, swir_folder: Path, output_dir: Path,
    ref_image_path: Optional[Path] = None, n_trials: int = 100,
    run_detection: bool = False, ground_truth_path: Optional[Path] = None
) -> None:
    """
    Complete optimization pipeline with Optuna.

    Parameters
    ----------
    visible_folder : Path
        Folder containing visible images.
    swir_folder : Path
        Folder containing SWIR images.
    output_dir : Path
        Directory where results will be saved.
    ref_image_path : Path, optional
        Path to the reference image.
    n_trials : int, default=100
        Number of optimization trials.
    run_detection : bool, default=False
        Whether to optimize based on detection (F1-score).
    ground_truth_path : Path, optional
        Path to ground truth annotations.

    Raises
    ------
    ValueError
        If the number of visible and SWIR images does not match.

    Notes
    -----
    - Saves the best parameters in `best_parameters.json`.
    - Creates or reuses an Optuna study with SQLite or external storage.
    """

    logger.info("🚀 Démarrage de l'optimisation avec Optuna...")

    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    visible_files = sorted([f for ext in image_extensions for f in Path(visible_folder).glob(ext)])
    swir_files = sorted([f for ext in image_extensions for f in Path(swir_folder).glob(ext)])

    if len(visible_files) != len(swir_files):
        raise ValueError("❌ Nombre d'images Visible et SWIR non correspondant.")

    # Charger config optuna
    configs = load_all_configs()
    optuna_cfg = configs["optuna"]

    study, best_params, best_values = optimize_parameters_for_group_v2(
        visible_files, swir_files, ref_image_path, output_dir,
        n_trials=optuna_cfg.get("n_trials", n_trials),
        run_detection=run_detection,
        ground_truth_path=ground_truth_path,
        sampler_name=optuna_cfg.get("sampler", "TPE"),
        pruner_name=optuna_cfg.get("pruner", "MedianPruner"),
        n_jobs=optuna_cfg.get("n_jobs", 4),
        storage=optuna_cfg.get("storage")
    )

    logger.info(f"⚙️ Best parameters found: {best_params}")
    logger.info(f"✅ Best metrics value: {best_values}")

    # Sauvegarde JSON
    with open(output_dir / "best_parameters.json", "w") as json_file:
        json.dump(best_params, json_file, indent=4)

    logger.info("💾 Optimization completed.")
