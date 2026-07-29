"""
Task manager for batch processing task coordination.
"""

# src/processing/task_manager.py

from pathlib import Path
import numpy as np
from typing import List, Optional

from common.logger import logger
from common.datatypes import FusionTask
from fusion.utils import prepare_ground_truth_list


def generate_tasks_in_memory(
    visible_files: List[Path],
    swir_files: List[Path],
    ref_image_path: Optional[Path],
    params: dict,
    run_detection: bool,
    ground_truth_path: Optional[Path],
    save_output: bool,
    output_dir: Path
) -> List[FusionTask]:
    """
    Generate a list of FusionTask objects in memory (fixed mode or exploration mode).

    This function creates tasks for each pair of visible and SWIR images,
    either with a single fixed set of parameters or by exploring all
    combinations of parameter ranges.

    Parameters
    ----------
    visible_files : list of Path
        List of visible image file paths.
    swir_files : list of Path
        List of SWIR image file paths.
    ref_image_path : Path or None
        Path to the reference image (optional).
    params : dict
        Fusion parameters. Can be:
        - Fixed mode: contains single values for each parameter.
        - Exploration mode: contains ranges (min, max, step) or lists of values.
    run_detection : bool
        Whether to enable YOLO detection.
    ground_truth_path : Path or None
        Path to ground truth annotations (optional).
    save_output : bool
        Whether to save intermediate results.
    output_dir : Path
        Directory where outputs will be saved.

    Returns
    -------
    list of FusionTask
        List of tasks ready for execution.

    Notes
    -----
    - In fixed mode, only one task per image pair is generated.
    - In exploration mode, all parameter combinations are expanded into tasks.
    - Ground truth files are aligned with visible images using
      `prepare_ground_truth_list`.
    """

    # Préparer la liste des ground truths
    ground_truth_list = prepare_ground_truth_list(
        visible_files=visible_files,
        run_detection=run_detection,
        ground_truth_path=ground_truth_path
    )

    logger.debug(f"✅ ground_truth_list générée : {len(ground_truth_list)} éléments")

    tasks: List[FusionTask] = []

    for visible_path, swir_path, gt_path in zip(visible_files, swir_files, ground_truth_list):

        if params.get("mode_fixe", False):
            # Mode fixe : un seul ensemble de paramètres
            task = FusionTask(
                visible_path=visible_path,
                swir_path=swir_path,
                ref_image_path=ref_image_path,
                ground_truth_path=gt_path,
                params={
                    "facteur_swir": params["facteur_swir"],
                    "beta": params["beta"],
                    "level": params["level"],
                    "apply_gamma": params["apply_gamma"],
                    "gamma_value": params["gamma_value"]
                },
                save_output=save_output,
                run_detection=run_detection,
                output_dir=output_dir
            )
            tasks.append(task)

        else:
            # Mode exploration : toutes les combinaisons
            for facteur_swir in np.arange(params["facteur_swir"]["min"], params["facteur_swir"]["max"] + 1e-8, params["facteur_swir"]["step"]):
                for beta in np.arange(params["beta"]["min"], params["beta"]["max"] + 1e-8, params["beta"]["step"]):
                    for level in range(params["level"]["min"], params["level"]["max"] + 1):
                        for apply_gamma in params["apply_gamma"]["values"]:
                            gamma_values = (
                                np.arange(params["gamma_value"]["min"], params["gamma_value"]["max"] + 1e-8, params["gamma_value"]["step"])
                                if apply_gamma else [1.0]
                            )
                            for gamma_value in gamma_values:
                                task = FusionTask(
                                    visible_path=visible_path,
                                    swir_path=swir_path,
                                    ref_image_path=ref_image_path,
                                    ground_truth_path=gt_path,
                                    params={
                                        "facteur_swir": float(facteur_swir),
                                        "beta": float(beta),
                                        "level": int(level),
                                        "apply_gamma": bool(apply_gamma),
                                        "gamma_value": float(gamma_value)
                                    },
                                    save_output=save_output,
                                    run_detection=run_detection,
                                    output_dir=output_dir
                                )
                                tasks.append(task)

    logger.info(f"🧮 {len(tasks)} tâches générées en mémoire")
    return tasks


def batchify_tasks(tasks: List[FusionTask], batch_size: int):
    """
    Split a list of tasks into smaller batches of a given size.

    Parameters
    ----------
    tasks : list of FusionTask
        List of tasks to split.
    batch_size : int
        Number of tasks per batch.

    Yields
    ------
    list of FusionTask
        A batch of tasks.

    Examples
    --------
    >>> tasks = [FusionTask(...), FusionTask(...), FusionTask(...)]
    >>> for batch in batchify_tasks(tasks, batch_size=2):
    ...     print(len(batch))
    2
    1
    """
    for i in range(0, len(tasks), batch_size):
        yield tasks[i:i + batch_size]
