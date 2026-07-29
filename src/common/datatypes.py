"""
Data structures and containers for the VISWIR project.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any#, List


# Résultat d'une fusion d'images
@dataclass
class ProcessResult:
    """
    Container for the result of a VIS-SWIR image fusion process.

    Attributes
    ----------
    visible_path : Path
        Path to the visible image used in the fusion.
    swir_path : Path
        Path to the SWIR image used in the fusion.
    ground_truth_path : Path, optional
        Path to the ground truth image, if available.
    params : dict of str to Any
        Dictionary of fusion parameters (e.g., {"facteur_swir": 0.89, "beta": 1.07}).
    metrics_fusion : dict of str to float, optional
        Evaluation metrics computed on the fused image.
    metrics_visible : dict of str to float, optional
        Evaluation metrics computed on the visible image.
    metrics_swir : dict of str to float, optional
        Evaluation metrics computed on the SWIR image.
    error : str, optional
        Error message if the fusion process failed.
    """
    
    visible_path: Path
    swir_path: Path
    ground_truth_path: Optional[Path]
    params: Dict[str, Any]                # ex: {"facteur_swir": 0.89, "beta": 1.07, ...}
    metrics_fusion: Optional[Dict[str, float]] = None
    metrics_visible: Optional[Dict[str, float]] = None
    metrics_swir: Optional[Dict[str, float]] = None
    error: Optional[str] = None           # message d’erreur éventuel


# Configuration générique (chargée depuis YAML/JSON) # Non utilisé
@dataclass
class FusionConfig:
    """
    Generic configuration for the VISWIR project (loaded from YAML/JSON).

    Attributes
    ----------
    visible_folder : Path
        Path to the folder containing visible images.
    swir_folder : Path
        Path to the folder containing SWIR images.
    output_folder : Path
        Path to the folder where fused images will be saved.
    ref_image_path : Path, optional
        Path to a reference image, if required.
    ground_truth_path : Path, optional
        Path to the ground truth image, if available.
    mode : str, default="fixed"
        Execution mode. Possible values: "fixed", "sql", "optuna".
    run_detection : bool, default=False
        Whether to run object detection after fusion.
    """

    visible_folder: Path
    swir_folder: Path
    output_folder: Path
    ref_image_path: Optional[Path] = None
    ground_truth_path: Optional[Path] = None
    mode: str = "fixed"                   # "fixed", "sql", "optuna"
    # n_trials: int = 100                   # utilisé si mode=optuna
    run_detection: bool = False


# Configuration spécifique à Optuna (chargée depuis optuna_config.yaml) # Non utilisé
@dataclass
class OptunaConfig:
    """
    Configuration for Optuna optimization (loaded from optuna_config.yaml).

    Attributes
    ----------
    n_trials : int
        Number of optimization trials.
    n_jobs : int
        Number of parallel jobs to run.
    sampler : str, default="TPE"
        Sampling strategy used by Optuna.
    pruner : str, default="MedianPruner"
        Pruning strategy used by Optuna.
    timeout : int, optional
        Maximum optimization time in seconds.
    """

    n_trials: int
    n_jobs: int
    sampler: str = "TPE"
    pruner: str = "MedianPruner"
    timeout: Optional[int] = None         # en secondes


# Tâche unitaire (utile pour batch_runner / task_manager)
@dataclass
class FusionTask:
    """
    Representation of a single fusion task (used by batch_runner / task_manager).

    Attributes
    ----------
    visible_path : Path
        Path to the visible image.
    swir_path : Path
        Path to the SWIR image.
    ref_image_path : Path, optional
        Path to the reference image, if available.
    ground_truth_path : Path, optional
        Path to the ground truth image, if available.
    params : dict of str to Any
        Dictionary of fusion parameters for this task.
    save_output : bool, default=True
        Whether to save the fused output image.
    run_detection : bool, default=False
        Whether to run object detection after fusion.
    output_dir : Path, optional
        Directory where results should be saved.
    """

    visible_path: Path
    swir_path: Path
    ref_image_path: Optional[Path]
    ground_truth_path: Optional[Path]
    params: Dict[str, Any]
    save_output: bool = True
    run_detection: bool = False
    output_dir: Optional[Path] = None

@dataclass
class OptunaResult:
    """
    Result container optimized for Optuna trials.

    Attributes
    ----------
    visible_path : Path
        Path to the visible image.
    swir_path : Path
        Path to the SWIR image.
    params : dict of str to Any
        Fusion parameters used in this trial.
    metrics_fusion : dict of str to float, optional
        Metrics computed on the fused image (NR-IQA or detection).
    error : str, optional
        Error message if the fusion process failed.
    """

    visible_path: Path
    swir_path: Path
    params: Dict[str, Any]
    metrics_fusion: Optional[Dict[str, float]] = None
    error: Optional[str] = None
