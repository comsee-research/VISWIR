"""
config_loader.py
----------------
Module centralisé pour charger et valider les fichiers de configuration
(YAML ou JSON) utilisés dans VISWIR_vQuasar.

Ce fichier est conçu pour être extensible :
- On peux ajouter de nouveaux fichiers de config (ex: logging_config.yaml).
- On peux définir des clés obligatoires par contexte (fixed, sql, optuna...).
- On peux fusionner plusieurs fichiers en un seul dictionnaire global.

Auteur : Alexandre Riffard
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict


# ============================================================
# Exceptions personnalisées
# ============================================================

class ConfigError(Exception):
    """Custom exception for configuration-related errors."""


# ============================================================
# Fonctions utilitaires
# ============================================================

def load_config(path: str | Path, defaults: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Load a YAML or JSON configuration file and merge it with default values.

    Parameters
    ----------
    path : str or Path
        Path to the configuration file (.yaml/.yml or .json).
    defaults : dict, optional
        Dictionary of default values to merge with the loaded configuration.

    Returns
    -------
    dict
        Final configuration dictionary (file values + defaults).

    Raises
    ------
    ConfigError
        If the file does not exist or has an unsupported format.
    """

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"❌ Fichier de configuration introuvable : {path}")

    # Lecture du fichier
    if path.suffix in [".yaml", ".yml"]:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        raise ConfigError(f"❌ Format non supporté : {path.suffix}")

    # Fusion avec les valeurs par défaut
    if defaults:
        merged = defaults.copy()
        merged.update(config)
        return merged
    return config


def validate_config(config: Dict[str, Any], required_keys: list[str], context: str = "") -> None:
    """
    Validate that required keys are present in the configuration.

    Parameters
    ----------
    config : dict
        Loaded configuration dictionary.
    required_keys : list of str
        List of mandatory keys that must be present.
    context : str, optional
        Context name (e.g., "optuna", "sql") used in error messages.

    Raises
    ------
    ConfigError
        If one or more required keys are missing.
    """

    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ConfigError(
            f"❌ Clés manquantes dans la configuration {context or ''} : {missing}"
        )


# ============================================================
# Chargement global (fusion de plusieurs fichiers)
# ============================================================

def load_all_configs(
    base_path: str | Path = "../config/config_viswir.yaml",
    params_path: str | Path = "../config/parameters.json",
    yolo_path: str | Path = "../config/yolo_config.json",
    optuna_path: str | Path = "../config/optuna_config.yaml",
    logging_path: str | Path = "../config/logging_config.yaml"
) -> Dict[str, Dict[str, Any]]:
    """
    Load and validate all configuration files required for the VISWIR_vQuasar project.

    Parameters
    ----------
    base_path : str or Path, optional
        General configuration (paths, mode, etc.).
    params_path : str or Path, optional
        Fusion parameters (fixed values or ranges).
    yolo_path : str or Path, optional
        YOLO configuration (model, thresholds, classes).
    optuna_path : str or Path, optional
        Optuna configuration (HPC optimization).
    logging_path : str or Path, optional
        Logger configuration.

    Returns
    -------
    dict of dict
        Structured dictionary containing all loaded configurations.
    """

    configs = {}

    # Config générale
    configs["base"] = load_config(base_path)
    validate_config(
        configs["base"],
        ["visible_folder", "swir_folder", "output_folder", "mode", "run_detection", "save_output"],
        context="base"
    )

    # Paramètres de fusion
    configs["params"] = load_config(params_path)
    validate_config(
        configs["params"],
        ["facteur_swir", "beta", "level", "apply_gamma", "gamma_value"],
        context="parameters"
    )

    # YOLO
    configs["yolo"] = load_config(yolo_path)
    validate_config(
        configs["yolo"],
        ["model_path", "confidence_threshold", "iou_threshold", "device"],
        context="yolo"
    )

    # Optuna (optionnel → valeurs par défaut si absent)
    try:
        configs["optuna"] = load_config(optuna_path, defaults={
            "n_trials": 100,
            "n_jobs": 4,
            "sampler": "TPE",
            "pruner": "MedianPruner",
            "timeout": None,
            "storage": None
        })
    except ConfigError:
        configs["optuna"] = {
            "n_trials": 100,
            "n_jobs": 4,
            "sampler": "TPE",
            "pruner": "MedianPruner",
            "timeout": None,
            "storage": None
        }

    # Logging (optionnel)
    try:
        configs["logging"] = load_config(logging_path, defaults={
            "level": "INFO",
            "log_to_file": True,
            "log_file": "../logs/viswir.log",
            "log_to_console": True,
            "format": "[%(asctime)s] [%(levelname)s] %(message)s"
        })
    except ConfigError:
        configs["logging"] = {
            "level": "INFO",
            "log_to_file": True,
            "log_file": "../logs/viswir.log",
            "log_to_console": True,
            "format": "[%(asctime)s] [%(levelname)s] %(message)s"
        }

    return configs

# ============================================================
# Chargement spécial pour Optuna (espace de recherche des paramètres)
# ============================================================

def load_optuna_search_space(path: str | Path = "../config/optuna_search_space.yaml") -> Dict[str, Any]:
    """
    Load the YAML file defining the Optuna search space.

    Parameters
    ----------
    path : str or Path, optional
        Path to the YAML file describing the search space.

    Returns
    -------
    dict
        Dictionary describing the Optuna search space.
    """
    
    return load_config(path)
