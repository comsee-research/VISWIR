"""
Sampler and pruner configurations for Optuna.
"""

# src/optimization/samplers.py
# ============================================================
# FILENAME:       samplers.py
# DESCRIPTION:    Fournit des fonctions utilitaires pour
#                 récupérer un sampler et un pruner Optuna
#                 en fonction d’un nom (défini dans la config).
# ============================================================

import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, GridSampler
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, NopPruner

from common.logger import logger


# ============================================================
# Samplers
# ============================================================

def get_sampler(name: str, **kwargs) -> optuna.samplers.BaseSampler:
    """
    Return an Optuna sampler based on the provided name.

    Parameters
    ----------
    name : str
        Name of the sampler. Supported values:
        - "TPE" : Tree-structured Parzen Estimator sampler.
        - "Random" : Random search sampler.
        - "CMAES" : Covariance Matrix Adaptation Evolution Strategy sampler.
        - "Grid" : Grid search sampler (requires explicit search space).
    **kwargs : dict
        Additional parameters passed to the sampler constructor.

    Returns
    -------
    optuna.samplers.BaseSampler
        The corresponding Optuna sampler instance.

    Raises
    ------
    ValueError
        If "Grid" is selected but no `search_space` is provided.
    """
    name = name.lower()
    if name == "tpe":
        return TPESampler(**kwargs)
    elif name == "random":
        return RandomSampler(**kwargs)
    elif name == "cmaes":
        return CmaEsSampler(**kwargs)
    elif name == "grid":
        # ⚠️ GridSampler nécessite un search_space explicite
        if "search_space" not in kwargs:
            raise ValueError("GridSampler requiert un argument 'search_space'.")
        return GridSampler(kwargs["search_space"])
    else:
        logger.warning(f"⚠️ Sampler inconnu '{name}', fallback sur TPE.")
        return TPESampler(**kwargs)


# ============================================================
# Pruners
# ============================================================

def get_pruner(name: str, **kwargs) -> optuna.pruners.BasePruner:
    """
    Return an Optuna pruner based on the provided name.

    Parameters
    ----------
    name : str
        Name of the pruner. Supported values:
        - "MedianPruner" : Stops unpromising trials using median of past trials.
        - "SHA" or "SuccessiveHalving" : Successive Halving pruner.
        - "None" or "Nop" : No pruning applied.
    **kwargs : dict
        Additional parameters passed to the pruner constructor.

    Returns
    -------
    optuna.pruners.BasePruner
        The corresponding Optuna pruner instance.

    Notes
    -----
    - If an unknown pruner name is provided, defaults to `MedianPruner`.
    """
    name = name.lower()
    if name == "medianpruner":
        return MedianPruner(**kwargs)
    elif name in ["sha", "successivehalving"]:
        return SuccessiveHalvingPruner(**kwargs)
    elif name in ["none", "nop"]:
        return NopPruner()
    else:
        logger.warning(f"⚠️ Pruner inconnu '{name}', fallback sur MedianPruner.")
        return MedianPruner(**kwargs)
