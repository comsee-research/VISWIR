"""
Optimization package
====================

This package provides modules and utilities for parameter optimization
within the VISWIR project. It is primarily focused on automating the
search for optimal fusion parameters using different strategies.

Submodules
----------
- optuna_runner : integration with Optuna for hyperparameter search.
- objective     : optimisation target management (NR-IQA or F1 score if detection is enabled).
- sampler       : sampler for Optuna.
- visualization : module for the result visualization if the environnement authorise it.

Features
--------
- Centralized entry points for launching optimization experiments.
- Logging and result tracking integrated with the VISWIR framework.
- Extensible design to add new optimization backends.

Notes
-----
This package is intended for research and experimentation. 
It may consume significant computational resources depending 
on the chosen optimization strategy.
"""

__all__ = []
