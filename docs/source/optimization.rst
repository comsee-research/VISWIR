optimization package
====================

The ``optimization`` package provides tools for **hyperparameter search,
objective evaluation, and visualization** of the VISWIR fusion pipeline.
It integrates with Optuna for automated optimization and includes custom
samplers and plotting utilities.

Submodules
----------

optimization.objective module
-----------------------------
Definition of objective functions used to evaluate fusion quality and
guide optimization.

.. automodule:: optimization.objective
   :members:
   :undoc-members:
   :show-inheritance:

optimization.optuna\_runner module
----------------------------------
Orchestration of fusion parameter optimisation with Optuna framework
to run optimization studies and manage trials (NR-IQA or detection).

.. automodule:: optimization.optuna_runner
   :members:
   :undoc-members:
   :show-inheritance:

optimization.optuna\_wrapper module
----------------------------------
Wrapper for Optuna, managing image fusion and metric calculation,
as well as detection if enabled.

.. automodule:: optimization.optuna_runner
   :members:
   :undoc-members:
   :show-inheritance:

optimization.samplers module
----------------------------
Custom Optuna samplers for exploring the hyperparameter space more
efficiently.

.. automodule:: optimization.samplers
   :members:
   :undoc-members:
   :show-inheritance:

optimization.visualization module
---------------------------------
Visualization utilities for analyzing optimization results, including
loss curves and parameter importance plots.

.. automodule:: optimization.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------
The top-level ``optimization`` module re-exports selected functions and
classes from its submodules for convenience.

.. automodule:: optimization
   :members:
   :undoc-members:
   :show-inheritance:
