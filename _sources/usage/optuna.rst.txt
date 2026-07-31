Optuna Optimization
===================

VISWIR integrates `Optuna <https://optuna.org/>`_ to automatically optimize
fusion parameters. This page explains how to configure, run, and analyze
Optuna experiments.

.. contents::
   :local:
   :depth: 2

Configuration
-------------

Three configuration files are required:

- ``config_viswir.yaml`` : general control parameters, with ``mode: "optuna"``.
- ``optuna_config.yaml`` : optimization settings (trials, jobs, sampler, pruner).
- ``optuna_search_space.yaml`` : definition of the hyperparameter ranges.

Example ``optuna_config.yaml``:

.. code-block:: yaml

   # Configuration spécifique à Optuna
   n_trials: 2
   n_jobs: 1
   sampler: "TPE"          # or "Random", "CMA-ES"…
   pruner: "MedianPruner"  # or "SuccessiveHalving"
   timeout: 7200           # in seconds
   storage: null           # e.g. "sqlite:///results/optuna.db"
   sample_size: 50

Example ``optuna_search_space.yaml``:

.. code-block:: yaml

   # Définition de l’espace de recherche pour Optuna
   facteur_swir:
     type: float
     min: 0.0
     max: 1.0
     step: 0.01

   beta:
     type: float
     min: 1.0
     max: 5.0
     step: 0.01

   level:
     type: int
     min: 1
     max: 6
     step: 1

   apply_gamma:
     type: categorical
     values: [True]

   gamma_value:
     type: float
     min: 0.01
     max: 4.0
     step: 0.01

Example ``config_viswir.yaml`` (pilotage):

.. code-block:: yaml

   visible_folder: "../data/VIS"
   swir_folder: "../data/SWIR"
   output_folder: "../results/optim"
   ref_image_path: null
   ground_truth_path: "../data/ground_truth"

   mode: "optuna"

   run_detection: false # true or false
   save_output: false # always true for optimization

.. important::
   You do not need include fusion parameters in ``parameters.json`` when using Optuna.
   They are defined and explored via ``optuna_search_space.yaml``.

Execution
---------

Run VISWIR in Optuna mode:

.. code-block:: bash

   python src/VISWIR_vQuasar.py

Each trial explores a different set of parameters. Results are stored in
``results/optuna_study.sqlite3`` (SQLite).

.. tip::
   Start with a small number of trials (10-20) to validate setup,
   then scale up for production runs.

.. note::
    Once optimisation is complete, the best parameters are stored in ``result/best_parameters.json``.

Resource Considerations
-----------------------

- **Workers (n_jobs)**: increasing jobs consumes more CPU/RAM.  
- **Detection optimization**: not recommended without a powerful machine
  (risk of crash or extremely long runtimes).  
- **Batch size & image size**: strongly influence optimization time.  

.. warning::
   Running detection optimization on limited hardware can lead to memory
   exhaustion or absurdly long computation times.

Results and Visualization
-------------------------

- Trials and metrics are saved in ``results/optuna_study.sqlite3``.
- The best parameters are stored in ``result/best_parameters.json``.
- Use ``tools/optuna_sqlite3_loader_mono_obj.py`` or
  ``tools/optuna_sqlite3_loader_multi_obj.py`` to explore results.
- On local machines, graphs are displayed in the browser at the end of optimization.
- The VS Code extension **Optuna Dashboard** can be used to visualize trials
  in real time.

.. note::
   Visualization requires a local environment with a browser. On HPC clusters,
   results must be exported and analyzed offline.

Advanced Usage
--------------

Optuna offers several advanced features that can be leveraged in VISWIR:

- **Multi-objective optimization**  
  Instead of optimizing a single metric, Optuna can handle multiple objectives
  (e.g. maximizing Entropy while minimizing BRISQUE, NIQE and PIQE).  
  Results can be explored with ``tools/optuna_sqlite3_loader_multi_obj.py``
  and visualized as a Pareto front.

  .. important::
      To activate multi-objective optimization, simply **disable detection** in
      ``config_viswir.yaml``. The optimization will then be performed on
      NR-IQA metrics instead of the F1 score.  
      This allows Optuna to balance several image quality metrics simultaneously.

- **Custom samplers**  
  By default, the TPE sampler is used. You can switch to other samplers such as
  Random, CMA-ES, or GridSearch by editing ``optuna_config.yaml``.  
  Each sampler has trade-offs:  
  - *TPE*: efficient for continuous spaces.  
  - *Random*: baseline, useful for debugging.  
  - *CMA-ES*: good for continuous, smooth search spaces.

- **Pruners**  
  Pruners stop unpromising trials early to save resources.  
  Common options:  
  - *MedianPruner*: stops trials worse than the median (default).  
  - *SuccessiveHalving*: aggressively prunes based on performance tiers.  
  Configure in ``optuna_config.yaml``.

- **Custom metrics**  
  You can extend VISWIR to optimize on custom metrics (e.g. perceptual quality,
  task-specific accuracy). This requires modifying the evaluation/objective function
  inside the pipeline.

- **Storage backends**  
  Optuna supports different storage backends for results:  
  - SQLite (default, local file).  
  - MySQL/PostgreSQL (for collaborative or large-scale experiments).  
  Configure via the ``storage`` field in ``optuna_config.yaml``.

.. tip::
   Use multi-objective optimisation when you want to optimise the quality
   of the output images independently of the detection scores.

Best Practices
--------------

- **Start small**  
  Begin with a limited number of trials (10-20) to validate your setup before
  scaling up.

- **Control resources**  
  The number of workers (``n_jobs``) directly impacts CPU and memory usage.
  Avoid setting it too high on limited hardware like laptop.

- **Detection optimization caution**  
  Running Optuna with detection enabled requires significant memory and compute.
  On modest machines, this can lead to crashes or extremely long runtimes.

- **Batch size & image size**  
  Larger batches and high-resolution images drastically increase optimization
  time. Adjust these parameters carefully.

- **Crash and restart**  
  If optimisation stops midway, simply restart it. As long as the ``*.sqlite3``
  file is still in the output folder, VISWIR and Optuna can pick up where they stopped.

- **Use pruners**  
  Enable pruners to stop unpromising trials early and save resources.

- **Visualize progress**  
  On local runs, Optuna automatically opens graphs in the browser at the end
  of optimization.  
  For real-time monitoring, the VS Code extension **Optuna Dashboard** is
  recommended but not mendatory.

.. warning::
   Misconfigured search spaces (too wide or unrealistic ranges) can lead to
   wasted trials and excessive runtimes. Always define sensible bounds.
