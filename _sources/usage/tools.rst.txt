Tools
=====

The ``tools/`` folder contains helper scripts and utilities that support
data conversion, visualization, HPC execution, and optimization workflows
in VISWIR.

.. warning::
   The scripts and utilities provided in the ``tools/`` folder are distributed
   **as-is**, without any warranty. Their use is entirely at the responsibility
   of the user.

   The authors cannot be held liable for any data loss, corruption, or other
   issues that may arise from their use. Please ensure you test scripts on
   non-critical data before applying them in production environments.


.. contents::
   :local:
   :depth: 1

Data Conversion
---------------
- ``coco_to_voc_converter.py`` : convert COCO annotations to Pascal VOC format.
- ``export_data_to_csv.py`` : export results or metrics to CSV.
- ``export_data_to_json.py`` : export results or metrics to JSON (optional).

Visualization
-------------
- ``draw_graph_from_csv.py`` : generate graphs from CSV outputs (To be used after using the ``export_data_to_csv.py`` script).
- ``draw_annotations_on_images.py`` : overlay annotations on images (optional).
- ``csv_graph_comparator.py`` : compare multiple CSV result files (optional).

HPC / Container
---------------
- ``Singularity`` : Singularity container definition file.
- ``job_viswir.slurm`` : SLURM job submission script for HPC environments.
- ``job_viswir_ex.slurm`` : extended SLURM job script (custom resources and option).
- ``job_viswir_test.slurm`` : lightweight SLURM script for testing.

Optimization
------------
- ``optuna_sqlite3_loader_mono_obj.py`` : load and analyze Optuna results (single objective - F1).
- ``optuna_sqlite3_loader_multi_obj.py`` : load and analyze Optuna results (multi-objective - NR-IQA).
- ``sql_explorer.py`` : explore the SQLite results database interactively.

Usage
-----
Each script is standalone and must be adapted to your environment.
Refer to the script headers for usage instructions.

.. tip::
   - HPC users should start with ``job_viswir.slurm`` and the ``Singularity`` file.  
   - For optimization analysis, use the Optuna loaders together with ``sql_explorer.py``.  
   - Visualization scripts are mainly useful for legacy or debugging workflows.
