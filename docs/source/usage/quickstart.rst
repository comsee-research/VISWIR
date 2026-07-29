Quickstart
==========

This step-by-step guide walks you from a fresh checkout to a successful run
of **VISWIR**. Follow the steps in order; optional paths (container/HPC)
are clearly marked.

.. contents::
   :local:
   :depth: 2

Prerequisites
-------------

- **OS:** Linux (recommended) or Windows (developed on), not tested on macOS
- **Python:** 3.10+ installed and on PATH
- **Git:** installed
- **Disk space:** ~3-5 GB (datasets + container optional)
- **Optional:** `Singularity <https://docs.sylabs.io/guides/latest/user-guide/>`_/Apptainer (for container), `SLURM <https://slurm.schedmd.com/overview.html>`_ (for HPC)

.. note::
   If you plan to use TIFF images, you'll need the Python package ``imagecodecs``.

Step 1 - Get the source
-----------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/comsee-research/viswir.git
      cd viswir

2. Inspect the layout (optional):

   .. code-block:: bash

      ls -1
      # README.md, HELP.md, src/, config/, data/, tools/, results/, requirements.txt ...

Step 2 - Choose your environment
--------------------------------

A. Local Python (virtualenv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install packages (script provided):

   .. code-block:: bash

      chmod +x install_packages_venv.sh
      ./install_packages_venv.sh

2. If you use TIFF images:

   .. code-block:: bash

      pip install imagecodecs

.. note::
   You can also install manually with ``pip install -r requirements.txt``.
   In this case, first create a virtualenv and activate it:

   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate

B. Container (Singularity/Apptainer)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Build the container (CPU-only example):

   .. code-block:: bash

      cd tools
      sudo singularity build VISWIR.sif Singularity

   The Singularity build file is available in the ``tools`` folder.

.. important::
   To use Singularity, you must be running Linux or WSL2 for Windows
   (tested via WSL2 with Debian).  
   See the `Singularity user guide <https://docs.sylabs.io/guides/latest/user-guide/>`_.

2. Confirm it runs:

   .. code-block:: bash

      singularity exec VISWIR.sif python -V

.. tip::
   Full container instructions are available in :doc:`../help` and :doc:`hpc_slurm`.

Step 3 - Prepare your data
--------------------------

1. Create a minimal dataset structure:

   .. code-block:: bash

      mkdir -p data/visible data/swir results
      cp path/to/your/visible_images/*.png data/visible/
      cp path/to/your/swir_images/*.png data/swir/

2. Ensure filenames are paired consistently:

   - Example: ``data/visible/img_001.png`` matches ``data/swir/img_001.png``.

3. Optional detection annotations:

   - Add XML files in ``data/ground_truth/`` if you plan to run detection.

.. warning::
   Misaligned pairs produce invalid fusion. Always verify filenames match
   across ``visible/`` and ``swir/``.

Step 4 - Configure the run
--------------------------

1. Edit ``config/config_viswir.yaml``:
   - Paths: ``visible_folder``, ``swir_folder``, ``output_folder``
   - Mode: ``sql`` (recommended), ``optuna``, or ``fixed`` (legacy)
   - Optional: ``ref_image_path``, ``ground_truth_path``, ``run_detection``, ``save_output``

2. Fusion parameters:
   - ``parameters.json`` (SQL/fixed modes)
   - ``fast_config.yaml`` (fast mode)

3. Detection (optional):
   - ``yolo_config.json`` (weights, thresholds, classes, device)

.. tip::
   Start simple—disable detection and enable ``save_output`` to validate the pipeline first.

Step 5 - First execution
------------------------

Fast mode (quick smoke test)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python src/VISWIR_vQuasar.py --fast \
       --visible ./data/visible/img_001.png \
       --swir ./data/swir/img_001.png \
       --out ./results/fused_img_001.png

.. note::
   Fast mode skips metric computation. Use it for rapid prototyping,
   not for final evaluation.

SQL mode (recommended full run)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # In config_viswir.yaml: mode: "sql"
   python src/VISWIR_vQuasar.py

Optuna mode (hyperparameter optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # In config_viswir.yaml: mode: "optuna"
   python src/VISWIR_vQuasar.py

.. tip::
   Start with fewer trials (10-20) to validate everything, then scale up.

Step 6 - Running with container
-------------------------------

Fast test inside container:

.. code-block:: bash

   singularity run VISWIR.sif --fast \
       --visible /VISWIR/data/visible/img_001.png \
       --swir /VISWIR/data/swir/img_001.png \
       --out /VISWIR/results/fused_img_001.png

Bind host directories:

.. code-block:: bash

   singularity run \
       -B $(pwd)/results:/VISWIR/results \
       -B $(pwd)/src/logs:/VISWIR/src/logs \
       -B $(pwd)/config:/VISWIR/config \
       -B $(pwd)/data:/VISWIR/data \
       VISWIR.sif

Step 7 - Optional: HPC (SLURM) submission
-----------------------------------------

1. Pick a script in ``tools/``:
   - ``job_viswir_test.slurm`` (quick test)
   - ``job_viswir.slurm`` (standard)
   - ``job_viswir_ex.slurm`` (custom resources)

2. Submit:

   .. code-block:: bash

      sbatch tools/job_viswir.slurm

3. Monitor:

   .. code-block:: bash

      squeue -u $USER
      scontrol show job <job_id>
      seff <job_id>

.. important::
   Adjust ``--mem``, ``--cpus-per-task``, and ``--time`` to match your quota
   and dataset size.

Step 8 - Verify outputs and logs
--------------------------------

- **Images:** check ``results/`` for fused outputs.
- **Database:** inspect ``results/results.db`` with ``tools/sql_explorer.py``.
- **Logs:** review ``src/logs/`` for warnings or errors.

.. note::
    Any errors related to input data (different sizes, unequal number of visible and SWIR images, etc.)
    will cause a fatal error in code execution.
    Errors that cause the code to crash will be displayed in your terminal.

Step 9 - Common issues and quick fixes
--------------------------------------

- Missing TIFF support → ``pip install imagecodecs``
- No fused image in fast mode → check ``--out`` path
- SQL run persists but no metrics → confirm ``save_output`` and dataset paths
- Detection crashes → check ``yolo_config.json`` and try ``device: "cpu"``
- Container path issues → verify ``-B`` bindings and internal paths

Step 10 - Next steps
--------------------

- Explore configuration details: see :doc:`configs`
- Prepare larger datasets: see :doc:`data`
- Use tools for analysis and export: see :doc:`tools`
- Run at scale on HPC: see :doc:`hpc_slurm`

.. tip::
   Once your pipeline is stable in SQL mode, move to Optuna to tune parameters
   and maximize metric performance for your dataset.
