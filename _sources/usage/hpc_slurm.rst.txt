Running VISWIR on HPC with SLURM
================================

This section explains how to use the provided SLURM script to run VISWIR
on a High Performance Computing (HPC) cluster.

Location
--------

The SLURM job script is available in the ``tools/`` directory:

- ``tools/Singularity`` : definition file to build the container
- ``tools/job_viswir.slurm`` : SLURM batch script to launch VISWIR

Script Overview
---------------

The script performs the following steps:

1. **Job configuration**  
   - Job name, partition, time limit, memory, CPUs  
   - Email notifications on job completion or failure

2. **Container execution**  
   - Loads Singularity  
   - Runs the VISWIR container with mounted volumes for results, logs, config, and data

3. **Safety checks**  
   - Verifies that the required ``.sif`` file exists before execution

4. **Job efficiency report**  
   - Calls ``seff $SLURM_JOB_ID`` to display resource usage statistics

5. **Result transfer**  
   - Copies the ``results.db`` file to a remote PC via ``scp`` for post-processing

Script Excerpt
--------------

Here is an exemple configuration section for a SLURM script:

.. code-block:: bash

   #SBATCH --job-name=VISWIR
   #SBATCH --output=viswir_output.log
   #SBATCH --error=viswir_error.log
   #SBATCH --partition=long
   #SBATCH --time=48:00:00
   #SBATCH --ntasks=1
   #SBATCH --mem=384G
   #SBATCH --cpus-per-task=16
   #SBATCH --mail-type=END,FAIL
   #SBATCH --mail-user=your.email@your.lab.xx

This configuration requests:

- **Partition**: ``long`` (long jobs)
- **Time limit**: 48 hours
- **Resources**: 1 task, 16 CPUs, 384 GB RAM
- **Notifications**: email sent at job end or failure

Customization
-------------

- Adjust ``--time``, ``--mem``, and ``--cpus-per-task`` depending on your cluster resources.
- Update the paths in the ``-B`` bindings to match your HPC environment.
- Replace the email address in ``#SBATCH --mail-user`` with your own.
- Modify the ``scp`` target to point to your workstation or storage server.

Submitting the Job
------------------

To submit the job:

.. code-block:: bash

   sbatch tools/run_viswir.slurm

Checking Job Status
-------------------

To check if the job has been accepted and is running:

.. code-block:: bash

   squeue -u $USER

To see detailed information about the job:

.. code-block:: bash

   scontrol show job <job_id>

After completion, you can check efficiency with:

.. code-block:: bash

   seff <job_id>

Logs and Results
----------------

- Standard output: ``viswir_output.log``
- Errors: ``viswir_error.log``
- Results: copied automatically to your workstation via ``scp`` (see script). Depending on your configuration, this command may not work. In this case, remove it or comment it out.

.. tip::
   - Use the ``--fast`` mode inside the container for quick parameter testing.  
   - Always check the ``logs/`` directory for error messages and debugging information.  
   - If your cluster supports GPUs, request them with ``#SBATCH --gres=gpu:1`` and ensure your container includes GPU-enabled PyTorch (not by default).
