# 🚀 VISWIR Container User Guide

Welcome to the **VISWIR** container!
This guide will help you understand how to use the container, its main features, and useful commands.

---

## 📌 Installing and Running the Container

### 1️⃣ **Download and Build**

To **rebuild** the container from the [`Singularity`](./tools/Singularity) definition file:

```bash
sudo singularity build VISWIR.sif Singularity
```

### 2️⃣ **Running the Container**

To launch the container with mounted directories:

```bash
singularity run \
    -B /path/to/results:/VISWIR/results \
    -B /path/to/logs:/VISWIR/src/logs \
    -B /path/to/config:/VISWIR/config \
    -B /path/to/data:/VISWIR/data \
    VISWIR.sif
```

You can pass arguments directly to the main script `VISWIR_vQuasar.py`:

```python
singularity run VISWIR.sif --fast --visible ../data/vis.png --swir ../data/swir.png --out ../results/fused.png
```

---

## 🛠 Features

✔️ Ready-to-use Python environment
✔️ Dynamic library installation (via `install_packages_container.sh`)
✔️ CPU-only PyTorch version (optimized for GPU-free HPC environments)
✔️ Informative welcome message displayed at startup
✔️ Easy access to documentation via `/VISWIR/HELP.md`:

```bash
singularity exec VISWIR.sif cat /VISWIR/HELP.md
```

---

## 🖥 Useful Commands

### 🔧 Check installed Python libraries

```bash
singularity exec VISWIR.sif pip list
```

### 🔍 Check PyTorch version

```bash
singularity exec VISWIR.sif python -c "import torch; print(torch.__version__)"
```

### 🔄 Reinstall / Update Python libraries

```bash
singularity exec VISWIR.sif /VISWIR/install_packages_container.sh
```

---

## 🏗 Container Structure

The container includes the following files:

```
 /VISWIR
 ├── config/                        (Configuration files)
 ├── data/                          (Input datasets)
 ├── results/                       (Processing results)
 ├── msc/                           (Funding and HPC logos)
 ├── src/                           (Main source code)
 │ ├── VISWIR_vQuasar.py            # Main entry point
 │ ├── logs/                        # Log files
 │ │ 
 │ ├── fusion/                      # Scientific core
 │ │ ├── NIQE/                      # NIQE implementation
 │ │ │ ├── *.mat                    # Matlab files for NIQE
 │ │ │ ├── niqe.py                  # NIQE computation
 │ │ ├── fusion.py                  # Main fusion functions
 │ │ ├── functions.py               # Direct support functions
 │ │ ├── metrics.py                 # Metrics computation (SSIM, NIQE, etc.)
 │ │ ├── detection_module.py        # Detection (YOLO + F1)
 │ │ └── utils.py                   # Utilities (I/O, normalization, etc.)
 │ │ 
 │ ├── processing/                  # Batch/SQL orchestration
 │ │ ├── batch_runner.py            # Batch processing
 │ │ ├── sql_runner.py              # SQL processing
 │ │ ├── task_manager.py            # Task management
 │ │ └── interruption.py            # Interruption handling
 │ │ 
 │ ├── optimization/                # Optimization (Optuna, HPC)
 │ │ ├── optuna_runner.py           # Main Optuna loop
 │ │ ├── objective.py               # Objective functions
 │ │ ├── visualization.py           # Result visualization
 │ │ └── samplers.py                # Sampler/pruner configuration
 │ │ 
 │ ├── realtime/                    
 │ │ ├── fast_fusion_runner.py      # Fast pipeline
 │ │ ├── fast_config.py             # Fast configuration
 │ │ └── fast_detection.py          # Fast detection
 │ │ 
 │ └── common/                      # Shared modules
 │ │ ├── logger.py                  # Centralized logging
 │ │ ├── ui.py                      # Terminal display
 │ │ ├── results_db.py              # Database connection and saving
 │ │ ├── config_loader.py           # YAML/JSON loader
 │ │ └── datatypes.py               # Dataclasses (ProcessResult, Config, etc.)
 │ 
 ├── test/                          (Test scripts)
 ├── tools/                         (Utilities and SQL)
 │
 ├── HELP.md                        (This guide)
 ├── install_packages_venv.sh       (Local installation via venv)
 ├── install_packages_container.sh  (Container installation)
 ├── LICENSE.txt                    (Project license)
 ├── README.md                      (Project overview)
 ├── requirements.txt               (List of Python dependencies)
```

---

## 🆘 Support and Contact

If you encounter an issue, check:

📄 [README.md](./README.md): General project overview.

📖 [HELP_fr.md](./HELP_fr.md) (this file but in french): Container usage guide.

📜 Logs are available in `/VISWIR/src/logs` in case of errors.

📬 Need help? Contact us at: [alexandre.riffard@uca.fr](mailto:alexandre.riffard@uca.fr).
