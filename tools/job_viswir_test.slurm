#!/bin/bash
set -euo pipefail

#===============================================================================

#SBATCH --job-name=Test_VISWIR    # Nom du job
#SBATCH --output=test_output.log  # Fichier de sortie
#SBATCH --error=test_error.log    # Fichier d'erreur
#SBATCH --partition=normal        # Partition normale pour les tests
#SBATCH --time=00:10:00           # Durée max de 10 minutes
#SBATCH --ntasks=1                # Une seule tâche
#SBATCH --cpus-per-task=2         # 2 cœurs CPU
#SBATCH --mem=4G                  # 4 Go de RAM

#===============================================================================
# Charger Singularity
module load singularity

# Exécuter le conteneur avec les bons volumes montés
singularity run \
    -B /home/alriffard/Desktop/workspace_alex/VISWIR/results:/VISWIR/results \
    -B /home/alriffard/Desktop/workspace_alex/VISWIR/src/logs:/VISWIR/src/logs \
    /home/alriffard/Desktop/workspace_alex/VISWIR_opt_23-24.sif

# 🔍 Résumé post-job (CPU, RAM, durée...)
seff $SLURM_JOB_ID