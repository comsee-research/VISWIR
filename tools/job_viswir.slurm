#!/bin/bash

#===============================================================================

#SBATCH --job-name=VISWIR
#SBATCH --output=viswir_output.log
#SBATCH --error=viswir_error.log
#SBATCH --partition=long
#SBATCH --time=48:00:00  # Durée max du job (48h ici, ajuste si besoin)
#SBATCH --ntasks=1
#SBATCH --mem=64G  # Allouer 128 Go de RAM (ajuste selon tes besoins)
#SBATCH --cpus-per-task=16  # Nombre de cœurs CPU (ajuste selon tes besoins)

#===============================================================================
# Charger Singularity
module load singularity

# Exécuter le conteneur avec les bons volumes montés
singularity run \
    -B /home/alriffard/Desktop/workspace_alex/VISWIR/results:/VISWIR/results \
    -B /home/alriffard/Desktop/workspace_alex/VISWIR/src/logs:/VISWIR/src/logs \
    /home/alriffard/Desktop/workspace_alex/VISWIR_ex_51.sif
