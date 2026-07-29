#!/bin/bash
# set -euo pipefail

#===============================================================================
#SBATCH --job-name=VISWIR
#SBATCH --output=viswir_output.log
#SBATCH --error=viswir_error.log
#SBATCH --partition=long
#SBATCH --time=48:00:00  # Durée max du job (48h ici, a ajuster si besoin)
#SBATCH --ntasks=1
#SBATCH --mem=384G  # Allouer 64 Go de RAM (ajuster selon les besoins)
#SBATCH --cpus-per-task=16  # Nombre de cœurs CPU (ajuster selon les besoins)
#SBATCH --mail-type=END,FAIL  # Envoi d'un email à la fin ou en cas d'échec
#SBATCH --mail-user=alexandre.riffard@doctorant.uca.fr  # Remplacer par le bon

#===============================================================================
# Charger Singularity
module load singularity

# Exécuter le conteneur avec les bons volumes montés
singularity run \
    -B /home/alriffard/Desktop/workspace_alex/VISWIR/results:/VISWIR/results \
    -B /home/alriffard/Desktop/workspace_alex/VISWIR/src/logs:/VISWIR/src/logs \
	-B /home/alriffard/Desktop/workspace_alex/VISWIR/config:/VISWIR/config \
    /home/alriffard/Desktop/workspace_alex/VISWIR_RASMD_Snow.sif

if [ ! -f "/home/alriffard/Desktop/workspace_alex/VISWIR_opt_23-24.sif" ]; then
    echo "❌ Fichier .sif introuvable ! Annulation du job."
    exit 1
fi


seff $SLURM_JOB_ID

#===============================================================================
# Transférer le fichier `results.db` vers un PC distant via SCP
echo "🚀 Transfert du fichier results.db vers le PC distant..."
scp -i /home/alriffard/.ssh/id_rsa /home/alriffard/Desktop/workspace_alex/VISWIR/results/results.db \
    riffard@DESKTOP-NVKQ3OV:/mnt/c/Users/Riffard/Documents/__WSL/Resultats_HPC
    #riffard@DESKTOP-NVKQ3OV:"C:\Users\Riffard\Documents\__WSL\Resultats_HPC"

if [ $? -ne 0 ]; then
    echo "❌ Échec du transfert SCP."
fi


echo "✅ Transfert terminé !"