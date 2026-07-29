#!/bin/bash
set -e  # stop on first error

# Détecter la langue du système
LANGUAGE=$(locale | grep LANG | cut -d= -f2 | cut -d_ -f1)

# Définition des couleurs ANSI
RED=$(tput setaf 1)
GREEN=$(tput setaf 2)
CYAN=$(tput setaf 6)
BOLD=$(tput bold)
RESET=$(tput sgr0)

# export LD_LIBRARY_PATH=/opt/glibc-2.35/lib:$LD_LIBRARY_PATH

# Définir les messages en fonction de la langue détectée
if [ "$LANGUAGE" == "fr" ]; then
    MSG_UPDATE_PIP="🔄 Mise à jour de pip..."
    MSG_INSTALL="📦 Installation des bibliothèques..."
    MSG_DONE="✅ Installation terminée !"
    MSG_ERROR="❌ Erreur : python3/pip n'est pas installé !"
    MSG_CPU="⚙️ Installation des versions CPU de PyTorch..."
else
    MSG_UPDATE_PIP="🔄 Updating pip..."
    MSG_INSTALL="📦 Installing packages..."
    MSG_DONE="✅ Installation completed!"
    MSG_ERROR="❌ Error: python3/pip is not installed!"
    MSG_CPU="⚙️ Installing CPU versions of PyTorch..."
fi

# Vérifier si pip est installé
if ! command -v python3 &> /dev/null; then
    echo "${RED}${BOLD}$MSG_ERROR${RESET}"
    exit 1
fi

# Mise à jour de pip
echo "${CYAN}${BOLD}$MSG_UPDATE_PIP${RESET}"
python3 -m pip install --upgrade pip setuptools wheel

# Liste des bibliothèques à installer
PACKAGES=(
    "numpy==2.2.6"
    "rich==14.0.0"
    "opencv-contrib-python==4.11.0.86"
    "scikit-image==0.25.2"
    "matplotlib==3.10.3"
    "typing-extensions==4.13.2"
    "loguru==0.7.3"
    "pytorch-msssim==1.0.0"
    "brisque==0.0.16"
    "SQLAlchemy==2.0.41"
    "libsvm-official==3.35.0"
    "imagecodecs==2025.3.30"
	"sewar==0.4.6"
    "pypiqe==1.2"
    "optuna==4.3.0"
    "kaleido==0.2.1"
	"plotly==6.0.1"
	"ultralytics==8.3.14"
)

# Installation des bibliothèques
echo "${GREEN}${BOLD}$MSG_INSTALL${RESET}"
python3 -m pip install "${PACKAGES[@]}"

echo "${CYAN}${BOLD}$MSG_CPU${RESET}"
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Nettoyage cache pip
python3 -m pip cache purge || true

# Message final
echo -e "\n${GREEN}${BOLD}$MSG_DONE${RESET}"