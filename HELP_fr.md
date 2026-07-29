# 🚀 Guide d'utilisation du conteneur VISWIR

Bienvenue dans le conteneur **VISWIR** ! Ce guide vous aidera à comprendre comment utiliser le conteneur, ses fonctionnalités principales et les commandes utiles.

---

## 📌 Installation et exécution du conteneur

### 1️⃣ **Téléchargement et construction**
Pour **reconstruire** le conteneur depuis le fichier [`Singularity`](./tools/Singularity) :

```bash
sudo singularity build VISWIR.sif Singularity
```

### 2️⃣ **Exécution du conteneur**
Pour lancer le conteneur avec les répertoires montés :
```bash
singularity run \
    -B /chemin/vers/results:/VISWIR/results \
    -B /chemin/vers/logs:/VISWIR/src/logs \
    -B /path/to/config:/VISWIR/config \
    -B /path/to/data:/VISWIR/data \
    VISWIR.sif
```
Vous pouvez passer des arguments directement au script principal ```VISWIR_vQuasar.py``` :
```python
singularity run VISWIR.sif --fast --visible ../data/vis.png --swir ../data/swir.png --out ../results/fused.png
```

---

## 🛠 Fonctionnalités
✔️ Environnement Python prêt à l'emploi
✔️ Installation dynamique des bibliothèques (via script ```install_packages_container.sh```)
✔️ Version CPU‑only de PyTorch (optimisée pour les environnements HPC sans GPU)
✔️ Affichage d’un message de bienvenue informatif à chaque lancement
✔️ Facilité d’accès à la documentation via ```/VISWIR/HELP.md``` :
	```bash
	singularity exec VISWIR.sif cat /VISWIR/HELP.md
	```

---

## 🖥 Commandes utiles

### 🔧 Vérifier les bibliothèques python installées
```bash
singularity exec VISWIR.sif pip list
```

### 🔍 Vérifier la version de PyTorch
```bash
singularity exec VISWIR.sif python -c "import torch; print(torch.__version__)"
```

### 🔄 Réinstaller / Mettre à jour les bibliothèques Python
```bash
singularity exec VISWIR.sif /VISWIR/install_packages_container.sh
```

---

## 🏗 Structure du conteneur
Le conteneur contient les fichiers suivants :

```
 /VISWIR
 ├── config/                        (Fichiers de configuration) 
 ├── data/                          (Jeux de données d'entrée) 
 ├── results/                       (Résultats des traitements) 
 ├── msc/                           (Logos CIR, I-SITE et Mésocentre) 
 ├── src/                           (Code source principal) 
 │ ├── VISWIR_vQuasar.py            # Point d’entrée principal 
 │ ├── logs/                        # Fichiers de logs 
 │ │ 
 │ ├── fusion/                      # Cœur scientifique 
 │ │ ├── NIQE/                      # Implémentation NIQE 
 │ │ │ ├── *.mat                    # Fichiers Matlab pour NIQE 
 │ │ │ ├── niqe.py                  # Calcul de NIQE 
 │ │ ├── fusion.py                  # Fonctions principales de fusion 
 │ │ ├── functions.py               # Fonctions de support direct 
 │ │ ├── metrics.py                 # Calcul des métriques (SSIM, NIQE, etc.) 
 │ │ ├── detection_module.py        # Détection (YOLO + F1) 
 │ │ └── utils.py                   # Utilitaires (I/O, normalisation…) 
 │ │ 
 │ ├── processing/                  # Orchestration batch/SQL 
 │ │ ├── batch_runner.py            # Traitement par lots 
 │ │ ├── sql_runner.py              # Traitement SQL 
 │ │ ├── task_manager.py            # Gestion des tâches 
 │ │ └── interruption.py            # Gestion des interruptions 
 │ │ 
 │ ├── optimization/                # Optimisation (Optuna, HPC) 
 │ │ ├── optuna_runner.py           # Boucle principale Optuna 
 │ │ ├── objective.py               # Fonctions objectif 
 │ │ ├── visualization.py           # Visualisation des résultats 
 │ │ └── samplers.py                # Config des samplers/pruners 
 │ │ 
 │ ├── realtime/                     
 │ │ ├── fast_fusion_runner.py      # Pipeline rapide 
 │ │ ├── fast_config.py             # Config rapide 
 │ │ └── fast_detection.py          # Détection rapide 
 │ │ 
 │ └── common/                      # Modules transverses 
 │ │ ├── logger.py                  # Logging centralisé 
 │ │ ├── ui.py                      # Affichage terminal 
 │ │ ├── results_db.py              # Connexion et sauvegarde DB 
 │ │ ├── config_loader.py           # Chargement YAML/JSON 
 │ │ └── datatypes.py               # Dataclasses (ProcessResult, Config, etc.)
 │ 
 ├── test/                          (Scripts de tests) 
 ├── tools/                         (Utilitaires et SQL) 
 │
 ├── HELP_fr.md                     (Ce guide) 
 ├── install_packages_venv.sh       (Installation locale via venv) 
 ├── install_packages_container.sh  (Installation dans le conteneur) 
 ├── LICENSE.txt                    (Licence du projet) 
 ├── README.md                      (Présentation du projet) 
 ├── requirements.txt               (Liste des dépendances Python)
```

---

## 🆘 Support et contact
Si vous rencontrez un problème, consultez :

📄 [README.md](./README.md) : Présentation générale du projet.

📖 [HELP.md](./HELP.md) (ce fichier mais en anglais) : Guide d’utilisation du conteneur.

📜 Logs disponibles dans ```/VISWIR/src/logs``` en cas d'erreurs.

📬 Besoin d’aide ? Contactez-nous à : [alexandre.riffard@uca.fr](mailto:alexandre.riffard@uca.fr).