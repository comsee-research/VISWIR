"""
Main entry point for the VISWIR Quasar project.

This function orchestrates the execution of the VISWIR pipeline
depending on the selected mode (fast, SQL, or Optuna).

Workflow
--------
1. Parse CLI arguments.
    - `--fast` : Run the fast pipeline (fusion + detection, no metrics).
    - `--visible`, `--swir`, `--out` : Required in fast mode.
2. If fast mode is enabled:
    - Load minimal config (`fast_config.yaml` and `yolo_config.json`).
    - Run the `FastFusionPipeline` on the provided visible and SWIR images.
    - Save the fused image to the specified output path.
3. Otherwise:
    - Load all configurations (`base`, `params`, `optuna`).
    - Validate that `mode_fixe` is enabled in `parameters.json`.
        If not, exit with a safety warning.
    - Dispatch execution depending on the mode:
        * `"optuna"` : Run Optuna optimization (`process_folder_optuna`).
        * `"sql"` : Run SQL-based batch processing (`process_folder_sql`).
        * `"fixed"` : Run fixed-parameter batch processing (`process_folder`).

Notes
-----
- The script enforces `mode_fixe` for safety reasons, to prevent
    accidental execution of exploratory modes that may consume
    excessive disk space or memory.
- Logs are written using the project-wide logger.
- Errors are caught at the top level and logged as critical.

Raises
------
SystemExit
    If required arguments are missing in fast mode, or if `mode_fixe`
    is disabled in the configuration.
"""
# =============================================================================
# FILENAME:       VISWIR_vQuasar.py
# DESCRIPTION:    Point d’entrée du projet VISWIR Quasar.
#                 Charge la configuration et lance le mode choisi (fixed, sql, optuna).
#  
# REPOSITORY:     https://github.com/comsee-research/VISWIR.git
#
# AUTHOR:         [Riffard Alexandre]
# EMAIL:          [alexandre.riffard@uca.fr]
# CREATED:        [28-10-2025]
# LAST UPDATED:   [28-10-2025]
# VERSION:        1.2 (Quasar)
#
# LICENSE:        GNU LESSER GENERAL PUBLIC LICENSE (voir LICENSE dans le dépôt)
#
# USAGE:          - Ne rien toucher et lancer l'éxécution du fichier seul.
#
# DEPENDENCIES:   - Aucune
#
# NOTES:
#   - ...
#
# CHANGELOG:
#   - [28-10-2025]: Création initiale du fichier à partir d'un ancien notebook.
#
# =============================================================================

import sys
import argparse
from pathlib import Path

from common.logger import logger
from common.config_loader import load_all_configs
from processing.batch_runner import process_folder
from processing.sql_runner import process_folder_sql
from optimization.optuna_runner import process_folder_optuna
from common.ui import print_viswir_header

# Import pipeline rapide
from realtime.fast_config import load_fast_config
from realtime.fast_fusion_runner import FastFusionPipeline
from skimage import io

def main():
    """
    Main orchestrator for VISWIR. Parses arguments and runs the chosen mode.
    """
    print_viswir_header()

    # --- Étape 0 : Parser les arguments CLI ---
    parser = argparse.ArgumentParser(description="VISWIR vQuasar")
    parser.add_argument("--fast", action="store_true", help="Utiliser la pipeline rapide (fusion + détection sans métriques)")
    parser.add_argument("--visible", type=Path, help="Image visible (mode --fast)")
    parser.add_argument("--swir", type=Path, help="Image SWIR (mode --fast)")
    parser.add_argument("--out", type=Path, help="Image de sortie (mode --fast)")
    args = parser.parse_args()

    # --- Mode rapide ---
    if args.fast:
        cfg = load_fast_config()
        pipeline = FastFusionPipeline(cfg)

        if not args.visible or not args.swir or not args.out:
            logger.error("❌ En mode --fast, vous devez fournir --visible, --swir et --out")
            sys.exit(1)

        logger.info(f"🚀 Start !")
        fused = pipeline.run(args.visible, args.swir)
        io.imsave(args.out, (fused * 255).astype("uint8"))
        logger.info(f"✅ Image fusionnée sauvegardée dans {args.out}")
        return

    # --- Étape 1 : Charger toutes les configs ---
    configs = load_all_configs()
    base_cfg = configs["base"]
    params = configs["params"]

    visible_folder = Path(base_cfg["visible_folder"])
    swir_folder = Path(base_cfg["swir_folder"])
    output_folder = Path(base_cfg["output_folder"])
    ref_image_path = Path(base_cfg["ref_image_path"]) if base_cfg.get("ref_image_path") else None
    ground_truth_path = Path(base_cfg["ground_truth_path"]) if base_cfg.get("ground_truth_path") else None
    mode = base_cfg.get("mode", "sql")
    run_detection = base_cfg.get("run_detection", False)
    save_output = base_cfg.get("save_output", False)

    # --- Étape 1 bis : Vérification du mode_fixe ---
    if not params.get("mode_fixe", False):
        logger.warning(
            "⚠️ ATTENTION : Vous n'utilisez pas le mode_fixe dans parameters.json.\n"
            "L'utilisation des modes d'exploration (pour les option Fixed / SQL) n'est PAS recommandée sans précautions.\n"
            "Ces modes peuvent consommer énormément d'espace disque et de mémoire vive.\n"
            "👉 Vérifiez vos ressources système avant de lancer ce mode.\n"
            "ℹ️ Le fichier parameters.json doit respecter un format particulier.\n"
            "   Un exemple est disponible dans le dossier config/parameter_exemple/parameters_exploratory_default.json."
            "❗ Décharge de responsabilité : Les auteurs du logiciel ne pourront être tenus responsables "
            "d'éventuels dommages matériels, pertes de données ou saturations de ressources liés à une "
            "utilisation inappropriée de ces modes."
        )
        # Crash volontaire pour protéger l'utilisateur et son matériel
        sys.exit("❌ Sécurité activée : exécution interrompue car mode_fixe est désactivé.")

    # --- Étape 2 : Dispatcher selon le mode ---
    if mode == "optuna":
        n_trials = base_cfg.get("n_trials", configs["optuna"]["n_trials"])
        process_folder_optuna(
            visible_folder, swir_folder, output_folder,
            ref_image_path=ref_image_path,
            n_trials=n_trials,
            run_detection=run_detection,
            ground_truth_path=ground_truth_path
        )
    elif mode == "sql":
        process_folder_sql(
            visible_folder, swir_folder, output_folder,
            ref_image_path=ref_image_path,
            ground_truth_path=ground_truth_path,
            run_detection=run_detection,
            save_output=save_output,
            params=params
        )
    else:  # mode fixe
        process_folder(
            visible_folder, swir_folder, output_folder,
            ref_image_path=ref_image_path,
            run_detection=run_detection,
            ground_truth_path=ground_truth_path,
            save_output=save_output,
            params=params
        )

    logger.info("✅ Script terminé.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"❌ Erreur critique : {e}")
        sys.exit(1)