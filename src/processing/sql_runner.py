"""
SQL database orchestration for batch runs and logging.
"""

# src/processing/sql_runner.py

from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
import os
import json

from common.logger import logger
from common.datatypes import ProcessResult
from processing.task_manager import generate_tasks_in_memory, batchify_tasks
from processing.batch_runner import process_image_wrapper
from fusion.utils import load_image_ref_skimage, generate_parameters_json
from common.results_db import get_session, save_result_to_db
from common.config_loader import load_all_configs
# from processing import interruption

from sqlalchemy import text # pour les requêtes SQL

# Legacy version :
# def process_folder_sql(
    # visible_folder: Path,
    # swir_folder: Path,
    # output_dir: Path,
    # batch_size: int = 20,
    # ref_image_path: Optional[Path] = None,
    # save_output: bool = False,
    # run_detection: bool = True,
    # ground_truth_path: Optional[Path] = None,
    # params: dict | None = None,
    # workers: int = 2
# ) -> None:
    # """
    # Legacy version of the function
    # Process a folder of visible and SWIR images for SQL-based analysis.

    # This function orchestrates the batch processing of image fusion tasks,
    # computes metrics, and stores results directly into a SQLite database
    # (`results.db`) located in the output directory.

    # Steps
    # -----
    # 1. Initialize the SQLite database session.
    # 2. Discover visible and SWIR image files in the provided folders.
    # 3. Validate the number of images and optionally load a reference image.
    # 4. Load fusion parameters (from config if not provided).
    # 5. Generate tasks in memory for all image pairs.
    # 6. Execute tasks in parallel batches with progress tracking.
    # 7. Save results (metrics and parameters) into the database.

    # Parameters
    # ----------
    # visible_folder : Path
        # Path to the folder containing visible images.
    # swir_folder : Path
        # Path to the folder containing SWIR images.
    # output_dir : Path
        # Directory where the SQLite database and logs will be saved.
    # batch_size : int, default=20
        # Number of tasks to process in parallel per batch.
    # ref_image_path : Path or None, optional
        # Path to the reference image (optional).
    # save_output : bool, default=False
        # Whether to save intermediate outputs (images, annotations).
    # run_detection : bool, default=True
        # Whether to run YOLO detection in addition to metric computation.
    # ground_truth_path : Path or None, optional
        # Path to ground truth annotations (used if detection is enabled).
    # params : dict or None, optional
        # Fusion parameters to apply. If None, parameters are loaded from config.

    # Returns
    # -------
    # None
        # Results are inserted into the SQLite database.

    # Raises
    # ------
    # ValueError
        # If the number of visible and SWIR images does not match.
    # FileNotFoundError
        # If the reference image path is provided but cannot be loaded.
    # RuntimeError
        # If an error occurs during task execution or saving results.

    # Notes
    # -----
    # - Results are stored in `results.db` inside the output directory.
    # - Progress is displayed in the console with a live progress bar.
    # - Updates `interruption.last_params` with the latest parameters for
      # potential resume after interruption.
    # """


    # logger.info("🚀 Démarrage du traitement SQL...")
    # logger.info(f"📂 Visible : {visible_folder}")
    # logger.info(f"📂 SWIR    : {swir_folder}")
    # logger.info(f"📁 Sortie  : {output_dir}")

    # output_dir.mkdir(parents=True, exist_ok=True)
    
    # # Initialisation de la base de données
    # session = get_session(output_dir / "results.db")

    # # --- Étape 1 : Découverte des fichiers ---
    # image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    # visible_files = sorted([f for ext in image_extensions for f in visible_folder.glob(ext)])
    # swir_files    = sorted([f for ext in image_extensions for f in swir_folder.glob(ext)])

    # if len(visible_files) != len(swir_files):
        # raise ValueError("❌ Nombre d'images Visible et SWIR non correspondant.")

    # # Vérification image de référence
    # if ref_image_path:
        # I_ref = load_image_ref_skimage(ref_image_path, as_gray=False, normalize=True)
        # if I_ref is None:
            # raise FileNotFoundError(f"❌ Impossible de charger l'image de référence : {ref_image_path}")
        # logger.info(f"✔ Image de référence détectée : {ref_image_path}")

    # # --- Étape 2 : Chargement paramètres ---
    # if params is None:
        # configs = load_all_configs()
        # params = configs["params"]   # on récupère directement parameters.json

    # # --- Étape 3 : Génération des tâches en mémoire ---
    # tasks = generate_tasks_in_memory(
        # visible_files, swir_files, ref_image_path,
        # params=params, run_detection=run_detection,
        # ground_truth_path=ground_truth_path,
        # save_output=save_output, output_dir=output_dir
    # )

    # total_batches = (len(tasks) + batch_size - 1) // batch_size
    # task_batches = batchify_tasks(tasks, batch_size)

    # logger.info(f"🧮 {len(tasks)} combinaisons générées en mémoire")
    
    # # Log du nombre de workers utilisés
    # num_workers = workers if workers is not None else 2 #os.cpu_count()
    # logger.info(f"⚙️ Exécution avec {num_workers} workers en parallèle.")

    # # --- Étape 4 : Exécution parallèle + sauvegarde DB ---
    # with Progress(
        # SpinnerColumn(),
        # TextColumn("[bold blue]{task.description}"),
        # BarColumn(bar_width=None),
        # "[progress.percentage]{task.percentage:>3.0f}%",
        # TimeElapsedColumn(),
        # TimeRemainingColumn(),
    # ) as progress:

        # batch_task = progress.add_task("Batchs", total=total_batches)

        # for batch in task_batches:
            # image_task = progress.add_task(
                # f"[green] ➤ Traitement batch {progress.tasks[batch_task].completed + 1}/{total_batches}",
                # total=len(batch)
            # )

            # results: list[ProcessResult] = []
            # with ProcessPoolExecutor(max_workers=workers) as executor:
                # for res in executor.map(process_image_wrapper, batch):
                    # results.append(res)
                    # progress.update(image_task, advance=1)

            # progress.remove_task(image_task)

            # # Sauvegarde DB
            # save_task = progress.add_task("[cyan] ✔ Enregistrement DB", total=len(results))
            # for res in results:
                # if res.error is None:
                    # save_result_to_db(
                        # session=session,
                        # visible_img=str(res.visible_path) if res.visible_path else None,
                        # swir_img=str(res.swir_path) if res.swir_path else None,
                        # ref_img=str(ref_image_path) if ref_image_path else None,
                        # grd_tr=str(res.ground_truth_path) if res.ground_truth_path else None,
                        # alpha=res.params["facteur_swir"],
                        # beta=res.params["beta"],
                        # level=res.params["level"],
                        # gamma=res.params["gamma_value"],
                        # metrics_dict_f=res.metrics_fusion,
                        # metrics_dict_v=res.metrics_visible,
                        # metrics_dict_s=res.metrics_swir,
                        # error=res.error
                    # )
                    # logger.debug(f"✔ {Path(res.visible_path).name} + {Path(res.swir_path).name} traitées avec succès.")
                # else:
                    # logger.warning(f"⚠ Erreur sur {res.visible_path} et {res.swir_path} : {res.error}")
                    # raise RuntimeError(f"Une erreur est survenue : {res.error}")

                # # Mise à jour des derniers paramètres (utile pour interruption/reprise)
                # # interruption.last_params.update(res.params)
                # progress.update(save_task, advance=1)

            # progress.remove_task(save_task)
            # progress.update(batch_task, advance=1)

    # session.close()
    # logger.info("✅ Traitement SQL terminé.")

# Version séquentiel et parallèle du code : (en cas de crash sur des gros jeu de données).
# src/processing/sql_runner.py
def process_folder_sql(
    visible_folder: Path,
    swir_folder: Path,
    output_dir: Path,
    batch_size: int = 10,
    ref_image_path: Optional[Path] = None,
    save_output: bool = False,
    run_detection: bool = True,
    ground_truth_path: Optional[Path] = None,
    params: dict | None = None,
    workers: int = 1  # Par défaut 1 pour la sécurité sur gros dataset
) -> None:
    """
    Process a folder of visible and SWIR images for SQL-based analysis with resume capability.
    Handles matching between images (e.g. 0000_rgb.jpg) and GT (e.g. 0000.xml).

    This function orchestrates the batch processing of image fusion tasks, computes metrics,
    and stores results directly into a SQLite database (`results.db`). It includes a robust
    "resume" feature that checks the database for previously processed images to avoid
    redundant calculations.

    Steps
    -----
    1. Create output directory and initialize SQLite database session.
    2. Query the database to identify images that have already been processed.
    3. Discover visible and SWIR image files in the provided folders.
    4. Filter out files that are already present in the database (Skip logic).
    5. Match remaining images with their corresponding Ground Truth files (if any).
    6. Generate tasks in memory for the remaining matched pairs.
    7. Execute tasks in batches.
    8. Save results into the database.

    Parameters
    ----------
    visible_folder : Path
        Path to the folder containing visible images.
    swir_folder : Path
        Path to the folder containing SWIR images.
    output_dir : Path
        Directory where the SQLite database and logs will be saved.
    batch_size : int, default=10
        Number of tasks to process per batch. A smaller size is recommended for stability
        on large datasets to allow frequent memory clearing and DB commits.
    ref_image_path : Path or None, optional
        Path to the reference image (optional).
    save_output : bool, default=False
        Whether to save intermediate outputs (images, annotations).
    run_detection : bool, default=True
        Whether to run YOLO detection in addition to metric computation.
    ground_truth_path : Path or None, optional
        Path to ground truth annotations (used if detection is enabled).
    params : dict or None, optional
        Fusion parameters to apply. If None, parameters are loaded from config.
    workers : int, default=1
        Number of parallel processes to use.
        - If set to 1 (default): Runs in sequential mode (Loop). Most stable for Windows/WSL and large images.
        - If set to > 1: Runs in parallel using ProcessPoolExecutor. Faster for small datasets.

    Returns
    -------
    None
        Results are inserted into the SQLite database.

    Raises
    ------
    ValueError
        If the number of visible and SWIR images does not match.
    FileNotFoundError
        If the reference image path is provided but cannot be loaded.

    Notes
    -----
    - Results are stored in `results.db` inside the output directory.
    - The function automatically detects the table name (e.g., `fusion_results`) to perform the skip check.
    - Progress is displayed in the console with a live progress bar.
    """
    
    logger.info("🚀 Démarrage du traitement SQL (version Quasar)...")
    
    # 1. Création dossier & DB
    output_dir.mkdir(parents=True, exist_ok=True)
    session = get_session(output_dir / "results.db")

    # --- ÉTAPE 0 : LOGIQUE DE REPRISE ---
    processed_filenames = set()
    try:
        query = text("SELECT visible_img FROM fusion_results")
        result = session.execute(query)
        for row in result:
            if row[0]:
                processed_filenames.add(Path(row[0]).name)
        
        if processed_filenames:
            logger.info(f"🔄 REPRISE DÉTECTÉE : {len(processed_filenames)} images déjà dans la base.")
            logger.info("👉 Ces images seront ignorées.")
            
    except Exception as e:
        logger.warning(f"ℹ️ Base neuve ou illisible. Démarrage à zéro.")

    # --- Étape 1 : Découverte des fichiers ---
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    visible_files = sorted([f for ext in image_extensions for f in visible_folder.glob(ext)])
    swir_files    = sorted([f for ext in image_extensions for f in swir_folder.glob(ext)])

    if len(visible_files) != len(swir_files):
        raise ValueError("❌ Nombre d'images Visible et SWIR non correspondant.")

    # --- Étape 2 : Filtrage ET Matching des GT ---
    
    # 2.1 Préparation du dictionnaire de GT pour recherche rapide
    gt_map = {}
    if ground_truth_path and run_detection:
        # On liste xml et txt
        all_gt_files = sorted([f for f in ground_truth_path.glob("*.xml")] + [f for f in ground_truth_path.glob("*.txt")])
        # On crée une map : "00000" -> Path(.../00000.xml)
        gt_map = {f.stem: f for f in all_gt_files}
        logger.info(f"📂 GT trouvés dans le dossier : {len(all_gt_files)}")

    # 2.2 Construction des listes synchronisées
    final_visible = []
    final_swir = []
    final_gt = []

    skipped_count = 0
    
    for v_file, s_file in zip(visible_files, swir_files):
        # A. Skip si déjà fait
        if v_file.name in processed_filenames:
            skipped_count += 1
            continue
        
        # B. Gestion du GT (Matching Intelligent)
        if ground_truth_path and run_detection:
            # 1. Tentative Exacte (ex: '00000' -> '00000.xml')
            match = gt_map.get(v_file.stem)
            
            # 2. Tentative Nettoyage Suffixe (ex: '00000_rgb' -> '00000.xml')
            if not match:
                clean_stem = v_file.stem
                # Liste des suffixes courants à supprimer pour trouver le GT
                for suffix in ["_rgb", "_vis", "_visible", "_swir"]:
                    if clean_stem.endswith(suffix):
                        clean_stem = clean_stem.replace(suffix, "")
                        break
                match = gt_map.get(clean_stem)

            if match:
                final_visible.append(v_file)
                final_swir.append(s_file)
                final_gt.append(match)
            else:
                # Pas de GT trouvé -> On loggue et on ignore l'image (pour éviter le crash plus loin)
                logger.warning(f"⚠️ Ignorée (Pas de GT correspondant) : {v_file.name} (Cherché: {v_file.stem} ou {clean_stem if 'clean_stem' in locals() else '?'})")
        else:
            # Pas de détection demandée
            final_visible.append(v_file)
            final_swir.append(s_file)
    
    if not final_visible:
        logger.info("✅ Toutes les images sont déjà traitées ! (100% Complete)")
        session.close()
        return

    logger.info(f"📊 Bilan : {skipped_count} ignorées | {len(final_visible)} à traiter.")

    # --- Étape 3 : Params & Ref ---
    if ref_image_path:
        I_ref = load_image_ref_skimage(ref_image_path, as_gray=False, normalize=True)
    else:
        I_ref = None

    if params is None:
        configs = load_all_configs()
        params = configs["params"]

    # --- Étape 4 : Génération des tâches ---
    gt_arg = final_gt if (ground_truth_path and run_detection) else None

    tasks = generate_tasks_in_memory(
        final_visible, 
        final_swir, 
        ref_image_path,
        params=params, 
        run_detection=run_detection,
        ground_truth_path=gt_arg, 
        save_output=save_output, 
        output_dir=output_dir
    )

    task_batches = list(batchify_tasks(tasks, batch_size))
    total_batches = len(task_batches)

    # --- Étape 5 : Exécution ---
    is_sequential = workers is None or workers <= 1
    mode_str = "SÉQUENTIEL (Stable)" if is_sequential else f"PARALLÈLE ({workers} workers)"
    logger.info(f"⚙️ Mode exécution : {mode_str}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:

        batch_task = progress.add_task("Batchs", total=total_batches)

        for batch in task_batches:
            image_task = progress.add_task(
                f"[green] ➤ Batch {progress.tasks[batch_task].completed + 1}/{total_batches}", 
                total=len(batch)
            )
            results = []

            if is_sequential:
                # Boucle simple
                for task in batch:
                    try:
                        res = process_image_wrapper(task)
                        results.append(res)
                        progress.update(image_task, advance=1)
                    except Exception as e:
                        logger.error(f"❌ Erreur sur {task.visible_path.name} : {e}")
            else:
                # Parallèle
                with ProcessPoolExecutor(max_workers=workers) as executor:
                    for res in executor.map(process_image_wrapper, batch):
                        results.append(res)
                        progress.update(image_task, advance=1)

            progress.remove_task(image_task)

            # Sauvegarde
            save_task = progress.add_task("Save DB", total=len(results))
            for res in results:
                if res.error is None:
                    save_result_to_db(
                        session=session,
                        visible_img=str(res.visible_path),
                        swir_img=str(res.swir_path),
                        ref_img=str(ref_image_path) if ref_image_path else None,
                        grd_tr=str(res.ground_truth_path) if res.ground_truth_path else None,
                        alpha=res.params["facteur_swir"],
                        beta=res.params["beta"],
                        level=res.params["level"],
                        gamma=res.params["gamma_value"],
                        metrics_dict_f=res.metrics_fusion,
                        metrics_dict_v=res.metrics_visible,
                        metrics_dict_s=res.metrics_swir,
                        error=res.error
                    )
                else:
                    logger.warning(f"⚠ Erreur sur {res.visible_path.name} : {res.error}")

                    error_msg = (
                        f"[bold red]❌ ÉCHEC SUR L'IMAGE :[/] [yellow]{Path(res.visible_path).name}[/]\n"
                        f"   └─ SWIR : {Path(res.swir_path).name}\n"
                        f"   └─ Erreur : [red]{res.error}[/]"
                    )
                    progress.console.print(error_msg)
                    
                    # raise RuntimeError(f"Une erreur est survenue : {res.error}")
                
                # Mise à jour des derniers paramètres (utile pour interruption/reprise)
                # interruption.last_params.update(res.params)
                progress.update(save_task, advance=1)
            
            session.commit()
            progress.remove_task(save_task)
            progress.update(batch_task, advance=1)

    session.close()
    logger.info("✅ Traitement terminé avec succès.")