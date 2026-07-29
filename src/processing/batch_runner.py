"""
Batch processing orchestration for running VISWIR on directories.
"""



from pathlib import Path
import os, csv, gc
import traceback

from concurrent.futures import ProcessPoolExecutor
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

from fusion.fusion import process_image
from fusion.detection_module import run_yolo_and_compute_f1
from fusion.metrics import compute_all_metrics, no_ref_metrics, full_ref_metrics
from fusion.utils import load_image_ref_skimage
from common.logger import logger
from common.datatypes import FusionTask, ProcessResult
from processing.task_manager import generate_tasks_in_memory, batchify_tasks

# def process_image_wrapper(args) -> ProcessResult:
#     """
#     Exécute la fusion d'une paire d'images avec paramètres donnés,
#     calcule les métriques et lance éventuellement la détection.
#     Retourne un ProcessResult.
#     """
#     (
#         visible_path, swir_path, ref_path, save_output, output_dir,
#         run_detection, ground_truth_path, facteur_swir, beta, level,
#         apply_gamma, gamma_value
#     ) = args

def process_image_wrapper(task: FusionTask) -> ProcessResult:
    """
    Execute the fusion of a pair of images from a FusionTask, compute metrics,
    and optionally run detection. Returns a ProcessResult object.

    This function performs the following steps:

    1. Fusion of visible and SWIR images using the provided parameters.
    2. Loading of the reference image (if available).
    3. Computation of quality metrics (fusion, visible, SWIR).
    4. Optional YOLOv8 detection and F1-score computation.
    5. Return of results in a ProcessResult object.

    Parameters
    ----------
    task : FusionTask
        Task object containing:

        * **visible_path** (str) - Path to the visible image.
        * **swir_path** (str) - Path to the SWIR image.
        * **ref_image_path** (str or None) - Path to the reference image (optional).
        * **ground_truth_path** (str or None) - Path to ground truth annotations (optional).
        * **params** (dict) - Fusion parameters (facteur_swir, beta, level, apply_gamma, gamma_value).
        * **save_output** (bool) - Whether to save intermediate results.
        * **run_detection** (bool) - Whether to run YOLO detection.
        * **output_dir** (str or Path) - Directory for saving outputs.

    Returns
    -------
    ProcessResult
        Object containing:

        * **visible_path** (str)
        * **swir_path** (str)
        * **ground_truth_path** (str or None)
        * **params** (dict) - Fusion parameters used.
        * **metrics_fusion** (dict) - Metrics computed on the fused image.
        * **metrics_visible** (dict) - Metrics computed on the visible image.
        * **metrics_swir** (dict) - Metrics computed on the SWIR image.
        * **error** (str or None) - Error message if the process failed.

    Notes
    -----
    - If fusion fails, returns a ProcessResult with the error message.
    - If detection is enabled, YOLOv8 is run on fused, visible, and SWIR images.
    - Memory cleanup is performed at the end to avoid leaks.
    """
    
    visible_path = task.visible_path
    swir_path = task.swir_path
    ref_path = task.ref_image_path
    ground_truth_path = task.ground_truth_path
    save_output = task.save_output
    run_detection = task.run_detection
    output_dir = task.output_dir
    params = task.params

    facteur_swir = params["facteur_swir"]
    beta = params["beta"]
    level = params["level"]
    apply_gamma = params["apply_gamma"]
    gamma_value = params["gamma_value"]

    # Initialisation pour éviter les UnboundLocalError
    I5 = I_out = I_visible = I_swir = I_ref = None

    try:
        logger.debug(
            f"→ START: {Path(visible_path).name} | "
            f"facteur SWIR={facteur_swir:.2f}, β={beta:.2f}, "
            f"level={level}, gamma={apply_gamma}:{gamma_value:.2f}"
        )

        # --- Étape 1 : Fusion ---
        I5, I_out, error = process_image(
            visible_path=visible_path,
            swir_path=swir_path,
            facteur_swir=facteur_swir,
            beta=beta,
            level=level,
            apply_gamma=apply_gamma,
            gamma_value=gamma_value,
            save_output=save_output,
            output_dir=output_dir
        )
        if error is not None:
            logger.error(f"❌ Fusion échouée pour {visible_path} : {error}")
            return ProcessResult(
                visible_path=visible_path,
                swir_path=swir_path,
                ground_truth_path=ground_truth_path,
                params={"facteur_swir": facteur_swir, "beta": beta,
                        "level": level, "apply_gamma": apply_gamma,
                        "gamma_value": gamma_value},
                error=error
            )

        # --- Étape 2 : Chargement référence ---
        I_ref = None
        if ref_path and Path(ref_path).exists():
            I_ref = load_image_ref_skimage(ref_path, as_gray=False, normalize=True)

        # --- Étape 3 : Calcul métriques ---
        metrics_fusion = compute_all_metrics(I_ref=I_ref, I_fused=I_out)
        I_visible = load_image_ref_skimage(visible_path, as_gray=False, normalize=True)
        I_swir = load_image_ref_skimage(swir_path, as_gray=False, normalize=True)
        metrics_visible = compute_all_metrics(I_ref=I_ref, I_fused=I_visible)
        metrics_swir = compute_all_metrics(I_ref=I_ref, I_fused=I_swir)

        # --- Étape 4 : Détection (optionnelle) ---
        if run_detection:
            try:
                det_fused = run_yolo_and_compute_f1(
                    I_out, ground_truth_path, output_dir=output_dir,
                    save_output=save_output, mode="fusion", image_filename=visible_path
                )
                if det_fused:
                    metrics_fusion.update(det_fused)
            except Exception as det_error:
                logger.warning(f"⚠️ Erreur détection fusion : {det_error}")

            try:
                det_visible = run_yolo_and_compute_f1(
                    I_visible, ground_truth_path, output_dir=output_dir,
                    save_output=save_output, mode="visible", image_filename=visible_path
                )
                if det_visible:
                    metrics_visible.update(det_visible)
            except Exception as det_error:
                logger.warning(f"⚠️ Erreur détection visible : {det_error}")

            try:
                det_swir = run_yolo_and_compute_f1(
                    I_swir, ground_truth_path, output_dir=output_dir,
                    save_output=save_output, mode="swir", image_filename=swir_path
                )
                if det_swir:
                    metrics_swir.update(det_swir)
            except Exception as det_error:
                logger.warning(f"⚠️ Erreur détection SWIR : {det_error}")
        else:
            logger.debug("🛑 Skipping detection step (run_detection=False)")

        # --- Étape 5 : Retour ---
        return ProcessResult(
            visible_path=visible_path,
            swir_path=swir_path,
            ground_truth_path=ground_truth_path,
            params={"facteur_swir": facteur_swir, "beta": beta,
                    "level": level, "apply_gamma": apply_gamma,
                    "gamma_value": gamma_value},
            metrics_fusion=metrics_fusion,
            metrics_visible=metrics_visible,
            metrics_swir=metrics_swir
        )

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"🔥 Exception dans process_image_wrapper : {traceback_str}")
        return ProcessResult(
            visible_path=visible_path,
            swir_path=swir_path,
            ground_truth_path=ground_truth_path,
            params={"facteur_swir": facteur_swir, "beta": beta,
                    "level": level, "apply_gamma": apply_gamma,
                    "gamma_value": gamma_value},
            error=str(e)
        )

    finally:
        # Nettoyage mémoire
        del I5, I_out, I_visible, I_swir, I_ref
        gc.collect()


def process_folder(visible_folder: str, swir_folder: str, output_dir: str,
                   batch_size: int = 20, ref_image_path: str | None = None,
                   params: dict | None = None, run_detection: bool = False,
                   ground_truth_path: str | None = None, save_output: bool = True):
    """
    Orchestrate batch processing of image fusion tasks and save results to CSV.

    This function performs the following steps:
    1. Discover visible and SWIR image files in the provided folders.
    2. Validate the number of images and optionally load a reference image.
    3. Generate fusion tasks with the given parameters.
    4. Execute tasks in parallel batches with progress tracking.
    5. Save computed metrics to a CSV file.

    Parameters
    ----------
    visible_folder : str
        Path to the folder containing visible images.
    swir_folder : str
        Path to the folder containing SWIR images.
    output_dir : str
        Directory where results (CSV and optional outputs) will be saved.
    batch_size : int, default=20
        Number of tasks to process in parallel per batch.
    ref_image_path : str or None, optional
        Path to the reference image (optional).
    params : dict or None, optional
        Fusion parameters to apply. If None, defaults are used.
    run_detection : bool, default=False
        Whether to run YOLO detection in addition to metric computation.
    ground_truth_path : str or None, optional
        Path to ground truth annotations (used if detection is enabled).
    save_output : bool, default=True
        Whether to save intermediate outputs (images, annotations).

    Returns
    -------
    None
        Results are written to a CSV file in the output directory.

    Raises
    ------
    ValueError
        If the number of visible and SWIR images does not match.
    FileNotFoundError
        If the reference image path is provided but cannot be loaded.

    Notes
    -----
    - The CSV file is saved as `combinations.csv` in the output directory.
    - Metrics include both no-reference and full-reference metrics.
    - Progress is displayed in the console with a live progress bar.
    """

    logger.info("🚀 Démarrage du traitement de fusion d'images...")
    logger.info(f"📂 Dossier Visible : {visible_folder}")
    logger.info(f"📂 Dossier SWIR : {swir_folder}")
    logger.info(f"📁 Dossier de sortie : {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # --- Étape 1 : Découverte des fichiers ---
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    visible_files, swir_files = [], []
    for ext in image_extensions:
        visible_files.extend(Path(visible_folder).glob(ext))
        swir_files.extend(Path(swir_folder).glob(ext))

    visible_files, swir_files = sorted(visible_files), sorted(swir_files)

    if len(visible_files) != len(swir_files):
        raise ValueError("❌ Le nombre d'images Visible et SWIR ne correspond pas.")

    # Vérification image de référence
    if ref_image_path:
        I_ref = load_image_ref_skimage(ref_image_path, as_gray=False, normalize=True)
        if I_ref is None:
            raise FileNotFoundError(f"❌ Impossible de charger l'image de référence : {ref_image_path}")
        logger.info(f"✔ Image de référence détectée : {ref_image_path}")

    # --- Étape 2 : Génération des tâches ---
    tasks = generate_tasks_in_memory(
        visible_files, swir_files, ref_image_path,
        params=params, run_detection=run_detection,
        ground_truth_path=ground_truth_path, save_output=save_output,
        output_dir=output_dir
    )

    total_tasks = len(tasks)
    logger.info(f"🧮 Génération de {total_tasks} combinaisons à traiter")
    task_batches = list(batchify_tasks(tasks, batch_size))
    total_batches = len(task_batches)

    # --- Étape 3 : Préparation CSV ---
    csv_path = Path(output_dir) / "combinations.csv"
    metric_names = list(no_ref_metrics.keys()) + list(full_ref_metrics.keys())
    header = ['visible_path', 'swir_path', 'ref_image_path',
              'facteur_swir', 'beta', 'level', 'apply_gamma', 'gamma_value'] + metric_names

    with open(csv_path, "w", newline='', encoding="utf-8-sig", buffering=1) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        # --- Étape 4 : Exécution parallèle ---
        with Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"),
                      BarColumn(bar_width=None), "[progress.percentage]{task.percentage:>3.0f}%",
                      TimeElapsedColumn(), TimeRemainingColumn()) as progress:

            batch_task = progress.add_task("Batchs", total=total_batches)

            for batch in task_batches:
                image_task = progress.add_task(
                    f"[green] ➤ Traitement batch {progress.tasks[batch_task].completed + 1}/{total_batches}",
                    total=len(batch)
                )

                results: list[ProcessResult] = []
                with ProcessPoolExecutor() as executor:
                    for res in executor.map(process_image_wrapper, batch):
                        results.append(res)
                        progress.update(image_task, advance=1)

                progress.remove_task(image_task)

                # --- Étape 5 : Sauvegarde des résultats ---
                save_task = progress.add_task("[cyan] ✔ Enregistrement des résultats", total=len(results))
                for res in results:
                    if res.error is None:
                        row = [
                            res.visible_path, res.swir_path, ref_image_path,
                            res.params["facteur_swir"], res.params["beta"], res.params["level"],
                            res.params["apply_gamma"], res.params["gamma_value"]
                        ]
                        row += [res.metrics_fusion.get(m) for m in metric_names]
                        writer.writerow(row)
                        csv_file.flush()
                    else:
                        logger.warning(f"⚠ Erreur sur {res.visible_path} et {res.swir_path} : {res.error}")
                    progress.update(save_task, advance=1)

                progress.remove_task(save_task)
                progress.update(batch_task, advance=1)

    logger.info("✅ Traitement terminé.")
