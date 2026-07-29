"""
Wrapper helper classes and functions for Optuna trials.
"""

from common.datatypes import FusionTask, OptunaResult
from fusion.fusion import process_image
from fusion.metrics import compute_all_metrics
from fusion.detection_module import run_yolo_and_compute_f1

import gc
import traceback

from common.logger import logger


# ============================================================
# Wrapper Optuna
# ============================================================

def process_image_wrapper_optuna(task: FusionTask) -> OptunaResult:
    """
    Optimized wrapper for Optuna trials.
    Only computes metrics on the fused image (NR-IQA or detection if enabled).

    Parameters
    ----------
    task : FusionTask
        Fusion task with parameters and paths.

    Returns
    -------
    OptunaResult
        Simplified result containing only fusion metrics.
    """
    visible_path = task.visible_path
    swir_path = task.swir_path
    params = task.params
    run_detection = task.run_detection
    save_output = False  # Optuna never saves outputs
    output_dir = None

    try:
        # --- Fusion ---
        I5, I_out, error = process_image(
            visible_path=visible_path,
            swir_path=swir_path,
            facteur_swir=params["facteur_swir"],
            beta=params["beta"],
            level=params["level"],
            apply_gamma=params["apply_gamma"],
            gamma_value=params["gamma_value"],
            save_output=save_output,
            output_dir=output_dir
        )
        if error is not None:
            return OptunaResult(
                visible_path=visible_path,
                swir_path=swir_path,
                params=params,
                error=error
            )

        # --- Metrics fusion only ---
        metrics_fusion = compute_all_metrics(I_ref=None, I_fused=I_out)

        # --- Detection (fusion only) ---
        if run_detection:
            try:
                det_fused = run_yolo_and_compute_f1(
                    I_out, task.ground_truth_path,
                    output_dir=output_dir,
                    save_output=save_output,
                    mode="fusion",
                    image_filename=visible_path
                )
                if det_fused:
                    metrics_fusion.update(det_fused)
            except Exception as det_error:
                logger.warning(f"⚠️ Erreur détection fusion : {det_error}")

        return OptunaResult(
            visible_path=visible_path,
            swir_path=swir_path,
            params=params,
            metrics_fusion=metrics_fusion
        )

    except Exception as e:
        traceback_str = traceback.format_exc()
        logger.error(f"🔥 Exception dans process_image_wrapper_optuna : {traceback_str}")
        return OptunaResult(
            visible_path=visible_path,
            swir_path=swir_path,
            params=params,
            error=str(e)
        )

    finally:
        del I5, I_out
        gc.collect()