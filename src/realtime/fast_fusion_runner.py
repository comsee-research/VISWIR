"""
Fast/real-time pipeline runner for quick VISWIR execution.
"""

# src/fast/fast_fusion_runner.py
from pathlib import Path
from typing import Optional
import numpy as np
from skimage import io, img_as_float

# from realtime.fast_config import load_fast_config
from realtime.fast_detection import FastDetector
from fusion.fusion import viswir_core_fusion

def read_image_visible(path: Path) -> np.ndarray:
    """
    Read a visible (RGB) image and convert it to float64 in [0, 1].

    Parameters
    ----------
    path : Path
        Path to the visible image file.

    Returns
    -------
    numpy.ndarray
        RGB image as float64 normalized to [0, 1].
    """
    return img_as_float(io.imread(path))  # RGB float64 [0,1]

def read_image_swir(path: Path) -> np.ndarray:
    """
    Read a SWIR (Short-Wave Infrared) image and convert it to float64 in [0, 1].

    Parameters
    ----------
    path : Path
        Path to the SWIR image file.

    Returns
    -------
    numpy.ndarray
        Grayscale image as float64 normalized to [0, 1].
    """
    return img_as_float(io.imread(path, as_gray=True))  # Grayscale float64 [0,1]

class FastFusionPipeline:
    """
    Fast pipeline for VISWIR image fusion with optional YOLO detection.

    This class loads configuration parameters, initializes the YOLO detector
    if required, and provides a ``run`` method to fuse visible and SWIR images.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary containing:

        * **base** (dict) – Base fusion parameters
          (facteur_swir, beta, level, apply_gamma, gamma_value).
        * **yolo** (dict) – YOLO detection parameters
          (model_path, thresholds, device, allowed_classes).

    Attributes
    ----------
    cfg : dict
        Full configuration dictionary.
    detector : FastDetector or None
        YOLO detector instance if detection is enabled, otherwise None.
    run_detection : bool
        Whether detection is enabled (from config).
    defaults : dict
        Default fusion parameters loaded from config.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        yolo_cfg = cfg["yolo"]

        self.detector = None
        self.run_detection = bool(cfg["base"].get("run_detection", False))  # pris du YAML
        if self.run_detection:
            model_path = Path(yolo_cfg["model_path"])
            self.detector = FastDetector(
                model_path=model_path,
                conf_thres=yolo_cfg.get("confidence_threshold", 0.25),
                iou_thres=yolo_cfg.get("iou_threshold", 0.3),
                device=yolo_cfg.get("device", "cpu"),
                allowed_classes=yolo_cfg.get("allowed_classes", [])
            )

        self.defaults = {
            "facteur_swir": float(self.cfg["base"].get("facteur_swir", 0.5)),
            "beta": float(self.cfg["base"].get("beta", 2.0)),
            "level": int(self.cfg["base"].get("level", 4)),
            "apply_gamma": bool(self.cfg["base"].get("apply_gamma", True)),
            "gamma_value": float(self.cfg["base"].get("gamma_value", 1.0)),
        }

    def run(self, visible_path: Path, swir_path: Path,
            override_params: Optional[dict] = None) -> np.ndarray:
        """
        Run the fusion pipeline on a pair of visible and SWIR images.

        Parameters
        ----------
        visible_path : Path
            Path to the visible (RGB) image.
        swir_path : Path
            Path to the SWIR (grayscale) image.
        override_params : dict, optional
            Dictionary of parameters to override defaults.

        Returns
        -------
        numpy.ndarray
            Fused RGB image as float64 normalized to [0, 1].
            If detection is enabled, bounding boxes and labels are drawn.
        """
        I1_RGB = read_image_visible(visible_path)
        I2 = read_image_swir(swir_path)

        p = {**self.defaults, **(override_params or {})}
        _, I_out = viswir_core_fusion(
            I1_RGB=I1_RGB,
            I2=I2,
            facteur_swir=p["facteur_swir"],
            beta=p["beta"],
            level=p["level"],
            apply_gamma=p["apply_gamma"],
            gamma_value=p["gamma_value"]
        )

        fused_rgb = np.clip(I_out, 0, 1)

        if self.run_detection and self.detector is not None:
            results = self.detector.predict((fused_rgb*255).astype(np.uint8))
            fused_rgb = self.detector.draw(fused_rgb, results)

        return fused_rgb
