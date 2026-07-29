"""
Configuration loader and classes for the fast real-time VISWIR pipeline.
"""

# src/fast/fast_config.py
from pathlib import Path
import json
import yaml

def load_fast_config(project_root: Path | None = None) -> dict:
    """
    Load the minimal configuration for the fast pipeline.

    This function loads:

    - Base fusion parameters from ``config/fast_config.yaml``.
    - YOLO detection parameters from ``config/yolo_config.json`` if available,
      otherwise falls back to default YOLO settings.

    Parameters
    ----------
    project_root : Path or None, optional
        Root directory of the project. If None, it is inferred automatically
        by traversing up from the current file location.

    Returns
    -------
    dict
        Dictionary containing the following keys:

        * **base** (dict) – Base fusion parameters loaded from YAML.
        * **yolo** (dict) – YOLO detection parameters (from JSON or defaults).
        * **paths** (dict) – Paths used in the configuration:
          
          - ``project_root`` (str)
          - ``config_dir`` (str)

    Notes
    -----
    - If ``yolo_config.json`` does not exist, default YOLO parameters are used:
      model path, confidence threshold, IoU threshold, device, and allowed classes.
    - Ensures consistent configuration loading for both fusion and detection.
    """
    if project_root is None:
        # src/fast/fast_config.py → src/fast → src → project root
        project_root = Path(__file__).resolve().parent.parent.parent

    cfg_dir = project_root / "config"

    # Base fusion params
    base_cfg_path = cfg_dir / "fast_config.yaml"
    with open(base_cfg_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    # YOLO params (optional)
    yolo_cfg_path = cfg_dir / "yolo_config.json"
    if yolo_cfg_path.exists():
        with open(yolo_cfg_path, "r", encoding="utf-8") as f:
            yolo_cfg = json.load(f)
    else:
        yolo_cfg = {
            "model_path": "yolov8x-seg.pt",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.3,
            "device": "cpu",
            "allowed_classes": ["truck", "person", "bus", "motorcycle", "bicycle", "car"]
        }

    return {
        "base": base_cfg,
        "yolo": yolo_cfg,
        "paths": {
            "project_root": str(project_root),
            "config_dir": str(cfg_dir)
        }
    }
