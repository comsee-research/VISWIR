"""
Real-time object detection module for the fast pipeline.
"""

# src/fast/fast_detection.py
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
from skimage.draw import rectangle_perimeter, set_color
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class FastDetector:
    """
    Fast object detector using Ultralytics YOLO.

    This class wraps a YOLO model for quick inference and visualization.
    It supports filtering by allowed classes and drawing bounding boxes
    with labels on images.

    Parameters
    ----------
    model_path : Path
        Path to the YOLO model weights.
    conf_thres : float, default=0.25
        Confidence threshold for predictions.
    iou_thres : float, default=0.3
        IoU threshold for non-maximum suppression.
    device : str, default="cpu"
        Device to run inference on ("cpu" or "cuda").
    allowed_classes : Sequence[str], optional
        List of class names to allow. If None, all classes are allowed.

    Raises
    ------
    RuntimeError
        If Ultralytics YOLO is not installed.
    FileNotFoundError
        If the YOLO weights file does not exist.
    """
    def __init__(self, model_path: Path, conf_thres=0.25, iou_thres=0.3,
                 device="cpu", allowed_classes: Optional[Sequence[str]]=None):
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not installed")

        self.model_path = Path(model_path)
        if not self.model_path.is_absolute():
            self.model_path = Path(__file__).resolve().parent.parent.parent / "config" / self.model_path

        if not self.model_path.exists():
            from common.logger import logger
            logger.warning(f"⚠️ YOLO weights not found locally at {self.model_path}. Trying to download automatically...")
            self.model = YOLO(self.model_path.name)
        else:
            self.model = YOLO(str(self.model_path))
        self.conf = conf_thres
        self.iou = iou_thres
        self.device = device
        self.allowed = set(allowed_classes or [])

        self.class_colors = {
            "person": (1, 0, 0),       # rouge
            "car": (0, 1, 0),          # vert
            "bus": (0, 0, 1),          # bleu
            "truck": (1, 1, 0),        # jaune
            "motorcycle": (1, 0, 1),   # magenta
            "bicycle": (0, 1, 1),      # cyan
        }

    def predict(self, img_rgb: np.ndarray):
        """
        Run YOLO prediction on an RGB image.

        Parameters
        ----------
        img_rgb : numpy.ndarray
            Input image in RGB format, normalized to [0, 1] or [0, 255].

        Returns
        -------
        list
            List of YOLO detection results.
        """
        return self.model.predict(img_rgb, conf=self.conf, iou=self.iou, device=self.device, verbose=False)
    
    @staticmethod
    def draw_label(img_rgb: np.ndarray, x1, y1, name, color=(0,255,0)):
        """
        Draw a text label on an image at the given coordinates.
        (Static method)

        Parameters
        ----------
        img_rgb : numpy.ndarray
            Input RGB image (float in [0, 1]).
        x1 : int
            X-coordinate of the label position.
        y1 : int
            Y-coordinate of the label position.
        name : str
            Class name to display.
        color : tuple of int, default=(0, 255, 0)
            RGB color of the text.

        Returns
        -------
        numpy.ndarray
            Image with the label drawn (float in [0, 1]).
        """
        # Convertir en image PIL
        img_pil = Image.fromarray((img_rgb * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_pil)

        # Police par défaut
        font = ImageFont.load_default()

        # Dessiner le texte
        draw.text((x1, max(0, y1 - 12)), name, font=font, fill=color)

        # Retourner en numpy [0,1]
        return np.array(img_pil) / 255.0

    def draw(self, img_rgb: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes and labels on an image based on YOLO results.

        Parameters
        ----------
        img_rgb : numpy.ndarray
            Input RGB image (float in [0, 1]).
        results : list
            YOLO detection results containing bounding boxes and class IDs.

        Returns
        -------
        numpy.ndarray
            Image with bounding boxes and labels drawn (float in [0, 1]).

        Notes
        -----
        - Bounding boxes are drawn using `skimage.draw.rectangle_perimeter`.
        - Labels are drawn using Pillow.
        - Colors are assigned per class (e.g., person=green, others=red).
        """
        out = img_rgb.copy()
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                name = r.names.get(cls_id, str(cls_id))
                if self.allowed and name not in self.allowed:
                    continue

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                # Couleur différente par classe
                color = (0, 255, 0) if name == "person" else (255, 0, 0)

                # Rectangle avec skimage
                rr, cc = rectangle_perimeter(start=(y1, x1), end=(y2, x2), shape=out.shape)
                set_color(out, (rr, cc), np.array(color) / 255.0)

                # Texte avec Pillow
                out = FastDetector.draw_label(out, x1, y1, name, color=color)

        return out
