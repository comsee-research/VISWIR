"""
Object detection and evaluation (YOLO, precision, recall, F1, IoU) for VISWIR.
"""

# detection_module.py

import os
import gc
import cv2
import numpy as np
from pathlib import Path
import json

from datetime import datetime
import uuid

from ultralytics import YOLO
# from ultralytics.utils.ops import scale_boxes
# from ultralytics.engine.results import Boxes
# import torch

import xml.etree.ElementTree as ET
from xml.dom import minidom
# from sklearn.metrics import precision_score, recall_score, f1_score

from fusion.utils import get_image_shape#, save_float64_image_as_uint16
from common.logger import logger

def prepare_image_for_yolo(image_input, mode="default"): # New version
    """
    Prepare an image for YOLOv8 prediction.

    This function accepts either a file path (str or Path) or a NumPy array.
    It handles specific preprocessing for different modes:
    - "swir": converts single-channel SWIR images into 3-channel pseudo-RGB.
    - "visible": ensures correct RGB ↔ BGR conversion.

    Parameters
    ----------
    image_input : str, Path, or np.ndarray
        Input image, either as a file path or a NumPy array.
    mode : str, default="default"
        Processing mode. Options:
        - "swir": preprocess SWIR images.
        - "visible": preprocess visible images.
        - "default": no special preprocessing.

    Returns
    -------
    np.ndarray
        Preprocessed image in BGR format, dtype=uint8.

    Raises
    ------
    ValueError
        If the image cannot be read or has an invalid shape.
    TypeError
        If the input type is unsupported.
    """

    # === Charger l'image si c'est un chemin ===
    if isinstance(image_input, (str, os.PathLike)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"❌ Image non lisible à l'emplacement : {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError("❌ Entrée non supportée. Fournis un chemin ou un tableau NumPy.")

    # === Traitement spécifique pour SWIR : image 2D float64 ===
    if mode == "swir":
        if img.ndim == 2:
            logger.debug("🌀 SWIR détectée : duplication canaux → faux RGB")
            img = np.stack([img]*3, axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            logger.debug("🌀 SWIR (3D, 1 canal) → duplication")
            img = np.repeat(img, 3, axis=2)
        else:
            logger.debug("ℹ️ Image SWIR déjà 3 canaux")

    # === Vérification du format final attendu ===
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"❌ Image invalide (attendu 3 canaux) : shape={img.shape}")


    # === Conversion float64 normalisé → uint8 ===
    if img.dtype == np.float64 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)

    # === Convertir RGB → BGR si nécessaire (mode visible) ===
    # if mode == "fusion":
    if mode == "visible":
        logger.debug("🎨 Mode 'fusion' : inversion RGB ↔ BGR")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # logger.debug(f"📐 Shape finale de l'image : {img.shape}, dtype : {img.dtype}")

    return img

def save_annotated_float64_image_as_uint16(path, float64_img, vis_img):
    """
    Overlay annotations on a float64 image and save the result as uint16.

    The function takes a normalized float64 image (values in [0, 1]),
    overlays annotations from a visualization image, and saves the
    result as a uint16 image.

    Parameters
    ----------
    path : str or Path
        Path where the annotated image will be saved.
    float64_img : np.ndarray
        Original normalized float64 image (values in [0, 1]).
    vis_img : np.ndarray
        Visualization image containing annotations.

    Raises
    ------
    ValueError
        If the input image is not a normalized float64 array.
    """
    if float64_img.dtype != np.float64 or float64_img.max() <= 1.0:
        raise ValueError("Image d’origine attendue en float64 normalisée [0, 1]")

    # Convertir l'image float64 en uint16
    base_uint16 = (float64_img * 65535).astype(np.uint16)

    # Redimensionner l’image d’annotations à la même taille, si besoin
    vis_img_resized = cv2.resize(vis_img, (base_uint16.shape[1], base_uint16.shape[0]))

    # Convertir les annotations en niveaux de gris ou masque
    vis_gray = cv2.cvtColor(vis_img_resized, cv2.COLOR_BGR2GRAY)
    vis_mask = vis_gray > 10  # Seuil empirique pour identifier les zones annotées

    # Fusion : on remplace les pixels dans la couche rouge par les annotations
    if len(base_uint16.shape) == 2:
        base_uint16[vis_mask] = 65535  # Si image grayscale
    else:
        base_uint16[vis_mask, 0] = 0       # B
        base_uint16[vis_mask, 1] = 0       # G
        base_uint16[vis_mask, 2] = 65535   # R, on force les annotations en rouge vif

    # Sauvegarde
    cv2.imwrite(str(path), base_uint16)

def parse_voc_annotations(xml_path):
    """
    Parse Pascal VOC XML annotations and extract bounding boxes.

    Parameters
    ----------
    xml_path : str or Path
        Path to the Pascal VOC XML annotation file.

    Returns
    -------
    list of list of int
        List of bounding boxes in the format [xmin, ymin, xmax, ymax].
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])

    return boxes

def iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Parameters
    ----------
    box1 : list of int
        First bounding box [xmin, ymin, xmax, ymax].
    box2 : list of int
        Second bounding box [xmin, ymin, xmax, ymax].

    Returns
    -------
    float
        IoU value between the two bounding boxes.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

def save_predictions_as_voc_xml(result, image_shape, save_path, image_filename):
    """
    Save YOLO predictions in Pascal VOC XML format.

    Parameters
    ----------
    result : ultralytics.engine.results.Results
        YOLO prediction result object containing bounding boxes and masks.
    image_shape : tuple of int
        Shape of the image as (height, width, depth).
    save_path : str or Path
        Path where the XML file will be saved.
    image_filename : str
        Name of the image file associated with the predictions.

    Notes
    -----
    - Bounding boxes and class names are extracted from YOLO results.
    - If segmentation masks are available, polygon coordinates are also saved.
    - The output XML follows the Pascal VOC annotation format.
    """

    height, width, depth = image_shape
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "VOC"
    ET.SubElement(annotation, "filename").text = image_filename
    ET.SubElement(annotation, "path").text = image_filename

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "roboflow.ai"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    if result.boxes is not None:
        boxes_data = result.boxes.data.cpu().numpy()
        for idx, box in enumerate(boxes_data):
            x1, y1, x2, y2, conf, cls_id = box[:6]
            cls_name = result.names[int(cls_id)]

            obj_tag = ET.SubElement(annotation, "object")
            ET.SubElement(obj_tag, "name").text = cls_name
            ET.SubElement(obj_tag, "pose").text = "Unspecified"
            ET.SubElement(obj_tag, "truncated").text = "0"
            ET.SubElement(obj_tag, "difficult").text = "0"
            ET.SubElement(obj_tag, "occluded").text = "0"

            bndbox = ET.SubElement(obj_tag, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(int(x1))
            ET.SubElement(bndbox, "ymin").text = str(int(y1))
            ET.SubElement(bndbox, "xmax").text = str(int(x2))
            ET.SubElement(bndbox, "ymax").text = str(int(y2))

            if result.masks is not None and hasattr(result.masks, "xy"):
                polygons = result.masks.xy[idx]
                polygon_tag = ET.SubElement(obj_tag, "polygon")
                for i, (x, y) in enumerate(polygons):
                    ET.SubElement(polygon_tag, f"x{i+1}").text = str(int(x))
                    ET.SubElement(polygon_tag, f"y{i+1}").text = str(int(y))
    else:
        logger.warning("Aucune boîte prédite trouvée dans les résultats de YOLO.")

    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
    with open(save_path, "w") as f:
        f.write(xml_str)

def run_yolo_and_compute_f1(image, ground_truth_path=None, iou_threshold=0.3, output_dir=None, save_output=False, mode="fusion", image_filename=None):
    """
    Run YOLOv8 object detection and compute evaluation metrics (Precision, Recall, F1-score, IoU).

    This function loads YOLO configuration from an external JSON file (or creates one with
    default parameters if missing), performs inference on the input image, compares predictions
    with ground truth annotations (Pascal VOC format), and computes detection metrics.

    Parameters
    ----------
    image : str, Path, or np.ndarray
        Input image, either as a file path or a NumPy array.
    ground_truth_path : str or Path, optional
        Path to the Pascal VOC XML file containing ground truth annotations.
    iou_threshold : float, default=0.3
        IoU threshold used to determine true positives.
    output_dir : str or Path, optional
        Directory where annotated images and XML files will be saved.
    save_output : bool, default=False
        Whether to save annotated images and XML predictions.
    mode : str, default="fusion"
        Processing mode for image preparation (e.g., "fusion", "visible", "swir").
    image_filename : str, optional
        Original image filename, used to generate output file names.

    Returns
    -------
    dict
        Dictionary containing detection metrics with the following keys:

        * **f1_score** (float) – F1-score of the detection.
        * **precision** (float) – Precision of the detection.
        * **recall** (float) – Recall of the detection.
        * **iou_mean** (float) – Mean IoU between predictions and ground truth.

    Notes
    -----
    - YOLO configuration is loaded from ``config/yolo_config.json``. If the file does not exist,
      it is created with default parameters.
    - Predictions are filtered to include only the allowed classes defined in the configuration.
    - Results can be saved as annotated images and Pascal VOC XML files if ``save_output=True``.
    - Heavy objects (YOLO model, predictions, images) are explicitly deleted to free memory.
    """

    logger.debug(f"🔎 Lancement de la detection...")

    # === Charger ou créer le fichier de configuration JSON pour YOLO ===
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "yolo_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            yolo_config = json.load(f)
    else:
        yolo_config = {
            "model_path": "yolov8x-seg.pt", # modèle *segmentation*
            "confidence_threshold": 0.25,
            "iou_threshold": 0.3, #0.5,
            "device": "cpu",
            "save_detection_results": False, # Permet de forcer la sauvegarde de la détection seule
            "allowed_classes": ["truck", "person", "bus", "motorcycle", "bicycle", "car"] # RASMD Dataset classes
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(yolo_config, f, indent=4)    

    # === Appliquer les paramètres ===
    # model_path = yolo_config["model_path"]
    model_path = Path(yolo_config["model_path"])  # conversion en Path
    if not model_path.is_absolute(): # Résoudre le chemin relatif du modèle par rapport à l’emplacement du JSON
        model_path = config_path.parent / model_path
    conf_thres = yolo_config["confidence_threshold"]
    iou_threshold = yolo_config.get("iou_threshold", iou_threshold)
    device = yolo_config.get("device", "cpu")
    allowed_classes = set(yolo_config.get("allowed_classes", []))

    # === Convertir l'image si besoin ===
    img_bgr = prepare_image_for_yolo(image, mode=mode)
    # logger.debug(f"🧪 Vérification image passée à YOLO : shape={img_bgr.shape}")

    # === Inference : Prédiction avec YOLOv8 ===
    if not model_path.exists():
        logger.warning(f"⚠️ YOLO weights not found locally at {model_path}. Trying to download automatically...")
        model = YOLO(model_path.name)
    else:
        model = YOLO(str(model_path))

    # # === Mapping classes autorisées ===
    class_names = model.names  # dict: id -> class name
    allowed_ids = [cls_id for cls_id, name in class_names.items() if name in allowed_classes]

    # === Prédiction
    results = model.predict(source=img_bgr, conf=conf_thres, device=device, verbose=False, classes=allowed_ids)
    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        logger.debug("⚠️ Aucune boîte détectée dans l'image fournie.")
        logger.debug(f"🧪 Détection limitée aux classes : {allowed_classes}")

    # === Bounding boxes prédites ===
    preds = result.boxes.xyxy.cpu().numpy().astype(int) if result.boxes else []

    # === Charger les Ground Truth ===
    if ground_truth_path is not None and Path(ground_truth_path).exists():
        gts = parse_voc_annotations(ground_truth_path)
    else:
        gts = []

    # === Calcul des métriques ===
    tp, fp, fn = 0, 0, 0
    matched_gt = set()

    for pred in preds:
        match_found = False
        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue
            if iou(pred, gt) >= iou_threshold:
                tp += 1
                matched_gt.add(i)
                match_found = True
                break
        if not match_found:
            fp += 1
    fn = len(gts) - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    # iou_mean = np.mean([iou(pred, gt) for pred in preds for gt in gts]) if preds and gts else 0.0
    iou_mean = np.mean([iou(pred, gt) for pred in preds for gt in gts]) if len(preds) > 0 and len(gts) > 0 else 0.0


    logger.debug(f"✅ Précision: {precision:.4f}, Rappel: {recall:.4f}, F1: {f1:.4f}, IoU moyen: {iou_mean:.4f}")

    # === Sauvegarde des résultats ===
    if save_output or yolo_config.get("save_detection_results", False):
        output_dir = Path(output_dir) if output_dir else Path("output")
        output_dir_images = output_dir / "annotated_images" / mode
        output_dir_xml = output_dir / "annotated_xml" / mode

        output_dir_images.mkdir(parents=True, exist_ok=True)
        output_dir_xml.mkdir(parents=True, exist_ok=True)

        # === Génération nom unique triable ===
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # unique_id = uuid.uuid4().hex[:8]
        # image_name = f"yolo_result_{timestamp}_{unique_id}.tiff"
        # xml_name = f"yolo_result_{timestamp}_{unique_id}.xml"
        if image_filename is not None:
            base_name = Path(image_filename).stem
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            unique_id = uuid.uuid4().hex[:8]
            base_name = f"result_{timestamp}_{unique_id}"

        image_name = f"{base_name}.tiff"
        xml_name = f"{base_name}.xml"

        # === Sauvegarder image prédite ===
        vis_img = result.plot()
        # if isinstance(image, np.ndarray) and image.dtype == np.float64 and image.min() >= 0.0 and image.max() <= 1.0:
        #     logger.debug("💾 Sauvegarde image avec utils.save_float64_image_as_uint16")
        #     save_float64_image_as_uint16(output_dir_images / image_name, image)
        # else:
        #     logger.debug("💾 Sauvegarde image via cv2.imwrite")
        #     cv2.imwrite(str(output_dir_images / image_name), vis_img)
        # Toujours sauvegarder l'image avec les annotations
        logger.debug("💾 Sauvegarde image via cv2.imwrite")
        cv2.imwrite(str(output_dir_images / f"Yolo_{image_name}"), vis_img)

        # === Sauvegarder les annotations ===
        save_predictions_as_voc_xml(
            result=result,
            # image_shape=image.shape if isinstance(image, np.ndarray) else (result.orig_shape[0], result.orig_shape[1], 3),
            image_shape=get_image_shape(image),
            save_path=output_dir_xml / xml_name,
            image_filename=image_name
        )
        del vis_img
        gc.collect()


    # return {
    #     "f1_score": round(float(f1), 4),
    #     "precision": round(float(precision), 4),
    #     "recall": round(float(recall), 4),
    #     "iou_mean": round(float(iou_mean), 4)
    # }
    # Résultat à retourner
    metrics = {
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "iou_mean": round(float(iou_mean), 4)
    }

    # 🔥 Suppression des objets lourds
    del img_bgr, results, result, preds, gts, model
    gc.collect()

    return metrics