# =============================================================================
#  Fichier       : coco_to_voc_converter.py
#  Projet        : VISWIR - Fusion d'images Visible/SWIR pour la perception des véhicules autonomes.
#  Auteur        : Alexandre Riffard
#  Description   : Convertit un fichier d'annotations COCO (au format JSON)
#                  vers des fichiers XML au format Pascal VOC, compatibles
#                  avec les outils de détection comme YOLOv8.
# 
#  Licence       : GNU Lesser General Public License v3.0 (LGPL-3.0)
#                  Vous pouvez redistribuer et/ou modifier ce fichier sous les
#                  termes de la LGPL telle que publiée par la Free Software Foundation.
# 
#                  Ce programme est distribué dans l’espoir qu’il sera utile,
#                  mais SANS AUCUNE GARANTIE ; sans même la garantie implicite
#                  de COMMERCIALISATION ou D’ADÉQUATION À UN BUT PARTICULIER.
#                  Voir la licence LGPL pour plus de détails.
#
#  Pour une copie complète de la licence, voir :
#  https://www.gnu.org/licenses/lgpl-3.0.html
# =============================================================================

import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

def coco_to_voc(coco_json_path, output_dir, image_dir=None):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # Dictionnaires pour accès rapide
    image_id_to_info = {img["id"]: img for img in coco["images"]}
    category_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    annotations_by_image = {}
    for ann in coco["annotations"]:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_id, image_info in image_id_to_info.items():
        filename = image_info["file_name"]
        width = image_info["width"]
        height = image_info["height"]
        depth = 3  # Hypothèse : image RGB

        # === Construction de la structure XML ===
        annotation = ET.Element("annotation")

        ET.SubElement(annotation, "folder").text = ""
        ET.SubElement(annotation, "filename").text = filename
        ET.SubElement(annotation, "path").text = str(Path(image_dir, filename)) if image_dir else filename

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "roboflow.com"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = str(depth)

        ET.SubElement(annotation, "segmented").text = "0"

        for ann in annotations_by_image.get(image_id, []):
            category_name = category_id_to_name[ann["category_id"]]
            x, y, w, h = ann["bbox"]
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w)
            y2 = int(y + h)

            obj_tag = ET.SubElement(annotation, "object")
            ET.SubElement(obj_tag, "name").text = category_name
            ET.SubElement(obj_tag, "pose").text = "Unspecified"
            ET.SubElement(obj_tag, "truncated").text = "0"
            ET.SubElement(obj_tag, "difficult").text = "0"
            ET.SubElement(obj_tag, "occluded").text = "0"

            bndbox = ET.SubElement(obj_tag, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(x1)
            ET.SubElement(bndbox, "xmax").text = str(x2)
            ET.SubElement(bndbox, "ymin").text = str(y1)
            ET.SubElement(bndbox, "ymax").text = str(y2)

            # Bloc <polygon> vide
            ET.SubElement(obj_tag, "polygon")

        # Sauvegarde du fichier XML
        xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
        xml_path = output_dir / (Path(filename).stem + ".xml")
        with open(xml_path, "w") as f:
            f.write(xml_str)

        print(f"✅ Annotation écrite : {xml_path}")

# Exemple d’utilisation :
# coco_to_voc("annotations_rasmd.json", "converted_annotations", image_dir="images_rgb")
coco_to_voc(r"D:\RASMD\RASMD_detection_annotation\train_swir_align.json", r"D:\RASMD\RASMD_VOC_XLM_Annotation\train\swir", image_dir=r"D:\RASMD\RASMD_detection\RASMD_detection\train\SWIR")
