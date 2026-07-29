"""
Script to verify that the Pascal VOC ground truth bounding boxes match the test images.
Overlays boxes on both VIS and SWIR images and saves them in results/verification_gt/.
"""

import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

def draw_boxes_from_voc(image_path, xml_path, output_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[!] Impossible de lire l'image : {image_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymin = int(float(bndbox.find("ymin").text))
        ymax = int(float(bndbox.find("ymax").text))

        # Dessin du rectangle et du nom de la classe
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Annotated image saved to: {output_path}")

def main():
    data_dir = Path("data")
    gt_dir = data_dir / "Ground truth"
    out_dir = Path("results/verification_gt")
    
    # Process clear, fog, rain
    for condition in ["clear", "fog", "rain"]:
        xml_path = gt_dir / f"{condition}.xml"
        if not xml_path.exists():
            print(f"XML not found: {xml_path}")
            continue
            
        # Draw on VIS
        vis_image = data_dir / "VIS" / f"{condition}.jpg"
        if vis_image.exists():
            draw_boxes_from_voc(vis_image, xml_path, out_dir / f"{condition}_vis.jpg")
            
        # Draw on SWIR
        swir_image = data_dir / "SWIR" / f"{condition}.jpg"
        if swir_image.exists():
            draw_boxes_from_voc(swir_image, xml_path, out_dir / f"{condition}_swir.jpg")

if __name__ == "__main__":
    main()
