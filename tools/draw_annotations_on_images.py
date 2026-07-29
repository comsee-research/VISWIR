import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

def draw_boxes_from_voc(image_path, xml_path, output_path):
    # Lecture de l'image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[!] Impossible de lire l'image : {image_path}")
        return

    # Parsing XML
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

    # Sauvegarde de l'image annotée
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

def process_directory(images_dir, annotations_dir, output_dir):
    images_dir = Path(images_dir)
    annotations_dir = Path(annotations_dir)
    output_dir = Path(output_dir)

    for xml_file in annotations_dir.glob("*.xml"):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text
        image_path = images_dir / filename
        output_path = output_dir / filename

        draw_boxes_from_voc(image_path, xml_file, output_path)

if __name__ == "__main__":
    # À adapter selon ton dossier
    images_dir = r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\data\RASMD_train\SWIR" #"chemin/vers/tes/images"
    annotations_dir = r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\data\RASMD_VOC_XML_Annotation\train\swir" #"chemin/vers/tes/annotations"
    output_dir = r"C:\Users\Riffard\Documents\code_python_git\Techniques_de_fusion\VISWIR\data\RASMD_VOC_XML_Annotation\annotated_images\SWIR" #"chemin/vers/dossier_resultat"

    process_directory(images_dir, annotations_dir, output_dir)
    print("✅ Traitement terminé.")
