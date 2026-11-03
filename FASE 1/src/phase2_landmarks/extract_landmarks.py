"""
Script de extraccion offline de landmarks para Fase 2 (MediaPipe Hands).
Procesa los splits train, val y test, extrayendo vectores de 126 caracteristicas normalizadas
y guardandolos en data/landmarks_phase2/ como archivos .npy.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Asegurar que el modulo local normalizer se pueda importar
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from normalizer import extract_and_normalize_2hands, render_skeleton_image

# Rutas por defecto calculadas respecto a la raiz del proyecto
PROJECT_ROOT = current_dir.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "landmarks_skeleton"
DEFAULT_NUMERIC_DIR = DEFAULT_DATA_DIR / "landmarks_phase2"


def process_dataset(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR, skip_unrecognized=False):
    """
    Extrae landmarks con MediaPipe y genera las imagenes sinteticas de esqueleto (224x224x3)
    organizadas por clase para entrenar MobileNetV3Large (Opcion A).
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    numeric_dir = Path(DEFAULT_NUMERIC_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    numeric_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(" GENERACION DE ESQUELETOS LANDMARKS — FASE 2 (MOBILENETV3) ".center(60, "="))
    print("=" * 60)
    print(f"Directorio de entrada  : {data_dir.resolve()}")
    print(f"Directorio de salida   : {output_dir.resolve()}")
    print(f"Omitir sin deteccion   : {skip_unrecognized}")
    print("=" * 60 + "\n")

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    stats = {}

    for split in ['train', 'val', 'test']:
        split_path = data_dir / split
        if not split_path.exists():
            print(f"[ADVERTENCIA] No existe la carpeta del split: {split_path}")
            continue

        split_out_path = output_dir / split
        split_out_path.mkdir(parents=True, exist_ok=True)

        classes = sorted([d.name for d in split_path.iterdir() if d.is_dir()])
        total_images = 0
        detected_images = 0
        saved_images = 0

        print(f"\n---> Procesando split '{split}' ({len(classes)} clases)...")

        for class_idx, class_name in enumerate(tqdm(classes, desc=f"Split {split}")):
            class_path = split_path / class_name
            class_out_path = split_out_path / class_name
            class_out_path.mkdir(parents=True, exist_ok=True)

            img_files = [f for f in class_path.glob("*.*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]

            for img_path in img_files:
                total_images += 1
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                skeleton_img, has_hand = render_skeleton_image(img, hands_detector, canvas_size=224, padding=16)

                if has_hand or not skip_unrecognized:
                    if has_hand:
                        detected_images += 1
                    out_img_path = class_out_path / f"{img_path.stem}.png"
                    # Convertir a BGR para que cv2.imwrite guarde los canales R, G, B correctamente en el PNG
                    skeleton_bgr = cv2.cvtColor(skeleton_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_img_path), skeleton_bgr)
                    saved_images += 1


        rate = (detected_images / total_images * 100) if total_images > 0 else 0
        stats[split] = {
            'total': total_images,
            'detected': detected_images,
            'saved': saved_images,
            'rate': rate
        }

        print(f"  [OK] Split {split}: {saved_images} imagenes guardadas ({rate:.1f}% con landmarks detectados)")

        # Guardar archivo de clases en ambas carpetas
        if split == 'train':
            for target_dir in [output_dir, numeric_dir]:
                classes_file = target_dir / "classes.txt"
                with open(classes_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(classes))
                print(f"  [OK] Clases guardadas ({len(classes)} clases) en: {classes_file}")

    hands_detector.close()

    print("\n" + "=" * 60)
    print(" RESUMEN DE GENERACION DE ESQUELETOS (OPCION A) ".center(60, "="))
    print("=" * 60)
    for s, data in stats.items():
        print(f"Split {s:<6} : {data['saved']:>6} guardadas | {data['detected']:>6} / {data['total']:>6} con landmarks ({data['rate']:.1f}%)")
    print(f"Dataset de esqueletos listo en: {output_dir.resolve()}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    process_dataset()

