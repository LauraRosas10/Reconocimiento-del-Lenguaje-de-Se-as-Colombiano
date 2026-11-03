"""
Inferencia en tiempo real para Fase 1 (MobileNetV3Large + MediaPipe Hand Crop).
Detecta la(s) mano(s) en el cuadro de la webcam con MediaPipe, extrae el cuadro delimitador (ROI) con margen,
redimensiona a 224x224x3 y clasifica con MobileNetV3Large.
Incluye suavizado temporal de predicciones para mayor estabilidad.
"""

import sys
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent.parent
MODEL_PATH = PROJECT_ROOT / "models_saved" / "phase1_mobilenetv3_lsc70.keras"
CLASSES_PATH = PROJECT_ROOT / "models_saved" / "classes_phase1.txt"

# Configuración de suavizado temporal
PREDICTION_BUFFER_SIZE = 5       # Promediar últimas N predicciones
CONFIDENCE_THRESHOLD = 0.75      # Umbral mínimo para mostrar predicción
STABLE_FRAMES_REQUIRED = 3       # Frames consecutivos con la misma clase para confirmar


def get_hands_bbox(multi_hand_landmarks, img_w, img_h, margin_ratio=0.25):
    """
    Calcula el cuadro delimitador (Bounding Box) global para una o dos manos con un margen de holgura.
    """
    all_x = []
    all_y = []
    for hand_lms in multi_hand_landmarks:
        all_x.extend([lm.x for lm in hand_lms.landmark])
        all_y.extend([lm.y for lm in hand_lms.landmark])

    if not all_x or not all_y:
        return None

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    x1 = int(x_min * img_w)
    y1 = int(y_min * img_h)
    x2 = int(x_max * img_w)
    y2 = int(y_max * img_h)

    # Añadir margen de seguridad
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    dx = int(bw * margin_ratio)
    dy = int(bh * margin_ratio)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(img_w, x2 + dx)
    y2 = min(img_h, y2 + dy)

    return x1, y1, x2, y2


def to_square(x1, y1, x2, y2, w, h):
    """Convierte el bbox en un cuadrado (lado = max(ancho, alto)) centrado en el mismo punto y
    desplazado para quedar dentro del frame. Evita deformar la mano al redimensionar a 224x224."""
    side = min(max(x2 - x1, y2 - y1), w, h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    nx1 = min(max(0, cx - side // 2), w - side)
    ny1 = min(max(0, cy - side // 2), h - side)
    return nx1, ny1, nx1 + side, ny1 + side


def open_webcam(camera_index=0):
    """
    Intenta abrir la cámara web utilizando DirectShow en Windows para evitar demoras,
    con fallback al backend estándar.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    return cap


def run_realtime_phase1(camera_index=0, flip=True, threshold=CONFIDENCE_THRESHOLD, debug=False):
    """
    Bucle principal de inferencia en tiempo real para la Fase 1.
    """
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] No se encontro el modelo entrenado en: {MODEL_PATH.resolve()}")
        print("Debes ejecutar primero el entrenamiento con: python src/phase1_raw_images/train_phase1.py\n")
        return

    if not CLASSES_PATH.exists():
        print(f"\n[ERROR] No se encontro el archivo de clases en: {CLASSES_PATH.resolve()}\n")
        return

    print("=" * 60)
    print(" INFERENCIA EN TIEMPO REAL — FASE 1 (MOBILENETV3 RGB) ".center(60, "="))
    print("=" * 60)
    print(f"Cargando modelo desde: {MODEL_PATH.name}...")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    print(f"[OK] Modelo cargado exitosamente. Clases disponibles: {len(class_names)}")

    # Inicializar detector de MediaPipe Hands (soporta hasta 2 manos)
    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )

    print(f"Abriendo camara web (indice {camera_index})...")
    cap = open_webcam(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"\n[ERROR] No se pudo abrir la camara web en el indice {camera_index}.")
        print("Sugerencias:")
        print("  1. Verifica que otra aplicacion (Zoom, Teams, Streamlit, etc.) no este usando la camara.")
        print("  2. Intenta con otro indice de camara: python src/phase1_raw_images/real_time_infer_phase1.py --camera 1\n")
        return

    print("\n Camara activa para Inferencia Fase 1 (Crop + MobileNetV3).")
    print(" Controles:")
    print("   - Presiona 'Q' o 'ESC' en la ventana para salir.\n")

    prev_time = 0
    prediction_buffer = deque(maxlen=PREDICTION_BUFFER_SIZE)
    last_stable_class = None
    stable_count = 0
    frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("[AVISO] No se pudo leer el cuadro de la camara web.")
                break

            if flip:
                frame = cv2.flip(frame, 1)  # Efecto espejo (desactivar con --no-flip)
            h, w, _ = frame.shape

            # Calculo de FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)

            if results.multi_hand_landmarks:
                bbox = get_hands_bbox(results.multi_hand_landmarks, w, h, margin_ratio=0.25)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    x1, y1, x2, y2 = to_square(x1, y1, x2, y2, w, h)  # sin deformar la mano
                    crop_w, crop_h = x2 - x1, y2 - y1

                    if crop_w > 30 and crop_h > 30:
                        hand_crop = frame[y1:y2, x1:x2]
                        # Usar INTER_AREA para downscale, INTER_LINEAR para upscale
                        interp = cv2.INTER_AREA if (crop_w > 224 or crop_h > 224) else cv2.INTER_LINEAR
                        resized_crop = cv2.resize(hand_crop, (224, 224), interpolation=interp)
                        # Convertir BGR a RGB para coincidir con la entrada esperada por MobileNetV3
                        rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
                        input_tensor = np.expand_dims(rgb_crop, axis=0).astype(np.float32)

                        # Prediccion
                        preds = model.predict(input_tensor, verbose=0)[0]
                        
                        # Suavizado temporal: acumular predicciones
                        prediction_buffer.append(preds)
                        avg_preds = np.mean(prediction_buffer, axis=0)
                        class_idx = int(np.argmax(avg_preds))
                        confidence = float(avg_preds[class_idx])
                        
                        # Estabilidad: exigir N frames consecutivos de la misma clase
                        if class_idx == last_stable_class:
                            stable_count += 1
                        else:
                            last_stable_class = class_idx
                            stable_count = 1

                        if debug:
                            frame_idx += 1
                            if frame_idx % 10 == 0:
                                top3 = np.argsort(avg_preds)[-3:][::-1]
                                print(" | ".join(f"{class_names[i]}={avg_preds[i]:.2f}" for i in top3),
                                      f"| estable={stable_count} | umbral={threshold}")
                                cv2.imwrite("debug_crop.png", resized_crop)  # BGR: lo que ve el modelo

                        # Siempre se muestra la clase con mayor probabilidad.
                        # Verde = confirmada (supera el umbral y es estable). Naranja con "?" = tentativa.
                        confirmada = confidence >= threshold and stable_count >= STABLE_FRAMES_REQUIRED
                        if confirmada:
                            color, thick = (0, 255, 0), 3
                            label = f"{class_names[class_idx]} ({confidence * 100:.0f}%)"
                        else:
                            color, thick = (0, 165, 255), 2
                            label = f"{class_names[class_idx]}? ({confidence * 100:.0f}%)"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                        cv2.rectangle(frame, (x1, max(0, y1 - 40)), (x1 + 350, max(0, y1)), (20, 20, 20), -1)
                        cv2.putText(frame, label, (x1 + 10, max(25, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

                        # Panel con las 3 clases mas probables
                        cv2.rectangle(frame, (10, 70), (270, 175), (20, 20, 20), -1)
                        for k, i in enumerate(np.argsort(avg_preds)[-3:][::-1]):
                            cv2.putText(frame, f"{k + 1}. {class_names[i]}  {avg_preds[i] * 100:.0f}%",
                                        (20, 102 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        color if k == 0 else (255, 255, 255), 2, cv2.LINE_AA)
            else:
                prediction_buffer.clear()
                last_stable_class = None
                stable_count = 0
                cv2.rectangle(frame, (10, 10), (460, 60), (20, 20, 20), -1)
                cv2.putText(frame, "Esperando deteccion de mano...", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2, cv2.LINE_AA)

            # Barra superior con FPS e informacion
            fps_text = f"FPS: {fps:.1f} | Fase 1: MobileNetV3 RGB"
            cv2.putText(frame, fps_text, (w - 400, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Inferencia LSC70 — Fase 1 (MobileNetV3 RGB Crop)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands_detector.close()
        print("[OK] Camara y recursos liberados correctamente.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Inferencia en tiempo real Fase 1 (MobileNetV3 RGB)")
    parser.add_argument("--camera", type=int, default=0, help="Indice de la camara web (default: 0)")
    parser.add_argument("--no-flip", action="store_true", help="No espejar el frame antes de inferir")
    parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Umbral de confianza (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--debug", action="store_true", help="Imprime top-3 y guarda debug_crop.png cada 10 frames")
    args = parser.parse_args()

    run_realtime_phase1(camera_index=args.camera, flip=not args.no_flip,
                        threshold=args.threshold, debug=args.debug)
