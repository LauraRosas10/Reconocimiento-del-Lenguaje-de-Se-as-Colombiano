"""
Inferencia en tiempo real para Fase 2 (MediaPipe Landmarks + MLP).
Visualiza:
  - Landmarks (puntos) coloreados por dedo.
  - Conexiones (lineas esqueleticas) coloreadas por dedo.
  - Handedness (Mano Izquierda / Mano Derecha).
  - Prediccion de la seña LSC y porcentaje de confianza en vivo.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

from normalizer import draw_styled_hand_with_handedness, render_skeleton_from_landmarks

current_dir = Path(__file__).resolve().parent
PROJECT_ROOT = current_dir.parent.parent
MODEL_PATH = PROJECT_ROOT / "models_saved" / "phase2_landmarks_mobilenetv3.keras"
LEGACY_MODEL_PATH = PROJECT_ROOT / "models_saved" / "phase2_landmarks_lsc70.keras"
CLASSES_PATH = PROJECT_ROOT / "models_saved" / "classes_phase2.txt"
LEGACY_CLASSES_PATH = PROJECT_ROOT / "data" / "landmarks_phase2" / "classes.txt"


def extract_features_and_draw(frame, results, h, w):
    """
    Extrae el vector de 126 dimensiones y dibuja los landmarks y handedness.
    """
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            raw_label = handedness.classification[0].label  # 'Left' or 'Right'
            label_es = "Derecha" if raw_label == "Right" else "Izquierda"

            # Dibujar en pantalla
            draw_styled_hand_with_handedness(frame, hand_lm, label_es, h, w)

            # Extraer y normalizar
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
            wrist = coords[0].copy()
            centered = coords - wrist
            scale = np.linalg.norm(centered[9])
            if scale < 1e-6:
                scale = 1.0
            normalized = (centered / scale).flatten()

            if raw_label == 'Left':
                left_hand = normalized
            else:
                right_hand = normalized

    return np.concatenate([left_hand, right_hand])


def run_realtime():
    chosen_model_path = MODEL_PATH if MODEL_PATH.exists() else LEGACY_MODEL_PATH
    if not chosen_model_path.exists():
        print(f"[ERROR] No se encontro el modelo entrenado en: {MODEL_PATH.resolve()}")
        print("Debes ejecutar primero el entrenamiento con:")
        print("  python src/phase2_landmarks/train_phase2.py")
        return

    chosen_classes_path = CLASSES_PATH if CLASSES_PATH.exists() else LEGACY_CLASSES_PATH
    if not chosen_classes_path.exists():
        print(f"[ERROR] No se encontro el archivo de clases en: {CLASSES_PATH.resolve()}")
        return

    print(f"Cargando modelo desde: {chosen_model_path.name}...")
    model = tf.keras.models.load_model(str(chosen_model_path))
    with open(chosen_classes_path, "r", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]

    is_mobilenet = (len(model.input_shape) == 4 and model.input_shape[-1] == 3)
    arch_name = "MobileNetV3Large (Esqueletos 224x224)" if is_mobilenet else "MLP Residual (126 dims)"
    print(f"Modelo cargado: {arch_name} | Clases: {len(class_names)}")

    hands_detector = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la camara web.")
        return

    print(f"\n Camara activa para Inferencia Fase 2 — {arch_name}.")
    print(" Presiona 'Q' o 'ESC' en la ventana para salir.\n")

    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Efecto espejo
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)

        # 1. Dibujar landmarks sobre el frame en vivo
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                raw_label = handedness.classification[0].label
                label_es = "Derecha" if raw_label == "Right" else "Izquierda"
                draw_styled_hand_with_handedness(frame, hand_lm, label_es, h, w)

        # 2. Generar imagen esquelética 224x224
        skeleton_canvas, has_hand = render_skeleton_from_landmarks(
            results.multi_hand_landmarks, canvas_size=224, padding=16
        )

        hands_count = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        # 3. Predicción
        if has_hand:
            if is_mobilenet:
                input_data = np.expand_dims(skeleton_canvas, axis=0).astype(np.float32)
            else:
                features = extract_features_and_draw(frame, results, h, w)
                input_data = np.expand_dims(features, axis=0)

            preds = model.predict(input_data, verbose=0)[0]
            class_idx = int(np.argmax(preds))
            confidence = float(preds[class_idx])

            color = (0, 255, 0) if confidence >= 0.70 else (0, 255, 255)

            # Panel informativo superior
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (560, 105), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, f"Seña: {class_names[class_idx]}", (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
            cv2.putText(frame, f"Confianza: {confidence * 100:.1f}% | Manos: {hands_count} | FPS: {int(fps)}",
                        (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

            # Recuadro PiP (Picture in Picture) del esqueleto sintético
            pip_size = 150
            pip_img = cv2.resize(skeleton_canvas, (pip_size, pip_size))
            pip_bgr = cv2.cvtColor(pip_img, cv2.COLOR_RGB2BGR)
            frame[15:15 + pip_size, w - pip_size - 15:w - 15] = pip_bgr
            cv2.rectangle(frame, (w - pip_size - 15, 15), (w - 15, 15 + pip_size), (0, 255, 255), 2)
            cv2.putText(frame, "Input MobileNet", (w - pip_size - 10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 75), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, "Esperando mano...", (25, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {int(fps)} | Manos detectadas: 0", (25, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        cv2.imshow("Inferencia LSC70 — Fase 2 (MobileNetV3 Landmarks Skeletons)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    print("Camara cerrada correctamente.")


if __name__ == '__main__':
    run_realtime()

