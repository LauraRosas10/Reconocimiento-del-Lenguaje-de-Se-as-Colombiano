
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from landmark_recognition.inference import (
    load_checkpoint_model,
    prepare_checkpoint_features,
)
from landmark_recognition.class_labels import display_class_name
from landmark_recognition.mediapipe_support import holistic_result_to_array


# =========================================================
# CONFIGURACIÓN VISUAL
# =========================================================

WIDTH = 1280
HEIGHT = 760

BLACK = (10, 12, 16)
PANEL = (20, 23, 30)
PANEL_LIGHT = (28, 32, 41)
WHITE = (245, 247, 250)
GRAY = (155, 164, 177)
DARK_GRAY = (55, 62, 74)
TURQUOISE = (210, 190, 50)  # BGR
GREEN = (100, 220, 145)
YELLOW = (60, 210, 255)
CORAL = (105, 105, 235)
POSE_COLOR = (190, 220, 80)
HAND_COLOR = (105, 210, 255)
FONT = cv2.FONT_HERSHEY_DUPLEX


def default_checkpoint_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "weights" / "transformer_intersection.pth"
    return Path(__file__).resolve().parents[1] / "checkpoints" / "transformer_intersection.pth"


def draw_text(
    image,
    text,
    position,
    size=0.7,
    color=WHITE,
    thickness=1,
):
    cv2.putText(
        image,
        str(text),
        position,
        FONT,
        size,
        color,
        thickness,
        cv2.LINE_AA,
    )


def rounded_rectangle(image, p1, p2, color, radius=12, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    radius = max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))

    if thickness < 0:
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.rectangle(image, p1, p2, color, thickness)


def draw_status_badge(image, text, color, position):
    x, y = position
    width = 175 if len(text) > 15 else 145
    rounded_rectangle(image, (x, y), (x + width, y + 34), (25, 29, 37), radius=17)
    cv2.circle(image, (x + 17, y + 17), 5, color, -1)
    draw_text(image, text, (x + 30, y + 22), size=0.38, color=WHITE, thickness=1)


def draw_video_frame(image, position, size):
    x, y = position
    width, height = size
    cv2.rectangle(image, (x, y), (x + width, y + height), (62, 72, 86), 1)
    corner = 18
    for start_x, start_y, end_x, end_y in (
        (x, y + corner, x, y),
        (x, y, x + corner, y),
        (x + width - corner, y, x + width, y),
        (x + width, y, x + width, y + corner),
        (x, y + height - corner, x, y + height),
        (x, y + height, x + corner, y + height),
        (x + width - corner, y + height, x + width, y + height),
        (x + width, y + height - corner, x + width, y + height),
    ):
        cv2.line(image, (start_x, start_y), (end_x, end_y), TURQUOISE, 2)


def fit_video(frame, width, height):
    """Ajusta el video sin deformar sus proporciones."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    x = (width - new_w) // 2
    y = (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def draw_probability_card(
    canvas,
    y,
    rank,
    name,
    probability,
    is_first=False,
):
    """Dibuja una predicción y su barra de probabilidad."""
    x1, x2 = 920, 1245
    card_height = 112

    background = PANEL_LIGHT if is_first else PANEL
    rounded_rectangle(
        canvas,
        (x1, y),
        (x2, y + card_height),
        background,
        radius=10,
    )

    accent = TURQUOISE if is_first else DARK_GRAY
    cv2.rectangle(canvas, (x1, y + 12), (x1 + 4, y + card_height - 12), accent, -1)

    draw_text(
        canvas,
        f"{rank:02d}",
        (x1 + 20, y + 31),
        size=0.55,
        color=GRAY,
    )

    # Nombre de la seña
    max_chars = 25
    visible_name = name if len(name) <= max_chars else name[:max_chars - 3] + "..."
    draw_text(
        canvas,
        visible_name,
        (x1 + 58, y + 34),
        size=0.62 if len(visible_name) < 19 else 0.48,
        color=WHITE if is_first else (205, 205, 205),
        thickness=2 if is_first else 1,
    )

    # Porcentaje
    percentage = probability * 100
    draw_text(
        canvas,
        f"{percentage:.1f}%",
        (x2 - 85, y + 34),
        size=0.62,
        color=TURQUOISE if is_first else WHITE,
        thickness=2,
    )

    # Barra de probabilidad
    bar_x1 = x1 + 20
    bar_x2 = x2 - 20
    bar_y = y + 68
    bar_height = 8

    rounded_rectangle(
        canvas,
        (bar_x1, bar_y),
        (bar_x2, bar_y + bar_height),
        DARK_GRAY,
        radius=4,
    )

    fill_width = int((bar_x2 - bar_x1) * np.clip(probability, 0, 1))

    if fill_width > 0:
        rounded_rectangle(
            canvas,
            (bar_x1, bar_y),
            (bar_x1 + fill_width, bar_y + bar_height),
            accent,
            radius=4,
        )


def draw_interface(
    frame,
    predictions,
    history_length,
    frame_count,
    hands_detected,
    fps,
):
    """Construye una interfaz negra con video y tres predicciones."""
    canvas = np.full((HEIGHT, WIDTH, 3), BLACK, dtype=np.uint8)

    # Encabezado
    draw_text(
        canvas,
        "RECONOCIMIENTO DE SEÑAS",
        (35, 43),
        size=0.82,
        color=WHITE,
        thickness=2,
    )
    draw_text(
        canvas,
        "LENGUA DE SEÑAS COLOMBIANA",
        (37, 70),
        size=0.43,
        color=GRAY,
    )

    # Indicador de estado
    status_color = GREEN if hands_detected else YELLOW
    status_text = "SENAL DETECTADA" if hands_detected else "BUSCANDO SENAL"
    draw_status_badge(canvas, status_text, status_color, (1055, 20))

    cv2.line(canvas, (35, 91), (1245, 91), (40, 40, 40), 1)

    # Área del video
    video_x, video_y = 35, 115
    video_w, video_h = 850, 570

    fitted = fit_video(frame, video_w, video_h)
    canvas[
        video_y:video_y + video_h,
        video_x:video_x + video_w,
    ] = fitted

    draw_video_frame(canvas, (video_x, video_y), (video_w, video_h))

    # Panel de probabilidades
    draw_text(canvas, "PREDICCIONES", (920, 133), size=0.63, color=WHITE, thickness=2)
    draw_text(canvas, "Las tres opciones mas probables", (920, 158), size=0.38, color=GRAY)

    if predictions:
        for index, (name, probability) in enumerate(predictions[:3]):
            draw_probability_card(
                canvas,
                y=183 + index * 126,
                rank=index + 1,
                name=name,
                probability=probability,
                is_first=(index == 0),
            )
    else:
        rounded_rectangle(
            canvas,
            (920, 183),
            (1245, 295),
            PANEL,
            radius=10,
        )
        draw_text(canvas, "Analizando movimiento...", (940, 228), size=0.55, color=WHITE)
        draw_text(canvas, "Esperando suficientes fotogramas", (940, 256), size=0.36, color=GRAY)

    # Progreso de captura
    draw_text(canvas, "CAPTURA TEMPORAL", (920, 600), size=0.43, color=GRAY)

    progress = min(history_length / max(frame_count, 1), 1.0)
    bar_x1, bar_x2 = 920, 1245
    bar_y = 615

    rounded_rectangle(canvas, (bar_x1, bar_y), (bar_x2, bar_y + 8), DARK_GRAY, radius=4)

    if progress > 0:
        fill_x = bar_x1 + int((bar_x2 - bar_x1) * progress)
        rounded_rectangle(
            canvas,
            (bar_x1, bar_y),
            (fill_x, bar_y + 8),
            TURQUOISE,
            radius=4,
        )

    draw_text(
        canvas,
        f"{history_length} / {frame_count} fotogramas",
        (920, 646),
        size=0.42,
        color=GRAY,
    )

    draw_text(canvas, f"{fps:.1f} FPS", (920, 679), size=0.42, color=GRAY)

    # Pie de página
    cv2.line(canvas, (35, 710), (1245, 710), (40, 40, 40), 1)
    draw_text(
        canvas,
        "Presiona Q para salir",
        (37, 738),
        size=0.4,
        color=GRAY,
    )
    draw_text(
        canvas,
        "Reconocimiento en tiempo real",
        (970, 738),
        size=0.4,
        color=GRAY,
    )

    return canvas


def main():
    parser = argparse.ArgumentParser(
        description="Reconocimiento de señas mediante landmarks."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="Checkpoint .pth; por defecto se usan los pesos incluidos.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Indice de webcam o ruta de un video.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    model.eval()

    classes = checkpoint["class_names"]
    frame_count = checkpoint["frame_count"]

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {args.source}")

    history = deque(maxlen=frame_count)

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    drawing = mp.solutions.drawing_utils
    latest_predictions = []
    previous_time = time.perf_counter()
    fps = 0.0
    frame_index = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)

            result = holistic.process(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            history.append(holistic_result_to_array(result))
            frame_index += 1

            hands_detected = bool(
                result.left_hand_landmarks or result.right_hand_landmarks
            )

            # Inferencia periódica para evitar procesar cada fotograma.
            if len(history) == frame_count and frame_index % 3 == 0:
                sequence = prepare_checkpoint_features(
                    np.asarray(history, dtype=np.float32),
                    checkpoint,
                )

                inputs = torch.from_numpy(sequence).unsqueeze(0).to(device)

                with torch.inference_mode():
                    probabilities = torch.softmax(model(inputs), dim=1)[0]

                k = min(3, probabilities.numel())
                top_probabilities, top_indices = torch.topk(probabilities, k=k)

                latest_predictions = [
                    (
                        display_class_name(classes[int(class_index)]),
                        float(probability),
                    )
                    for probability, class_index in zip(
                        top_probabilities.cpu().tolist(),
                        top_indices.cpu().tolist(),
                    )
                ]

            # Dibujar landmarks sobre el video original.
            if result.pose_landmarks:
                drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp.solutions.holistic.POSE_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=POSE_COLOR, thickness=2, circle_radius=2
                    ),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=POSE_COLOR, thickness=2
                    ),
                )

            if result.left_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    result.left_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=HAND_COLOR, thickness=2, circle_radius=3
                    ),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=HAND_COLOR, thickness=2
                    ),
                )

            if result.right_hand_landmarks:
                drawing.draw_landmarks(
                    frame,
                    result.right_hand_landmarks,
                    mp.solutions.holistic.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=CORAL, thickness=2, circle_radius=3
                    ),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=CORAL, thickness=2
                    ),
                )

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            if elapsed > 0:
                instantaneous_fps = 1.0 / elapsed
                fps = instantaneous_fps if fps == 0 else 0.85 * fps + 0.15 * instantaneous_fps

            interface = draw_interface(
                frame=frame,
                predictions=latest_predictions,
                history_length=len(history),
                frame_count=frame_count,
                hands_detected=hands_detected,
                fps=fps,
            )

            cv2.imshow("Reconocimiento de senas", interface)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        holistic.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
