import numpy as np
import cv2
import mediapipe as mp

def extract_and_normalize_2hands(image_bgr, hands_detector):
    """
    Extrae landmarks de hasta 2 manos (126 dimensiones).
    Si solo detecta 1 mano, rellena la otra con ceros.
    (Mantenido para compatibilidad).
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label # 'Left' or 'Right'
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)

            # Normalización (Centrar en muñeca y escalar por nudillo)
            wrist = coords[0].copy()
            centered = coords - wrist
            scale = np.linalg.norm(centered[9])
            if scale < 1e-6: scale = 1.0
            normalized = (centered / scale).flatten()

            if label == 'Left':
                left_hand = normalized
            else:
                right_hand = normalized

    # Vector fijo de 126 características (si no hay manos, retorna vector de ceros)
    return np.concatenate([left_hand, right_hand])


# Paleta de colores RGB distintiva por articulación / dedo (Formato RGB nativo para MobileNetV3)
COLORS = {
    'wrist': (255, 255, 255),      # Blanco (Muñeca)
    'palm': (200, 200, 200),       # Gris claro (Palma)
    'thumb': (255, 165, 0),        # Naranja (R=255, G=165, B=0)
    'index': (0, 255, 255),        # Cyan (R=0, G=255, B=255)
    'middle': (0, 255, 0),         # Verde (R=0, G=255, B=0)
    'ring': (255, 0, 255),         # Magenta / Rosa (R=255, G=0, B=255)
    'pinky': (255, 69, 0)          # Rojo-Naranja (R=255, G=69, B=0)
}


FINGER_CONNECTIONS = {
    'thumb': [(0, 1), (1, 2), (2, 3), (3, 4)],
    'index': [(0, 5), (5, 6), (6, 7), (7, 8)],
    'middle': [(9, 10), (10, 11), (11, 12)],
    'ring': [(13, 14), (14, 15), (15, 16)],
    'pinky': [(0, 17), (17, 18), (18, 19), (19, 20)],
    'palm': [(5, 9), (9, 13), (13, 17)]
}

FINGER_LANDMARKS = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}


def render_skeleton_from_landmarks(multi_hand_landmarks, canvas_size=224, padding=16):
    """
    Renderiza los landmarks detectados sobre un lienzo negro de (canvas_size, canvas_size, 3).
    Escala y centra las manos manteniendo la proporción espacial relativa.
    """
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    if not multi_hand_landmarks:
        return canvas, False

    # Recolectar todas las coordenadas de todas las manos visibles
    all_x = []
    all_y = []
    for hand_lm in multi_hand_landmarks:
        for lm in hand_lm.landmark:
            all_x.append(lm.x)
            all_y.append(lm.y)

    if not all_x or not all_y:
        return canvas, False

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    box_w = max(max_x - min_x, 1e-5)
    box_h = max(max_y - min_y, 1e-5)

    # Calcular factor de escala para ajustar al lienzo con margen
    target_span = canvas_size - 2 * padding
    scale = target_span / max(box_w, box_h)

    # Centrar en el lienzo
    offset_x = padding + (target_span - box_w * scale) / 2.0
    offset_y = padding + (target_span - box_h * scale) / 2.0

    # Dibujar cada mano
    for hand_lm in multi_hand_landmarks:
        points = {}
        for idx, lm in enumerate(hand_lm.landmark):
            px = int(offset_x + (lm.x - min_x) * scale)
            py = int(offset_y + (lm.y - min_y) * scale)
            px = np.clip(px, 0, canvas_size - 1)
            py = np.clip(py, 0, canvas_size - 1)
            points[idx] = (px, py)

        # Dibujar líneas óseas por dedo
        for finger, conns in FINGER_CONNECTIONS.items():
            color = COLORS[finger]
            for p1, p2 in conns:
                if p1 in points and p2 in points:
                    cv2.line(canvas, points[p1], points[p2], color, 2, cv2.LINE_AA)

        # Dibujar articulaciones
        if 0 in points:
            cv2.circle(canvas, points[0], 5, COLORS['wrist'], -1, cv2.LINE_AA)

        for finger, idx_list in FINGER_LANDMARKS.items():
            color = COLORS[finger]
            for idx in idx_list:
                if idx in points:
                    radius = 5 if idx in [4, 8, 12, 16, 20] else 4
                    cv2.circle(canvas, points[idx], radius, color, -1, cv2.LINE_AA)

    return canvas, True


def draw_styled_hand_with_handedness(frame_bgr, hand_landmarks, label_es, h, w):
    """
    Dibuja los landmarks y conexiones de una mano sobre el frame BGR en vivo.
    Colorea cada dedo según la paleta COLORS y muestra la etiqueta de handedness.

    Args:
        frame_bgr: Frame BGR de la cámara (modificado in-place).
        hand_landmarks: Objeto landmarks de MediaPipe para una mano.
        label_es: Etiqueta en español ('Derecha' o 'Izquierda').
        h: Alto del frame.
        w: Ancho del frame.
    """
    # Convertir landmarks normalizados a píxeles
    points = {}
    for idx, lm in enumerate(hand_landmarks.landmark):
        px = int(lm.x * w)
        py = int(lm.y * h)
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        points[idx] = (px, py)

    # Dibujar conexiones por dedo (BGR: invertir R y B de la paleta RGB)
    for finger, conns in FINGER_CONNECTIONS.items():
        rgb_color = COLORS[finger]
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        for p1, p2 in conns:
            if p1 in points and p2 in points:
                cv2.line(frame_bgr, points[p1], points[p2], bgr_color, 2, cv2.LINE_AA)

    # Dibujar muñeca
    if 0 in points:
        wrist_rgb = COLORS['wrist']
        wrist_bgr = (wrist_rgb[2], wrist_rgb[1], wrist_rgb[0])
        cv2.circle(frame_bgr, points[0], 6, wrist_bgr, -1, cv2.LINE_AA)

    # Dibujar articulaciones por dedo
    for finger, idx_list in FINGER_LANDMARKS.items():
        rgb_color = COLORS[finger]
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
        for idx in idx_list:
            if idx in points:
                radius = 6 if idx in [4, 8, 12, 16, 20] else 4
                cv2.circle(frame_bgr, points[idx], radius, bgr_color, -1, cv2.LINE_AA)

    # Etiqueta de handedness cerca de la muñeca
    if 0 in points:
        label_color_bgr = (0, 165, 255) if label_es == "Derecha" else (255, 165, 0)
        cv2.putText(frame_bgr, label_es, (points[0][0] - 20, points[0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color_bgr, 2, cv2.LINE_AA)


def render_skeleton_image(image_bgr, hands_detector, canvas_size=224, padding=16):
    """
    Procesa un frame BGR con MediaPipe y genera la imagen esquelética sintética (224, 224, 3)
    optimizada para entrada directa a MobileNetV3.
    """
    if image_bgr is None or not hasattr(image_bgr, 'shape') or image_bgr.size == 0:
        return np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8), False

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)
    if results.multi_hand_landmarks:
        return render_skeleton_from_landmarks(results.multi_hand_landmarks, canvas_size=canvas_size, padding=padding)
    return np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8), False


