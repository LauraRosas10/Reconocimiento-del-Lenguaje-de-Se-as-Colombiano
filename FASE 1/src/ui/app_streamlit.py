"""
Dashboard Interactivo Neumórfico — Reconocimiento de Señas LSC70.
Desarrollado aplicando los principios de la skill UI/UX Pro Max:
- Contenedores Neumórficos delimitados (soft 3D depth) donde cada bloque de contenido
  está completamente encuadrado, sin cajas flotantes que actúen como simples líneas o divisores.
- Inversión cromática estricta de contraste:
    * Modo Dark: Fondo oscuro (#181b22), textos blancos/claros (#f8fafc) y secundarios (#94a3b8).
    * Modo Light: Fondo claro (#e6ecf4), textos negros/oscuros (#0f172a) y secundarios (#475569).
- Tipografía 'Inter' técnica, eliminación de emojis informales y distribución Bento Grid balanceada.
"""

import sys
from pathlib import Path
import time
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import streamlit as st

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models_saved"
DATA_P2_DIR = PROJECT_ROOT / "data" / "landmarks_phase2"
DATA_SKELETON_DIR = PROJECT_ROOT / "data" / "landmarks_skeleton"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase2_landmarks.normalizer import render_skeleton_from_landmarks

# Paleta de colores para landmarks sobre el feed de cámara
COLORS = {
    'wrist': (240, 240, 240),
    'thumb': (0, 165, 255),
    'index': (255, 215, 0),
    'middle': (0, 230, 118),
    'ring': (224, 64, 251),
    'pinky': (255, 82, 82)
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


def draw_styled_landmarks_and_handedness(frame, hand_landmarks, handedness_label, h, w):
    points = {}
    for idx, lm in enumerate(hand_landmarks.landmark):
        points[idx] = (int(lm.x * w), int(lm.y * h))

    for finger, conns in FINGER_CONNECTIONS.items():
        color = COLORS.get(finger, (200, 200, 200))
        for p1, p2 in conns:
            if p1 in points and p2 in points:
                cv2.line(frame, points[p1], points[p2], color, 2, cv2.LINE_AA)

    if 0 in points:
        cv2.circle(frame, points[0], 6, COLORS['wrist'], -1, cv2.LINE_AA)

    for finger, idx_list in FINGER_LANDMARKS.items():
        color = COLORS[finger]
        for idx in idx_list:
            if idx in points:
                rad = 7 if idx in [4, 8, 12, 16, 20] else 5
                cv2.circle(frame, points[idx], rad, color, -1, cv2.LINE_AA)

    if 0 in points:
        wx, wy = points[0]
        label = f"Mano {handedness_label}"
        cv2.putText(frame, label, (wx - 20, wy + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def extract_features(results, frame, draw=True):
    h, w, _ = frame.shape
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            raw_label = handedness.classification[0].label
            label_es = "Derecha" if raw_label == "Right" else "Izquierda"

            if draw:
                draw_styled_landmarks_and_handedness(frame, hand_lm, label_es, h, w)

            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32)
            wrist = coords[0].copy()
            centered = coords - wrist
            scale = np.linalg.norm(centered[9])
            if scale < 1e-6:
                scale = 1.0
            norm = (centered / scale).flatten()

            if raw_label == 'Left':
                left_hand = norm
            else:
                right_hand = norm

    return np.concatenate([left_hand, right_hand])


def get_hand_bbox(hand_landmarks, img_w, img_h, margin_ratio=0.25):
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    x1 = int(max(0, (min(x_coords) - margin_ratio * (max(x_coords) - min(x_coords))) * img_w))
    y1 = int(max(0, (min(y_coords) - margin_ratio * (max(y_coords) - min(y_coords))) * img_h))
    x2 = int(min(img_w, (max(x_coords) + margin_ratio * (max(x_coords) - min(x_coords))) * img_w))
    y2 = int(min(img_h, (max(y_coords) + margin_ratio * (max(y_coords) - min(y_coords))) * img_h))
    return x1, y1, x2, y2


# Configuración de página
st.set_page_config(
    page_title="LSC70 Recognition System",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_custom_neumorphic_theme(theme_mode="Dark"):
    """
    Inyecta estilos CSS Neumórficos con encuadre completo de texto y contraste riguroso:
    - Modo Light: Fondo claro, texto oscuro/negro (#0f172a, #334155).
    - Modo Dark: Fondo oscuro, texto blanco/claro (#f8fafc, #94a3b8).
    - Estructura limpia donde los textos y widgets viven dentro de tarjetas encuadradas unificadas.
    """
    if theme_mode == "Light":
        bg_main = "#ebf0f7"
        bg_card = "#ebf0f7"
        text_primary = "#0f172a"      # Negro / azul muy oscuro (alto contraste)
        text_secondary = "#475569"    # Gris oscuro legible
        text_muted = "#64748b"
        shadow_card = "6px 6px 14px #cad3df, -6px -6px 14px #ffffff"
        shadow_inset = "inset 3px 3px 6px #cad3df, inset -3px -3px 6px #ffffff"
        accent_color = "#2563eb"      # Azul vibrante accesible
        border_subtle = "rgba(255, 255, 255, 0.7)"
        card_border = "1px solid rgba(255, 255, 255, 0.8)"
    else:
        bg_main = "#14171f"
        bg_card = "#14171f"
        text_primary = "#f8fafc"      # Blanco puro / tiza
        text_secondary = "#cbd5e1"    # Gris claro legible
        text_muted = "#94a3b8"
        shadow_card = "6px 6px 16px #0b0d12, -6px -6px 16px #1d212c"
        shadow_inset = "inset 3px 3px 8px #0b0d12, inset -3px -3px 8px #1d212c"
        accent_color = "#60a5fa"      # Azul cielo
        border_subtle = "rgba(255, 255, 255, 0.05)"
        card_border = "1px solid rgba(255, 255, 255, 0.06)"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"], [class*="st-"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    .stApp {{
        background-color: {bg_main} !important;
        color: {text_primary} !important;
    }}

    /* Sobrescribir colores globales de Streamlit para asegurar contraste */
    .stApp p, .stApp span, .stApp label, .stApp div {{
        color: {text_primary};
    }}

    /* Sidebar neumórfico unificado */
    [data-testid="stSidebar"] {{
        background-color: {bg_main} !important;
        border-right: {card_border};
    }}
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {bg_main} !important;
    }}
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {{
        color: {text_primary} !important;
        font-weight: 500;
    }}

    /* Contenedor Card Neumórfico (Encuadre Completo) */
    .neu-card {{
        background: {bg_card};
        border-radius: 16px;
        box-shadow: {shadow_card};
        padding: 20px 24px;
        margin-bottom: 22px;
        border: {card_border};
        transition: transform 0.15s ease;
    }}

    /* Panel hundido (Inset / Concave) */
    .neu-inset {{
        background: {bg_card};
        border-radius: 12px;
        box-shadow: {shadow_inset};
        padding: 14px 18px;
        border: {card_border};
        color: {text_primary} !important;
        font-weight: 500;
    }}

    /* Títulos dentro de las tarjetas */
    .neu-title {{
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: {text_muted};
        margin: 0 0 10px 0;
    }}

    .neu-value {{
        font-size: 2.1rem;
        font-weight: 700;
        color: {accent_color};
        margin: 4px 0;
        line-height: 1.1;
    }}

    .neu-badge {{
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
        background: {bg_card};
        box-shadow: {shadow_card};
        color: {text_secondary};
    }}

    .neu-badge-active {{
        color: #10b981 !important;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }}

    /* Botones y controles nativos */
    .stButton>button {{
        border-radius: 12px !important;
        border: {card_border} !important;
        box-shadow: {shadow_card} !important;
        background: {bg_card} !important;
        color: {text_primary} !important;
        font-weight: 600 !important;
        padding: 10px 22px !important;
        transition: all 0.15s ease !important;
    }}

    .stButton>button:hover {{
        box-shadow: {shadow_inset} !important;
        color: {accent_color} !important;
    }}

    /* Sliders y radios */
    .stRadio > div {{
        background: {bg_card};
        padding: 10px;
        border-radius: 12px;
        box-shadow: {shadow_inset};
    }}

    /* Header encuadrado */
    .app-header-card {{
        background: {bg_card};
        border-radius: 16px;
        box-shadow: {shadow_card};
        padding: 24px 28px;
        margin-bottom: 24px;
        border: {card_border};
    }}

    .app-title {{
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {text_primary};
        margin: 0;
    }}

    .app-subtitle {{
        font-size: 0.95rem;
        color: {text_secondary};
        margin: 6px 0 0 0;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource
def load_phase2_model():
    model_path = MODELS_DIR / "phase2_landmarks_mobilenetv3.keras"
    if not model_path.exists():
        model_path = MODELS_DIR / "phase2_landmarks_lsc70.keras"

    classes_path = MODELS_DIR / "classes_phase2.txt"
    if not classes_path.exists():
        classes_path = DATA_SKELETON_DIR / "classes.txt"
    if not classes_path.exists():
        classes_path = DATA_P2_DIR / "classes.txt"

    if model_path.exists() and classes_path.exists():
        try:
            model = tf.keras.models.load_model(str(model_path))
            with open(classes_path, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
            return model, classes
        except Exception as e:
            st.sidebar.error(f"Fase 2 error: {e}")
    return None, None


@st.cache_resource
def load_phase1_model():
    model_path = MODELS_DIR / "phase1_mobilenetv3_lsc70.keras"
    classes_path = MODELS_DIR / "classes_phase1.txt"
    if model_path.exists() and classes_path.exists():
        try:
            model = tf.keras.models.load_model(str(model_path))
            with open(classes_path, "r", encoding="utf-8") as f:
                classes = [line.strip() for line in f if line.strip()]
            return model, classes
        except Exception as e:
            st.sidebar.error(f"Fase 1 error: {e}")
    return None, None


def main():
    # 1. Configuración de tema en la barra lateral (encuadrada)
    st.sidebar.markdown("""
    <div class='neu-card'>
        <p class='neu-title'>Apariencia Visual</p>
    </div>
    """, unsafe_allow_html=True)
    theme_choice = st.sidebar.radio(
        "Modo de Contraste:",
        ("Dark", "Light"),
        horizontal=True,
        label_visibility="collapsed"
    )

    inject_custom_neumorphic_theme(theme_choice)

    # 2. Header completamente encuadrado en tarjeta neumórfica
    st.markdown("""
    <div class="app-header-card">
        <h1 class="app-title">Sistema de Reconocimiento LSC70</h1>
        <p class="app-subtitle">Plataforma de Inferencia y Análisis en Tiempo Real — Lengua de Señas Colombiana</p>
    </div>
    """, unsafe_allow_html=True)

    # 3. Sidebar: Motor de inferencia
    st.sidebar.markdown("""
    <div class='neu-card'>
        <p class='neu-title'>Motor de Inferencia</p>
    </div>
    """, unsafe_allow_html=True)
    phase_option = st.sidebar.radio(
        "Selección de Pipeline:",
        ("Fase 2: Landmarks Esqueléticos (MobileNetV3)", "Fase 1: Recorte RGB (MobileNetV3)")
    )

    conf_threshold = st.sidebar.slider("Umbral de Confianza (%)", 30, 95, 60) / 100.0

    # 4. Sidebar: Estado de modelos
    p2_model, p2_classes = load_phase2_model()
    p1_model, p1_classes = load_phase1_model()

    p2_status = "Operativo" if p2_model else "No Disponible"
    p1_status = "Operativo" if p1_model else "No Disponible"
    p2_badge = "neu-badge-active" if p2_model else ""
    p1_badge = "neu-badge-active" if p1_model else ""

    st.sidebar.markdown(f"""
    <div class='neu-card'>
        <p class='neu-title'>Estado del Sistema</p>
        <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85rem; font-weight: 500;">Fase 2 (Esqueletos):</span> 
            <span class="neu-badge {p2_badge}">{p2_status}</span>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.85rem; font-weight: 500;">Fase 1 (RGB):</span> 
            <span class="neu-badge {p1_badge}">{p1_status}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 5. Distribución Principal: Feed (Izquierda) vs Métricas (Derecha)
    col_video, col_stats = st.columns([13, 10], gap="large")

    with col_video:
        st.markdown("""
        <div class='neu-card'>
            <p class='neu-title'>Transmisión de Video en Directo</p>
        </div>
        """, unsafe_allow_html=True)
        run_camera = st.toggle("Activar Dispositivo de Captura", value=False)
        frame_window = st.image([], use_container_width=True)

    with col_stats:
        # Tarjeta 1: Predicción Principal
        st.markdown("<div class='neu-card'>", unsafe_allow_html=True)
        st.markdown("<p class='neu-title'>Seña Identificada</p>", unsafe_allow_html=True)
        pred_placeholder = st.empty()
        conf_placeholder = st.empty()
        pred_placeholder.markdown("<div class='neu-value'>—</div>", unsafe_allow_html=True)
        conf_placeholder.markdown("<span class='neu-badge'>En espera de detección</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Tarjeta 2: Historial Acumulativo
        st.markdown("<div class='neu-card'>", unsafe_allow_html=True)
        st.markdown("<p class='neu-title'>Secuencia Traducida</p>", unsafe_allow_html=True)
        hist_placeholder = st.empty()
        hist_placeholder.markdown("<div class='neu-inset'>Sin actividad reciente</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Tarjeta 3: Distribución Top-5
        st.markdown("<div class='neu-card'>", unsafe_allow_html=True)
        st.markdown("<p class='neu-title'>Distribución de Probabilidad Top-5</p>", unsafe_allow_html=True)
        chart_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)

    # 6. Bucle de Cámara e Inferencia
    if run_camera:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        hands_detector = mp.solutions.hands.Hands(
            max_num_hands=2 if "Fase 2" in phase_option else 1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        current_model = p2_model if "Fase 2" in phase_option else p1_model
        current_classes = p2_classes if "Fase 2" in phase_option else p1_classes

        if current_model is None:
            st.error(f"El modelo correspondiente ({phase_option}) no se encuentra cargado.")
            cap.release()
            return

        recent_predictions = []

        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Error al establecer comunicación con el dispositivo de captura.")
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb_frame)

            preds = None
            detected_sign = "Sin detección"
            detected_conf = 0.0

            if "Fase 2" in phase_option:
                extract_features(results, frame, draw=True)
                is_mobilenet_p2 = (len(current_model.input_shape) == 4 and current_model.input_shape[-1] == 3)
                if is_mobilenet_p2:
                    if results.multi_hand_landmarks:
                        skeleton_canvas, has_hand = render_skeleton_from_landmarks(results.multi_hand_landmarks, canvas_size=224)
                        if has_hand:
                            preds = current_model.predict(np.expand_dims(skeleton_canvas, axis=0).astype(np.float32), verbose=0)[0]
                else:
                    features = extract_features(results, frame, draw=False)
                    if np.any(features != 0):
                        preds = current_model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
            else:
                if results.multi_hand_landmarks:
                    for h_lm in results.multi_hand_landmarks:
                        x1, y1, x2, y2 = get_hand_bbox(h_lm, w, h)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 118), 2)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            resized = cv2.resize(crop, (224, 224))
                            crop_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                            preds = current_model.predict(np.expand_dims(crop_rgb, axis=0).astype(np.float32), verbose=0)[0]

            if preds is not None:
                top_idx = int(np.argmax(preds))
                detected_conf = float(preds[top_idx])

                if detected_conf >= conf_threshold:
                    detected_sign = current_classes[top_idx]
                    if not recent_predictions or recent_predictions[-1] != detected_sign:
                        recent_predictions.append(detected_sign)
                        if len(recent_predictions) > 7:
                            recent_predictions.pop(0)

                pred_placeholder.markdown(f"<div class='neu-value'>{detected_sign}</div>", unsafe_allow_html=True)
                conf_placeholder.markdown(f"<span class='neu-badge neu-badge-active'>Confianza: {detected_conf * 100:.1f}%</span>", unsafe_allow_html=True)

                top5_indices = np.argsort(preds)[-5:][::-1]
                top5_data = {current_classes[i]: float(preds[i]) for i in top5_indices}
                chart_placeholder.bar_chart(top5_data, height=180)

                if recent_predictions:
                    hist_items = " &nbsp;&rarr;&nbsp; ".join([f"<strong>{p}</strong>" for p in recent_predictions])
                    hist_placeholder.markdown(f"<div class='neu-inset'>{hist_items}</div>", unsafe_allow_html=True)
            else:
                pred_placeholder.markdown("<div class='neu-value'>—</div>", unsafe_allow_html=True)
                conf_placeholder.markdown("<span class='neu-badge'>Sin manos en encuadre</span>", unsafe_allow_html=True)

            frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            time.sleep(0.01)

        cap.release()


if __name__ == "__main__":
    main()
