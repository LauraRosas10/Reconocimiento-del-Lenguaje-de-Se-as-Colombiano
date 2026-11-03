# 🤟 Reconocimiento LSC70 (TensorFlow/Keras + MediaPipe)

Sistema integral para el reconocimiento de señas estáticas de la Lengua de Señas Colombiana (37 clases en el dataset actual LSC70) implementado con TensorFlow 2.x, MediaPipe Hands oficial de Google (soporte de 1 o 2 manos con distinción de lateralidad) y una interfaz gráfica interactiva en Streamlit.

---

## 🏛️ Arquitectura Dual

El proyecto compara y ofrece dos enfoques complementarios de visión por computador:

1. **Fase 1 (`src/phase1_raw_images/`) — RGB Crudo con Paridad y Regularización:**
   - Detección de mano con Bounding Box holgado (25% margen).
   - Pipeline de **Data Augmentation** on-the-fly (rotación leve, escala, traslación, brillo y contraste, excluyendo flip horizontal).
   - Backbone **MobileNetV3Large** preentrenado con cabeza densa compacta (256 neuronas, Dropout 0.5 y regularización $L_2$).
   - Protocolo de entrenamiento en dos etapas: cabeza congelada + fine-tuning selectivo de 30 capas superiores.
   - **Rendimiento en Test:** **71.30% Accuracy** / **71.64% Macro F1**.

2. **Fase 2 (`src/phase2_landmarks/`) — Esqueletos Sintéticos 2D (MediaPipe + CNN):**
   - Extracción de landmarks articulares y detección de lateralidad (*Handedness*).
   - Rasterizado dinámico a imágenes sintéticas $224 \times 224 \times 3$ sobre fondo negro con conexiones codificadas por color.
   - Clasificación mediante **MobileNetV3Large** preentrenado (garantizando paridad metodológica y ceteris paribus con Fase 1).
   - **Rendimiento en Test:** **73.87% Accuracy** / **74.73% Macro F1**.

3. **Dashboard Web Interactivo (`src/ui/app_streamlit.py`):**
   - Interfaz Neumórfica (Soft UI) desarrollada con soporte persistente para modo Claro y Oscuro (WCAG AA).
   - Selector en caliente entre el motor de **Fase 1 (RGB)** y **Fase 2 (Esqueletos)**.
   - Distribución Bento Grid con métricas de confianza, historial de predicción y visualización Top-5 de probabilidades.

---

## 📁 Estructura del Repositorio

```text
LSC70ANH/
├── data/
│   ├── train/                          # Split de entrenamiento RGB (37 clases)
│   ├── val/                            # Split de validación RGB (37 clases)
│   ├── test/                           # Split de prueba RGB (37 clases)
│   └── landmarks_skeleton/             # Esqueletos 224x224 pre-renderizados
├── models_saved/
│   ├── phase1_mobilenetv3_lsc70.keras  # Modelo entrenado Fase 1 (RGB + Fine-Tuning)
│   ├── phase2_landmarks_mobilenetv3.keras # Modelo entrenado Fase 2 (Esqueletos)
│   ├── classes_phase1.txt              # Vocabulario de clases Fase 1
│   └── classes_phase2.txt              # Vocabulario de clases Fase 2
├── results/                            # Reportes numéricos, curvas y matrices de confusión
│   ├── metrics_phase1.json             # Métricas en Test Fase 1 (71.30% Acc)
│   ├── metrics_phase2.json             # Métricas en Test Fase 2 (73.87% Acc)
│   ├── learning_curves_phase1.png
│   ├── learning_curves_phase2.png
│   ├── confusion_matrix_phase1.png
│   └── confusion_matrix_phase2.png
├── src/
│   ├── common/
│   │   └── dataset_utils.py            # Utilidades de verificación de datos
│   ├── phase1_raw_images/
│   │   ├── mobilenet_model.py          # Definición MobileNetV3 + Data Aug + L2
│   │   ├── train_phase1.py             # Entrenamiento en 2 etapas (Feature Extraction + Fine-Tuning)
│   │   └── real_time_infer_phase1.py   # Inferencia en vivo con webcam (RGB + BBox)
│   ├── phase2_landmarks/
│   │   ├── normalizer.py               # Generación del lienzo 224x224 con esqueleto
│   │   ├── extract_landmarks.py        # Generación masiva offline de esqueletos
│   │   ├── landmark_model.py           # Definición MobileNetV3Large para esqueletos
│   │   ├── train_phase2.py             # Entrenamiento y evaluación Fase 2
│   │   └── real_time_infer.py          # Inferencia en vivo con webcam (Esqueletos + PiP)
│   └── ui/
│       └── app_streamlit.py            # Dashboard Neumórfico en Streamlit
└── requirements.txt
```

---

## 🚀 Guía Rápida de Ejecución

### 1. Inferencia en Tiempo Real por Webcam
- **Fase 1 (RGB):**
  ```powershell
  python src/phase1_raw_images/real_time_infer_phase1.py
  ```
- **Fase 2 (Esqueletos):**
  ```powershell
  python src/phase2_landmarks/real_time_infer.py
  ```

### 2. Dashboard Web (Streamlit)
```powershell
streamlit run src/ui/app_streamlit.py
```

### 3. Re-entrenamiento de Modelos
- **Entrenar Fase 1 (RGB optimizado en 2 etapas):**
  ```powershell
  python src/phase1_raw_images/train_phase1.py
  ```
- **Extraer y entrenar Fase 2 (Esqueletos):**
  ```powershell
  python src/phase2_landmarks/extract_landmarks.py
  python src/phase2_landmarks/train_phase2.py
  ```
