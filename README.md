<div align="center">

# Reconocimiento de Lenguaje de Señas Colombiano mediante Aprendizaje Profundo

<img src="Banner e imágenes/Banner LSC.jpg" alt="Banner del proyecto" width="100%"/>

**Autores:** Laura Vanesa Rosas, Harold Esteban Duran, Yeison Steven Ovalle

</div>

---

## Tabla de contenido

1. [Motivación](#1-motivación)
2. [Alcance](#2-alcance)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Requisitos](#4-requisitos)
5. [Instalación](#5-instalación)
6. [Ejecución](#6-ejecución-en-tiempo-real)
7. [Referencias](#7-referencias)
8. [Licencia y citación](#8-licencia-y-citación)

---

## 1. Motivación

En Colombia, más de medio millón de personas sordas (DANE) enfrentan brechas de comunicación por la escasez de intérpretes de Lengua de Señas Colombiana (LSC). El aprendizaje profundo ofrece alternativas viables para mitigar este desafío mediante el procesamiento automático de imágenes y video.

## 2. Alcance

Implementar un sistema de reconocimiento del Lenguaje de Señas Colombiano (LSC), a partir de modelos de aprendizaje profundo.

| Fase | Entrada | Salida | 
|------|---------|--------|
| 1. Señas estáticas | Imagen / frame de cámara | Letra o número + confianza |
| 2. Señas dinámicas | Video (ventana de N frames) | Palabra del LSC | 


## 3. Estructura del repositorio

> Ajusta esta sección si los nombres de carpetas o scripts de tu repositorio difieren.

```
Reconocimiento-del-Lenguaje-de-Se-as-Colombiano/
├── Banner e imágenes/                 # Banner e imágenes del README
├── Codigo/
│   ├── preprocessing/      # Extracción de landmarks y normalización
│   ├── models/             # Definición de CNN / BiLSTM
│   ├── train.py            # Entrenamiento
│   ├── evaluate.py         # Evaluación
│   └── realtime.py         # Inferencia en tiempo real con webcam
├── SPOTER/                 # Base del transformer
├── requirements.txt
└── README.md
```

## 4. Requisitos

| Componente | Recomendado |
|------------|-------------|
| Sistema operativo | Windows 10/11, Ubuntu 20.04+ o macOS |
| Python | 3.9 – 3.11 |
| Cámara web | Necesaria para la ejecución en tiempo real |
| GPU (opcional) | NVIDIA con CUDA para acelerar el entrenamiento |
| Espacio en disco | ~10 GB para LSC70 y más si se descarga LSC50 completo |

Librerías principales: `tensorflow` (o `torch`), `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyter`.

## 5. Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/LauraRosas10/Reconocimiento-del-Lenguaje-de-Se-as-Colombiano.git
cd Reconocimiento-del-Lenguaje-de-Se-as-Colombiano

# 2. Crear y activar un entorno virtual
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt
```

Si no existe `requirements.txt`, instala manualmente:

```bash
pip install tensorflow mediapipe opencv-python numpy pandas scikit-learn matplotlib jupyter
```

Verificación rápida:

```bash
python -c "import cv2, mediapipe, tensorflow as tf; print('OK', tf.__version__)"
```


## 6. Referencias

1. Muñoz Galindez et al. *LSC70 dataset*. Data in Brief, 2025. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340924011752) · Mendeley Data DOI `10.17632/9ssyn8tff5.2`.
2. Flórez-Sierra et al. *LSC50 dataset*. Scientific Data, 2024. [Nature](https://www.nature.com/articles/s41597-024-04172-5) · Figshare DOI `10.6084/m9.figshare.27383016`.
3. *MediaPipe y CNN para el reconocimiento de gestos de LSA en tiempo real*, 2023. [arXiv:2305.05296](https://arxiv.org/pdf/2305.05296).
4. Bohacek & Hruz. *Sign Pose-based Transformer for Word-level Sign Language Recognition*, WACVW 2022. [PDF](https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf).
5. *Sistema de interpretación dinámica para signos alfanuméricos de la LSC mediante la combinación de redes de aprendizaje profundo*. Multimedia Tools and Applications. [Springer](https://link.springer.com/article/10.1007/s11042-023-17361-y).

## 7. Licencia y citación

