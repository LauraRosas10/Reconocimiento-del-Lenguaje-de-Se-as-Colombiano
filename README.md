<div align="center">

# Reconocimiento de Lenguaje de Señas Colombiano mediante Aprendizaje Profundo

<img src="Imagenes/Banner LSC.jpg" alt="Banner del proyecto" width="100%"/>

**Autores:** Laura Vanesa Rosas, Harold Esteban Duran, Yeison Steven Ovalle

</div>

---
ENLACE DIAPOSITIVAS: https://canva.link/buep2znmswtpwn6
## Tabla de contenido

1. [Motivación](#1-motivación)
2. [Alcance](#2-alcance)
3. [Requisitos](#4-requisitos)
4. [Referencias](#7-referencias)

---

## 1. Motivación

En Colombia, más de medio millón de personas sordas (DANE) enfrentan brechas de comunicación por la escasez de intérpretes de Lengua de Señas Colombiana (LSC). El aprendizaje profundo ofrece alternativas viables para mitigar este desafío mediante el procesamiento automático de imágenes y video.

## 2. Alcance

Implementar un sistema de reconocimiento del Lenguaje de Señas Colombiano (LSC), a partir de modelos de aprendizaje profundo.

| Fase | Entrada | Salida | 
|------|---------|--------|
| 1. Señas estáticas | Imagen / frame de cámara | Letra o número + confianza |
| 2. Señas dinámicas | Video (ventana de N frames) | Palabra del LSC | 


## 3. Requisitos

Librerías principales: `tensorflow` (o `torch`), `mediapipe`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `jupyter`.



## 4. Referencias

1. Muñoz Galindez et al. *LSC70 dataset*. Data in Brief, 2025. [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340924011752) · Mendeley Data DOI `10.17632/9ssyn8tff5.2`.
2. Flórez-Sierra et al. *LSC50 dataset*. Scientific Data, 2024. [Nature](https://www.nature.com/articles/s41597-024-04172-5) · Figshare DOI `10.6084/m9.figshare.27383016`.
3. *MediaPipe y CNN para el reconocimiento de gestos de LSA en tiempo real*, 2023. [arXiv:2305.05296](https://arxiv.org/pdf/2305.05296).
4. Bohacek & Hruz. *Sign Pose-based Transformer for Word-level Sign Language Recognition*, WACVW 2022. [PDF](https://openaccess.thecvf.com/content/WACV2022W/HADCV/papers/Bohacek_Sign_Pose-Based_Transformer_for_Word-Level_Sign_Language_Recognition_WACVW_2022_paper.pdf).



