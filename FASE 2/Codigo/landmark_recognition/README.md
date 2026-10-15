# Reconocimiento de señas con landmarks

Este flujo independiente del proyecto SPOTER prepara tus secuencias CSV, entrena un Transformer o una alternativa recurrente LSTM, extrae landmarks de video/webcam con MediaPipe y predice sobre video con OpenCV. No requiere ni modifica archivos de SPOTER.

## LANDMARKS + Videos Courtesy (flujo recomendado nuevo)

Para entrenar con el ZIP de videos y conservar solo las clases acordadas, abre [entrenamiento_datasets_combinados.ipynb](./entrenamiento_datasets_combinados.ipynb) y ejecútalo en orden. Extrae los landmarks con MediaPipe, guarda los CSV y el manifiesto aparte en [courtesy_preprocessing/](./courtesy_preprocessing), prepara ambos datasets, limita LANDMARKS a las 13 coincidencias directas y añade cuatro señas Courtesy como clases nuevas e independientes: `0050` CÓMO ESTÁ (Courtesy), `0051` CHAO, `0052` DE NADA y `0053` LO SIENTO. No se fusionan con `0026` ¿CÓMO ESTÁS?, `0031` ADIÓS, `0033` CON GUSTO ni `0035` PERDÓN.

El inventario inicial de `Videos Courtesy.zip` encontró 1.228 videos y 43 rótulos distintos; 737 rutas corresponden a las 17 clases retenidas en 18 IDs de participante. El notebook vuelve a verificar los resultados efectivos de extracción, genera gráficas de distribución, curvas de aprendizaje y matriz de confusión, y guarda los gráficos y métricas bajo `reports/courtesy_intersection`. Los IDs de participante se prefijan por dataset para evitar colisiones entre fuentes.

El notebook busca automáticamente el proyecto y los ZIP tanto en Google Drive como en la carpeta del proyecto y sus directorios superiores al ejecutarse localmente. En local, abre el cuaderno desde `Fase 2`, `Codigo` o `landmark_recognition`; en Drive, guarda `Videos Courtesy.zip` en Mi unidad, `FASE 2 IA/Fase 2` o dentro de `landmark_recognition`. Si aún no existe `data/landmarks_dataset.npz`, también busca `LANDMARKS.zip` en esas ubicaciones. Las gráficas de distribución de extracción, balance de clases, curvas de entrenamiento y matriz de confusión, además del CSV de métricas por clase, quedan guardadas en `landmark_recognition/reports/courtesy_intersection`.

## Formato de entrada

Se usan 75 puntos por frame: 33 de pose y 21 de cada mano, con coordenadas `x`, `y`, `z` (225 valores por frame). No se incluye la cara, que tiene 468 puntos y no está usada por estos clasificadores.

El ZIP proporcionado se inspeccionó directamente: contiene 1.000 secuencias en `BODY_LANDMARKS`, 1.000 en `FACE_LANDMARKS` y 1.000 para cada mano. Los archivos alineados tienen nombres como `0000_0000_0000.csv`. La primera parte se utiliza como etiqueta de clase (`0000` a `0049`) y la segunda como identificador de participante. El mapeo de los 50 códigos a los nombres de señas proporcionados está en [class_labels.py](./class_labels.py); la webcam muestra esos nombres y los índices internos del modelo siguen siendo numéricos.

## Instalación

Se recomienda Python 3.11 en un entorno virtual (MediaPipe y PyTorch no siempre publican wheels para las versiones más recientes de Python):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r landmark_recognition/requirements.txt
```

## Preparar el ZIP de landmarks

Ejecuta desde la carpeta `Codigo`:

```powershell
python -m landmark_recognition.prepare_landmark_dataset `
  --input ..\LANDMARKS.zip `
  --output landmark_recognition\data\landmarks_dataset.npz `
  --frames 64
```

El preparador lee el ZIP sin extraerlo, alinea las tres modalidades por nombre, combina pose y manos, y remuestrea cada secuencia a 64 frames. El `.npz` contiene `X`, `y`, `signers`, `sample_ids` y `class_names`. En el ZIP inspeccionado hay etiquetas numéricas, no nombres de señas.

## Entrenar Transformer

Para preparar, entrenar y visualizar resultados desde Jupyter, abre [analisis_resultados.ipynb](./analisis_resultados.ipynb) y ejecuta sus celdas en orden. El cuaderno grafica la distribución de clases y participantes, las curvas de entrenamiento, la matriz de confusión del conjunto de prueba, métricas por clase y trayectorias de landmarks.

### Ejecutar en Google Colab con Drive

1. Copia la carpeta completa `landmark_recognition` (incluidos los archivos Python y el notebook) a `Mi unidad/FASE 2 IA/Fase 2/Codigo/landmark_recognition` o `Mi unidad/landmark_recognition`.
2. Sube `LANDMARKS.zip` a la raíz de Mi unidad, a `Mi unidad/FASE 2 IA/Fase 2` o a la carpeta `landmark_recognition`.
3. Abre `analisis_resultados.ipynb` desde Drive con Google Colab, autoriza el montaje de Drive y ejecuta las celdas en orden.

El notebook detecta Colab, monta Drive y busca automáticamente la ruta anidada del proyecto mostrada arriba o `Mi unidad/landmark_recognition`. Guarda el dataset y el checkpoint en las subcarpetas `data` y `checkpoints` del proyecto detectado. Si usas otras ubicaciones, cambia `PROJECT_CANDIDATES` o `ARCHIVE_CANDIDATES` en la primera celda. Colab proporciona PyTorch, NumPy y Matplotlib.

```powershell
python -m landmark_recognition.train_model `
  --dataset landmark_recognition\data\landmarks_dataset.npz `
  --architecture transformer `
  --output landmark_recognition\checkpoints\transformer.pth `
  --epochs 50
```

Con tres o más participantes, por defecto se reserva el último para prueba y el penúltimo para validación; el resto se usa para entrenar. En el ZIP actual hay cinco participantes y 50 clases, con 4 secuencias por clase y participante; por eso el split 600/200/200 es intencional y balanceado. Esta medida evalúa la generalización a un participante no visto y resulta más difícil que una partición aleatoria. Puedes elegir los participantes reservados con `--validation-signer` y `--test-signer`.

Para diagnosticar si el modelo aprende las clases al compartir participantes entre particiones, se puede solicitar una partición aleatoria estratificada por clase:

```powershell
python -m landmark_recognition.train_model `
  --dataset landmark_recognition\data\landmarks_dataset.npz `
  --architecture transformer `
  --output landmark_recognition\checkpoints\transformer_random.pth `
  --split-mode random `
  --epochs 100
```

El modo `random` es solo diagnóstico de aprendizaje: grabaciones del mismo participante estarán en entrenamiento, validación y prueba, por lo que su exactitud no representa generalización a firmantes no vistos. La línea base de prototipos sobre descriptores temporales del dataset actual obtuvo aproximadamente 19.5% en partición aleatoria frente a 6.5% con firmante reservado; esto evidencia un cambio de distribución por participante, no un problema con el tamaño del split. El notebook permite comparar ambos modos.

El entrenamiento centra cada frame en el punto medio de los hombros, escala por su distancia mediana y estandariza cada característica usando únicamente el conjunto de entrenamiento. El Transformer usa un token de clase para agregar información temporal; las métricas muestran pérdida/exactitud de entrenamiento y validación, además de reducir automáticamente la tasa de aprendizaje cuando la exactitud de validación se estanca. La matriz de confusión y exactitud por clase se calculan sobre el participante de prueba reservado. Estos cambios requieren reentrenar: los checkpoints de una versión anterior del Transformer no son compatibles.

La aumentación de datos está activada por defecto y se genera en línea durante cada época, solo para entrenamiento: deformación temporal monótona, interpolación de un pequeño tramo de frames, oclusión aleatoria leve de landmarks, pequeñas rotaciones en el plano, escalas, traslaciones y ruido gaussiano leve. No se duplican archivos ni se alteran las etiquetas. Validación y prueba usan siempre los landmarks originales; la exactitud de entrenamiento también se calcula con ejemplos originales para que sea comparable. Para desactivarla en CLI, añade `--no-augmentation`.

Para reducir el sobreajuste en el conjunto actual (solo 12 muestras originales por clase en entrenamiento), el Transformer usa una sola capa y 64 unidades ocultas, más dropout, weight decay y parada temprana por pérdida de validación. El learning rate se reduce si la pérdida de validación se estanca. El checkpoint se selecciona por menor pérdida de validación y conserva la época seleccionada.

La alternativa LSTM tiene instrucciones en [lstm_recurrente/README.md](./lstm_recurrente/README.md).

## Extraer landmarks desde video o webcam

Una grabación se guarda como CSV dentro de una carpeta cuyo nombre es la etiqueta:

```powershell
python -m landmark_recognition.extract_mediapipe `
  --source .\mi_video.mp4 `
  --label hola `
  --output-dir landmark_recognition\data_mediapipe `
  --display
```

Para abrir la webcam usa `--source 0`. Repite la captura para reunir varios ejemplos de cada clase. Puedes preparar los CSV de MediaPipe con el mismo convertidor, que asigna una clase por carpeta:

```powershell
python -m landmark_recognition.prepare_landmark_dataset `
  --input landmark_recognition\data_mediapipe `
  --output landmark_recognition\data\mis_senas.npz
```

Para evaluar en personas nuevas, conserva el participante de cada grabación. Usa `--signer` al capturar; se guardará en `participante/seña/archivo.csv`, una estructura que el preparador reconoce:

```powershell
python -m landmark_recognition.extract_mediapipe `
  --source 0 `
  --label 0000 `
  --signer participante_06 `
  --output-dir landmark_recognition\data_capturas `
  --display

python -m landmark_recognition.prepare_landmark_dataset `
  --input landmark_recognition\data_capturas `
  --output landmark_recognition\data\capturas_participante_06.npz
```

Graba varias tomas separadas por seña y repite con más participantes. El conjunto actual tiene solo 4 tomas por seña y participante, así que la diversidad real de firmantes y grabaciones aporta más que duplicar ejemplos sintéticamente. Combina el dataset original con las capturas nuevas:

```powershell
python -m landmark_recognition.merge_landmark_datasets `
  --inputs landmark_recognition\data\landmarks_dataset.npz landmark_recognition\data\capturas_participante_06.npz `
  --output landmark_recognition\data\landmarks_ampliado.npz
```

El combinador alinea clases por nombre, exige la misma cantidad de frames y conserva los IDs de participante. Entrena luego con `landmarks_ampliado.npz` y define explícitamente `--validation-signer` y `--test-signer` para reservar firmantes completos. No uses `--split-mode random` para afirmar generalización a personas nuevas.

```powershell
python -m landmark_recognition.train_model `
  --dataset landmark_recognition\data\landmarks_ampliado.npz `
  --architecture transformer `
  --split-mode subject `
  --signer-invariant `
  --validation-signer participante_05 `
  --test-signer participante_06 `
  --epochs 100 `
  --output landmark_recognition\checkpoints\transformer_ampliado.pth
```

En este ejemplo `participante_05` y `participante_06` no participan en el ajuste de pesos; uno se usa para selección/early stopping y el otro para prueba final. Sustituye ambos IDs por participantes que realmente estén presentes en el `.npz`. El nuevo checkpoint se genera solo después de ampliar el dataset y ejecutar el entrenamiento.

## Probar un checkpoint en webcam o video

```powershell
python -m landmark_recognition.predict_video `
  --checkpoint landmark_recognition\checkpoints\transformer.pth `
  --source 0
```

También se acepta una ruta de video como `--source .\prueba.mp4`. Pulsa `q` para cerrar la ventana. La predicción usa una ventana temporal de frames del checkpoint y muestra la clase y confianza. `predict_video` admite checkpoints Transformer, LSTM, BiGRU, Conv1D e híbridos; aplica la normalización relativa a muñeca y las velocidades cuando el checkpoint fue entrenado con esas opciones.

Para probar en la cámara local el checkpoint Transformer signer-invariant del experimento, descárgalo primero desde Drive a `Codigo\checkpoints` en la computadora y ejecuta desde `Codigo`:

```powershell
python -m landmark_recognition.predict_video `
  --checkpoint checkpoints\transformer_subject_invariant.pth `
  --source 0
```

`cv2.VideoCapture(0)` requiere ejecutar el comando en la computadora conectada a la webcam. No abre la webcam local desde el runtime remoto de Colab.

### Usar la webcam local desde el navegador de Colab

El runtime de Colab es remoto: `cv2.VideoCapture(0)` intenta abrir una cámara conectada al servidor y no la webcam de tu computadora. Para usar la cámara local, copia también `colab_webcam.py` a Drive y ejecuta la celda de instalación de MediaPipe 0.10.21 del notebook. Si MediaPipe ya estaba cargado en el runtime, reinícialo; luego vuelve a ejecutar la primera celda de configuración y la celda final de cámara. No hace falta volver a entrenar si ya tienes un checkpoint. La cámara solicita permiso al navegador, captura frames con JavaScript y ejecuta la inferencia en Python. Pulsa **Detener cámara** para finalizar. Puedes usarlo con un checkpoint Transformer o LSTM.

La inferencia aplica la misma normalización, estandarización guardada en el checkpoint y ventana temporal que el flujo de video local. Se presenta el top 3; la puntuación Softmax no está calibrada y una cámara en vivo no reemplaza la evaluación con nuevos ejemplos etiquetados. Para uso local con `--source 0`, ejecuta `predict_video` en el mismo equipo que tiene conectada la webcam.

## Nota sobre el SPOTER original

El `train.py` original de este repositorio consume un CSV agregado con un esquema de puntos 2D específico. Los landmarks de este ZIP son CSV separados por modalidad y contienen 3 coordenadas, por lo que este flujo independiente normaliza el formato y no modifica el código de SPOTER.
