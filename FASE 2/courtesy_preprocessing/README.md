# Preprocesamiento de Videos Courtesy

Este flujo trabaja aparte del ZIP original de landmarks y no modifica SPOTER. Lee los AVI del archivo comprimido, conserva el directorio numérico como participante, extrae 33 puntos de pose y 21 puntos por mano con MediaPipe, y escribe CSV con la convención:

```text
NNNN/NNNN_NNNN_NNNN.csv
 clase  participante  toma
```

El preparador conserva el participante que va en el nombre del archivo. Las clases se limitan a las 13 coincidencias directas y cuatro clases adicionales del conjunto Courtesy. Las cuatro clases adicionales quedan separadas aunque sean semánticamente próximas a señas del dataset original:

| Código | Carpeta Courtesy | Tratamiento |
|---|---|---|
| 0000 | Gracias | Coincidencia directa |
| 0001 | Buenos dias | Coincidencia directa |
| 0002 | Buenas tardes | Coincidencia directa |
| 0003 | Buenas Noches | Coincidencia directa |
| 0024 | Bien | Coincidencia directa |
| 0025 | Mal | Coincidencia directa |
| 0027 | Maso Menos | Coincidencia directa |
| 0030 | Hola | Coincidencia directa |
| 0032 | Por favor | Coincidencia directa |
| 0033 | Con gusto | Coincidencia directa |
| 0034 | Bienvenido | Coincidencia directa |
| 0035 | Perdon | Coincidencia directa |
| 0036 | Permiso | Coincidencia directa |
| 0050 | Como esta | Clase separada; no se fusiona con 0026 |
| 0051 | Chao | Clase separada; no se fusiona con 0031 |
| 0052 | Denada | Clase separada; no se fusiona con 0033 |
| 0053 | Lo siento | Clase separada; no se fusiona con 0035 |

Se inspeccionó el ZIP suministrado: 1.228 AVI y 43 rótulos distintos. El mapeo conserva 737 clips de 18 IDs de participante, sujeto a que el preprocesamiento y la decodificación final terminen correctamente. Estos totales son del inventario de rutas, no de landmarks validados.

Abre `entrenamiento_datasets_combinados.ipynb` desde Drive/Colab, revisa la ruta del ZIP y ejecuta las celdas en orden. Los CSV y el manifiesto se guardan bajo `courtesy_preprocessing/outputs`; dataset combinado y checkpoints se guardan bajo `data` y `checkpoints`. La celda de extracción puede tardar, pues ejecuta MediaPipe para cada video seleccionado.
