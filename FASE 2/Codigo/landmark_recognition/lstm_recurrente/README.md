# Alternativa recurrente LSTM

Este directorio contiene un clasificador recurrente bidireccional LSTM para secuencias de landmarks. Comparte el formato de datos, la preparacion, la validacion por participante y el script de prediccion con el modelo Transformer.

Para entrenar y visualizar curvas y métricas en Google Colab, abre [entrenamiento_lstm.ipynb](./entrenamiento_lstm.ipynb) y ejecuta las celdas en orden. El notebook monta Drive y guarda `checkpoints/lstm_subject.pth` en la carpeta del proyecto. Copia primero la carpeta `landmark_recognition` completa a Drive.

Desde la carpeta `Codigo`, ejecuta:

```powershell
python -m landmark_recognition.lstm_recurrente.train_lstm `
  --dataset landmark_recognition/data/landmarks_dataset.npz `
  --output landmark_recognition/checkpoints/lstm.pth `
  --epochs 50
```

Para probar el checkpoint con webcam:

```powershell
python -m landmark_recognition.predict_video `
  --checkpoint landmark_recognition/checkpoints/lstm.pth `
  --source 0
```
