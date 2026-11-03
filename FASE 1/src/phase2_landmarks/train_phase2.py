"""
Script de entrenamiento y evaluacion completa para Fase 2 (MobileNetV3Large con Esqueletos de Landmarks).
Entrena la red convolucional sobre las imagenes de esqueleto 224x224x3 generadas en data/landmarks_skeleton/.
Evalua formalmente en test y exporta reportes y metricas.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Asegurar import de modulos locales
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from landmark_model import build_landmark_classifier

# Rutas del proyecto
PROJECT_ROOT = current_dir.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "landmarks_skeleton"
MODELS_DIR = PROJECT_ROOT / "models_saved"
RESULTS_DIR = PROJECT_ROOT / "results"


def plot_and_save_learning_curves(history, output_dir: Path):
    """
    Genera y guarda las curvas de aprendizaje:
    1. Loss (train vs val) para diagnóstico de sobreajuste.
    2. Accuracy (train vs val).
    3. Panel combinado de ambas curvas.
    4. Archivo JSON con el historial de metricas por epoca.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hist = history.history
    epochs_range = range(1, len(hist['loss']) + 1)

    # Guardar historial numerico en JSON
    history_file = output_dir / "history_phase2.json"
    serializable_hist = {k: [float(v) for v in vals] for k, vals in hist.items()}
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(serializable_hist, f, indent=4)
    print(f"[OK] Historial de entrenamiento exportado: {history_file.resolve()}")

    # 1. Grafica de Perdida (Loss vs Val Loss) - Diagnostico de Sobreajuste
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, hist['loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, hist['val_loss'], 's--', label='Validation Loss', color='#d62728', linewidth=2)

    best_loss_idx = int(np.argmin(hist['val_loss']))
    best_loss_epoch = best_loss_idx + 1
    best_val_loss = hist['val_loss'][best_loss_idx]
    plt.axvline(x=best_loss_epoch, color='gray', linestyle=':', label=f'Mejor Val Loss ({best_val_loss:.4f} en Epoca {best_loss_epoch})')
    plt.scatter([best_loss_epoch], [best_val_loss], color='red', s=90, zorder=5)

    plt.title('Curva de Perdida — Fase 2 (Diagnostico de Sobreajuste)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Loss (Crossentropy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_path = output_dir / "loss_curve_phase2.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"[OK] Grafica de Perdida guardada: {loss_path.resolve()}")

    # 2. Grafica de Accuracy (Train vs Val)
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, hist['accuracy'], 'o-', label='Train Accuracy', color='#2ca02c', linewidth=2)
    plt.plot(epochs_range, hist['val_accuracy'], 's--', label='Validation Accuracy', color='#ff7f0e', linewidth=2)

    best_acc_idx = int(np.argmax(hist['val_accuracy']))
    best_acc_epoch = best_acc_idx + 1
    best_val_acc = hist['val_accuracy'][best_acc_idx]
    plt.axvline(x=best_acc_epoch, color='gray', linestyle=':', label=f'Mejor Val Accuracy ({best_val_acc*100:.2f}% en Epoca {best_acc_epoch})')
    plt.scatter([best_acc_epoch], [best_val_acc], color='green', s=90, zorder=5)

    plt.title('Curva de Rendimiento / Accuracy — Fase 2', fontsize=14, fontweight='bold')
    plt.xlabel('Epoca', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    acc_path = output_dir / "accuracy_curve_phase2.png"
    plt.savefig(acc_path, dpi=200)
    plt.close()
    print(f"[OK] Grafica de Accuracy guardada: {acc_path.resolve()}")

    # 3. Panel Combinado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs_range, hist['loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs_range, hist['val_loss'], 's--', label='Val Loss', color='#d62728', linewidth=2)
    ax1.axvline(x=best_loss_epoch, color='gray', linestyle=':', label=f'Min Val Loss ({best_val_loss:.4f})')
    ax1.set_title('Perdida (Loss vs Val Loss)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoca', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.plot(epochs_range, hist['accuracy'], 'o-', label='Train Accuracy', color='#2ca02c', linewidth=2)
    ax2.plot(epochs_range, hist['val_accuracy'], 's--', label='Val Accuracy', color='#ff7f0e', linewidth=2)
    ax2.axvline(x=best_acc_epoch, color='gray', linestyle=':', label=f'Max Val Acc ({best_val_acc*100:.2f}%)')
    ax2.set_title('Exactitud (Accuracy vs Val Accuracy)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoca', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    fig.suptitle('Curvas de Aprendizaje y Validacion — Fase 2 (MobileNetV3 Esqueletos)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    combined_path = output_dir / "learning_curves_phase2.png"
    plt.savefig(combined_path, dpi=200)
    plt.close()
    print(f"[OK] Panel combinado de curvas guardado: {combined_path.resolve()}")


def train_and_evaluate_phase2(epochs=35, batch_size=32, img_size=(224, 224)):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(" ENTRENAMIENTO FASE 2 — MOBILENETV3 (ESQUELETOS LANDMARKS) ".center(60, "="))
    print("=" * 60)

    train_path = DATA_DIR / "train"
    val_path = DATA_DIR / "val"
    test_path = DATA_DIR / "test"

    if not train_path.exists() or not val_path.exists():
        print(f"\n[ERROR] No se encontro el dataset de esqueletos en: {DATA_DIR.resolve()}")
        print("Debes ejecutar primero la extraccion y generacion de esqueletos:")
        print("  python src/phase2_landmarks/extract_landmarks.py\n")
        return

    print(f"Cargando datasets desde: {DATA_DIR.resolve()}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(train_path),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(val_path),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_ds = None
    if test_path.exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            str(test_path),
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False
        )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Total clases detectadas: {num_classes}")

    # Guardar lista de clases oficial de Fase 2
    classes_file = MODELS_DIR / "classes_phase2.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    # Optimizar pipeline tf.data
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # 2. Construir modelo MobileNetV3Large
    model = build_landmark_classifier(input_shape=(*img_size, 3), num_classes=num_classes, trainable_base=False)
    model.summary()

    # 3. Callbacks con EarlyStopping y monitoreo estricto de sobreajuste
    model_path = MODELS_DIR / "phase2_landmarks_mobilenetv3.keras"
    legacy_model_path = MODELS_DIR / "phase2_landmarks_lsc70.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 4. Entrenar
    print("\nIniciando entrenamiento con MobileNetV3Large sobre Esqueletos...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Generar y guardar graficas de aprendizaje (Loss para sobreajuste, Accuracy/Rendimiento)
    plot_and_save_learning_curves(history, RESULTS_DIR)

    # Copiar checkpoint al path general de fase 2 para compatibilidad
    if model_path.exists():
        import shutil
        shutil.copy2(model_path, legacy_model_path)
    print(f"\n[OK] Modelo optimo guardado en: {model_path.resolve()}")

    # 5. Evaluacion en Test Set
    if test_ds is not None:
        print("\n" + "=" * 60)
        print(" EVALUACION EN TEST SET (METRICAS COMPARATIVAS) ".center(60, "="))
        print("=" * 60)

        best_model = tf.keras.models.load_model(str(model_path))

        y_true = []
        y_pred = []
        for images, labels in test_ds:
            preds = best_model.predict(images, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            y_true.extend(labels.numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        macro_prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

        print(f"\n--- METRICAS GLOBALES EN TEST (FASE 2 MOBILENETV3) ---")
        print(f"  Accuracy Test   : {acc * 100:.2f}%")
        print(f"  Macro F1-Score  : {macro_f1:.4f}")
        print(f"  Weighted F1     : {weighted_f1:.4f}")
        print(f"  Macro Precision : {macro_prec:.4f}")
        print(f"  Macro Recall    : {macro_rec:.4f}")

        # Guardar metricas en JSON para comparativa
        metrics = {
            "model": "Phase 2 - MobileNetV3 (Landmark Skeletons 224x224)",
            "test_accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "num_test_samples": int(len(y_true)),
            "num_classes": num_classes
        }

        metrics_file = RESULTS_DIR / "metrics_phase2.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n[OK] Metricas exportadas en JSON: {metrics_file.resolve()}")

        # Classification report detallado
        report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        report_file = RESULTS_DIR / "classification_report_phase2.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"[OK] Reporte por clase exportado: {report_file.resolve()}")

        # Matriz de confusion
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(14, 12))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Matriz de Confusion — Fase 2 (MobileNetV3 Esqueletos)')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
        plt.yticks(tick_marks, class_names, fontsize=8)
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Prediccion')
        plt.tight_layout()
        cm_path = RESULTS_DIR / "confusion_matrix_phase2.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[OK] Grafica de Matriz de Confusion guardada: {cm_path.resolve()}")

    print("\n" + "=" * 60)
    print(" FASE 2 COMPLETADA EXITOSAMENTE ".center(60, "="))
    print("=" * 60 + "\n")


if __name__ == '__main__':
    train_and_evaluate_phase2()

