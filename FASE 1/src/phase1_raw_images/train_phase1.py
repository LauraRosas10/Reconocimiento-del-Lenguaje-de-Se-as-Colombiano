"""
Script de entrenamiento y evaluacion completa para Fase 1 (MobileNetV3Large con Imagenes RGB).
Incorpora Data Augmentation, Regularizacion L2, Dropout 0.5 y protocolo de Fine-Tuning en 2 etapas.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

# Asegurar importacion de modulos locales
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from mobilenet_model import build_mobilenetv3_lsc70

# Rutas del proyecto
PROJECT_ROOT = current_dir.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models_saved"
RESULTS_DIR = PROJECT_ROOT / "results"


def plot_and_save_learning_curves_phase1(combined_history, output_dir: Path):
    """
    Genera y guarda las curvas de aprendizaje para la Fase 1:
    1. Loss (train vs val) para diagnóstico de sobreajuste.
    2. Accuracy (train vs val).
    3. Panel combinado de ambas curvas.
    4. Archivo JSON con el historial de métricas por época.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(combined_history['loss']) + 1)

    # Guardar historial numérico en JSON
    history_file = output_dir / "history_phase1.json"
    serializable_hist = {k: [float(v) for v in vals] for k, vals in combined_history.items()}
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(serializable_hist, f, indent=4)
    print(f"[OK] Historial Fase 1 exportado: {history_file.resolve()}")

    # 1. Gráfica de Pérdida (Loss vs Val Loss)
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, combined_history['loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    plt.plot(epochs_range, combined_history['val_loss'], 's--', label='Validation Loss', color='#d62728', linewidth=2)

    best_loss_idx = int(np.argmin(combined_history['val_loss']))
    best_loss_epoch = best_loss_idx + 1
    best_val_loss = combined_history['val_loss'][best_loss_idx]
    plt.axvline(x=best_loss_epoch, color='gray', linestyle=':', label=f'Mejor Val Loss ({best_val_loss:.4f} en Época {best_loss_epoch})')
    plt.scatter([best_loss_epoch], [best_val_loss], color='red', s=90, zorder=5)

    plt.title('Curva de Pérdida — Fase 1 RGB (Diagnóstico Anti-Overfitting)', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Loss (Crossentropy)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    loss_path = output_dir / "loss_curve_phase1.png"
    plt.savefig(loss_path, dpi=200)
    plt.close()
    print(f"[OK] Gráfica de Pérdida Fase 1 guardada: {loss_path.resolve()}")

    # 2. Gráfica de Accuracy (Train vs Val)
    plt.figure(figsize=(9, 6))
    plt.plot(epochs_range, combined_history['accuracy'], 'o-', label='Train Accuracy', color='#2ca02c', linewidth=2)
    plt.plot(epochs_range, combined_history['val_accuracy'], 's--', label='Validation Accuracy', color='#ff7f0e', linewidth=2)

    best_acc_idx = int(np.argmax(combined_history['val_accuracy']))
    best_acc_epoch = best_acc_idx + 1
    best_val_acc = combined_history['val_accuracy'][best_acc_idx]
    plt.axvline(x=best_acc_epoch, color='gray', linestyle=':', label=f'Mejor Val Accuracy ({best_val_acc*100:.2f}% en Época {best_acc_epoch})')
    plt.scatter([best_acc_epoch], [best_val_acc], color='green', s=90, zorder=5)

    plt.title('Curva de Rendimiento / Accuracy — Fase 1 RGB', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    acc_path = output_dir / "accuracy_curve_phase1.png"
    plt.savefig(acc_path, dpi=200)
    plt.close()
    print(f"[OK] Gráfica de Accuracy Fase 1 guardada: {acc_path.resolve()}")

    # 3. Panel Combinado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(epochs_range, combined_history['loss'], 'o-', label='Train Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs_range, combined_history['val_loss'], 's--', label='Val Loss', color='#d62728', linewidth=2)
    ax1.axvline(x=best_loss_epoch, color='gray', linestyle=':', label=f'Mín Val Loss ({best_val_loss:.4f})')
    ax1.set_title('Pérdida (Loss vs Val Loss)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Época', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)

    ax2.plot(epochs_range, combined_history['accuracy'], 'o-', label='Train Accuracy', color='#2ca02c', linewidth=2)
    ax2.plot(epochs_range, combined_history['val_accuracy'], 's--', label='Val Accuracy', color='#ff7f0e', linewidth=2)
    ax2.axvline(x=best_acc_epoch, color='gray', linestyle=':', label=f'Máx Val Acc ({best_val_acc*100:.2f}%)')
    ax2.set_title('Exactitud (Accuracy vs Val Accuracy)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Época', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)

    fig.suptitle('Curvas de Aprendizaje y Validación — Fase 1 (MobileNetV3 RGB Optimizado)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    combined_path = output_dir / "learning_curves_phase1.png"
    plt.savefig(combined_path, dpi=200)
    plt.close()
    print(f"[OK] Panel combinado Fase 1 guardado: {combined_path.resolve()}")


def merge_histories(h1, h2):
    """Fusiona los historiales de entrenamiento de la Etapa 1 y la Etapa 2 (Fine-Tuning)."""
    merged = {}
    keys = set(h1.history.keys()).union(set(h2.history.keys()))
    for k in keys:
        v1 = h1.history.get(k, [])
        v2 = h2.history.get(k, [])
        merged[k] = list(v1) + list(v2)
    return merged

def color_aug(x, y):
    """
    Augmentación de color más agresiva para simular condiciones de webcam:
    - Mayor variación de hue/saturación
    - Ruido gaussiano para simular ruido de sensor de cámara
    - Conversión aleatoria a escala de grises
    """
    x = x / 255.0
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.5, 1.5)
    # Ruido gaussiano para simular ruido de webcam
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02)
    x = x + noise
    x = tf.cond(tf.random.uniform([]) < 0.15,
                lambda: tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(x)),
                lambda: x)
    return tf.clip_by_value(x, 0.0, 1.0) * 255.0, y


def train_and_evaluate_phase1(
    stage1_epochs=12,
    stage2_epochs=40,
    batch_size=32,
    img_size=(224, 224),
    fine_tune_layers=60
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(" ENTRENAMIENTO OPTIMIZADO FASE 1 — MOBILENETV3 (RGB) ".center(65, "="))
    print("=" * 65)

    train_path = DATA_DIR / "train"
    val_path = DATA_DIR / "val"
    test_path = DATA_DIR / "test"

    print("Cargando datasets...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
    str(train_path), image_size=img_size, batch_size=batch_size,
    shuffle=True, label_mode="categorical", seed=42)
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Total clases detectadas: {num_classes}")

    val_ds = tf.keras.utils.image_dataset_from_directory(
    str(val_path), image_size=img_size, batch_size=batch_size,
    shuffle=False, label_mode="categorical", class_names=class_names)

    test_ds = None
    if test_path.exists():
        # FIX: Usar label_mode="categorical" y class_names para alinear con train/val
        test_ds = tf.keras.utils.image_dataset_from_directory(
            str(test_path),
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False,
            label_mode="categorical",
            class_names=class_names
        )



    classes_file = MODELS_DIR / "classes_phase1.txt"
    with open(classes_file, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = (train_ds.unbatch().map(color_aug, tf.data.AUTOTUNE)
            .batch(batch_size).prefetch(tf.data.AUTOTUNE))
    
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # 1. Construir modelo con base congelada
    print("\n[INFO] Construyendo modelo (Etapa 1: Backbone Congelado + Data Augmentation)...")
    model = build_mobilenetv3_lsc70(
        input_shape=(img_size[0], img_size[1], 3),
        num_classes=num_classes,
        trainable_base=False
    )
    model.summary()


    print("\n=== Información de capas ===")
    for i, layer in enumerate(model.layers):
        print(
            f"{i}: {layer.name} | "
            f"Tipo: {layer.__class__.__name__} | "
            f"Salida: {layer.output.shape}"
    )
        
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model_path = MODELS_DIR / "phase1_mobilenetv3_lsc70.keras"

    # Callbacks Etapa 1
    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
        )
    ]

    print(f"\n--- INICIO ETAPA 1: Entrenando Cabeza Densa (Max {stage1_epochs} épocas, LR=1e-3) ---")
    history_s1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    best_s1 = max(history_s1.history["val_accuracy"])
    tf.keras.callbacks.ModelCheckpoint(
        str(model_path), monitor="val_accuracy", mode="max",
        save_best_only=True, initial_value_threshold=best_s1)

    # 2. Etapa 2: Fine-Tuning Selectivo
    print(f"\n" + "-" * 65)
    print(f"--- INICIO ETAPA 2: Fine-Tuning de las ultimas {fine_tune_layers} capas de MobileNetV3 ---")
    print("-" * 65)

    # Cargar los mejores pesos encontrados en Etapa 1
    if model_path.exists():
        print(f"Cargando mejores pesos de Etapa 1 desde: {model_path.name}")
        model = tf.keras.models.load_model(str(model_path))

    # Obtener el sub-modelo backbone de MobileNetV3
    base_model = None
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        print("[ERROR] Capas del modelo:", [l.name for l in model.layers])
        raise RuntimeError("No se encontró el backbone; no se descongela el modelo completo.")

    base_model.trainable = True
    # Congelar todas excepto las ultimas 'fine_tune_layers'
    for layer in base_model.layers[:-fine_tune_layers]:
        layer.trainable = False

    # FIX CRÍTICO: Congelar TODAS las BatchNormalization del backbone
    # para preservar las estadísticas de ImageNet y evitar colapso
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    print(f"Capas descongeladas en Backbone: {fine_tune_layers} de {len(base_model.layers)}")

    # Recompilar con Learning Rate muy bajo para fine-tuning
    fine_tune_lr = 1e-5
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    
    bns = [l for l in base_model.layers
       if isinstance(l, tf.keras.layers.BatchNormalization)] if base_model else []
    bn_before = bns[-1].moving_mean.numpy().copy() if bns else None

    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history_s2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        callbacks=callbacks_stage2,
        verbose=1
    )
    
    if bns:
        print("BN moving_mean cambió:", not np.allclose(bn_before, bns[-1].moving_mean.numpy()))

    # Unir historiales de ambas etapas para las graficas
    combined_hist = merge_histories(history_s1, history_s2)
    



    plot_and_save_learning_curves_phase1(combined_hist, RESULTS_DIR)

    # 3. Evaluacion formal en Test Set
    if test_ds is not None:
        print("\n" + "=" * 65)
        print(" EVALUACION EN TEST SET (METRICAS COMPARATIVAS FASE 1) ".center(65, "="))
        print("=" * 65)

        best_model = tf.keras.models.load_model(str(model_path))

        y_true = []
        y_pred = []

        for images, labels in test_ds:
            preds = best_model.predict(images, verbose=0)
            y_pred.extend(np.argmax(preds, axis=1))
            # FIX: labels ahora son categorical (one-hot), convertir a índice
            y_true.extend(np.argmax(labels.numpy(), axis=1))

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        macro_prec = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

        print(f"\n--- METRICAS GLOBALES EN TEST (FASE 1 OPTIMIZADA) ---")
        print(f"  Accuracy Test   : {acc * 100:.2f}%")
        print(f"  Macro F1-Score  : {macro_f1:.4f}")
        print(f"  Weighted F1     : {weighted_f1:.4f}")
        print(f"  Macro Precision : {macro_prec:.4f}")
        print(f"  Macro Recall    : {macro_rec:.4f}")

        metrics = {
            "model": "Phase 1 - MobileNetV3Large RGB (DataAug + L2 + FineTuning)",
            "test_accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "num_test_samples": int(len(y_true)),
            "num_classes": num_classes
        }

        metrics_file = RESULTS_DIR / "metrics_phase1.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print(f"\n[OK] Metricas Fase 1 exportadas en JSON: {metrics_file.resolve()}")

        report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
        report_file = RESULTS_DIR / "classification_report_phase1.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_str)
        print(f"[OK] Reporte por clase exportado: {report_file.resolve()}")

        # Matriz de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        off = cm.copy(); np.fill_diagonal(off, 0)
        top = np.dstack(np.unravel_index(np.argsort(-off, axis=None)[:15], off.shape))[0]
        for i, j in top:
            print(f"{class_names[i]} -> {class_names[j]}: {off[i, j]}")
            
            
        plt.figure(figsize=(14, 12))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
        plt.title('Matriz de Confusion — Fase 1 (MobileNetV3 Optimizado)')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
        plt.yticks(tick_marks, class_names, fontsize=8)
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Prediccion')
        plt.tight_layout()
        cm_path = RESULTS_DIR / "confusion_matrix_phase1.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"[OK] Grafica de Matriz de Confusion guardada: {cm_path.resolve()}")

    print("\n" + "=" * 65)
    print(" ENTRENAMIENTO OPTIMIZADO FASE 1 COMPLETADO ".center(65, "="))
    print("=" * 65 + "\n")


if __name__ == '__main__':
    train_and_evaluate_phase1()
