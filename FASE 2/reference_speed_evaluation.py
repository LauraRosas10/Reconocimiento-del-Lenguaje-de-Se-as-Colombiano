from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import Image as NotebookImage
from IPython.display import display
from PIL import Image, ImageDraw

from landmark_recognition.data import HAND_LANDMARKS, LANDMARK_COUNT, POSE_LANDMARKS, resample_sequence
from landmark_recognition.inference import prepare_checkpoint_features


REFERENCE_SIGNER = "25"
REFERENCE_LABELS = {
    "gracias": ("0000", "GRACIAS"),
    "buenos dias": ("0001", "BUENOS DÍAS"),
    "buenas noches": ("0003", "BUENAS NOCHES"),
}
SPEED_FACTORS = (0.5, 1.0, 2.0)
EXTRACTOR_SIGNATURE = "mediapipe-solutions-holistic-0.10.21"
POSE_EDGES = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
)
HAND_EDGES = tuple(
    (start + offset, start + offset + 1)
    for start in (1, 5, 9, 13, 17)
    for offset in range(3)
) + ((0, 1), (0, 5), (0, 9), (0, 13), (0, 17))


def _result_to_landmarks(result) -> np.ndarray:
    sequence = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    for offset, detected, expected in (
        (0, getattr(result, "pose_landmarks", None), POSE_LANDMARKS),
        (POSE_LANDMARKS, getattr(result, "left_hand_landmarks", None), HAND_LANDMARKS),
        (POSE_LANDMARKS + HAND_LANDMARKS, getattr(result, "right_hand_landmarks", None), HAND_LANDMARKS),
    ):
        points = detected.landmark if hasattr(detected, "landmark") else detected
        if points is None:
            continue
        for index, point in enumerate(points[:expected]):
            sequence[offset + index] = (point.x, point.y, point.z)
    return sequence


def _extract_landmarks_for_reference(video_path: Path, namespace: dict) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV no pudo abrir el video: {video_path}")
    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frame_total < 1 or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise RuntimeError(f"Metadatos inválidos para {video_path.name}: frames={frame_total}, FPS={fps}")
    max_frames = int(namespace.get("MAX_EXTRACTION_FRAMES", 96))
    source_indices = np.unique(
        np.linspace(0, frame_total - 1, min(frame_total, max_frames)).round().astype(np.int64)
    )
    mp = namespace["mp"]
    landmarks = []
    try:
        if not namespace["USE_SOLUTIONS"]:
            raise RuntimeError(
                "La comparación requiere MediaPipe Solutions Holistic, el mismo "
                "backend usado para entrenar el checkpoint. Instala "
                "mediapipe==0.10.21 y reinicia el kernel; no uses MediaPipe Tasks."
            )
        detector = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        )

        with detector as landmarker:
            for source_index in source_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_index))
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(
                        f"No se pudo leer el frame {source_index} de {video_path.name}"
                    )
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.process(rgb)
                landmarks.append(_result_to_landmarks(result))
    finally:
        capture.release()
    if not landmarks:
        raise RuntimeError(f"No se extrajeron landmarks: {video_path.name}")
    return np.stack(landmarks).astype(np.float32, copy=False), source_indices


def _stretch_sequence(sequence: np.ndarray, speed_factor: float) -> np.ndarray:
    if speed_factor <= 0:
        raise ValueError("speed_factor debe ser positivo.")
    target_frames = max(2, int(round((len(sequence) - 1) / speed_factor)) + 1)
    source_time = np.linspace(0.0, 1.0, len(sequence))
    target_time = np.linspace(0.0, 1.0, target_frames)
    flat = sequence.reshape(len(sequence), -1)
    stretched = np.stack(
        [np.interp(target_time, source_time, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    )
    return stretched.reshape(target_frames, LANDMARK_COUNT, 3).astype(np.float32)


def _predict(sequence: np.ndarray, namespace: dict) -> tuple[np.ndarray, float]:
    frame_count = int(namespace["frame_count"])
    sampled = resample_sequence(
        sequence.reshape(len(sequence), LANDMARK_COUNT * 3),
        frame_count,
    ).reshape(frame_count, LANDMARK_COUNT, 3)
    features = prepare_checkpoint_features(sampled, namespace["checkpoint"])
    started = time.perf_counter()
    with torch.inference_mode():
        logits = namespace["model"](torch.from_numpy(features[None]))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probabilities, elapsed_ms


def _draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def point_xy(index: int) -> tuple[int, int]:
        x, y = landmarks[index, :2]
        return int(np.clip(x, 0, 1) * (width - 1)), int(np.clip(y, 0, 1) * (height - 1))

    parts = (
        (0, POSE_LANDMARKS, POSE_EDGES, "#42a5f5"),
        (POSE_LANDMARKS, POSE_LANDMARKS + HAND_LANDMARKS, HAND_EDGES, "#ffb300"),
        (POSE_LANDMARKS + HAND_LANDMARKS, LANDMARK_COUNT, HAND_EDGES, "#66bb6a"),
    )
    for start, end, edges, color in parts:
        for left, right in edges:
            left += start
            right += start
            if right >= end:
                continue
            if np.any(np.abs(landmarks[left]) > 1e-8) and np.any(np.abs(landmarks[right]) > 1e-8):
                draw.line((*point_xy(left), *point_xy(right)), fill=color, width=2)
        for index in range(start, end):
            if np.any(np.abs(landmarks[index]) > 1e-8):
                x, y = point_xy(index)
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
    return image


def _make_landmark_gif(video_path: Path, raw: np.ndarray, source_indices: np.ndarray, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video de referencia: {video_path}")
    sample_positions = np.unique(
        np.linspace(0, len(source_indices) - 1, min(18, len(source_indices))).round().astype(int)
    )
    images = []
    try:
        for position in sample_positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(source_indices[position]))
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"No se pudo leer el frame {source_indices[position]} de {video_path.name}")
            image = _draw_landmarks(frame, raw[position])
            image.thumbnail((360, 480))
            images.append(image)
    finally:
        capture.release()
    if not images:
        raise RuntimeError(f"No se generaron frames de vista previa para {video_path.name}")
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=110,
        loop=0,
        optimize=True,
    )


def _confusion_matrix(frame: pd.DataFrame, class_codes: list[str]) -> pd.DataFrame:
    matrix = pd.crosstab(frame["codigo_real"], frame["prediccion"])
    return matrix.reindex(index=class_codes, columns=class_codes, fill_value=0)


def _show_saved_test_baseline(namespace: dict, target_codes: list[str]) -> None:
    dataset_path = Path(namespace["PROJECT_DIR"]) / "data" / "landmarks_intersection.npz"
    summary_path = (
        Path(namespace["PROJECT_DIR"])
        / "reports"
        / "courtesy_intersection"
        / "training_summary.json"
    )
    if not dataset_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Faltan los landmarks y el resumen del test del checkpoint; "
            f"se requieren {dataset_path} y {summary_path}."
        )
    with summary_path.open(encoding="utf-8") as handle:
        training_summary = json.load(handle)
    if training_summary.get("test_signer") != "courtesy_videos::0025":
        raise ValueError(
            "El split guardado no reserva courtesy_videos::0025; no se puede afirmar "
            "que estos videos sean un test independiente para este checkpoint."
        )

    with np.load(dataset_path, allow_pickle=False) as archive:
        sequences = archive["X"].astype(np.float32)
        labels = archive["y"].astype(np.int64)
        signers = archive["signers"].astype(str)
        dataset_classes = archive["class_names"].astype(str).tolist()
    checkpoint_classes = [str(code) for code in namespace["class_codes"]]
    if dataset_classes != checkpoint_classes:
        raise ValueError("El orden de clases del dataset no coincide con el checkpoint.")

    target_indices = [
        index
        for index, (signer, label) in enumerate(zip(signers, labels))
        if signer == training_summary["test_signer"]
        and dataset_classes[int(label)] in target_codes
    ]
    if len(target_indices) != 9:
        raise ValueError(
            f"El split de prueba debía contener nueve clips de las tres señas; "
            f"se encontraron {len(target_indices)}."
        )
    features = np.stack(
        [
            prepare_checkpoint_features(sequences[index], namespace["checkpoint"])
            for index in target_indices
        ]
    )
    with torch.inference_mode():
        logits = namespace["model"](torch.from_numpy(features))
    predictions = logits.argmax(dim=1).cpu().numpy()
    targets = labels[target_indices]
    baseline = pd.DataFrame(
        {
            "codigo_real": [dataset_classes[int(label)] for label in targets],
            "prediccion": [dataset_classes[int(label)] for label in predictions],
        }
    )
    matrix = _confusion_matrix(baseline, target_codes)
    print(
        "Referencia checkpoint con los landmarks guardados del test original: "
        f"{int(np.sum(predictions == targets))}/9 correctos "
        f"(accuracy={np.mean(predictions == targets):.1%}); "
        f"test completo guardado={training_summary['test_correct']}/"
        f"{training_summary['test_count']} "
        f"({training_summary['test_accuracy']:.1%})."
    )
    print("Matriz de referencia con landmarks guardados (filas=real, columnas=predicción)")
    display(matrix)


def run_reference_speed_evaluation(
    namespace: dict,
    *,
    show_details: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive_path = Path(r"D:\Noveno\Fase 2\Videos Courtesy.zip")
    if not archive_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de videos etiquetados: {archive_path}")

    output_dir = Path(namespace["EXPERIMENT_DIR"]) / "referencia_prueba_firmante_0025"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "videos_prueba_manifest.csv"
    video_records = []

    with zipfile.ZipFile(archive_path) as archive:
        members = []
        for member in sorted(archive.namelist()):
            path = PurePosixPath(member)
            if len(path.parts) < 5 or path.parts[1] != REFERENCE_SIGNER or path.suffix.lower() not in {".avi", ".mp4"}:
                continue
            label = path.parent.name.strip().casefold()
            if label in REFERENCE_LABELS:
                members.append((member, REFERENCE_LABELS[label]))

        counts = {code: 0 for code, _ in REFERENCE_LABELS.values()}
        for member, (code, display_name) in members:
            path = PurePosixPath(member)
            video_dir = output_dir / code
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / path.name
            if not video_path.is_file():
                with archive.open(member) as source, video_path.open("wb") as target:
                    shutil.copyfileobj(source, target)

            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"OpenCV no pudo abrir el video extraído: {video_path}")
            frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            if frame_total < 1 or not np.isfinite(fps) or fps <= 0:
                raise RuntimeError(f"Metadatos inválidos para {video_path.name}: frames={frame_total}, FPS={fps}")

            extractor = namespace.get("extract_landmarks", _extract_landmarks_for_reference)
            if extractor is _extract_landmarks_for_reference:
                raw, source_indices = extractor(video_path, namespace)
            else:
                raw, source_indices = extractor(video_path)
            if raw.shape != (len(source_indices), LANDMARK_COUNT, 3):
                raise ValueError(f"Forma inesperada de landmarks en {video_path.name}: {raw.shape}")
            landmarks_path = output_dir / f"{code}_{video_path.stem}_landmarks.npz"
            np.savez_compressed(
                landmarks_path,
                landmarks=raw,
                source_frame_indices=source_indices,
                source_video=np.asarray(video_path.name),
                source_archive_member=np.asarray(member),
                extractor_signature=np.asarray(EXTRACTOR_SIGNATURE),
            )
            gif_path = output_dir / f"{code}_{video_path.stem}_landmarks.gif"
            _make_landmark_gif(video_path, raw, source_indices, gif_path)
            counts[code] += 1
            video_records.append(
                {
                    "archivo": video_path.name,
                    "video_path": video_path,
                    "landmarks_path": landmarks_path,
                    "gif_path": gif_path,
                    "codigo_real": code,
                    "sena": display_name,
                    "participante": REFERENCE_SIGNER,
                    "repeticion": video_path.stem,
                    "fotogramas": frame_total,
                    "fps": fps,
                    "duracion_s": frame_total / fps,
                    "landmarks": raw,
                    "source_indices": source_indices,
                }
            )

    expected_codes = set(counts)
    if any(count != 3 for count in counts.values()):
        raise ValueError(f"Se esperaban 3 videos por seña del firmante reservado; encontrados: {counts}")
    if len(video_records) != 9:
        raise ValueError(f"Se esperaban 9 videos en el split de prueba; encontrados: {len(video_records)}")

    catalog = pd.DataFrame(
        [{key: value for key, value in row.items() if key not in {"landmarks", "source_indices"}}
         for row in video_records]
    )
    catalog.drop(columns=["video_path", "landmarks_path", "gif_path"]).to_csv(
        manifest_path, index=False, encoding="utf-8-sig"
    )

    if show_details:
        print(f"Archivo fuente: {archive_path}")
        print("Firmante 0025: prueba reservada del checkpoint; 3 videos por clase.")
        print(f"Extractor del notebook: {namespace['EXTRACTOR_BACKEND']}")
        print(f"Videos y landmarks guardados en: {output_dir}")
        display(catalog[["archivo", "sena", "fotogramas", "fps", "duracion_s"]].round(3))

        print("Previsualización animada: video original con pose (azul), mano izquierda (naranja), mano derecha (verde).")
        for row in video_records:
            display(NotebookImage(filename=str(row["gif_path"])))
            print(f"{row['archivo']} | real={row['codigo_real']} {row['sena']} | {row['duracion_s']:.2f} s")

    model_codes = [str(code) for code in namespace["class_codes"]]
    if any(code not in model_codes for code in expected_codes):
        raise ValueError(f"El checkpoint no incluye alguna clase de prueba: {expected_codes - set(model_codes)}")
    if show_details:
        _show_saved_test_baseline(namespace, sorted(expected_codes))

    prediction_records = []
    for row in video_records:
        for speed_factor in SPEED_FACTORS:
            speed_sequence = _stretch_sequence(row["landmarks"], speed_factor)
            probabilities, inference_ms = _predict(speed_sequence, namespace)
            predicted_index = int(np.argmax(probabilities))
            top_indices = np.argsort(probabilities)[::-1][:3]
            prediction_records.append(
                {
                    "archivo": row["archivo"],
                    "codigo_real": row["codigo_real"],
                    "sena_real": row["sena"],
                    "factor_velocidad": speed_factor,
                    "duracion_simulada_s": row["duracion_s"] / speed_factor,
                    "prediccion": model_codes[predicted_index],
                    "acierto": model_codes[predicted_index] == row["codigo_real"],
                    "confianza_top1": float(probabilities[predicted_index]),
                    "probabilidad_real": float(probabilities[model_codes.index(row["codigo_real"])]),
                    "tiempo_inferencia_ms": inference_ms,
                    "top3": " | ".join(
                        f"{model_codes[index]}: {probabilities[index]:.1%}" for index in top_indices
                    ),
                }
            )

    predictions = pd.DataFrame(prediction_records)
    predictions_path = output_dir / "predicciones_por_velocidad.csv"
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    target_codes = sorted(expected_codes)
    summary = (
        predictions.groupby("factor_velocidad", sort=True)
        .agg(
            duracion_media_s=("duracion_simulada_s", "mean"),
            aciertos=("acierto", "sum"),
            total=("acierto", "count"),
            accuracy=("acierto", "mean"),
            confianza_media=("confianza_top1", "mean"),
            tiempo_inferencia_ms_medio=("tiempo_inferencia_ms", "mean"),
        )
        .reset_index()
    )
    if show_details:
        display(summary.round(3))

    for speed_factor in SPEED_FACTORS:
        speed_rows = predictions.loc[predictions["factor_velocidad"] == speed_factor]
        matrix = _confusion_matrix(speed_rows, target_codes)
        if show_details:
            print(f"Matriz de confusión | factor temporal {speed_factor:g}x")
            display(matrix)
        matrix.to_csv(output_dir / f"matriz_confusion_{speed_factor:g}x.csv", encoding="utf-8-sig")

    if show_details:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(summary["factor_velocidad"], summary["accuracy"], marker="o")
        axes[0].set(title="Acierto según velocidad temporal simulada", xlabel="Factor (1x = original)", ylabel="Accuracy", ylim=(0, 1.05), xticks=SPEED_FACTORS)
        axes[1].plot(summary["factor_velocidad"], summary["confianza_media"], marker="o", color="tab:orange")
        axes[1].set(title="Confianza media (no calibrada)", xlabel="Factor (1x = original)", ylabel="Probabilidad top-1", ylim=(0, 1.05), xticks=SPEED_FACTORS)
        fig.tight_layout()
        plt.show()

        print(f"Predicciones por velocidad: {predictions_path}")
        print(
            "Interpretación: el modelo recibe una secuencia remuestreada a una cantidad fija de frames y no recibe "
            "la duración en segundos. Por eso un cambio uniforme de velocidad puede conservar casi la misma predicción; "
            "la duración absoluta se analiza aparte en la tabla. Esta prueba evalúa estabilidad, no reentrena el modelo."
        )
    return catalog, predictions
