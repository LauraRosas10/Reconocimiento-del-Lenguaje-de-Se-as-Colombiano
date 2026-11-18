from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageOps

from landmark_recognition.data import HAND_LANDMARKS, LANDMARK_COUNT, POSE_LANDMARKS
from landmark_recognition.reference_speed_evaluation import (
    EXTRACTOR_SIGNATURE,
    REFERENCE_LABELS,
    SPEED_FACTORS,
    _draw_landmarks,
    _predict,
    _stretch_sequence,
    run_reference_speed_evaluation,
)


REFERENCE_DIRNAME = "referencia_prueba_firmante_0025"
COMPARISON_DIRNAME = "comparacion_firmante_0025_vs_propio"
TARGET_LABELS = {code: name for code, name in REFERENCE_LABELS.values()}
GROUP_LABELS = {
    "firmante25": "Firmante 0025 (test)",
    "propio": "Videos propios",
}
GROUP_COLORS = {"firmante25": "tab:blue", "propio": "tab:orange"}
LANDMARK_PHASES = (0.2, 0.5, 0.8)


def _read_landmarks(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de landmarks: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if (
            "landmarks" not in archive
            or "source_frame_indices" not in archive
            or "extractor_signature" not in archive
            or str(archive["extractor_signature"].item()) != EXTRACTOR_SIGNATURE
        ):
            raise ValueError(f"El archivo no contiene landmarks e índices de frames: {path}")
        landmarks = archive["landmarks"].astype(np.float32, copy=False)
        source_indices = archive["source_frame_indices"].astype(np.int64, copy=False)
    if landmarks.ndim != 3 or landmarks.shape[1:] != (LANDMARK_COUNT, 3):
        raise ValueError(f"Forma inesperada en {path.name}: {landmarks.shape}")
    if len(landmarks) != len(source_indices) or not len(landmarks):
        raise ValueError(f"Frames y landmarks no coinciden en {path.name}.")
    if not np.isfinite(landmarks).all():
        raise ValueError(f"Hay valores no finitos en los landmarks de {path.name}.")
    return landmarks, source_indices


def _video_metadata(video_path: Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(width, height, frame_count) < 1 or not np.isfinite(fps) or fps <= 0:
        raise RuntimeError(
            f"Metadatos inválidos en {video_path.name}: "
            f"{width}x{height}, {frame_count} frames, {fps} FPS."
        )
    return {
        "ancho_px": width,
        "alto_px": height,
        "fotogramas": frame_count,
        "fps": fps,
        "duracion_s": frame_count / fps,
        "relacion_aspecto": width / height,
    }


def _landmark_metrics(
    landmarks: np.ndarray,
    width: int,
    height: int,
) -> dict[str, float]:
    visibility = np.any(np.abs(landmarks) > 1e-8, axis=2)
    metrics: dict[str, float] = {
        "cobertura_pose": float(visibility[:, :POSE_LANDMARKS].mean()),
        "cobertura_mano_izquierda": float(
            visibility[:, POSE_LANDMARKS : POSE_LANDMARKS + HAND_LANDMARKS].mean()
        ),
        "cobertura_mano_derecha": float(
            visibility[:, POSE_LANDMARKS + HAND_LANDMARKS :].mean()
        ),
    }

    pose = landmarks[:, :POSE_LANDMARKS].copy()
    required = pose[:, [11, 12, 23, 24]]
    valid_points = np.any(np.abs(required) > 1e-8, axis=2)
    valid_frames = valid_points.all(axis=1)
    if not valid_frames.any():
        raise ValueError("No hay frames con hombros y caderas detectados para medir orientación.")

    # MediaPipe normaliza x por el ancho y y por el alto; corregir la relación
    # de aspecto evita sesgar las medidas al comparar videos verticales y horizontales.
    aspect = width / height
    pose[:, :, 0] *= aspect
    pose[:, :, 2] *= aspect
    pose = pose[valid_frames]
    left_shoulder, right_shoulder = pose[:, 11], pose[:, 12]
    left_hip, right_hip = pose[:, 23], pose[:, 24]

    shoulder_vector = right_shoulder - left_shoulder
    hip_vector = right_hip - left_hip
    shoulder_span = np.linalg.norm(shoulder_vector[:, :2], axis=1)
    hip_span = np.linalg.norm(hip_vector[:, :2], axis=1)
    shoulder_midpoint = (left_shoulder[:, :2] + right_shoulder[:, :2]) * 0.5
    hip_midpoint = (left_hip[:, :2] + right_hip[:, :2]) * 0.5
    torso_height = np.linalg.norm(hip_midpoint - shoulder_midpoint, axis=1)

    shoulder_depth_asymmetry = np.abs(
        right_shoulder[:, 2] - left_shoulder[:, 2]
    ) / np.maximum(shoulder_span, 1e-6)
    hip_depth_asymmetry = np.abs(right_hip[:, 2] - left_hip[:, 2]) / np.maximum(
        hip_span, 1e-6
    )
    orientation_proxy = (shoulder_depth_asymmetry + hip_depth_asymmetry) * 0.5
    shoulder_angle = np.abs(
        np.degrees(np.arctan2(shoulder_vector[:, 1], shoulder_vector[:, 0]))
    )
    shoulder_tilt = np.minimum(shoulder_angle % 180, 180 - shoulder_angle % 180)

    metrics.update(
        {
            "frames_orientacion_validos": int(valid_frames.sum()),
            "proxy_oblicuidad_mediana": float(np.median(orientation_proxy)),
            "proxy_oblicuidad_p75": float(np.percentile(orientation_proxy, 75)),
            "ancho_hombros_altura_torso": float(
                np.median(shoulder_span / np.maximum(torso_height, 1e-6))
            ),
            "inclinacion_hombros_deg": float(np.median(shoulder_tilt)),
        }
    )
    return metrics


def _reference_video_paths(reference_dir: Path) -> list[tuple[str, Path]]:
    videos: list[tuple[str, Path]] = []
    for code in TARGET_LABELS:
        class_dir = reference_dir / code
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in {".avi", ".mp4"}:
                videos.append((code, path))
    return videos


def _ensure_reference_landmarks(
    namespace: dict[str, Any],
    reference_dir: Path,
) -> list[tuple[str, Path]]:
    videos = _reference_video_paths(reference_dir)
    complete = len(videos) == 9 and all(
        (reference_dir / f"{code}_{video.stem}_landmarks.npz").is_file()
        and _has_current_extractor_signature(
            reference_dir / f"{code}_{video.stem}_landmarks.npz"
        )
        for code, video in videos
    )
    if not complete:
        run_reference_speed_evaluation(namespace, show_details=False)
        videos = _reference_video_paths(reference_dir)
    if len(videos) != 9 or not all(
        (reference_dir / f"{code}_{video.stem}_landmarks.npz").is_file()
        and _has_current_extractor_signature(
            reference_dir / f"{code}_{video.stem}_landmarks.npz"
        )
        for code, video in videos
    ):
        raise RuntimeError(
            "La referencia de firmante 0025 debe tener nueve videos y nueve "
            "archivos de landmarks completos."
        )
    return videos


def _has_current_extractor_signature(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                "extractor_signature" in archive
                and str(archive["extractor_signature"].item()) == EXTRACTOR_SIGNATURE
            )
    except (OSError, ValueError):
        return False


def _load_own_records(
    namespace: dict[str, Any],
    target_codes: set[str],
) -> list[dict[str, Any]]:
    video_dir = Path(namespace["VIDEO_DIR"])
    landmark_dir = Path(namespace["LANDMARK_DIR"])
    filename_pattern = namespace.get("FILENAME_PATTERN")
    videos = sorted(
        path for path in video_dir.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    )
    if not videos:
        raise FileNotFoundError(f"No hay videos propios en {video_dir}")

    records: list[dict[str, Any]] = []
    for video_path in videos:
        match = filename_pattern.match(video_path.stem) if filename_pattern else None
        if match is None:
            raise ValueError(
                f"El nombre del video no sigue el patrón participante_repetición_seña: "
                f"{video_path.name}"
            )
        values = match.groupdict()
        code = str(values["sign"])
        if code not in target_codes:
            continue

        landmarks_path = landmark_dir / f"{video_path.stem}.npz"
        if not landmarks_path.is_file() or not _has_current_extractor_signature(landmarks_path):
            extractor = namespace.get("extract_landmarks")
            if not callable(extractor):
                raise FileNotFoundError(
                    f"Faltan landmarks para {video_path.name}. Ejecuta primero "
                    "la sección de extracción de landmarks propios."
                )
            landmarks, source_indices = extractor(video_path)
            np.savez_compressed(
                landmarks_path,
                landmarks=landmarks,
                source_frame_indices=source_indices,
                source_video=np.asarray(video_path.name),
                extractor_signature=np.asarray(EXTRACTOR_SIGNATURE),
            )

        landmarks, source_indices = _read_landmarks(landmarks_path)
        metadata = _video_metadata(video_path)
        records.append(
            _make_record(
                group="propio",
                code=code,
                participant=str(values["participant"]),
                repetition=str(values["repetition"]),
                video_path=video_path,
                landmarks_path=landmarks_path,
                landmarks=landmarks,
                source_indices=source_indices,
                metadata=metadata,
            )
        )
    return records


def _make_record(
    *,
    group: str,
    code: str,
    participant: str,
    repetition: str,
    video_path: Path,
    landmarks_path: Path,
    landmarks: np.ndarray,
    source_indices: np.ndarray,
    metadata: dict[str, float | int],
) -> dict[str, Any]:
    metrics = _landmark_metrics(
        landmarks,
        int(metadata["ancho_px"]),
        int(metadata["alto_px"]),
    )
    return {
        "grupo": group,
        "grupo_nombre": GROUP_LABELS[group],
        "codigo_real": code,
        "sena": TARGET_LABELS[code],
        "participante": participant,
        "repeticion": repetition,
        "archivo": video_path.name,
        "ruta_video": str(video_path),
        "ruta_landmarks": str(landmarks_path),
        **metadata,
        "fotogramas_landmarks": len(landmarks),
        **metrics,
        "_landmarks": landmarks,
        "_source_indices": source_indices,
        "_video_path": video_path,
    }


def _load_reference_records(
    namespace: dict[str, Any],
    reference_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for code, video_path in _ensure_reference_landmarks(namespace, reference_dir):
        landmarks_path = reference_dir / f"{code}_{video_path.stem}_landmarks.npz"
        landmarks, source_indices = _read_landmarks(landmarks_path)
        metadata = _video_metadata(video_path)
        records.append(
            _make_record(
                group="firmante25",
                code=code,
                participant="0025",
                repetition=video_path.stem,
                video_path=video_path,
                landmarks_path=landmarks_path,
                landmarks=landmarks,
                source_indices=source_indices,
                metadata=metadata,
            )
        )
    return records


def _save_catalog(catalog: pd.DataFrame, output_dir: Path) -> None:
    non_serializable = {"_landmarks", "_source_indices", "_video_path"}
    catalog.drop(columns=list(non_serializable)).to_csv(
        output_dir / "catalogo_comparacion.csv",
        index=False,
        encoding="utf-8-sig",
    )


def load_subject_comparison(
    namespace: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Load both signers, measure orientation proxies, and evaluate temporal speeds."""
    experiment_dir = Path(namespace["EXPERIMENT_DIR"])
    reference_dir = experiment_dir / REFERENCE_DIRNAME
    output_dir = experiment_dir / COMPARISON_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_records = _load_reference_records(namespace, reference_dir)
    own_records = _load_own_records(
        namespace,
        set(TARGET_LABELS),
    )
    if not own_records:
        raise ValueError("No hay videos propios para las tres señas de comparación.")

    catalog = pd.DataFrame(reference_records + own_records)
    for group in GROUP_LABELS:
        group_codes = set(catalog.loc[catalog["grupo"] == group, "codigo_real"])
        missing_codes = set(TARGET_LABELS) - group_codes
        if missing_codes:
            raise ValueError(f"Faltan señas para {GROUP_LABELS[group]}: {sorted(missing_codes)}")
    _save_catalog(catalog, output_dir)

    model_codes = [str(code) for code in namespace["class_codes"]]
    missing_model_codes = set(TARGET_LABELS) - set(model_codes)
    if missing_model_codes:
        raise ValueError(f"El checkpoint no contiene estas señas: {sorted(missing_model_codes)}")

    prediction_records: list[dict[str, Any]] = []
    for record in catalog.to_dict(orient="records"):
        true_index = model_codes.index(record["codigo_real"])
        for speed_factor in SPEED_FACTORS:
            stretched = _stretch_sequence(record["_landmarks"], speed_factor)
            probabilities, inference_ms = _predict(stretched, namespace)
            predicted_index = int(np.argmax(probabilities))
            top_indices = np.argsort(probabilities)[::-1][:3]
            prediction_records.append(
                {
                    "grupo": record["grupo"],
                    "grupo_nombre": record["grupo_nombre"],
                    "archivo": record["archivo"],
                    "codigo_real": record["codigo_real"],
                    "sena_real": record["sena"],
                    "factor_velocidad": speed_factor,
                    "duracion_s": record["duracion_s"],
                    "duracion_simulada_s": record["duracion_s"] / speed_factor,
                    "prediccion": model_codes[predicted_index],
                    "acierto": model_codes[predicted_index] == record["codigo_real"],
                    "confianza_top1": float(probabilities[predicted_index]),
                    "probabilidad_real": float(probabilities[true_index]),
                    "tiempo_inferencia_ms": inference_ms,
                    "top3": " | ".join(
                        f"{model_codes[index]}: {probabilities[index]:.1%}"
                        for index in top_indices
                    ),
                }
            )

    predictions = pd.DataFrame(prediction_records)
    predictions.to_csv(
        output_dir / "predicciones_comparacion_velocidad.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        f"Comparación lista: {len(reference_records)} videos de firmante 0025 y "
        f"{len(own_records)} videos propios. Resultados: {output_dir}"
    )
    return catalog, predictions, output_dir


def summarize_speed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions.groupby(["grupo", "grupo_nombre", "factor_velocidad"], sort=False)
        .agg(
            aciertos=("acierto", "sum"),
            total=("acierto", "count"),
            accuracy=("acierto", "mean"),
            probabilidad_real_media=("probabilidad_real", "mean"),
            confianza_top1_media=("confianza_top1", "mean"),
            duracion_media_s=("duracion_simulada_s", "mean"),
        )
        .reset_index()
    )


def plot_angle_diagnostics(
    catalog: pd.DataFrame,
    output_dir: Path,
) -> plt.Figure:
    codes = list(TARGET_LABELS)
    groups = list(GROUP_LABELS)
    x_positions = np.arange(len(codes), dtype=float)
    offsets = {"firmante25": -0.16, "propio": 0.16}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for group in groups:
        subset = catalog.loc[catalog["grupo"] == group]
        for code_index, code in enumerate(codes):
            samples = subset.loc[subset["codigo_real"] == code].sort_values("repeticion")
            jitter = np.linspace(-0.035, 0.035, max(len(samples), 1))
            x = x_positions[code_index] + offsets[group] + jitter[: len(samples)]
            axes[0].scatter(
                x,
                samples["proxy_oblicuidad_mediana"],
                color=GROUP_COLORS[group],
                label=GROUP_LABELS[group] if code_index == 0 else None,
                s=52,
                alpha=0.85,
            )
            axes[1].scatter(
                x,
                samples["ancho_hombros_altura_torso"],
                color=GROUP_COLORS[group],
                label=GROUP_LABELS[group] if code_index == 0 else None,
                s=52,
                alpha=0.85,
            )
            for x_value, (_, row) in zip(x, samples.iterrows()):
                axes[0].annotate(
                    str(row["repeticion"]),
                    (x_value, row["proxy_oblicuidad_mediana"]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
                axes[1].annotate(
                    str(row["repeticion"]),
                    (x_value, row["ancho_hombros_altura_torso"]),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )

    labels = [f"{TARGET_LABELS[code]}\n{code}" for code in codes]
    for axis in axes:
        axis.set_xticks(x_positions, labels)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    axes[0].set_title("Asimetría de profundidad hombros/caderas")
    axes[0].set_ylabel("Proxy relativo (no son grados)")
    axes[1].set_title("Ancho aparente de hombros / altura del torso")
    axes[1].set_ylabel("Razón en coordenadas corregidas por aspecto")
    fig.suptitle(
        "Orientación y encuadre por video — las repeticiones son puntos individuales",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "comparacion_proxy_orientacion.png", dpi=150, bbox_inches="tight")
    return fig


def _angle_extremes(catalog: pd.DataFrame, group: str) -> list[tuple[pd.Series, str]]:
    selected: list[tuple[pd.Series, str]] = []
    group_rows = catalog.loc[catalog["grupo"] == group]
    for code in TARGET_LABELS:
        samples = group_rows.loc[group_rows["codigo_real"] == code].sort_values(
            "proxy_oblicuidad_mediana"
        )
        if samples.empty:
            continue
        selected.append((samples.iloc[0], "Proxy menor"))
        selected.append((samples.iloc[-1], "Proxy mayor"))
    return selected


def plot_parallel_landmark_examples(
    catalog: pd.DataFrame,
    output_dir: Path,
) -> plt.Figure:
    examples_by_group = {
        group: _angle_extremes(catalog, group)
        for group in GROUP_LABELS
    }
    if any(len(examples) != 2 * len(TARGET_LABELS) for examples in examples_by_group.values()):
        raise ValueError("Se requieren ambos extremos del proxy para cada seña y grupo.")

    fig, axes = plt.subplots(
        2 * len(TARGET_LABELS),
        2 * len(LANDMARK_PHASES),
        figsize=(18, 12),
        squeeze=False,
    )
    for row_index in range(2 * len(TARGET_LABELS)):
        for group_index, group in enumerate(GROUP_LABELS):
            record, extreme_label = examples_by_group[group][row_index]
            indices = record["_source_indices"]
            landmarks = record["_landmarks"]
            video_path = record["_video_path"]
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"No se pudo abrir el video: {video_path}")
            try:
                for phase_index, phase in enumerate(LANDMARK_PHASES):
                    landmark_index = int(round(phase * (len(indices) - 1)))
                    frame_index = int(indices[landmark_index])
                    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    success, frame = capture.read()
                    if not success:
                        raise RuntimeError(
                            f"No se pudo leer el frame {frame_index} de {record['archivo']}."
                        )
                    image = _draw_landmarks(frame, landmarks[landmark_index])
                    image = ImageOps.pad(image, (280, 190), color=(245, 245, 245))
                    column_index = group_index * len(LANDMARK_PHASES) + phase_index
                    axis = axes[row_index, column_index]
                    axis.imshow(image)
                    axis.axis("off")
                    if row_index == 0:
                        axis.set_title(
                            f"{GROUP_LABELS[group]} · {phase:.0%}",
                            fontsize=9,
                        )
                    if phase_index == 0:
                        axis.text(
                            0.02,
                            0.03,
                            f"{record['codigo_real']} {record['sena']} · {extreme_label}\n"
                            f"{record['archivo']} · proxy={record['proxy_oblicuidad_mediana']:.3f}",
                            transform=axis.transAxes,
                            fontsize=7,
                            color="black",
                            va="bottom",
                            bbox={"facecolor": "white", "alpha": 0.82, "pad": 3},
                        )
            finally:
                capture.release()

    fig.suptitle(
        "Mismo gesto, landmarks en paralelo — extremos del proxy, no ángulos calibrados",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(
        output_dir / "landmarks_paralelo_firmante25_vs_propio.png",
        dpi=130,
        bbox_inches="tight",
    )
    return fig


def plot_speed_summary(
    summary: pd.DataFrame,
    output_dir: Path,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for group in GROUP_LABELS:
        rows = summary.loc[summary["grupo"] == group].sort_values("factor_velocidad")
        color = GROUP_COLORS[group]
        axes[0].plot(
            rows["factor_velocidad"],
            rows["accuracy"],
            marker="o",
            color=color,
            label=GROUP_LABELS[group],
        )
        axes[1].plot(
            rows["factor_velocidad"],
            rows["probabilidad_real_media"],
            marker="o",
            color=color,
            label=GROUP_LABELS[group],
        )
        for axis, column in (
            (axes[0], "accuracy"),
            (axes[1], "probabilidad_real_media"),
        ):
            for _, row in rows.iterrows():
                annotation = (
                    f"{int(row['aciertos'])}/{int(row['total'])}"
                    if column == "accuracy"
                    else f"{row[column]:.2f}"
                )
                axis.annotate(
                    annotation,
                    (row["factor_velocidad"], row[column]),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    for axis in axes:
        axis.set_xticks(SPEED_FACTORS, [f"{factor:g}x" for factor in SPEED_FACTORS])
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("Factor temporal simulado (1x = original)")
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_title("Accuracy por grupo")
    axes[0].set_ylabel("Fracción correcta")
    axes[1].set_title("Probabilidad media de la seña real")
    axes[1].set_ylabel("Probabilidad (no calibrada)")
    fig.suptitle(
        "Velocidad: secuencia remuestreada a 64 frames; no mide duración absoluta",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "comparacion_velocidad_resumen.png", dpi=150, bbox_inches="tight")
    return fig


def plot_speed_confusions(
    predictions: pd.DataFrame,
    output_dir: Path,
) -> plt.Figure:
    codes = list(TARGET_LABELS)
    matrix_columns = codes + ["OTRAS"]
    fig, axes = plt.subplots(
        2,
        len(SPEED_FACTORS),
        figsize=(14, 8),
        constrained_layout=True,
    )

    for row_index, group in enumerate(GROUP_LABELS):
        group_rows = predictions.loc[predictions["grupo"] == group]
        for column_index, speed_factor in enumerate(SPEED_FACTORS):
            subset = group_rows.loc[group_rows["factor_velocidad"] == speed_factor].copy()
            subset["prediccion_matriz"] = subset["prediccion"].where(
                subset["prediccion"].isin(codes), "OTRAS"
            )
            matrix = pd.crosstab(
                subset["codigo_real"],
                subset["prediccion_matriz"],
            ).reindex(index=codes, columns=matrix_columns, fill_value=0)
            matrix.to_csv(
                output_dir / f"matriz_{group}_{speed_factor:g}x.csv",
                encoding="utf-8-sig",
            )
            axis = axes[row_index, column_index]
            axis.imshow(matrix.to_numpy(), cmap="Blues", vmin=0)
            axis.set_xticks(
                np.arange(len(matrix_columns)),
                [*codes, "OTRAS"],
                rotation=45,
                ha="right",
            )
            axis.set_yticks(np.arange(len(codes)), codes)
            if row_index == len(GROUP_LABELS) - 1:
                axis.set_xlabel("Predicción")
            if column_index == 0:
                axis.set_ylabel(f"{GROUP_LABELS[group]}\nReal")
            axis.set_title(f"{speed_factor:g}x")
            for matrix_row in range(matrix.shape[0]):
                for matrix_column in range(matrix.shape[1]):
                    axis.text(
                        matrix_column,
                        matrix_row,
                        str(int(matrix.iat[matrix_row, matrix_column])),
                        ha="center",
                        va="center",
                        color="black",
                    )
    fig.suptitle("Matrices de confusión — OTRAS agrupa las otras 14 clases")
    fig.savefig(output_dir / "matrices_confusion_velocidad.png", dpi=150, bbox_inches="tight")
    return fig
