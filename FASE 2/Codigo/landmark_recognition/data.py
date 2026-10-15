import csv
from pathlib import Path

import numpy as np


POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
LANDMARK_COUNT = POSE_LANDMARKS + 2 * HAND_LANDMARKS
COORDINATES = ("x", "y", "z")
FEATURE_COLUMNS = [
    f"{part}_{index}_{axis}"
    for part, count in (
        ("pose", POSE_LANDMARKS),
        ("left_hand", HAND_LANDMARKS),
        ("right_hand", HAND_LANDMARKS),
    )
    for index in range(count)
    for axis in COORDINATES
]


def resample_sequence(sequence: np.ndarray, frame_count: int) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] != LANDMARK_COUNT * len(COORDINATES):
        raise ValueError(
            f"Expected a sequence shaped [frames, {LANDMARK_COUNT * len(COORDINATES)}], "
            f"received {sequence.shape}."
        )
    if sequence.shape[0] == 0:
        raise ValueError("Cannot resample an empty landmark sequence.")
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1.")

    sequence = np.nan_to_num(sequence, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if sequence.shape[0] == frame_count:
        return sequence.astype(np.float32, copy=False)

    old_positions = np.linspace(0.0, 1.0, num=sequence.shape[0])
    new_positions = np.linspace(0.0, 1.0, num=frame_count)
    return np.stack(
        [np.interp(new_positions, old_positions, sequence[:, column]) for column in range(sequence.shape[1])],
        axis=1,
    ).astype(np.float32)


def normalize_landmark_sequence(
    sequence: np.ndarray,
    wrist_relative_hands: bool = False,
    decouple_hands: bool = False,
    preserve_zeros: bool = True,
) -> np.ndarray:
    if sequence.ndim != 3 or sequence.shape[1:] != (LANDMARK_COUNT, len(COORDINATES)):
        raise ValueError(
            f"Expected a sequence shaped [frames, {LANDMARK_COUNT}, 3], received {sequence.shape}."
        )
    if sequence.shape[0] == 0:
        raise ValueError("Cannot normalize an empty landmark sequence.")

    normalized = np.nan_to_num(sequence, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    valid_mask = (normalized != 0.0).any(axis=-1)
    shoulder_midpoint = (normalized[:, 11] + normalized[:, 12]) * 0.5
    shoulder_width = np.linalg.norm(normalized[:, 11, :2] - normalized[:, 12, :2], axis=1)
    scale = float(np.median(shoulder_width))
    if not np.isfinite(scale) or scale < 1e-4:
        visible_points = normalized[:, :POSE_LANDMARKS, :2]
        scale = float(np.median(np.linalg.norm(visible_points - visible_points.mean(axis=1, keepdims=True), axis=2)))
    scale = max(scale, 1e-4)

    if wrist_relative_hands or decouple_hands:
        left_start = POSE_LANDMARKS
        right_start = left_start + HAND_LANDMARKS
        normalized[:, :POSE_LANDMARKS] -= shoulder_midpoint[:, None, :]
        normalized[:, :POSE_LANDMARKS] /= scale
        for start in (left_start, right_start):
            end = start + HAND_LANDMARKS
            for frame_index in range(normalized.shape[0]):
                if valid_mask[frame_index, start:end].any():
                    wrist = sequence[frame_index, start].copy()
                    normalized[frame_index, start] = (wrist - shoulder_midpoint[frame_index]) / scale
                    normalized[frame_index, start + 1:end] = (
                        sequence[frame_index, start + 1:end] - wrist
                    ) / scale
    else:
        normalized -= shoulder_midpoint[:, None, :]
        normalized /= scale
    if preserve_zeros:
        normalized[~valid_mask] = 0.0
    return normalized


def compute_motion_features(sequence: np.ndarray) -> np.ndarray:
    """Computes first-order temporal velocity (delta) between consecutive frames."""
    velocities = np.zeros_like(sequence)
    velocities[1:] = sequence[1:] - sequence[:-1]
    return velocities.astype(np.float32, copy=False)


def prepare_features(sequence: np.ndarray, include_velocity: bool = True) -> np.ndarray:
    """Flattens landmark coordinates and optionally appends temporal velocities."""
    flattened = sequence.reshape(sequence.shape[0], -1).astype(np.float32, copy=False)
    if not include_velocity:
        return flattened
    return np.concatenate((flattened, compute_motion_features(flattened)), axis=-1)


def augment_landmark_sequence(
    sequence: np.ndarray,
    rng: np.random.Generator | None = None,
    rotation_degrees: float = 10.0,
    scale_range: tuple[float, float] = (0.9, 1.1),
    translation: float = 0.05,
    noise_std: float = 0.008,
    temporal_warp_probability: float = 0.5,
    landmark_dropout_probability: float = 0.03,
    frame_dropout_probability: float = 0.1,
) -> np.ndarray:
    if sequence.ndim != 3 or sequence.shape[1:] != (LANDMARK_COUNT, len(COORDINATES)):
        raise ValueError(
            f"Expected a sequence shaped [frames, {LANDMARK_COUNT}, 3], received {sequence.shape}."
        )
    if sequence.shape[0] < 2:
        raise ValueError("At least two frames are required for landmark augmentation.")
    if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
        raise ValueError("scale_range must contain positive values in ascending order.")
    if rotation_degrees < 0 or translation < 0 or noise_std < 0:
        raise ValueError("Rotation, translation, and noise magnitudes must be non-negative.")
    if not 0.0 <= temporal_warp_probability <= 1.0:
        raise ValueError("temporal_warp_probability must be between 0 and 1.")
    if not 0.0 <= landmark_dropout_probability <= 1.0:
        raise ValueError("landmark_dropout_probability must be between 0 and 1.")
    if not 0.0 <= frame_dropout_probability <= 1.0:
        raise ValueError("frame_dropout_probability must be between 0 and 1.")

    generator = rng or np.random.default_rng()
    augmented = np.nan_to_num(sequence, copy=True, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    frame_count = augmented.shape[0]

    if generator.random() < temporal_warp_probability:
        knot_count = min(5, frame_count)
        source_knots = np.linspace(0.0, frame_count - 1, knot_count)
        segment_lengths = generator.uniform(0.8, 1.2, size=knot_count - 1)
        warped_knots = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        warped_knots *= (frame_count - 1) / warped_knots[-1]
        target_positions = np.linspace(0.0, frame_count - 1, frame_count)
        source_positions = np.interp(target_positions, warped_knots, source_knots)
        lower = np.floor(source_positions).astype(np.intp)
        upper = np.minimum(lower + 1, frame_count - 1)
        weights = (source_positions - lower).astype(np.float32)[:, None, None]
        augmented = augmented[lower] * (1.0 - weights) + augmented[upper] * weights

    if generator.random() < frame_dropout_probability:
        gap_length = int(generator.integers(1, max(2, min(5, frame_count // 8 + 1))))
        start = int(generator.integers(0, frame_count - gap_length + 1))
        left = max(0, start - 1)
        right = min(frame_count - 1, start + gap_length)
        if right > left:
            for frame_index in range(start, start + gap_length):
                fraction = (frame_index - left) / (right - left)
                augmented[frame_index] = (
                    augmented[left] * (1.0 - fraction) + augmented[right] * fraction
                )

    angle = np.deg2rad(generator.uniform(-rotation_degrees, rotation_degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    spatial_scale = generator.uniform(*scale_range)
    augmented[..., :2] = (augmented[..., :2] @ rotation.T) * spatial_scale
    augmented[..., 2] *= spatial_scale
    augmented[..., 0] += generator.uniform(-translation, translation)
    augmented[..., 1] += generator.uniform(-translation, translation)
    if noise_std:
        augmented += generator.normal(0.0, noise_std, size=augmented.shape).astype(np.float32)
    if landmark_dropout_probability:
        dropped = generator.random(augmented.shape[1]) < landmark_dropout_probability
        augmented[:, dropped] = 0.0
    return augmented.astype(np.float32, copy=False)


def read_feature_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = set(FEATURE_COLUMNS).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing feature columns, for example: {sorted(missing)[:3]}")
        rows = [
            [float(row[column]) for column in FEATURE_COLUMNS]
            for row in reader
        ]
    if not rows:
        raise ValueError(f"{path} contains no landmark frames.")
    return np.asarray(rows, dtype=np.float32)


def write_feature_csv(path: Path, sequence: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", *FEATURE_COLUMNS])
        for frame_index, frame in enumerate(sequence):
            writer.writerow([frame_index, *frame.reshape(-1).tolist()])
