import argparse
import csv
import re
import zipfile
from pathlib import Path

import numpy as np

from landmark_recognition.data import FEATURE_COLUMNS, LANDMARK_COUNT, resample_sequence


ARCHIVE_ROOTS = {
    "pose": "LANDMARKS/BODY_LANDMARKS",
    "left_hand": "LANDMARKS/HANDS_LANDMARKS/LEFT_HAND_LANDMARKS",
    "right_hand": "LANDMARKS/HANDS_LANDMARKS/RIGHT_HAND_LANDMARKS",
}
ARCHIVE_NAME = re.compile(r"(?P<label>\d{4})_(?P<signer>\d{4})_(?P<take>\d{4})\.csv$")
DIRECTORY_ARCHIVE_NAME = re.compile(r"(?P<label>\d{4})_(?P<signer>\d{4})_(?P<take>\d{4})\.csv$")


def _read_archive_modality(archive: zipfile.ZipFile, path: str, count: int) -> np.ndarray:
    with archive.open(path) as raw_file:
        text_file = (line.decode("utf-8-sig") for line in raw_file)
        reader = csv.DictReader(text_file)
        required = [f"landmark_{index}_{axis}" for index in range(count) for axis in ("x", "y", "z")]
        missing = set(required).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing landmark columns, for example: {sorted(missing)[:3]}")
        rows = [[float(row[column]) for column in required] for row in reader]
    if not rows:
        raise ValueError(f"{path} contains no landmark frames.")
    return np.asarray(rows, dtype=np.float32)


def _archive_sequences(archive_path: Path):
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        stems = set()
        for path in names:
            if path.startswith(ARCHIVE_ROOTS["pose"] + "/") and path.endswith(".csv"):
                name_match = ARCHIVE_NAME.fullmatch(Path(path).name)
                if name_match:
                    stems.add((name_match.group(0)[:-4], name_match))

        if not stems:
            raise ValueError(
                "No files matching LANDMARKS/BODY_LANDMARKS/NNNN_NNNN_NNNN.csv were found."
            )

        for stem, match in sorted(stems):
            modalities = []
            for part, count in (
                ("pose", 33),
                ("left_hand", 21),
                ("right_hand", 21),
            ):
                member = f"{ARCHIVE_ROOTS[part]}/{stem}.csv"
                if member not in names:
                    raise ValueError(f"Missing aligned modality file in ZIP: {member}")
                modalities.append(_read_archive_modality(archive, member, count))

            frame_count = min(len(modality) for modality in modalities)
            if frame_count == 0:
                raise ValueError(f"Sequence {stem} has no aligned frames.")
            sequence = np.concatenate([modality[:frame_count] for modality in modalities], axis=1)
            yield match.group("label"), match.group("signer"), stem, sequence


def _directory_sequences(input_dir: Path):
    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        raise ValueError(f"No feature CSVs found under {input_dir}.")
    for path in csv_files:
        relative_path = path.relative_to(input_dir)
        if len(relative_path.parts) == 2:
            label = path.parent.name
            archive_name = DIRECTORY_ARCHIVE_NAME.fullmatch(path.name)
            if archive_name:
                if archive_name.group("label") != label:
                    raise ValueError(f"CSV label {archive_name.group('label')} does not match folder {label}: {path}")
                signer = archive_name.group("signer")
            else:
                signer = "unknown"
        elif len(relative_path.parts) == 3:
            signer = relative_path.parts[0]
            label = relative_path.parts[1]
        else:
            raise ValueError(
                f"CSV {path} must be organized as label/recording.csv or "
                "signer/label/recording.csv."
            )
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            missing = set(FEATURE_COLUMNS).difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing feature columns, for example: {sorted(missing)[:3]}")
            rows = [[float(row[column]) for column in FEATURE_COLUMNS] for row in reader]
        if not rows:
            raise ValueError(f"{path} contains no landmark frames.")
        yield label, signer, relative_path.as_posix(), np.asarray(rows, dtype=np.float32)


def prepare_dataset(input_path: Path, output_path: Path, frame_count: int) -> None:
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        sequences = _archive_sequences(input_path)
    elif input_path.is_dir():
        sequences = _directory_sequences(input_path)
    else:
        raise ValueError(f"Input must be a ZIP archive or a directory: {input_path}")

    records = list(sequences)
    if not records:
        raise ValueError("No landmark sequences were loaded.")
    classes = sorted({record[0] for record in records})
    class_to_index = {name: index for index, name in enumerate(classes)}
    features = np.stack([resample_sequence(record[3], frame_count) for record in records])
    features = features.reshape(len(records), frame_count, LANDMARK_COUNT, 3)
    if features.shape[2:] != (LANDMARK_COUNT, 3):
        raise ValueError(f"Unexpected feature shape: {features.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=features,
        y=np.asarray([class_to_index[record[0]] for record in records], dtype=np.int64),
        signers=np.asarray([record[1] for record in records], dtype=str),
        sample_ids=np.asarray([record[2] for record in records], dtype=str),
        class_names=np.asarray(classes, dtype=str),
    )
    print(f"Saved {len(records)} sequences, {len(classes)} classes, shape {features.shape} to {output_path}")
    print(f"Feature columns: {len(FEATURE_COLUMNS)} (33 pose + 21 left hand + 21 right hand, x/y/z)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert landmark CSVs or LANDMARKS.zip into a training dataset.")
    parser.add_argument("--input", type=Path, required=True, help="Path to LANDMARKS.zip or labeled CSV directory.")
    parser.add_argument("--output", type=Path, default=Path("data/landmarks_dataset.npz"))
    parser.add_argument("--frames", type=int, default=64, help="Fixed frame count after temporal resampling.")
    args = parser.parse_args()
    prepare_dataset(args.input, args.output, args.frames)


if __name__ == "__main__":
    main()
