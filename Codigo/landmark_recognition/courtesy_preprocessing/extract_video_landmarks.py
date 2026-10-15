import argparse
import csv
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import cv2
import mediapipe as mp
import numpy as np

from landmark_recognition.data import write_feature_csv
from landmark_recognition.mediapipe_support import holistic_result_to_array
from landmark_recognition.courtesy_preprocessing.class_mapping import (
    COMBINED_CLASSES,
    map_video_label,
)


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}
MANIFEST_FIELDS = (
    "video_member",
    "source_label",
    "class_name",
    "signer",
    "take",
    "sampled_frames",
    "landmark_csv",
    "status",
    "error",
    "processed_at",
)


def _eligible_videos(archive: zipfile.ZipFile):
    for member in sorted(archive.namelist()):
        path = PurePosixPath(member)
        if path.suffix.lower() not in VIDEO_EXTENSIONS or len(path.parts) < 5:
            continue
        class_name = map_video_label(path.parent.name)
        if class_name is None:
            continue
        signer = path.parts[1]
        take = path.stem
        if not signer.isdecimal() or not take.isdecimal():
            raise ValueError(f"Expected numeric participant and take in video path: {member}")
        yield member, path.parent.name, class_name, f"{int(signer):04d}", f"{int(take):04d}"


def _extract_sampled_frames(video_path: Path, max_frames: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise RuntimeError(f"Video has no readable frame-count metadata: {video_path}")
        positions = np.unique(
            np.linspace(0, frame_count - 1, min(frame_count, max_frames)).round().astype(np.int64)
        )
        frames = []
        with mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
        ) as holistic:
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(
                        f"Could not decode frame {int(position)} of {frame_count} in {video_path}"
                    )
                result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames.append(holistic_result_to_array(result))
        if not frames:
            raise RuntimeError(f"No frames were extracted from {video_path}")
        return np.stack(frames).astype(np.float32, copy=False)
    finally:
        capture.release()


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    limit: int | None = None,
    overwrite: bool = False,
    max_frames: int = 96,
) -> Path:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Video archive does not exist: {archive_path}")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2.")

    landmarks_dir = output_dir / "landmarks_csv"
    landmarks_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "extraction_manifest.csv"
    failures = []
    processed = 0
    seen_outputs = set()
    with zipfile.ZipFile(archive_path) as archive, manifest_path.open(
        "w", encoding="utf-8", newline=""
    ) as manifest_file:
        manifest = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        manifest.writeheader()
        candidates = _eligible_videos(archive)
        for candidate_index, candidate in enumerate(candidates):
            if limit is not None and candidate_index >= limit:
                break
            member, source_label, class_name, signer, take = candidate
            if class_name not in COMBINED_CLASSES:
                raise ValueError(f"Unregistered output class {class_name!r} for {member}")
            output_path = landmarks_dir / class_name / f"{class_name}_{signer}_{take}.csv"
            if output_path in seen_outputs:
                raise ValueError(
                    f"Two videos map to the same class/signer/take filename: {output_path}. "
                    "Inspect the archive's class-folder structure before extracting."
                )
            seen_outputs.add(output_path)
            row = {
                "video_member": member,
                "source_label": source_label,
                "class_name": class_name,
                "signer": signer,
                "take": take,
                "sampled_frames": 0,
                "landmark_csv": str(output_path),
                "status": "",
                "error": "",
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
            if output_path.is_file() and not overwrite:
                row["status"] = "skipped_existing"
                manifest.writerow(row)
                continue

            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory() as temporary_dir:
                    temporary_path = Path(temporary_dir) / f"video{PurePosixPath(member).suffix}"
                    with archive.open(member) as source:
                        with temporary_path.open("wb") as temporary_video:
                            shutil.copyfileobj(source, temporary_video)
                    frames = _extract_sampled_frames(temporary_path, max_frames)
                write_feature_csv(output_path, frames)
                row["sampled_frames"] = len(frames)
                row["status"] = "ok"
                processed += 1
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                row["status"] = "error"
                row["error"] = f"{type(error).__name__}: {error}"
                failures.append((member, row["error"]))
            manifest.writerow(row)
            if (candidate_index + 1) % 25 == 0:
                print(f"Inspected {candidate_index + 1} selected videos; extracted {processed}.")

    print(f"Extracted {processed} videos. Manifest: {manifest_path}")
    if failures:
        examples = "\n".join(f"- {member}: {error}" for member, error in failures[:10])
        raise RuntimeError(
            f"{len(failures)} video(s) failed extraction. Review {manifest_path}.\n{examples}"
        )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe pose and hand landmarks from selected Courtesy videos in a ZIP."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("landmark_recognition/courtesy_preprocessing/outputs"),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for a small extraction test.")
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract_archive(args.archive, args.output_dir, args.limit, args.overwrite, args.max_frames)


if __name__ == "__main__":
    main()
