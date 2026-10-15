import argparse
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from landmark_recognition.data import write_feature_csv
from landmark_recognition.mediapipe_support import holistic_result_to_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pose and hand landmarks from a video or webcam.")
    parser.add_argument("--source", default="0", help="Video file path or webcam index (default: 0).")
    parser.add_argument("--label", required=True, help="Class folder name for the recording.")
    parser.add_argument(
        "--signer",
        default="",
        help="Optional participant ID. When set, recordings are saved under signer/label.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data_mediapipe"))
    parser.add_argument("--display", action="store_true", help="Show the video and detected skeleton while extracting.")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    frames = []
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )
    drawing = mp.solutions.drawing_utils
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(holistic_result_to_array(result))
            if args.display:
                if result.pose_landmarks:
                    drawing.draw_landmarks(frame, result.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
                if result.left_hand_landmarks:
                    drawing.draw_landmarks(frame, result.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
                if result.right_hand_landmarks:
                    drawing.draw_landmarks(frame, result.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
                cv2.imshow("MediaPipe landmark extraction (press q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        holistic.close()
        if args.display:
            cv2.destroyAllWindows()

    if not frames:
        raise RuntimeError(f"No frames were read from source: {args.source}")
    output_dir = args.output_dir
    if args.signer:
        output_dir /= args.signer
    output = output_dir / args.label / f"{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    write_feature_csv(output, np.stack(frames))
    print(f"Saved {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
