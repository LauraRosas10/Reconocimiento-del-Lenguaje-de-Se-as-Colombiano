import argparse
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from landmark_recognition.inference import load_checkpoint_model, prepare_checkpoint_features
from landmark_recognition.class_labels import display_class_name
from landmark_recognition.mediapipe_support import holistic_result_to_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained landmark classifier on a video or webcam.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", default="0", help="Video file path or webcam index (default: 0).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    classes = checkpoint["class_names"]
    frame_count = checkpoint["frame_count"]
    print(
        f"Loaded {checkpoint['architecture']} checkpoint "
        f"({len(classes)} classes, {frame_count} frames, "
        f"signer-invariant={checkpoint.get('signer_invariant', False)})."
    )
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    history = deque(maxlen=frame_count)
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )
    drawing = mp.solutions.drawing_utils
    latest_prediction = "Collecting frames..."
    latest_confidence = 0.0
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            history.append(holistic_result_to_array(result))
            frame_index += 1

            if len(history) == frame_count and frame_index % 5 == 0:
                sequence = prepare_checkpoint_features(
                    np.asarray(history, dtype=np.float32),
                    checkpoint,
                )
                inputs = torch.from_numpy(sequence).unsqueeze(0).to(device)
                with torch.no_grad():
                    probabilities = torch.softmax(model(inputs), dim=1)[0]
                class_index = int(probabilities.argmax())
                latest_prediction = display_class_name(classes[class_index])
                latest_confidence = float(probabilities[class_index])

            if result.pose_landmarks:
                drawing.draw_landmarks(frame, result.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            if result.left_hand_landmarks:
                drawing.draw_landmarks(frame, result.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            if result.right_hand_landmarks:
                drawing.draw_landmarks(frame, result.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            cv2.putText(
                frame,
                f"{latest_prediction} ({latest_confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Landmark sign prediction (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        holistic.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
