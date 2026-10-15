import numpy as np

from landmark_recognition.data import HAND_LANDMARKS, LANDMARK_COUNT, POSE_LANDMARKS


def holistic_result_to_array(result) -> np.ndarray:
    landmarks = np.zeros((LANDMARK_COUNT, 3), dtype=np.float32)
    for destination, detected, expected_count in (
        (0, result.pose_landmarks, POSE_LANDMARKS),
        (POSE_LANDMARKS, result.left_hand_landmarks, HAND_LANDMARKS),
        (POSE_LANDMARKS + HAND_LANDMARKS, result.right_hand_landmarks, HAND_LANDMARKS),
    ):
        if detected is None:
            continue
        for index, point in enumerate(detected.landmark[:expected_count]):
            landmarks[destination + index] = (point.x, point.y, point.z)
    return landmarks
